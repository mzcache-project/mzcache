#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#if defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

/*--------------------------------- hyper-params --------------------------------
 * One source covers every configuration; the host injects the tuning constants
 * as build options (see the mzcache flash-attn block in ggml-opencl.cpp):
 *   -DD=64|128        head dim (model-dependent: EXAONE4=64, Qwen3=128)
 *   -DNWARPS=2|4      work-group warps   (Adreno 7xx=2, 8xx=4)
 *   -DKQ_TILE=64|128  keys per iteration (Adreno 7xx=64, 8xx=128)
 * NWARPS/KQ_TILE must match the host launch config (get_kernel_config), and
 * the defaults below only make the file self-contained for offline syntax
 * checking. */
#define WARP      64                /* warp size, 64 threads per warp              */
#ifndef NWARPS
#define NWARPS     4                 /* work-group y warps                   */
#endif
#define WG_SIZE  (WARP*NWARPS)

#ifndef D
#define D         128                /* head-dim, compile-time constant      */
#endif
#define NCOLS      8                /* query cols per tile                  */
#ifndef KQ_TILE
#define KQ_TILE    128                /* keys processed per iteration         */
#endif

#define HALF_MAX_HALF ((half)32752.0h)


/* --------------------------------svm params--------------------------------- */
#define TOKENS_PER_CHUNK 256
#define MAX_CHUNKS_PER_TENSOR 256

typedef struct {
    __global void *c[MAX_CHUNKS_PER_TENSOR];
} chunk_ptrs;


/*--------------------------------- helpers ------------------------------------*/
inline half hmax2(half a, half b) { return a > b ? a : b; }

inline float get_alibi_slope(
    const float max_bias,
    const uint h,
    const uint n_head_log2,
    const float m0,
    const float m1
) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }

    float base = (h < n_head_log2) ? m0 : m1;
    int   exph = (h < n_head_log2) ? (int)(h + 1) : (int)(2 * (h - n_head_log2) + 1);

    return pow(base, (float)exph);
}

inline half reduce8(half8 v) {

    return v.s0 + v.s1 + v.s2 + v.s3 +
           v.s4 + v.s5 + v.s6 + v.s7;
}


/*--------------------------------- kernel -------------------------------------*/
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
__kernel void kernel_mzcache_flash_attn_f16_kvchunk(
    /* base pointers + offsets --------------------------------------------------*/
    __global char  *Q,    ulong offQ,
    __global char  *K_chunks,
    __global char  *V_chunks,
    __local  half  *KQ,
    __global char  *mask, ulong offM,
    __global char  *dst,  ulong offDst,

    /* op params --------------------------------------------------------------- */
    const float scale,
    const float max_bias,  const float m0,   const float m1,      /* unused   */
    const uint  n_head_log2,               const float logit_softcap, /* =1    */

    /* shapes / strides -------------------------------------------------------- */
    const int ne00, const int ne01, const int ne02, const int ne03,   /* Q */
    const int ne10, const int ne11, const int ne12, const int ne13,   /* K/V */
    const int ne31, const ulong nb31,
    const ulong nb01, const ulong nb02, const ulong nb03,             /* Q nb */
    const ulong nb11, const ulong nb12, const ulong nb13,             /* K nb */
    const ulong nb21, const ulong nb22, const ulong nb23,             /* V nb */
    const int ne0,  const int ne1,  const int ne2,  const int ne3)     /* dst */
{
    /*---------------------- apply base-offsets --------------------------------*/
    Q = (global char *)((global char *)Q + offQ);
    mask = (global char *)((global char *)mask + offM);
    dst = (global char *)((global char *)dst + offDst);


    /*---------------------- thread / group indices ----------------------------*/
    const int lane = get_local_id(0);                /* 0‥31               */
    const int warp = get_local_id(1);                /* 0‥3                */
    const int head = get_group_id(2);                /* 0‥ne02-1 (16)      */

    /* query-tile grid: work-group x handles queries [ic0, ic0+NCOLS); the host
     * launches ceil(nq/NCOLS) groups in x, so any nq works (last tile ragged) */
    const int ic0  = get_group_id(0) * NCOLS;

    /*---------------------- element strides (in elements) ---------------------*/
    const int sQ1 = nb01 / sizeof(float2);   /* 8192 / sizeof(float2) = 64      */
    const int sKV = nb11 / sizeof(half2);    /* 2048 / sizeof(half2) = 32    */
#if D == 64
    const int sV  = nb11 / sizeof(half);     /* 2048 / sizeof(half) = 64      */
#endif
    const int sKV8 = sKV >> 2;
    const int sM  = nb31 / sizeof(float);

    /* grouped-query attention pointer adjust */
    const int gqa_ratio = ne02 / ne12;       /* 16 / 8 = 2 */

    global float2 *Qf2 = (global float2 *)(Q + nb02 * head);
    global const half *Mh = (global const half *)mask + ne11*ic0;
    global float *dst_f = (global float *)dst;


    const float slopef = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    /* ---- per-column accumulators (one “register file” per warp group) ---- */
    half  kqmax[NCOLS/NWARPS];
    half  kqsum[NCOLS/NWARPS];
#if D == 64
    /* D=64: one scalar half register per column (D/WARP == 1) */
    half  VKQ [NCOLS/NWARPS][D/WARP];
#else
    /* D=128: one half2 register per column (D/2/WARP == 1) */
    half2 VKQ [NCOLS/NWARPS][D/2/WARP];
#endif

    /* ── per-column accumulator init (every element, not just [0]) ────────── */
    for (int i = 0; i < NCOLS / NWARPS; ++i) {
        kqmax[i] = (half)(-HALF_MAX_HALF);              // == CUDA initial value
        kqsum[i] = (half)0.0f;

#if D == 64
        for (int j = 0; j < D / WARP; ++j)
            VKQ[i][j] = 0.0h;
#else
        for (int j = 0; j < D / 2 / WARP; ++j)
            VKQ[i][j] = (half2)(0.0h, 0.0h);
#endif
    }


    /*====================== 2. iterate over K chunks ==========================*/
    for(int chunk_idx = 0; chunk_idx < ne11/TOKENS_PER_CHUNK; chunk_idx += 1) {
        global char *K = (global char *)(((chunk_ptrs *)K_chunks)->c[chunk_idx]);
        global char *V = (global char *)((chunk_ptrs *)V_chunks)->c[chunk_idx];

        global half8 *Kh8 = (global half8*)(K + nb12 * (head / gqa_ratio));
#if D == 64
        global half   *Vh = (global half   *)(V + nb12 * (head / gqa_ratio));
#else
        global half2  *Vh2 = (global half2 *)(V + nb12 * (head / gqa_ratio));
#endif

        // 2. iterate over K chunks
        for(int k_VKQ_0 = 0; k_VKQ_0 < TOKENS_PER_CHUNK; k_VKQ_0 += KQ_TILE) {
            //
            // 2-A. init kqmax_new = previous kqmax
            //
            half kqmax_new[NCOLS/NWARPS];
            for(int jg = 0; jg < NCOLS/NWARPS; ++jg){
                kqmax_new[jg] = kqmax[jg];
            }

            half8 sum8[KQ_TILE/WARP][NCOLS/NWARPS];
            for(int i = 0; i < KQ_TILE/WARP; ++i){
                for(int jg = 0; jg < NCOLS/NWARPS; ++jg){
                    sum8[i][jg] = (half8)(0,0,0,0,0,0,0,0);
                }
            }

            //
            // 2-C. compute logits = Q·K + mask*slope
            for(int k8 = 0; k8 < D/8; ++k8) {
                half8 K_k[KQ_TILE/WARP];
                half8 Q_k[NCOLS/NWARPS];

                for(int i_KQ_0 = 0; i_KQ_0 < KQ_TILE; i_KQ_0 += WARP){
                    const int i_KQ = i_KQ_0 + lane;
                    K_k[i_KQ_0/WARP] = Kh8[(k_VKQ_0 + i_KQ)*sKV8 + k8];
                }

                for (int j0 = 0; j0 < NCOLS; j0 += NWARPS) {
                    int j = j0 + warp;
                    ushort valid = (ushort)(ic0 + j < ne01);
                    /* ragged last tile: clamp the row so the load stays in-bounds,
                     * the value is zeroed via `valid` anyway */
                    const int jq = min(ic0 + j, ne01 - 1);
                    float8 qf8 = vload8(0, (global float*)(&Qf2[jq*sQ1 + 4*k8])); // 2×float2
                    float8 qf8_masked = qf8 * (float8)(valid);
                    Q_k[j0/NWARPS] = convert_half8(qf8_masked * scale);
                }

                for(int i_KQ_0 = 0; i_KQ_0 < KQ_TILE; i_KQ_0 += WARP){
                    for(int j_KQ_0 = 0; j_KQ_0 < NCOLS; j_KQ_0 += NWARPS){
                        sum8[i_KQ_0/WARP][j_KQ_0/NWARPS] +=
                            K_k[i_KQ_0/WARP] * Q_k[j_KQ_0/NWARPS];
                    }
                }
            }

            for(int i_KQ_0 = 0; i_KQ_0 < KQ_TILE; i_KQ_0 += WARP){
                const int i_KQ = i_KQ_0 + lane;
                for(int j_KQ_0 = 0; j_KQ_0 < NCOLS; j_KQ_0 += NWARPS){
                    const int j_KQ = j_KQ_0 + warp;
                    half sum = reduce8(sum8[i_KQ_0/WARP][j_KQ_0/NWARPS]);
                    // apply mask & slope
                    if(Mh) {
                        int gk = chunk_idx * TOKENS_PER_CHUNK + k_VKQ_0 + i_KQ;
                        half m_val = Mh[j_KQ * ne11 + gk];
                        sum += (slopef * (half)m_val);
                    }
                    // update kqmax_new
                    kqmax_new[j_KQ_0/NWARPS] = hmax2(kqmax_new[j_KQ_0/NWARPS], sum);

                    KQ[j_KQ*KQ_TILE + i_KQ] = sum;
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            //
            // 2-D. re-scale previous accumulators to new max, then softmax & V-accum
            //
            for(int j0 = 0; j0 < NCOLS; j0 += NWARPS) {
                int j = j0 + warp;

                // warp_reduce_max
                kqmax_new[j0 / NWARPS] = sub_group_reduce_max(kqmax_new[j0 / NWARPS]);

                // only lane 0 needs to compute the scale factor
                half old_max = kqmax[j0/NWARPS];
                half new_max = kqmax_new[j0/NWARPS];
                // exp(old_max - new_max) packed into half2
                half KQ_max_scale = (half)native_exp((float)(old_max - new_max));
                // update kqmax
                kqmax[j0/NWARPS] = new_max;
                // re-scale previous rowsum

                float kqsum_add = 0.0f;
                for(int i = lane; i < KQ_TILE; i += WARP){

                    const half diff = KQ[j*KQ_TILE + i] - new_max;
                    const half val = (half)native_exp((float)diff);
                    kqsum_add += (float)val;

                    KQ[j*KQ_TILE + i] = val;
                }
                kqsum[j0/NWARPS] = kqsum[j0/NWARPS] * KQ_max_scale + kqsum_add;

#if D == 64
                for(int i0 = 0; i0 < D; i0 += WARP){
                    VKQ[j0/NWARPS][i0/WARP] *= KQ_max_scale;
                }
#else
                for(int i0 = 0; i0 < D/2; i0 += WARP){
                    VKQ[j0/NWARPS][i0/WARP] *= KQ_max_scale;
                }
#endif
            }

#if D == 64
            // 2-F. accumulate V with softmax weights (scalar V, 4 keys/step)
            for(int k0 = 0; k0 < KQ_TILE; k0 += 4) {
                half V_k[D/WARP][4];
                half4 KQ_k[NCOLS/NWARPS];

                for(int i0 = 0; i0 < D; i0 += WARP){
                    const int i = i0 + lane;

                    V_k[i0/WARP][0] = Vh[(k_VKQ_0 + k0    )*sV + i];
                    V_k[i0/WARP][1] = Vh[(k_VKQ_0 + k0 + 1)*sV + i];
                    V_k[i0/WARP][2] = Vh[(k_VKQ_0 + k0 + 2)*sV + i];
                    V_k[i0/WARP][3] = Vh[(k_VKQ_0 + k0 + 3)*sV + i];
                }

                for(int j0 = 0; j0 < NCOLS; j0 += NWARPS){
                    const int j = j0 + warp;

                    KQ_k[j0/NWARPS] = (half4)(KQ[j*KQ_TILE + k0], KQ[j*KQ_TILE + k0 + 1],
                                               KQ[j*KQ_TILE + k0 + 2], KQ[j*KQ_TILE + k0 + 3]);
                }

                for(int i0 = 0; i0 < D; i0 += WARP){
                    for(int j0 = 0; j0 < NCOLS; j0 += NWARPS){
                        const int i = i0 + lane;
                        const int j = j0 + warp;

                        // accumulate V with softmax weights
                        VKQ[j0/NWARPS][i0/WARP] += V_k[i0/WARP][0] * KQ_k[j0/NWARPS].x;
                        VKQ[j0/NWARPS][i0/WARP] += V_k[i0/WARP][1] * KQ_k[j0/NWARPS].y;
                        VKQ[j0/NWARPS][i0/WARP] += V_k[i0/WARP][2] * KQ_k[j0/NWARPS].z;
                        VKQ[j0/NWARPS][i0/WARP] += V_k[i0/WARP][3] * KQ_k[j0/NWARPS].w;
                    }
                }
            }
#else
            // 2-F. accumulate V with softmax weights (half2 V, 2 keys/step)
            for(int k0 = 0; k0 < KQ_TILE; k0 += 2) {
                half2 V_k[(D/2)/WARP][2];
                half2 KQ_k[NCOLS/NWARPS];

                for(int i0 = 0; i0 < D/2; i0 += WARP){
                    const int i = i0 + lane;

                    V_k[i0/WARP][0] = Vh2[(k_VKQ_0 + k0    )*sKV + i];
                    V_k[i0/WARP][1] = Vh2[(k_VKQ_0 + k0 + 1)*sKV + i];
                }

                for(int j0 = 0; j0 < NCOLS; j0 += NWARPS){
                    const int j = j0 + warp;

                    KQ_k[j0/NWARPS] = (half2)(KQ[j*KQ_TILE + k0], KQ[j*KQ_TILE + k0 + 1]);
                }

                for(int i0 = 0; i0 < D/2; i0 += WARP){
                    for(int j0 = 0; j0 < NCOLS; j0 += NWARPS){
                        const int i = i0 + lane;
                        const int j = j0 + warp;

                        // accumulate V with softmax weights
                        VKQ[j0/NWARPS][i0/WARP] += V_k[i0/WARP][0] * KQ_k[j0/NWARPS].x;
                        VKQ[j0/NWARPS][i0/WARP] += V_k[i0/WARP][1] * KQ_k[j0/NWARPS].y;
                    }
                }
            }
#endif
        } // end K-chunk loop
    }

    /*====================== 3. write results ==================================*/
    for(int j_VKQ_0 = 0; j_VKQ_0 < NCOLS; j_VKQ_0 += NWARPS) {
        const int j_VKQ = j_VKQ_0 + warp;

        if(ic0 + j_VKQ >= ne01) {
            // if this column is out of bounds, skip it
            return;
        }

        half kqsum_j = sub_group_reduce_add((float)kqsum[j_VKQ_0/NWARPS]);

#if D == 64
        for(int i00 = 0; i00 < D; i00 += WARP) {
            const int i0 = i00 + lane;

            if(i0 < D) {
                VKQ[j_VKQ_0/NWARPS][i0/WARP] /= kqsum_j;

                const int j_dst = ic0 + j_VKQ;
                dst_f[j_dst * D * ne02 + D * head + i0] = (float)VKQ[j_VKQ_0/NWARPS][i0/WARP];
            }
        }
#else
        for(int i00 = 0; i00 < D; i00 += 2*WARP) {
            const int i0 = i00 + 2*lane;

            if(i0 < D) {
                VKQ[j_VKQ_0/NWARPS][i0/(2*WARP)] /= kqsum_j;

                const int j_dst = ic0 + j_VKQ;
                dst_f[j_dst * D * ne02 + D * head + i0] = (float)VKQ[j_VKQ_0/NWARPS][i0/(2*WARP)].x;
                dst_f[j_dst * D * ne02 + D * head + i0 + 1] = (float)VKQ[j_VKQ_0/NWARPS][i0/(2*WARP)].y;
            }
        }
#endif
    }
}
