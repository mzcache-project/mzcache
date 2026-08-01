#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#if defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

/*--------------------------------- hyper-params --------------------------------*/
#define WARP      64                /* warp size, 64 threads per warp              */
#define NWARPS     4                 /* work-group y = 4 warps               */
#define WG_SIZE  (WARP*NWARPS)       /* 128 threads                          */

#define D         64                /* head-dim, compile-time constant      */
#define NCOLS      16                /* query cols per tile                  */
#define KQ_TILE    128                /* keys processed per iteration         */

#define HALF_MAX_HALF ((half)32752.0h) 

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


/*--------------------------------- kernel -------------------------------------*/
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
__kernel void kernel_mzcache_flash_attn_f16_nolocalKV_D64(
    /* base pointers + offsets --------------------------------------------------*/
    __global char  *Q,    ulong offQ,
    __global char  *K,    ulong offK,
    __global char  *V,    ulong offV,
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
    K = (global char *)((global char *)K + offK);
    V = (global char *)((global char *)V + offV);
    mask = (global char *)((global char *)mask + offM);
    dst = (global char *)((global char *)dst + offDst);



    // printf("ne00, ne01, ne02, ne03: %d, %d, %d, %d\n",
    //        ne00, ne01, ne02, ne03);

    // printf("ne10, ne11, ne12, ne13: %d, %d, %d, %d\n",
    //        ne10, ne11, ne12, ne13);
    
    // printf("ne31, nb31: %d, %lu\n", ne31, nb31);

    // printf("nb01, nb02, nb03: %lu, %lu, %lu\n",
    //        nb01, nb02, nb03);
    
    // printf("nb11, nb12, nb13: %lu, %lu, %lu\n",
    //        nb11, nb12, nb13);

    // printf("nb21, nb22, nb23: %lu, %lu, %lu\n",
    //        nb21, nb22, nb23);
           
    // printf("ne0, ne1, ne2, ne3: %d, %d, %d, %d\n",
    //        ne0, ne1, ne2, ne3);


    /*---------------------- thread / group indices ----------------------------*/
    const int lane = get_local_id(0);                /* 0‥31               */
    const int warp = get_local_id(1);                /* 0‥3                */
    const int head = get_group_id(2);                /* 0‥ne02-1 (16)      */

    const int ic0  = 0;    /* this kernel is built for the single-tile (nq≤16) */

    /*---------------------- element strides (in elements) ---------------------*/
    const int sQ1 = nb01 / sizeof(float2);   /* stride in q-rows     (1024)   */
    const int sQ2 = nb02 / sizeof(float2);   /* stride per head      (  64)   */
    const int sK  = nb11 / sizeof(half2);    /* stride in k/v rows   ( 512)   */
    const int sV  = nb11 / sizeof(half);     /* stride in v rows     (1024)   */
    const int sM  = nb31 / sizeof(float);    /* stride between q rows (nkv)   */

    /* grouped-query attention pointer adjust */
    const int gqa_ratio = ne02 / ne12;       /* 16 / 8 = 2 */

    global float2 *Qf2 = (global float2 *)(Q + nb02 * head);
    global half2  *Kh2 = (global half2 *)(K + nb12 * (head / gqa_ratio));
    global half  *Vh = (global half *)(V + nb12 * (head / gqa_ratio));
    global const half *Mh = (global const half *)mask + ne11*ic0;
    global float *dst_f = (global float *)dst;


    const float slopef = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);


    /*---------------------- shared buffers ------------------------------------*/
    __local half Ql[NCOLS][D];           /* 4 KB  */
    // __local half2 KV[KQ_TILE][D/2+1];       /* 16.6 KB */
    __local half  KQ[NCOLS*KQ_TILE];        /* 2 KB  */
    // __local half  red[WG_SIZE];             /* 256 B */

    /* ---- per-column accumulators (one “register file” per warp group) ---- */
    half  kqmax[NCOLS/NWARPS];
    half  kqsum[NCOLS/NWARPS];
    half VKQ [NCOLS/NWARPS][D/WARP]; // (16/4 , 64/32) = (4, 2) registers per column

    /* ── thread-cooperative zeroing & min-init ───────────────────────────── */
    for (int i = 0; i < NCOLS / NWARPS; i += WARP) {
        kqmax[i] = (half)(-HALF_MAX_HALF);           
        kqsum[i] = 0.0h; // init to 0.0h

        /* VKQ has two half2 registers per column (D/2 == 64, 64/WARP == 2) */
        for (int j = 0; j < D / WARP; ++j)
            VKQ[i][j] = 0.0h; //
    }

    /*---------------------- 1. load & scale Q ---------------------------------*/
    for(int j=warp; j<NCOLS; j+=NWARPS)
        for(int i=lane; i<D/2; i+=WARP){
            float2 v = (ic0+j < ne01) ? Qf2[j*sQ1 + i] : (float2)(0,0);
            // Ql[j][i] = (half2)(v.x * scale, v.y * scale);
            Ql[j][2*i]    = (half)v.x * scale; // Ql[j][2*i] = Ql[j][2*i+1] = v.x * scale
            Ql[j][2*i+1]  = (half)v.y * scale; // Ql[j][2*i+1] = v.y * scale
        }
    barrier(CLK_LOCAL_MEM_FENCE);

    /*====================== 2. iterate over K chunks ==========================*/
    // 2. iterate over K chunks
    for(int k_VKQ_0 = 0; k_VKQ_0 < ne11; k_VKQ_0 += KQ_TILE) {
        //
        // 2-A. init kqmax_new = previous kqmax
        //
        half kqmax_new[NCOLS/NWARPS];
        for(int jg = 0; jg < NCOLS/NWARPS; ++jg){
            kqmax_new[jg] = kqmax[jg];
        }

        half2 sum2[KQ_TILE/WARP][NCOLS/NWARPS];
        for(int i = 0; i < KQ_TILE/WARP; ++i){
            for(int jg = 0; jg < NCOLS/NWARPS; ++jg){
                sum2[i][jg] = (half2)(0.0h, 0.0h);
            }
        }

        //
        // 2-B. load K chunk into __local[KQ_TILE][D/2+1]
        //
        // for(int i_KQ = warp; i_KQ < KQ_TILE; i_KQ += NWARPS){
        //     for(int k_KQ = lane; k_KQ < D/2; k_KQ += WARP){
        //         KV[i_KQ][k_KQ] = Kh2[(k_VKQ_0 + i_KQ)*sK + k_KQ];
        //     }
        // }
        // barrier(CLK_LOCAL_MEM_FENCE);

        //
        // 2-C. compute logits = Q·K + mask*slope
        for(int k_KQ = 0; k_KQ < D/2; ++k_KQ) {
            half2 K_k[KQ_TILE/WARP];
            half2 Q_k[NCOLS/NWARPS];

            for(int i_KQ_0 = 0; i_KQ_0 < KQ_TILE; i_KQ_0 += WARP){
                const int i_KQ = i_KQ_0 + lane;
                K_k[i_KQ_0/WARP] = Kh2[(k_VKQ_0 + i_KQ)*sK + k_KQ];
            }

            for(int j_KQ_0 = 0; j_KQ_0 < NCOLS; j_KQ_0 += NWARPS){
                const int j_KQ = j_KQ_0 + warp;
                // Q_k[j_KQ_0/NWARPS] = Ql[j_KQ][k_KQ];
                Q_k[j_KQ_0/NWARPS] = (half2)(Ql[j_KQ][2*k_KQ], Ql[j_KQ][2*k_KQ + 1]);
            }

            for(int i_KQ_0 = 0; i_KQ_0 < KQ_TILE; i_KQ_0 += WARP){
                for(int j_KQ_0 = 0; j_KQ_0 < NCOLS; j_KQ_0 += NWARPS){
                    sum2[i_KQ_0/WARP][j_KQ_0/NWARPS] += 
                        K_k[i_KQ_0/WARP] * Q_k[j_KQ_0/NWARPS];
                }
            }
        }

        for(int i_KQ_0 = 0; i_KQ_0 < KQ_TILE; i_KQ_0 += WARP){
            const int i_KQ = i_KQ_0 + lane;
            for(int j_KQ_0 = 0; j_KQ_0 < NCOLS; j_KQ_0 += NWARPS){
                const int j_KQ = j_KQ_0 + warp;
                half sum = sum2[i_KQ_0/WARP][j_KQ_0/NWARPS].x + sum2[i_KQ_0/WARP][j_KQ_0/NWARPS].y;
                // apply mask & slope
                if(Mh) {
                    half m_val = Mh[j_KQ*ne11 + (k_VKQ_0 + i_KQ)];
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

            for(int i0 = 0; i0 < D; i0 += WARP){
                VKQ[j0/NWARPS][i0/WARP] *= KQ_max_scale;
            }
            
        }

        // for(int k = warp; k < KQ_TILE; k += NWARPS){
        //     for(int i = lane; i < D/2; i += WARP){
        //         KV[k][i] = Vh[(k_VKQ_0 + k)*sK + i];
        //     }
        // }
        barrier(CLK_LOCAL_MEM_FENCE);

        // 2-F. accumulate V with softmax weights
        for(int k0 = 0; k0 < KQ_TILE; k0 += 2) {
            half V_k[D/WARP][2];
            half2 KQ_k[NCOLS/NWARPS];

            for(int i0 = 0; i0 < D; i0 += WARP){
                const int i = i0 + lane;

                V_k[i0/WARP][0] = Vh[(k_VKQ_0 + k0    )*sV + i];
                V_k[i0/WARP][1] = Vh[(k_VKQ_0 + k0 + 1)*sV + i];

            }

            for(int j0 = 0; j0 < NCOLS; j0 += NWARPS){
                const int j = j0 + warp;

                KQ_k[j0/NWARPS] = (half2)(KQ[j*KQ_TILE + k0], KQ[j*KQ_TILE + k0 + 1]);
            }

            for(int i0 = 0; i0 < D; i0 += WARP){
                for(int j0 = 0; j0 < NCOLS; j0 += NWARPS){
                    const int i = i0 + lane;
                    const int j = j0 + warp;

                    // accumulate V with softmax weights
                    VKQ[j0/NWARPS][i0/WARP] += V_k[i0/WARP][0] * KQ_k[j0/NWARPS].x;
                    VKQ[j0/NWARPS][i0/WARP] += V_k[i0/WARP][1] * KQ_k[j0/NWARPS].y;
                }
            }

        }
        barrier(CLK_LOCAL_MEM_FENCE);
    } // end K-chunk loop

    /*====================== 3. write results ==================================*/
    for(int j_VKQ_0 = 0; j_VKQ_0 < NCOLS; j_VKQ_0 += NWARPS) {
        const int j_VKQ = j_VKQ_0 + warp;

        if(ic0 + j_VKQ >= ne01) {
            // if this column is out of bounds, skip it
            return;
        }

        half kqsum_j = sub_group_reduce_add((float)kqsum[j_VKQ_0/NWARPS]);

        for(int i00 = 0; i00 < D; i00 += WARP) {
            const int i0 = i00 + lane;

            if(i0 < D) {
                
                VKQ[j_VKQ_0/NWARPS][i0/WARP] /= kqsum_j;

                const int j_dst = j_VKQ;
                dst_f[j_dst * D * ne02 + D * head + i0] = (float)VKQ[j_VKQ_0/NWARPS][i0/WARP];
            }
        }
    }
}
