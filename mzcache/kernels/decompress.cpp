#include "mzcache_kernels.h"
#include <arm_neon.h>
#include <thread>
#include <vector>
#include <algorithm>
#include <future>
#include <chrono>

#include "mzcache_types.h"
#include "mzcache_threadpool.h"

const uint8x16_t V_MASK = vdupq_n_u8(0x0F); // 4-bit mask for low nibble

void flexgen_decompress(
    const uint8_t* input,
    int            hidden_dim,
    __fp16*        output,
    __fp16*        mins,
    __fp16*        maxs,

    ThreadPool*    pool,
    int            num_threads
) {
    constexpr int GROUP_SZ   = 64;  // elements per group

    constexpr int COMP_BYTES  = GROUP_SZ / 2;   // 32 bytes
    constexpr int OUT_PER_BLK = GROUP_SZ / 2;   // 32 output per iteration


    const int     elems      = hidden_dim * TOKENS_PER_CHUNK;
    const int     num_groups = elems / GROUP_SZ;


    auto worker = [&](int start, int end) {
        const float inv15 = 1.0f / 15.0f;

        for (int g = start; g < end; ++g) {
            // This group's min/max and step
            __fp16  min_h   = mins[g];
            __fp16  max_h   = maxs[g];
            float   step_f  = (float(max_h) - float(min_h)) * inv15;
            __fp16  step_h  = (__fp16)step_f;

            float16x8_t v_min  = vdupq_n_f16(min_h);
            float16x8_t v_step = vdupq_n_f16(step_h);

            const uint8_t* grp_in  = input  + g * COMP_BYTES;
            __fp16*  grp_out = output + g * GROUP_SZ;

            __builtin_prefetch(grp_in + 32, 0, 1);

            // Unpack in groups of 8 → reconstruct
            #pragma unroll 2
            for (int blk = 0; blk < 2; ++blk) {
                uint8x16_t cin = vld1q_u8(grp_in + blk * 16);
                uint8x16_t low  = vandq_u8(cin, vdupq_n_u8(0x0F));
                uint8x16_t high = vshrq_n_u8(cin, 4);
                uint8x16x2_t zip = vzipq_u8(low, high);

                // part 0
                {
                    uint8x16_t inter = zip.val[0];
                    int16x8_t i0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8 (inter)));
                    int16x8_t i1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(inter)));
                    float16x8_t f0 = vcvtq_f16_s16(i0);
                    float16x8_t f1 = vcvtq_f16_s16(i1);
                    // single-cycle FMA
                    float16x8_t o0 = vfmaq_f16(v_min, f0, v_step);
                    float16x8_t o1 = vfmaq_f16(v_min, f1, v_step);
                    // store
                    vst1q_f16(grp_out + blk*32 +  0, o0);
                    vst1q_f16(grp_out + blk*32 +  8, o1);
                }
                // part 1
                {
                    uint8x16_t inter = zip.val[1];
                    int16x8_t i0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8 (inter)));
                    int16x8_t i1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(inter)));
                    float16x8_t f0 = vcvtq_f16_s16(i0);
                    float16x8_t f1 = vcvtq_f16_s16(i1);
                    float16x8_t o0 = vfmaq_f16(v_min, f0, v_step);
                    float16x8_t o1 = vfmaq_f16(v_min, f1, v_step);
                    vst1q_f16(grp_out + blk*32 + 16, o0);
                    vst1q_f16(grp_out + blk*32 + 24, o1);
                }
            }
        }
    };

    // Launch threads
    std::vector<std::future<void>> futures;
    int per = (num_groups + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        int start = t * per;
        int end   = std::min(start + per, num_groups);
        if (start < end) {
            futures.emplace_back(pool->enqueue(CoreType::DECOMP_KERNEL, worker, start, end));
        }
    }
    for (auto &f : futures) f.get();
}

void flexgen_decompress_8bit(
    const uint8_t* input,
    int            hidden_dim,
    __fp16*        output,
    __fp16*        mins,
    __fp16*        maxs
) {
    constexpr int GROUP_SZ   = 64;
    constexpr int COMP_BYTES = GROUP_SZ; // 64 bytes per group

    const int elems      = hidden_dim * TOKENS_PER_CHUNK;
    const int num_groups = elems / GROUP_SZ;

    for (int g = 0; g < num_groups; ++g) {
        float min_f  = static_cast<float>(mins[g]);
        float max_f  = static_cast<float>(maxs[g]);
        float step_f = (max_f - min_f) / 255.0f;

        float32x4_t v_min_f32  = vdupq_n_f32(min_f);
        float32x4_t v_step_f32 = vdupq_n_f32(step_f);

        const uint8_t* grp_in  = input  + g * COMP_BYTES;
        __fp16*        grp_out = output + g * GROUP_SZ;

        __builtin_prefetch(grp_in + 64, 0, 1);

        for (int blk = 0; blk < 4; ++blk) {
            uint8x16_t qbytes = vld1q_u8(grp_in + blk * 16);

            uint16x8_t q_lo_u16 = vmovl_u8(vget_low_u8(qbytes));
            uint16x8_t q_hi_u16 = vmovl_u8(vget_high_u8(qbytes));

            uint32x4_t q0 = vmovl_u16(vget_low_u16(q_lo_u16));
            uint32x4_t q1 = vmovl_u16(vget_high_u16(q_lo_u16));
            uint32x4_t q2 = vmovl_u16(vget_low_u16(q_hi_u16));
            uint32x4_t q3 = vmovl_u16(vget_high_u16(q_hi_u16));

            float32x4_t f0 = vcvtq_f32_u32(q0);
            float32x4_t f1 = vcvtq_f32_u32(q1);
            float32x4_t f2 = vcvtq_f32_u32(q2);
            float32x4_t f3 = vcvtq_f32_u32(q3);

            float32x4_t o0 = vfmaq_f32(v_min_f32, f0, v_step_f32);
            float32x4_t o1 = vfmaq_f32(v_min_f32, f1, v_step_f32);
            float32x4_t o2 = vfmaq_f32(v_min_f32, f2, v_step_f32);
            float32x4_t o3 = vfmaq_f32(v_min_f32, f3, v_step_f32);

            float16x4_t h0 = vcvt_f16_f32(o0);
            float16x4_t h1 = vcvt_f16_f32(o1);
            float16x4_t h2 = vcvt_f16_f32(o2);
            float16x4_t h3 = vcvt_f16_f32(o3);

            float16x8_t out0 = vcombine_f16(h0, h1);
            float16x8_t out1 = vcombine_f16(h2, h3);

            vst1q_f16(grp_out + blk * 16 +  0, out0);
            vst1q_f16(grp_out + blk * 16 +  8, out1);
        }
    }
}


// --------------------- header ---------------------
void flexgen_decompress_single_thread(
    const uint8_t* input,
    int            hidden_dim,
    __fp16*        output,
    __fp16*        mins,
    __fp16*        maxs)
{
    constexpr int GROUP_SZ   = 64;   // 4-bit 64 elem per group
    constexpr int COMP_BYTES = GROUP_SZ / 2;   // 32B
    const int elems      = hidden_dim * TOKENS_PER_CHUNK;
    const int num_groups = elems / GROUP_SZ;

    const float inv15 = 1.0f / 15.0f;

    for (int g = 0; g < num_groups; ++g) {
        __fp16  min_h   = mins[g];
        __fp16  max_h   = maxs[g];
        float   step_f  = (float(max_h) - float(min_h)) * inv15;
        __fp16  step_h  = (__fp16)step_f;

        float16x8_t v_min  = vdupq_n_f16(min_h);
        float16x8_t v_step = vdupq_n_f16(step_h);

        const uint8_t* grp_in = input  + g * COMP_BYTES;
        __fp16*       grp_out = output + g * GROUP_SZ;

        __builtin_prefetch(grp_in + 32, 0, 1);

        #pragma unroll 2
        for (int blk = 0; blk < 2; ++blk) {
            uint8x16_t cin  = vld1q_u8(grp_in + blk * 16);
            uint8x16_t low  = vandq_u8(cin, vdupq_n_u8(0x0F));
            uint8x16_t high = vshrq_n_u8(cin, 4);
            uint8x16x2_t zip = vzipq_u8(low, high);

            // part 0
            {
                uint8x16_t inter = zip.val[0];
                int16x8_t i0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8 (inter)));
                int16x8_t i1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(inter)));
                float16x8_t f0 = vcvtq_f16_s16(i0);
                float16x8_t f1 = vcvtq_f16_s16(i1);
                float16x8_t o0 = vfmaq_f16(v_min, f0, v_step);
                float16x8_t o1 = vfmaq_f16(v_min, f1, v_step);
                vst1q_f16(grp_out + blk*32 +  0, o0);
                vst1q_f16(grp_out + blk*32 +  8, o1);
            }
            // part 1
            {
                uint8x16_t inter = zip.val[1];
                int16x8_t i0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8 (inter)));
                int16x8_t i1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(inter)));
                float16x8_t f0 = vcvtq_f16_s16(i0);
                float16x8_t f1 = vcvtq_f16_s16(i1);
                float16x8_t o0 = vfmaq_f16(v_min, f0, v_step);
                float16x8_t o1 = vfmaq_f16(v_min, f1, v_step);
                vst1q_f16(grp_out + blk*32 + 16, o0);
                vst1q_f16(grp_out + blk*32 + 24, o1);
            }
        }
    }
}

/////////

#include "cachegen_utils.h"


void decode_function_one_chunk(
    const EncoderOutput&    enc_out_cpu,
    const EncoderMeta&      enc_meta_cpu,
    const layer_kv_chunks&  layer_chunks,
    int total_tokens,             // == T (n_past), always a multiple of 256 (already padded)
    int tokens_per_chunk,         // == TOKENS_PER_CHUNK (256)
    int chunk_idx                 // index of the "dst raw chunk" to decode (0-based)
) {
    const int C = enc_meta_cpu.C;

    // Basic guards
    if ((int)enc_meta_cpu.max_key.size()   < total_tokens ||
        (int)enc_meta_cpu.max_value.size() < total_tokens) {
        throw std::runtime_error("decode_function_one_chunk: max_key/value size < total_tokens");
    }
    if (!layer_chunks.k_chunks.raw_chunks[chunk_idx].c ||
        !layer_chunks.v_chunks.raw_chunks[chunk_idx].c) {
        throw std::runtime_error("decode_function_one_chunk: null dst raw chunk pointer");
    }

    // Global token range of this raw chunk: [g_chunk_start, g_chunk_start + 256)
    const int g_chunk_start = chunk_idx * TOKENS_PER_CHUNK;

    // Bytestream chunk range corresponding to this raw chunk
    static_assert(TOKENS_PER_CHUNK % MAX_TOKENS_PER_CHUNK == 0, "must be multiple");
    const int n_bs_per_raw = TOKENS_PER_CHUNK / MAX_TOKENS_PER_CHUNK; // 16
    const int bs_start = chunk_idx * n_bs_per_raw;                     // starting bytestream index

    // dst base pointers (C-fastest)
    __fp16* dst_base_k = (__fp16*) layer_chunks.k_chunks.raw_chunks[chunk_idx].c;
    __fp16* dst_base_v = (__fp16*) layer_chunks.v_chunks.raw_chunks[chunk_idx].c;

    // Bounds guard (optional): make sure this raw chunk stays within total_tokens
    if (g_chunk_start + TOKENS_PER_CHUNK > total_tokens) {
        // Could conservatively return early, or add partial-processing code if needed;
        // since everything is assumed padded to a multiple of 256, treat it as an error
        throw std::runtime_error("decode_function_one_chunk: dst chunk range exceeds total_tokens");
    }

    // Sequentially reconstruct the 16 bytestream chunks for key and for value

    auto t_decode_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_bs_per_raw; ++i) {
        const int bs_idx = bs_start + i;                 // bytestream chunk index
        const int t_in_dst = i * MAX_TOKENS_PER_CHUNK;   // token offset within the raw chunk (0,16,32,...,240)
        const int g_start  = g_chunk_start + t_in_dst;   // global token start offset

        if (bs_idx < 0 ||
            bs_idx >= (int)enc_out_cpu.chunks_key.size() ||
            bs_idx >= (int)enc_out_cpu.chunks_value.size()) {
            throw std::runtime_error("decode_function_one_chunk: bytestream chunk idx OOR");
        }

        // ===== KEY =====
        {
            const auto& ch = enc_out_cpu.chunks_key[(size_t)bs_idx];
            // Optional guard: check that ch.t_chunk is 16
            // if (ch.t_chunk != MAX_TOKENS_PER_CHUNK) throw std::runtime_error("unexpected t_chunk");

            const int bins = enc_meta_cpu.cdf_key.bins_p1 - 1;
            std::vector<int8_t> q16 = decode_ntokens_cpu(
                /*cdf*/ enc_meta_cpu.cdf_key.data.data(),
                /*L*/ 1, /*C*/ C, /*bins_p1*/ enc_meta_cpu.cdf_key.bins_p1,
                /*bytestream*/ ch);

            __fp16* dst_ptr    = dst_base_k + (size_t)t_in_dst * C;
            const float* m_ptr = enc_meta_cpu.max_key.data() + (size_t)g_start;

            dequantize_cpu_into(
                q16.data(), m_ptr,
                /*L*/ 1, /*T*/ MAX_TOKENS_PER_CHUNK, /*C*/ C,
                /*bins*/ bins,
                /*out*/ dst_ptr
            );
        }

        // ===== VALUE =====
        {
            const auto& ch = enc_out_cpu.chunks_value[(size_t)bs_idx];

            const int bins = enc_meta_cpu.cdf_value.bins_p1 - 1;
            std::vector<int8_t> q16 = decode_ntokens_cpu(
                /*cdf*/ enc_meta_cpu.cdf_value.data.data(),
                /*L*/ 1, /*C*/ C, /*bins_p1*/ enc_meta_cpu.cdf_value.bins_p1,
                /*bytestream*/ ch);

            __fp16* dst_ptr    = dst_base_v + (size_t)t_in_dst * C;
            const float* m_ptr = enc_meta_cpu.max_value.data() + (size_t)g_start;

            dequantize_cpu_into(
                q16.data(), m_ptr,
                /*L*/ 1, /*T*/ MAX_TOKENS_PER_CHUNK, /*C*/ C,
                /*bins*/ bins,
                /*out*/ dst_ptr
            );
        }
    }

    auto t_decode_end = std::chrono::high_resolution_clock::now();
    double t_decode_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
    // printf("[Timing] Decoded chunk %d in %.2f ms\n", chunk_idx, t_decode_ms);
}


void decode_function_new(
    const uint8_t* input,
    int hidden_dim,
    __fp16*        output,

    std::vector<uint8_t>* lengths_arr,
    bool is_key,
    CdfRaw& cdf,
    const float* m_base,
    uint16_t* byte_sums_arr
) {

    const int C = hidden_dim;
    __fp16* dst_base = output;

    size_t current_offset = 0;

    // Basic guards
    static_assert(TOKENS_PER_CHUNK % MAX_TOKENS_PER_CHUNK == 0, "must be multiple");
    const int n_bs_per_raw = TOKENS_PER_CHUNK / MAX_TOKENS_PER_CHUNK; // 16


    auto t_decode_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_bs_per_raw; ++i) {

        //printf("[DECODE DEBUG] sub-chunk %d, offset = %zu, length = %u\n", i, current_offset, byte_sums_arr[i]);
        const int t_in_dst = i * MAX_TOKENS_PER_CHUNK;   // token offset within the raw chunk (0,16,32,...,240)

        {
            const int bins = cdf.bins_p1 - 1;
            std::vector<int8_t> q16 = decode_ntokens_cpu_new_range(
                /*cdf*/ cdf.data.data(),
                /*C*/ C, /*bins_p1*/ cdf.bins_p1,
                input + current_offset, lengths_arr[i]
            );

            // std::vector<int8_t> q16 = decode_ntokens_cpu_new_simd_range(
            //     /*cdf*/ cdf.data.data(),
            //     /*C*/ C, /*bins_p1*/ cdf.bins_p1,
            //     input + current_offset, lengths_arr[i]
            // );

            current_offset += byte_sums_arr[i];

            __fp16* dst_ptr  = dst_base + (size_t)t_in_dst * C;
            float* m_ptr  = (float*)(m_base + (size_t)t_in_dst);

            // dequantize_cpu_into_new(
            //     q16.data(), m_ptr,
            //     /*T*/ MAX_TOKENS_PER_CHUNK, /*C*/ C,
            //     /*bins*/ bins,
            //     /*out*/ dst_ptr
            // );

            dequantize_cpu_into_new_neon(
                q16.data(), m_ptr,
                /*T*/ MAX_TOKENS_PER_CHUNK, /*C*/ C,
                /*bins*/ bins,
                /*out*/ dst_ptr
            );
        }
    }

    auto t_decode_end = std::chrono::high_resolution_clock::now();
    double t_decode_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
    // printf("[Timing] Decoded chunk %d in %.2f ms\n", chunk_idx, t_decode_ms);

}