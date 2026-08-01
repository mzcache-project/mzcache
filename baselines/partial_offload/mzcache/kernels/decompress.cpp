#include "mzcache_kernels.h"
#include <arm_neon.h>
#include <thread>
#include <vector>
#include <algorithm>
#include <future>

#include "mzCache_types.h"
#include "mzCache_threadpool.h"

const uint8x16_t V_MASK = vdupq_n_u8(0x0F); // 4-bit mask for low nibble

void decompress_4bit_to_fp16(
    const uint8_t* input,
    int            hidden_dim,
    __fp16*        output,
    __fp16*        mins,
    __fp16*        maxs,

    ThreadPool&    pool,
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
            // this group's min/max and step
            __fp16  min_h   = mins[g];
            __fp16  max_h   = maxs[g];
            float   step_f  = (float(max_h) - float(min_h)) * inv15;
            __fp16  step_h  = (__fp16)step_f;

            float16x8_t v_min  = vdupq_n_f16(min_h);
            float16x8_t v_step = vdupq_n_f16(step_h);

            const uint8_t* grp_in  = input  + g * COMP_BYTES;
            __fp16*  grp_out = output + g * GROUP_SZ;

            __builtin_prefetch(grp_in + 32, 0, 1);

            // unpack 8 at a time → restore
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
                    // FMA in one cycle
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

    // launch threads
    std::vector<std::future<void>> futures;
    int per = (num_groups + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        int start = t * per;
        int end   = std::min(start + per, num_groups);
        if (start < end) {
            futures.emplace_back(pool.enqueue(CoreType::DECOMP_KERNEL, worker, start, end));
        }
    }
    for (auto &f : futures) f.get();
}


// --------------------- header ---------------------
void decompress_4bit_to_fp16_single_thread(
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
