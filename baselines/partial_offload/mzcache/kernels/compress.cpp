#include "mzcache_kernels.h"
#include <arm_neon.h>
#include <thread>
#include <vector>
#include <future>

#include "mzCache_types.h"
#include "mzCache_threadpool.h"


void compress_fp16_to_4bit(
    const __fp16*  input,      // [262144]
    int hidden_dim, // 51
    uint8_t*  output,     // [131072]
    __fp16*   mins,
    __fp16*   maxs,

    ThreadPool&    pool,
    int            num_threads
) {
    constexpr int GROUP_SZ    = 64;
    const int num_groups  = TOKENS_PER_CHUNK * hidden_dim / GROUP_SZ; // 4096

    // printf("Hi~\n");

    auto worker = [&](int start, int end) {
        for (int g = start; g < end; ++g) {
            const __fp16* src = input + g * GROUP_SZ;
            uint8_t*      dst = output + g * (GROUP_SZ/2);

            // 1) vld1q_f16 → v0…v7, compute vmin/vmax
            float16x8_t v0 = vld1q_f16(src +   0);
            float16x8_t v1 = vld1q_f16(src +   8);
            float16x8_t v2 = vld1q_f16(src +  16);
            float16x8_t v3 = vld1q_f16(src +  24);
            float16x8_t v4 = vld1q_f16(src +  32);
            float16x8_t v5 = vld1q_f16(src +  40);
            float16x8_t v6 = vld1q_f16(src +  48);
            float16x8_t v7 = vld1q_f16(src +  56);
            
            // 2) horizontal reduce to grp_min, grp_max → meta[g]

            float16x8_t vmin = vminq_f16(v0, v1);
            vmin = vminq_f16(vmin, v2);
            vmin = vminq_f16(vmin, v3);
            vmin = vminq_f16(vmin, v4);
            vmin = vminq_f16(vmin, v5);
            vmin = vminq_f16(vmin, v6);
            vmin = vminq_f16(vmin, v7);

            float16x8_t vmax = vmaxq_f16(v0, v1);
            vmax = vmaxq_f16(vmax, v2);
            vmax = vmaxq_f16(vmax, v3);
            vmax = vmaxq_f16(vmax, v4);
            vmax = vmaxq_f16(vmax, v5);
            vmax = vmaxq_f16(vmax, v6);
            vmax = vmaxq_f16(vmax, v7);

            // horizontal reduction (extract per-vector min/max)
            float16x4_t vmin_lo = vget_low_f16(vmin);
            float16x4_t vmin_hi = vget_high_f16(vmin);
            float16x4_t p1 = vpmin_f16(vmin_lo, vmin_hi);
            float16x4_t p2 = vpmin_f16(p1, p1);
            __fp16 grp_min = vget_lane_f16(vpmin_f16(p2, p2), 0);

            float16x4_t vmax_lo = vget_low_f16(vmax);
            float16x4_t vmax_hi = vget_high_f16(vmax);
            float16x4_t q1 = vpmax_f16(vmax_lo, vmax_hi);
            float16x4_t q2 = vpmax_f16(q1, q1);
            __fp16 grp_max = vget_lane_f16(vpmax_f16(q2, q2), 0);

            mins[g] = grp_min; // Store the minimum value for this group
            maxs[g] = grp_max; // Store the maximum value for this group

            // 3) float scale, bias
            float range = float(grp_max - grp_min);
            float scale = (range > 0) ? (15.0f / range) : 0.0f;
            float bias  = -float(grp_min) * scale;
            float32x4_t v_scale = vdupq_n_f32(scale);
            float32x4_t v_bias  = vdupq_n_f32(bias);
            
            for (int block = 0; block < 8; ++block) {
                // load 8 FP16 values
                float16x8_t vf16 = vld1q_f16(src + block * 8);

                // FP16 → FP32(lo/hi)
                float32x4_t lo = vcvt_f32_f16(vget_low_f16(vf16));
                float32x4_t hi = vcvt_f32_f16(vget_high_f16(vf16));

                // x * scale + bias
                float32x4_t q_lo = vmlaq_f32(v_bias, lo, v_scale);
                float32x4_t q_hi = vmlaq_f32(v_bias, hi, v_scale);

                // round to nearest integer
                int32x4_t i_lo = vcvtnq_s32_f32(q_lo);
                int32x4_t i_hi = vcvtnq_s32_f32(q_hi);

                // clamp [0,15]
                int32x4_t zero = vdupq_n_s32(0);
                int32x4_t fif  = vdupq_n_s32(15);
                i_lo = vmaxq_s32(zero, vminq_s32(i_lo, fif));
                i_hi = vmaxq_s32(zero, vminq_s32(i_hi, fif));

                // narrow → int8x8
                int16x4_t h_lo = vmovn_s32(i_lo);
                int16x4_t h_hi = vmovn_s32(i_hi);
                int8x8_t  b8   = vmovn_s16(vcombine_s16(h_lo, h_hi));

                // reduce vector to scalar array
                int8_t buf[8];
                vst1_s8(buf, b8);

                // pack pairs as 4-bit into one byte
                for (int i = 0; i < 4; ++i) {
                    dst[block * 4 + i] = uint8_t((buf[2*i] & 0xF) 
                                            | ((buf[2*i+1] & 0xF) << 4));
                }
            }
        }
    };

    std::vector<std::future<void>> futures;
    int per = (num_groups + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        int start = t * per;
        int end   = std::min(start + per, num_groups);
        if (start < end) {
            futures.emplace_back(pool.enqueue(CoreType::DECOMP, worker, start, end));
        }
    }
    for (auto &f : futures) f.get();
}
