#include "mzcache_kernels.h"
#include <arm_neon.h>
#include <thread>
#include <vector>
#include <future>

#include "mzcache_types.h"
#include "mzcache_threadpool.h"

#include <fstream>
#include <string>


void flexgen_compress(
    const __fp16*  input,      // [262144]
    int hidden_dim, // 51
    uint8_t*  output,     // [131072]
    __fp16*   mins,
    __fp16*   maxs,

    ThreadPool*    pool,
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

            // horizontal reduction (extract the min/max within each vector)
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
                // Load 8 FP16 values
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

                // spill the vector to a scalar array
                int8_t buf[8];
                vst1_s8(buf, b8);

                // pack pairs of values into one byte as 4-bit nibbles
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
            futures.emplace_back(pool->enqueue(CoreType::DECOMP, worker, start, end));
        }
    }
    for (auto &f : futures) f.get();
}

void flexgen_compress_8bit(
    const __fp16*  input,
    int            hidden_dim,
    uint8_t*       output,
    __fp16*        mins,
    __fp16*        maxs,

    ThreadPool*    pool,
    int            num_threads
) {
    constexpr int GROUP_SZ = 64;
    const int num_groups = TOKENS_PER_CHUNK * hidden_dim / GROUP_SZ;

    auto worker = [&](int start, int end) {
        for (int g = start; g < end; ++g) {
            const __fp16* src = input + g * GROUP_SZ;
            uint8_t*      dst = output + g * GROUP_SZ;

            // Load data and compute group min/max
            float16x8_t v0 = vld1q_f16(src +  0);
            float16x8_t v1 = vld1q_f16(src +  8);
            float16x8_t v2 = vld1q_f16(src + 16);
            float16x8_t v3 = vld1q_f16(src + 24);
            float16x8_t v4 = vld1q_f16(src + 32);
            float16x8_t v5 = vld1q_f16(src + 40);
            float16x8_t v6 = vld1q_f16(src + 48);
            float16x8_t v7 = vld1q_f16(src + 56);

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

            mins[g] = grp_min;
            maxs[g] = grp_max;

            float range = float(grp_max - grp_min);
            float scale = (range > 0.0f) ? (255.0f / range) : 0.0f;
            float bias  = -float(grp_min) * scale;
            float32x4_t v_scale = vdupq_n_f32(scale);
            float32x4_t v_bias  = vdupq_n_f32(bias);

            uint32x4_t v_zero = vdupq_n_u32(0);
            uint32x4_t v_255  = vdupq_n_u32(255);

            for (int block = 0; block < 8; ++block) {
                float16x8_t vf16 = vld1q_f16(src + block * 8);

                float32x4_t lo = vcvt_f32_f16(vget_low_f16(vf16));
                float32x4_t hi = vcvt_f32_f16(vget_high_f16(vf16));

                float32x4_t q_lo = vmlaq_f32(v_bias, lo, v_scale);
                float32x4_t q_hi = vmlaq_f32(v_bias, hi, v_scale);

                uint32x4_t u_lo = vcvtnq_u32_f32(q_lo);
                uint32x4_t u_hi = vcvtnq_u32_f32(q_hi);

                u_lo = vmaxq_u32(v_zero, vminq_u32(u_lo, v_255));
                u_hi = vmaxq_u32(v_zero, vminq_u32(u_hi, v_255));

                uint16x4_t h_lo = vmovn_u32(u_lo);
                uint16x4_t h_hi = vmovn_u32(u_hi);
                uint16x8_t combined = vcombine_u16(h_lo, h_hi);
                uint8x8_t packed = vmovn_u16(combined);

                vst1_u8(dst + block * 8, packed);
            }
        }
    };

    std::vector<std::future<void>> futures;
    int per = (num_groups + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        int start = t * per;
        int end   = std::min(start + per, num_groups);
        if (start < end) {
            futures.emplace_back(pool->enqueue(CoreType::DECOMP, worker, start, end));
        }
    }
    for (auto &f : futures) f.get();
}


/////////////////

#include "cachegen_utils.h"


static inline void dump_q_preview(const char* tag,
                                  const QuantResultRaw& qr,
                                  int T_show = 2,
                                  int C_show = 8,
                                  int flatN  = 64) {
    std::cout << "==== Q PREVIEW: " << tag
              << " (L=" << qr.L << ", T=" << qr.T << ", C=" << qr.C << ") ====\n";

    // 1) print the flat prefix
    const size_t total = (size_t)qr.L * qr.T * qr.C;
    size_t n = std::min((size_t)flatN, total);
    std::cout << "flat prefix (" << n << " elems):";
    for (size_t i = 0; i < n; ++i) {
        int v = (int) qr.q[i];
        std::cout << ' ' << v;
    }
    std::cout << "\n";

    // 2) (l=0) slice: t=0..T_show-1, c=0..C_show-1
    auto idx = [&](int l, int t, int c) -> size_t {
        return (size_t)c + (size_t)qr.C * ((size_t)t + (size_t)qr.T * (size_t)l);
    };

    int Ts = std::min(T_show, qr.T);
    int Cs = std::min(C_show, qr.C);

    for (int t = 0; t < Ts; ++t) {
        std::cout << "q[l=0, t=" << t << ", c=0.." << (Cs-1) << "]:";
        for (int c = 0; c < Cs; ++c) {
            int v = (int) qr.q[idx(0, t, c)];
            std::cout << ' ' << v;
        }
        std::cout << "\n";
    }

    // 3) a slice of max1
    // max1: [L,T] with idx = t + T*l
    auto idx_m = [&](int l, int t) -> size_t {
        return (size_t)t + (size_t)qr.T * (size_t)l;
    };
    std::cout << "max1[l=0, t=0.." << (Ts-1) << "]:";
    for (int t = 0; t < Ts; ++t) {
        float m = qr.max1[idx_m(0, t)];
        std::cout << ' ' << m;
    }
    std::cout << "\n";
}

static void dump_cdf_preview(const char* tag,
                             const CdfRaw& cdf,
                             int C_show = 4,
                             int bins_show = 16) {
    std::cout << "==== CDF PREVIEW: " << tag
              << " (C=" << cdf.C
              << ", bins_p1=" << cdf.bins_p1
              << ", total=" << cdf.data.size() << ") ====\n";

    int Cs = std::min(C_show, cdf.C);
    int Bs = std::min(bins_show, cdf.bins_p1);

    for (int c = 0; c < Cs; ++c) {
        std::cout << "cdf[c=" << c << "][0.." << (Bs-1) << "]:";
        for (int b = 0; b < Bs; ++b) {
            size_t idx = (size_t)b + (size_t)cdf.bins_p1 * (size_t)c;
            std::cout << ' ' << cdf.data[idx];
        }
        std::cout << "\n";
    }

    // printing the flat prefix too makes overall differences quick to spot
    size_t flatN = std::min<size_t>(64, cdf.data.size());
    std::cout << "flat prefix (" << flatN << "):";
    for (size_t i = 0; i < flatN; ++i) {
        std::cout << ' ' << cdf.data[i];
    }
    std::cout << "\n";
}

static inline void dump_q_to_file(const char* tag,
                                  const QuantResultRaw& qr,
                                  const std::string& filename) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open dump file: " << filename << "\n";
        return;
    }

    ofs << "==== Q DUMP: " << tag
        << " (L=" << qr.L << ", T=" << qr.T << ", C=" << qr.C << ") ====\n";

    // dump the entire q array
    ofs << "[q array] total=" << ((size_t)qr.L * qr.T * qr.C) << " elements\n";
    for (int l = 0; l < qr.L; ++l) {
        for (int t = 0; t < qr.T; ++t) {
            ofs << "l=" << l << ", t=" << t << ":";
            for (int c = 0; c < qr.C; ++c) {
                size_t idx = (size_t)c + (size_t)qr.C * ((size_t)t + (size_t)qr.T * (size_t)l);
                ofs << ' ' << (int)qr.q[idx];
            }
            ofs << "\n";
        }
    }

    // dump the entire max1 array
    ofs << "\n[max1 array] total=" << ((size_t)qr.L * qr.T) << " elements\n";
    for (int l = 0; l < qr.L; ++l) {
        for (int t = 0; t < qr.T; ++t) {
            size_t idx = (size_t)t + (size_t)qr.T * (size_t)l;
            ofs << "l=" << l << ", t=" << t << ": " << qr.max1[idx] << "\n";
        }
    }

    ofs.close();
    std::cout << "Dumped q and max1 to file: " << filename << "\n";
}

static inline void dump_chunked_half_to_file(const char* filename,
                                             const array_of_chunks& chunks,
                                             int T, int C,
                                             int tokens_per_chunk,
                                             int stride_C_elements /*= C*/) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open file: " << filename << "\n";
        return;
    }
    ofs << std::fixed << std::setprecision(3);

    // Basic guards
    const int expect_chunks = (T + tokens_per_chunk - 1) / tokens_per_chunk;
    // if ((int)chunks.raw_chunks.size() < expect_chunks) {
    //     std::cerr << "[dump] warning: chunks.size()=" << chunks.raw_chunks.size()
    //               << " < expected " << expect_chunks << " for T=" << T
    //               << " and tokens_per_chunk=" << tokens_per_chunk << "\n";
    // }

    for (int t = 0; t < T; ++t) {
        const int chunk_idx  = t / tokens_per_chunk;
        const int t_in_chunk = t % tokens_per_chunk;

        // if (chunk_idx >= (int)chunks.raw_chunks.size() || !chunks.raw_chunks[chunk_idx].c) {
        //     std::cerr << "[dump] error: invalid chunk_idx=" << chunk_idx << " at t=" << t << "\n";
        //     break;
        // }

        const __fp16* base = reinterpret_cast<const __fp16*>(chunks.raw_chunks[(size_t)chunk_idx].c);
        ofs << "t=" << t << ":";

        // start position of one token (row), stride applied in half elements
        const size_t row_off = (size_t)t_in_chunk * (size_t)stride_C_elements;

        for (int c = 0; c < C; ++c) {
            const size_t idx = row_off + (size_t)c;  // (c + stride * t_in_chunk)
            // points at the right location even when stride_C_elements != C
            float v = (float)base[idx];  // __fp16 → float
            ofs << ' ' << v;
        }
        ofs << '\n';
    }

    ofs.close();
    std::cout << "[dump] wrote " << (size_t)T * C
              << " values to " << filename
              << " (T=" << T << ", C=" << C
              << ", tokens_per_chunk=" << tokens_per_chunk
              << ", stride_C=" << stride_C_elements << ")\n";
}


void encode_meta_function(
    const layer_kv_chunks& layer_chunks,
    int total_tokens,            // == T
    int hidden_dim,              // == C
    EncoderMeta& enc_meta_cpu
) {
    const int L = 1;
    const int T = total_tokens;
    const int C = hidden_dim;

    std::unique_ptr<int[]> key_bins  (make_key_bins_cpp(L));
    std::unique_ptr<int[]> value_bins(make_value_bins_cpp(L));

    // dump_chunked_half_to_file("k_chunked_dump.txt",
    //                         layer_chunks.k_chunks,
    //                         /*T=*/total_tokens,
    //                         /*C=*/hidden_dim,
    //                         /*tokens_per_chunk=*/256,      // if the actual TPC differs, put the real value here!
    //                         /*stride_C_elements=*/hidden_dim); // if the row stride differs from C, use the actual stride

    // ★ quantize from chunked input
    QuantResultRaw qk = quantize_from_bins_raw_chunked(
        key_bins.get(), L, T, C, layer_chunks.k_chunks);
    QuantResultRaw qv = quantize_from_bins_raw_chunked(
        value_bins.get(), L, T, C, layer_chunks.v_chunks);

    enc_meta_cpu.T = T;
    enc_meta_cpu.C = C;

    const int key_max   = key_bins[0];
    const int value_max = value_bins[0];

    // dump_q_to_file("chunked/key", qk, "qk_dump.txt");

    // dump_q_preview("chunked/key",   qk, /*T_show=*/2, /*C_show=*/8, /*flatN=*/64);
    // dump_q_preview("chunked/value", qv, /*T_show=*/2, /*C_show=*/8, /*flatN=*/64);

    // CDF as before (based on q)
    enc_meta_cpu.cdf_key   = cdf_calculate_cpu(qk.q, L, T, C, key_max);
    enc_meta_cpu.cdf_value = cdf_calculate_cpu(qv.q, L, T, C, value_max);

    // dump_cdf_preview("chunked/key",   enc_meta_cpu.cdf_key,   /*C_show=*/4, /*bins_show=*/16);
    // dump_cdf_preview("chunked/value", enc_meta_cpu.cdf_value, /*C_show=*/4, /*bins_show=*/16);

    // store max1
    enc_meta_cpu.max_key.assign  (qk.max1, qk.max1 + (size_t)T);
    enc_meta_cpu.max_value.assign(qv.max1, qv.max1 + (size_t)T);
}



void encode_function(
    const __fp16* k_l,
    const __fp16* v_l,
    int n_tokens,
    const EncoderMeta& enc_meta_cpu,
    EncoderOutput& enc_out_cpu,
    bool log
) {

    // Print the shape of the input tensors
    // printf("k_l shape: [%lld, %lld, %lld]\n", k_l->ne[0], k_l->ne[1], k_l->ne[2]);
    // printf("v_l shape: [%lld, %lld, %lld]\n", v_l->ne[0], v_l->ne[1], v_l->ne[2]);

    // int hidden_dim = k_l->ne[1] * k_l->ne[2];  // Assuming k_l and v_l have the same shape
    int n_layers = 1;

    const int C = enc_meta_cpu.C;

    // bins for quantization (must match the precomputed CDFs' ranges)
    std::unique_ptr<int[]> key_bins  (make_key_bins_cpp(1));
    std::unique_ptr<int[]> value_bins(make_value_bins_cpp(1));

    // Optional sanity check (bins must match CDF max)
    if ((int)key_bins[0]   != enc_meta_cpu.cdf_key.bins_p1 - 1 ||
        (int)value_bins[0] != enc_meta_cpu.cdf_value.bins_p1 - 1) {
        throw std::runtime_error("encode_function: bins(max) mismatch with precomputed CDFs");
    }

    // load fp16 from ggml
    // std::unique_ptr<__fp16[]> k_cpu(new __fp16[(size_t)n_tokens*C]);
    // std::unique_ptr<__fp16[]> v_cpu(new __fp16[(size_t)n_tokens*C]);
    // ggml_backend_tensor_get(k_l, k_cpu.get(), sizeof(__fp16)*(size_t)offset_tokens*C, sizeof(__fp16)*(size_t)n_tokens*C);
    // ggml_backend_tensor_get(v_l, v_cpu.get(), sizeof(__fp16)*(size_t)offset_tokens*C, sizeof(__fp16)*(size_t)n_tokens*C);

    // quantize (L = 1)
    QuantResultRaw qk = quantize_from_bins_raw(key_bins.get(),   1, n_tokens, C, k_l);
    QuantResultRaw qv = quantize_from_bins_raw(value_bins.get(), 1, n_tokens, C, v_l);

    // if(log) {
    //     dump_q_preview("full/key",   qk, /*T_show=*/2, /*C_show=*/8, /*flatN=*/64);
    //     dump_q_preview("full/value", qv, /*T_show=*/2, /*C_show=*/8, /*flatN=*/64);
    // }



    // encode helper: accumulate bytes
    auto encode_one_stream = [&](const int8_t* q_base,
                                 const CdfRaw& cdf,
                                 std::vector<EncodeCPUResult>& out_chunks,
                                 size_t& bytes_sum) {
        for (int s = 0; s < n_tokens; s += MAX_TOKENS_PER_CHUNK) {
            const int e       = std::min(s + MAX_TOKENS_PER_CHUNK, n_tokens);
            const int t_chunk = e - s;
            const int8_t* q_chunk_ptr = q_base + (size_t)C * s;

            EncodeCPUResult r = encode_ntokens_cpu(
                /*cdf=*/cdf.data.data(), /*L=*/1, C, cdf.bins_p1,
                /*q_chunk=*/q_chunk_ptr, t_chunk
            );
            bytes_sum += r.bytestream.size();
            out_chunks.push_back(std::move(r));
        }
    };

    // encode both streams and accumulate sizes
    encode_one_stream(qk.q, enc_meta_cpu.cdf_key,   enc_out_cpu.chunks_key,   enc_out_cpu.bytes_key);
    encode_one_stream(qv.q, enc_meta_cpu.cdf_value, enc_out_cpu.chunks_value, enc_out_cpu.bytes_value);

    free_quant(qk);
    free_quant(qv);
}


void encode_meta_function_new(
    const layer_kv_chunks& layer_chunks,
    int total_tokens,            // == T
    int hidden_dim,              // == C
    EncoderMeta* enc_meta_cpu
) {
    const int L = 1;
    const int T = total_tokens;
    const int C = hidden_dim;

    std::unique_ptr<int[]> key_bins  (make_key_bins_cpp(L));
    std::unique_ptr<int[]> value_bins(make_value_bins_cpp(L));

    // dump_chunked_half_to_file("k_chunked_dump.txt",
    //                         layer_chunks.k_chunks,
    //                         /*T=*/total_tokens,
    //                         /*C=*/hidden_dim,
    //                         /*tokens_per_chunk=*/256,      // if the actual TPC differs, put the real value here!
    //                         /*stride_C_elements=*/hidden_dim); // if the row stride differs from C, use the actual stride

    // ★ quantize from chunked input
    QuantResultRaw qk = quantize_from_bins_raw_chunked(
        key_bins.get(), L, T, C, layer_chunks.k_chunks);
    QuantResultRaw qv = quantize_from_bins_raw_chunked(
        value_bins.get(), L, T, C, layer_chunks.v_chunks);

    enc_meta_cpu->T = T;
    enc_meta_cpu->C = C;

    const int key_max   = key_bins[0];
    const int value_max = value_bins[0];

    // dump_q_to_file("chunked/key", qk, "qk_dump.txt");

    // dump_q_preview("chunked/key",   qk, /*T_show=*/2, /*C_show=*/8, /*flatN=*/64);
    // dump_q_preview("chunked/value", qv, /*T_show=*/2, /*C_show=*/8, /*flatN=*/64);

    // CDF as before (based on q)
    enc_meta_cpu->cdf_key   = cdf_calculate_cpu(qk.q, L, T, C, key_max);
    enc_meta_cpu->cdf_value = cdf_calculate_cpu(qv.q, L, T, C, value_max);

    // dump_cdf_preview("chunked/key",   enc_meta_cpu.cdf_key,   /*C_show=*/4, /*bins_show=*/16);
    // dump_cdf_preview("chunked/value", enc_meta_cpu.cdf_value, /*C_show=*/4, /*bins_show=*/16);

    // store max1
    enc_meta_cpu->max_key.assign  (qk.max1, qk.max1 + (size_t)T);
    enc_meta_cpu->max_value.assign(qv.max1, qv.max1 + (size_t)T);
}


size_t encode_function_new(
    const __fp16* input,
    int hidden_dim,
    uint8_t*  output,

    std::vector<uint8_t>* lengths_arr,
    bool is_key,
    CdfRaw& cdf,
    uint16_t* byte_sums_arr
) {

    // Print the shape of the input tensors
    // printf("k_l shape: [%lld, %lld, %lld]\n", k_l->ne[0], k_l->ne[1], k_l->ne[2]);
    // printf("v_l shape: [%lld, %lld, %lld]\n", v_l->ne[0], v_l->ne[1], v_l->ne[2]);

    // int hidden_dim = k_l->ne[1] * k_l->ne[2];  // Assuming k_l and v_l have the same shape
    int n_layers = 1;

    const int C = hidden_dim;

    // bins for quantization (must match the precomputed CDFs' ranges)
    // std::unique_ptr<int[]> key_bins  (make_key_bins_cpp(1));
    // std::unique_ptr<int[]> value_bins(make_value_bins_cpp(1));

    std::unique_ptr<int[]> bins = is_key ?
        std::unique_ptr<int[]>(make_key_bins_cpp(1)) :
        std::unique_ptr<int[]>(make_value_bins_cpp(1));

    size_t bytes_sum = 0;

    // quantize (L = 1)
    QuantResultRaw quantized_input = quantize_from_bins_raw(bins.get(), 1, TOKENS_PER_CHUNK, C, input);

    // encode helper: accumulate bytes
    auto encode_one_stream = [&](const int8_t* q_base,
                                 const CdfRaw& cdf,
                                 uint8_t*  output,
                                 size_t& bytes_sum) {

        for (int s = 0; s < TOKENS_PER_CHUNK; s += MAX_TOKENS_PER_CHUNK) {
            const int t_chunk = MAX_TOKENS_PER_CHUNK;
            const int8_t* q_chunk_ptr = q_base + (size_t)C * s;

            int bytes_sum_temp = encode_ntokens_cpu_new_range(
                /*cdf=*/cdf.data.data(), /*C=*/C, /*lp=*/cdf.bins_p1,
                /*q_chunk=*/q_chunk_ptr, t_chunk,
                output + bytes_sum, lengths_arr[s / MAX_TOKENS_PER_CHUNK]
            );

            // DEBUG: Print the calculated length for each sub-chunk
            //printf("[ENCODE DEBUG] sub-chunk %d, length = %d\n", s / MAX_TOKENS_PER_CHUNK, bytes_sum_temp);

            bytes_sum += bytes_sum_temp;
            byte_sums_arr[s / MAX_TOKENS_PER_CHUNK] = bytes_sum_temp;
        }
    };

    encode_one_stream(quantized_input.q, cdf, output, bytes_sum);

    free_quant(quantized_input);

    return bytes_sum;
}

