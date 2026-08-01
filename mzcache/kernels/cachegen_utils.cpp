#include "cachegen_utils.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <cstdint>
#include <vector>

#include "mzcache_types.h"

#include <arm_neon.h>


int key_first_layers  = 1;
int key_second_layers = 1;
int key_third_layers  = 1;

int key_first_bins    = 32;
int key_second_bins   = 32;
int key_third_bins    = 32;

int value_first_layers = 2;
int value_first_bins   = 32;
int value_second_bins  = 32;

// ---------------- bins builders ----------------

int* make_key_bins_cpp(int n_layers) {
    int* bins = new int[n_layers];
    for (int l = 0; l < n_layers; ++l) {
        if (l < key_first_layers)        bins[l] = key_first_bins;
        else if (l < key_first_layers + key_second_layers)
                                         bins[l] = key_second_bins;
        else                              bins[l] = key_third_bins;
    }
    return bins;
}

int* make_value_bins_cpp(int n_layers) {
    int* bins = new int[n_layers];
    for (int l = 0; l < n_layers; ++l) {
        if (l < value_first_layers) bins[l] = value_first_bins;
        else                        bins[l] = value_second_bins;
    }
    return bins;
}

// ---------------- quantize (functionality-first scalar) ----------------
// layout: idx = c + C*(t + T*l)  (same as ggml's [C,T,L] layout)
QuantResultRaw quantize_from_bins_raw(
    const int*    bins_L,
    int           L,
    int           T,
    int           C,
    const __fp16* x)
{
    if (!bins_L) throw std::runtime_error("quantize_from_bins_raw: bins_L is null");
    if (!x)      throw std::runtime_error("quantize_from_bins_raw: x is null");
    if (L <= 0 || T <= 0 || C <= 0) throw std::runtime_error("quantize_from_bins_raw: invalid dims");

    QuantResultRaw out{};
    out.L = L; out.T = T; out.C = C;

    const size_t N  = static_cast<size_t>(L) * T * C;
    const size_t NT = static_cast<size_t>(L) * T;

    out.q   = new int8_t[N];
    out.max1 = new float[NT];

    const size_t strideC = 1;
    const size_t strideT = static_cast<size_t>(C);
    const size_t strideL = static_cast<size_t>(C) * T;

    for (int l = 0; l < L; ++l) {
        const int bins = bins_L[l];
        const int MAX_q = (bins / 2) - 1;      // ex) bins=32 → MAX=15
        const float MAXf = static_cast<float>(MAX_q);
        const float clamp_lo = 0.0f;
        const float clamp_hi = 2.0f * MAXf;

        for (int t = 0; t < T; ++t) {
            // 1) per-(l,t) max |x|
            float acc_max = 0.0f;
            for (int c = 0; c < C; ++c) {
                size_t idx = static_cast<size_t>(l)*strideL + static_cast<size_t>(t)*strideT + static_cast<size_t>(c)*strideC;
                float v = (float) x[idx];
                float av = std::fabs(v);
                if (av > acc_max) acc_max = av;
            }
            out.max1[static_cast<size_t>(t) + static_cast<size_t>(T)*l] = acc_max;

            // 2) factor
            const float factor = (acc_max > 0.0f) ? (MAXf / acc_max) : 0.0f;

            // 3) quantize: q = clamp(round(x*factor + MAX), 0..2*MAX)
            for (int c = 0; c < C; ++c) {
                size_t idx = static_cast<size_t>(l)*strideL + static_cast<size_t>(t)*strideT + static_cast<size_t>(c)*strideC;
                float vx = (float) x[idx];
                float qf = std::round(vx * factor + MAXf);
                if (qf < clamp_lo) qf = clamp_lo;
                if (qf > clamp_hi)  qf = clamp_hi;
                int qi = static_cast<int>(qf);         // 0..2*MAX (e.g. <= 30)
                // safe for int8_t (within the current bins range)
                out.q[idx] = static_cast<int8_t>(qi);
            }
        }
    }

    return out;
}

QuantResultRaw quantize_from_bins_raw_chunked(
    const int*   bins_L,   // [L]
    int          L,        // usually 1
    int          T,        // total tokens
    int          C,        // hidden dim
    const array_of_chunks& chunks // layer_chunks.k_chunks or v_chunks
) {
    if (!bins_L) throw std::runtime_error("quantize_from_bins_raw_chunked: bins_L is null");
    if (L <= 0 || T <= 0 || C <= 0) throw std::runtime_error("quantize_from_bins_raw_chunked: invalid dims");

    QuantResultRaw out{};
    out.L = L; out.T = T; out.C = C;

    const size_t N  = (size_t)L * T * C;
    const size_t NT = (size_t)L * T;

    out.q    = new int8_t[N];
    out.max1 = new float[NT];

    // Chunk-count guard
    const int num_chunks_expect = (T + TOKENS_PER_CHUNK - 1) / TOKENS_PER_CHUNK;
    // if ((int)chunks.raw_chunks.size() != num_chunks_expect) {
    //     // If the contract requires exact equality, assert/throw
    //     // Even if the last chunk is over-allocated, t is only accessed up to T, so it is safe by itself
    //     // The line below can be commented out if needed, but it is useful for debugging
    //     fprintf(stderr, "Warning: quantize_from_bins_raw_chunked: unexpected num_chunks %zu (expect %d for T=%d)\n",
    //             chunks.raw_chunks.size(), num_chunks_expect, T);
    // }

    // When writing (l,t,c) global indices into out.q/out.max1, keep the existing [L,T,C] contiguous indexing
    auto idx_q  = [&](int l, int t, int c) -> size_t {
        return (size_t)c + (size_t)C * ((size_t)t + (size_t)T * (size_t)l);
    };
    auto idx_m  = [&](int l, int t) -> size_t {
        return (size_t)t + (size_t)T * (size_t)l;
    };

    // Helper to load the half at position (t,c) from the chunks
    auto load_from_chunks = [&](int t, int c) -> float {
        const int chunk_idx   = t / TOKENS_PER_CHUNK;
        const int t_in_chunk  = t % TOKENS_PER_CHUNK;
        const __fp16* base    = (const __fp16*) chunks.raw_chunks[(size_t)chunk_idx].c;
        const size_t idx_in   = (size_t)c + (size_t)C * (size_t)t_in_chunk; // C-fastest
        // Reuse the half loader the project already has
        return (float) base[idx_in];
    };

    for (int l = 0; l < L; ++l) {
        const int   bins    = bins_L[l];
        const int   MAX_q   = (bins / 2) - 1;  // ex) 32 -> 15
        const float MAXf    = (float)MAX_q;
        const float lo      = 0.0f;
        const float hi      = 2.0f * MAXf;

        for (int t = 0; t < T; ++t) {
            // 1) per-(l,t) max |x|
            float acc_max = 0.f;
            for (int c = 0; c < C; ++c) {
                float v  = load_from_chunks(t, c);
                float av = std::fabs(v);
                if (av > acc_max) acc_max = av;
            }
            out.max1[idx_m(l, t)] = acc_max;

            // 2) factor
            const float factor = (acc_max > 0.f) ? (MAXf / acc_max) : 0.f;

            // 3) quantize: q = clamp(round(x*factor + MAX), 0..2*MAX)
            for (int c = 0; c < C; ++c) {
                float vx = load_from_chunks(t, c);
                float qf = std::round(vx * factor + MAXf);
                if (qf < lo) qf = lo;
                if (qf > hi) qf = hi;
                out.q[idx_q(l, t, c)] = (int8_t)((int)qf); // 0..2*MAX (e.g. 0..30)
            }
        }
    }

    return out;
}

static void dump_q_as_chars(const int8_t* q, int L, int T, int C, int max_show = 128) {
    if (!q) {
        std::cout << "[dump_q_as_chars] q is null\n";
        return;
    }
    size_t total = (size_t)L * T * C;
    size_t showN = std::min<size_t>(max_show, total);

    std::cout << "[dump_q_as_chars] total=" << total
              << " showing first " << showN << " elements\n";

    std::cout << "as chars: ";
    for (size_t i = 0; i < showN; ++i) {
        char c = static_cast<char>(q[i]);
        if (std::isprint(static_cast<unsigned char>(c)))
            std::cout << c;
        else
            std::cout << '.'; // print '.' for non-printable characters
    }
    std::cout << "\n";

    std::cout << "as ints :";
    for (size_t i = 0; i < showN; ++i) {
        std::cout << " " << (int)q[i];
    }
    std::cout << "\n";
}


CdfRaw cdf_calculate_cpu(const int8_t* q, int L, int T, int C, int max_bins) {
    if (!q) throw std::runtime_error("cdf_calculate_cpu: q is null");
    if (L <= 0 || T <= 0 || C <= 0) throw std::runtime_error("cdf_calculate_cpu: invalid dims");
    if (max_bins <= 0) throw std::runtime_error("cdf_calculate_cpu: invalid max_bins");

    const int Bp1 = max_bins + 1;     // bins+1
    const size_t out_elems = (size_t)L * C * Bp1;

    CdfRaw out;
    out.C = C;
    out.bins_p1 = Bp1;
    out.data.resize(out_elems);

    // dump_q_as_chars(q, L, T, C, 1024);

    // Layout convention:
    //  - input q:   [L,T,C], idx_in  = c + C*(t + T*l)        (C fastest)
    //  - output out: [L,C,Bp1], idx_out = b + Bp1*(c + C*l)   (bin fastest)
    auto idx_in  = [=](int l, int t, int c) -> size_t {
        return (size_t)c + (size_t)C * ((size_t)t + (size_t)T * (size_t)l);
    };
    auto idx_out = [=](int l, int c, int b) -> size_t {
        return (size_t)b + (size_t)Bp1 * ((size_t)c + (size_t)C * (size_t)l);
    };

    // Same logic as the CUDA kernel:
    // hist[] size: max_bins+1; hist[0] is the "below 0" slot, actual counts accumulate at value+1
    // 1) accumulate a histogram for each (l,c)
    // 2) prefix sum over hist[1..max_bins]
    // 3) normalize:  (0xFFFF - max_bins) * hist[i] / total  + i
    //    total = final prefix sum of hist[1..max_bins] == T (ntokens)
    const uint32_t MAX_U16_FREE = 0xFFFFu - (uint32_t)max_bins;

    // --- replacement loop body: begin ---
    std::vector<uint16_t> hist(Bp1);

    for (int l = 0; l < L; ++l) {
        for (int c_ = 0; c_ < C; ++c_) {
            // 1) clear hist
            std::fill(hist.begin(), hist.end(), 0u);

            // accumulate: value in [0..max_bins] → hist[value+1]++
            for (int t = 0; t < T; ++t) {
                uint8_t value = (uint8_t) q[idx_in(l, t, c_)];
                if (value > (uint8_t)max_bins) value = (uint8_t)max_bins;
                hist[(int)value + 1]++;
            }

            // 2) prefix sum on hist[1..max_bins]
            uint16_t local_sum = 0;
            for (int i = 0; i < max_bins; ++i) {
                uint16_t v = hist[i + 1];
                hist[i + 1] = (uint16_t)(hist[i + 1] + local_sum);
                local_sum = (uint16_t)(local_sum + v);
            }
            // usually local_sum == T

            // 3) normalize + offset(+i), store the result in out
            const uint32_t denom = std::max<uint32_t>(1, (uint32_t)local_sum);
            for (int b = 0; b <= max_bins; ++b) {
                const uint32_t norm = (MAX_U16_FREE * (uint32_t)hist[b]) / denom;
                const uint16_t v = (uint16_t)(norm + (uint32_t)b);
                out.data[idx_out(l, c_, b)] = v;
            }
        }
    }
    // --- replacement loop body: end ---


    return out;
}


// ============================
// Bit packer (MSB-first) that exactly mirrors the CUDA helpers:
//
// - add_bits_to_output(bit, num, output_reg, output_reg_len, ...)
//   Accumulates bits into a 32-bit register (MSB-first). When full (32),
//   spills 4 bytes in big-endian order.
//
// - spill_reg_to_shared(output_reg, output_reg_len, ...)
//   Writes exactly 4 bytes (register must be full).
//
// - spill_partial_reg_to_shared(...)
//   Writes the remaining (output_reg_len) bits as ceil(len/8) bytes,
//   left-aligned in the MSBs of the first byte (same left shift).
// ============================

static inline void spill_reg_to_vec(uint32_t &output_reg,
                                    int      &output_reg_len,
                                    std::vector<uint8_t> &out) {
    // Left align the 32 bits we accumulated, then write 4 bytes big-endian
    output_reg <<= (32 - output_reg_len);
    output_reg_len = 0;

    out.push_back(static_cast<uint8_t>(output_reg >> 24));
    out.push_back(static_cast<uint8_t>((output_reg >> 16) & 0xFF));
    out.push_back(static_cast<uint8_t>((output_reg >> 8)  & 0xFF));
    out.push_back(static_cast<uint8_t>(output_reg & 0xFF));

    output_reg = 0;
}

// Write leftover bits (not multiple of 32) as full bytes, MSB-first.
static inline void spill_partial_reg_to_vec(uint32_t &output_reg,
                                            int      &output_reg_len,
                                            std::vector<uint8_t> &out) {
    if (output_reg_len <= 0) return;

    // Align to MSB side
    uint32_t reg = output_reg << (32 - output_reg_len);
    int bytes = (output_reg_len + 7) / 8;

    for (int i = 0; i < bytes; ++i) {
        uint8_t b = static_cast<uint8_t>(reg >> 24);
        out.push_back(b);
        reg <<= 8;
    }
    output_reg      = 0;
    output_reg_len  = 0;
}

static inline void add_bits_slow(uint32_t bit, int num,
                                 uint32_t &output_reg,
                                 int      &output_reg_len,
                                 std::vector<uint8_t> &out)
{
    while (num > 0) {
        int can = 32 - output_reg_len;
        int take = (num < can) ? num : can;

        // Avoid UB: when take==32, fill directly instead of shifting
        if (take == 32) {
            // To get the same effect as left-align followed by spill,
            // empty the current register, fill all 32 bits at once, then spill
            output_reg <<= (32 - output_reg_len);
            output_reg_len = 32; // full
            // bit pattern to fill with (all ones or all zeros)
            uint32_t fill = bit ? 0xFFFFFFFFu : 0x00000000u;
            // already full, so spill
            out.push_back(static_cast<uint8_t>(fill >> 24));
            out.push_back(static_cast<uint8_t>((fill >> 16) & 0xFF));
            out.push_back(static_cast<uint8_t>((fill >>  8) & 0xFF));
            out.push_back(static_cast<uint8_t>( fill        & 0xFF));
            // reset the register
            output_reg = 0;
            output_reg_len = 0;
        } else {
            // normal path
            output_reg <<= take;
            if (bit) {
                // compute (1<<take)-1 safely
                uint32_t mask = (take == 32) ? 0xFFFFFFFFu : ((1u << take) - 1u);
                output_reg |= mask;
            }
            output_reg_len += take;
            if (output_reg_len == 32) {
                spill_reg_to_vec(output_reg, output_reg_len, out);
            }
        }
        num -= take;
    }
}

// Byte-boundary fast-path: quickly write long runs of identical bits
static inline void add_bits_run(uint32_t bit, uint64_t num,
                                uint32_t &output_reg,
                                int      &output_reg_len,
                                std::vector<uint8_t> &out)
{
    if (num == 0) return;

    // 1) First align the register to a byte boundary (multiple of 8)
    int mod8 = output_reg_len & 7;            // positions used in the current partial byte
    if (mod8 != 0) {
        int fill = 8 - mod8;                  // filling this many reaches a byte boundary
        int take = (num < (uint64_t)fill) ? (int)num : fill;
        add_bits_slow(bit, take, output_reg, output_reg_len, out);
        num -= take;
    }

    // 2) Now on a byte boundary. The register does not force 32-bit-granular spills:
    //    output_reg_len is one of 0,8,16,24; at 32, spill_reg_to_vec emits 4 bytes in BE.
    //    But at byte granularity, writing bytes straight into out is simpler/faster.
    //    => bulk byte writes only when the register is empty (= len==0). If len is 8/16/24, fill up to 32 via the slow path first.
    if (num >= 8 && output_reg_len == 0) {
        uint64_t nbytes = num >> 3;          // num / 8
        // fill with 0x00 or 0xFF
        uint8_t bytev = bit ? 0xFF : 0x00;

        // append to out in one go (reserve + insert preferred over a push_back loop)
        size_t old = out.size();
        out.resize(old + (size_t)nbytes);
        std::memset(out.data() + old, bytev, (size_t)nbytes);

        num &= 7; // remaining bits (num % 8)
    }

    // 3) handle the remaining 0-7 tail bits
    if (num) {
        add_bits_slow(bit, (int)num, output_reg, output_reg_len, out);
    }
}

// Append 1 bit and then 'pending_bits' of inverted bit, GPU logic
static inline void append_bit_and_pending(uint32_t bit,
                                          uint64_t &pending_bits,
                                          uint32_t &output_reg,
                                          int      &output_reg_len,
                                          std::vector<uint8_t> &out)
{
    // first the 1 bit
    add_bits_run(bit, 1, output_reg, output_reg_len, out);

    // pending (run of inverted bits)
    if (pending_bits) {
        uint32_t inv = 1u - bit;
        add_bits_run(inv, pending_bits, output_reg, output_reg_len, out);
        pending_bits = 0;
    }
}

// index helpers (contiguous layouts)
// q_chunk: [L, t_chunk, C]  (C fastest)
static inline size_t idx_q(int L, int C, int t_chunk, int l, int t, int c) {
    return (size_t)c + (size_t)C * ((size_t)t + (size_t)t_chunk * (size_t)l);
}
// cdf: [L, C, lp] (bin fastest)
static inline size_t idx_cdf(int L, int C, int lp, int l, int c, int b) {
    return (size_t)b + (size_t)lp * ((size_t)c + (size_t)C * (size_t)l);
}

EncodeCPUResult encode_ntokens_cpu(
    const uint16_t* cdf, int L, int C, int lp,
    const int8_t*  q_chunk, int t_chunk
) {
    if (!cdf || !q_chunk) throw std::runtime_error("encode_ntokens_cpu: null ptr");
    if (L <= 0 || C <= 0 || lp <= 2 || t_chunk <= 0)
        throw std::runtime_error("encode_ntokens_cpu: bad dims");

    // === constants (match CUDA kernel) ===
    constexpr uint32_t precision = 16;
    constexpr uint32_t FULL      = 0xFFFFFFFFu;
    constexpr uint32_t HALF      = 0x80000000u; // 0.5
    constexpr uint32_t QUARTER   = 0x40000000u; // 0.25
    constexpr uint32_t THREEQ    = 0xC0000000u; // 0.75
    constexpr uint32_t CDF_TOP   = 0x10000u;    // 1<<precision

    EncodeCPUResult out;
    out.t_chunk = t_chunk;
    out.lengths.assign((size_t)L * C, 0);

    // rough reserve (can be tuned/removed if desired)
    out.bytestream.reserve((size_t)L * C * (size_t)t_chunk / 2);

    // For each (l,c), arithmetic encode t_chunk symbols
    for (int l = 0; l < L; ++l) {
        for (int c_ = 0; c_ < C; ++c_) {

            // --- state init ---
            uint32_t low = 0u;
            uint32_t high = FULL;
            uint64_t pending_bits = 0;
            const int max_symbol = lp - 2;

            uint32_t output_reg = 0;
            int      output_reg_len = 0;

            // this channel's output start offset (for length computation)
            const size_t start = out.bytestream.size();

            // a little buffer headroom (fewer reallocations; optional)
            out.bytestream.reserve(out.bytestream.size() + (size_t)t_chunk / 2 + 16);

            // linear pointer access
            // q_chunk: [L, t_chunk, C] with idx = c + C*(t + t_chunk*l)
            const int8_t* qptr = q_chunk
                               + (size_t)C * ((size_t)t_chunk * (size_t)l)
                               + (size_t)c_;

            // cdf: [L, C, lp] with idx = sym + lp*(c + C*l)
            const uint16_t* cdf_row = cdf + (size_t)lp * ((size_t)c_ + (size_t)C * (size_t)l);

            // --- main encoding loop ---
            for (int t = 0; t < t_chunk; ++t, qptr += C) {
                // symbol ∈ [0..max_symbol]
                uint8_t sym = (uint8_t)(*qptr);
                if (sym > (uint8_t)max_symbol) sym = (uint8_t)max_symbol; // defensive clamp

                // interval update
                const uint64_t span = (uint64_t)high - (uint64_t)low + 1ull;

                const uint32_t c_low  = (uint16_t) cdf_row[sym];
                const uint32_t c_high = (sym == max_symbol)
                                        ? CDF_TOP
                                        : (uint16_t) cdf_row[sym + 1];

                high = (uint32_t)((uint64_t)(low - 1u) + ((span * (uint64_t)c_high) >> precision));
                low  = (uint32_t)((uint64_t)(low)       + ((span * (uint64_t)c_low ) >> precision));

                // E1/E2/E3 renormalization (keeps the original logic as-is)
                while (true) {
                    if (high < HALF) {
                        // E1: emit 0
                        append_bit_and_pending(0, pending_bits, output_reg, output_reg_len, out.bytestream);
                        low <<= 1; high <<= 1; high |= 1u;
                    } else if (low >= HALF) {
                        // E2: emit 1
                        append_bit_and_pending(1, pending_bits, output_reg, output_reg_len, out.bytestream);
                        low <<= 1; high <<= 1; high |= 1u;
                    } else if (low >= QUARTER && high < THREEQ) {
                        // E3: carry
                        pending_bits++;
                        low  = (low  << 1) & 0x7FFFFFFFu;
                        high = (high << 1) | 0x80000001u;
                    } else {
                        break;
                    }
                }
            }

            // termination (same as the original)
            pending_bits += 1ull;
            if (low < QUARTER) {
                append_bit_and_pending(0, pending_bits, output_reg, output_reg_len, out.bytestream);
            } else {
                append_bit_and_pending(1, pending_bits, output_reg, output_reg_len, out.bytestream);
            }

            // flush tail (partial 32-bits) — directly into the bytestream
            spill_partial_reg_to_vec(output_reg, output_reg_len, out.bytestream);

            // record length (bytes)
            out.lengths[(size_t)c_ + (size_t)C * (size_t)l] =
                (uint8_t)(out.bytestream.size() - start);
        }
    }

    return out;
}


int encode_ntokens_cpu_new(
    const uint16_t* cdf, int C, int lp,
    const int8_t*  q_chunk, int t_chunk,
    uint8_t* bytestream, std::vector<uint8_t>& lengths
) {
    if (!cdf || !q_chunk) throw std::runtime_error("encode_ntokens_cpu: null ptr");
    if (C <= 0 || lp <= 2 || t_chunk <= 0)
        throw std::runtime_error("encode_ntokens_cpu: bad dims");

    // === constants (match CUDA kernel) ===
    constexpr uint32_t precision = 16;
    constexpr uint32_t FULL      = 0xFFFFFFFFu;
    constexpr uint32_t HALF      = 0x80000000u; // 0.5
    constexpr uint32_t QUARTER   = 0x40000000u; // 0.25
    constexpr uint32_t THREEQ    = 0xC0000000u; // 0.75
    constexpr uint32_t CDF_TOP   = 0x10000u;    // 1<<precision

    lengths.assign((size_t)C, 0);

    int bytes_sum = 0;

    // For each (l,c), arithmetic encode t_chunk symbols
    for (int c_ = 0; c_ < C; ++c_) {

        // --- state init ---
        uint32_t low = 0u;
        uint32_t high = FULL;
        uint64_t pending_bits = 0;
        const int max_symbol = lp - 2;

        uint32_t output_reg = 0;
        int      output_reg_len = 0;

        std::vector<uint8_t> bytestream_vec;

        // cdf: [L, C, lp] with idx = sym + lp*(c + C*l)
        const uint16_t* cdf_row = cdf + (size_t)lp * (size_t)c_;

        // --- main encoding loop ---
        for (int t = 0; t < t_chunk; ++t) {
            // symbol ∈ [0..max_symbol]
            uint8_t sym = (uint8_t)q_chunk[t * C + c_];
            if (sym > (uint8_t)max_symbol) sym = (uint8_t)max_symbol; // defensive clamp

            // interval update
            const uint64_t span = (uint64_t)high - (uint64_t)low + 1ull;

            const uint32_t c_low  = (uint16_t) cdf_row[sym];
            const uint32_t c_high = (sym == max_symbol)
                                    ? CDF_TOP
                                    : (uint16_t) cdf_row[sym + 1];

            high = (uint32_t)((uint64_t)(low - 1u) + ((span * (uint64_t)c_high) >> precision));
            low  = (uint32_t)((uint64_t)(low)       + ((span * (uint64_t)c_low ) >> precision));

            // E1/E2/E3 renormalization (keeps the original logic as-is)
            while (true) {
                if (high < HALF) {
                    // E1: emit 0
                    append_bit_and_pending(0, pending_bits, output_reg, output_reg_len, bytestream_vec);
                    low <<= 1; high <<= 1; high |= 1u;
                } else if (low >= HALF) {
                    // E2: emit 1
                    append_bit_and_pending(1, pending_bits, output_reg, output_reg_len, bytestream_vec);
                    low <<= 1; high <<= 1; high |= 1u;
                } else if (low >= QUARTER && high < THREEQ) {
                    // E3: carry
                    pending_bits++;
                    low  = (low  << 1) & 0x7FFFFFFFu;
                    high = (high << 1) | 0x80000001u;
                } else {
                    break;
                }
            }
        }

        // termination (same as the original)
        pending_bits += 1ull;
        if (low < QUARTER) {
            append_bit_and_pending(0, pending_bits, output_reg, output_reg_len, bytestream_vec);
        } else {
            append_bit_and_pending(1, pending_bits, output_reg, output_reg_len, bytestream_vec);
        }

        // flush tail (partial 32-bits) — directly into the bytestream
        spill_partial_reg_to_vec(output_reg, output_reg_len, bytestream_vec);

        // copy to bytestream
        std::memcpy(bytestream + bytes_sum, bytestream_vec.data(), bytestream_vec.size());

        // record length (bytes)
        lengths[(size_t)c_] = (uint8_t)(bytestream_vec.size());

        bytes_sum += bytestream_vec.size();
    }

    return bytes_sum;
}

int encode_ntokens_cpu_new_range(
    const uint16_t* cdf, int C, int lp,
    const int8_t*  q_chunk, int t_chunk,
    uint8_t* bytestream, std::vector<uint8_t>& lengths
) {
    if (!cdf || !q_chunk) throw std::runtime_error("encode_ntokens_cpu: null ptr");
    if (C <= 0 || lp <= 2 || t_chunk <= 0)
        throw std::runtime_error("encode_ntokens_cpu: bad dims");

    // === constants (match CUDA kernel) ===
    constexpr uint32_t precision = 16;
    constexpr uint32_t FULL      = 0xFFFFFFFFu;   // initial range
    constexpr uint32_t HALF      = 0x80000000u;   // 0.5
    constexpr uint32_t QUARTER   = 0x40000000u;   // 0.25
    constexpr uint32_t CDF_TOP   = 0x10000u;      // 1<<precision

    lengths.assign((size_t)C, 0);
    int bytes_sum = 0;

    for (int c_ = 0; c_ < C; ++c_) {
        // --- state: (low, range) ---
        uint32_t low   = 0u;
        uint32_t range = FULL;
        uint64_t pending_bits = 0;
        const int max_symbol = lp - 2;

        uint32_t output_reg = 0;
        int      output_reg_len = 0;

        std::vector<uint8_t> bytestream_vec;
        bytestream_vec.reserve((size_t)t_chunk / 2 + 16);

        // cdf: [C, lp] with idx = sym + lp*c
        const uint16_t* cdf_row = cdf + (size_t)lp * (size_t)c_;

        // --- main encoding loop ---
        for (int t = 0; t < t_chunk; ++t) {
            // symbol ∈ [0..max_symbol]
            uint32_t sym = (uint8_t)q_chunk[(size_t)t * (size_t)C + (size_t)c_];
            if ((int)sym > max_symbol) sym = (uint32_t)max_symbol; // defensive clamp

            const uint32_t c_lo = (uint32_t)cdf_row[sym];
            const uint32_t c_hi = (sym == (uint32_t)max_symbol) ? CDF_TOP
                                                                : (uint32_t)cdf_row[sym + 1];

            // --- (low,range) update ---
            // base = floor(range * c_lo / 2^precision)
            // size = floor(range * (c_hi - c_lo) / 2^precision)
            const uint64_t base64 = ((uint64_t)range * (uint64_t)c_lo) >> precision;
            const uint64_t size64 = ((uint64_t)range * (uint64_t)(c_hi - c_lo)) >> precision;

            low   = (uint32_t)((uint64_t)low + base64);
            range = (uint32_t)size64;

            // --- renorm (E1/E2/E3) ---
            // Repeat until range grows large enough.
            // E1: low < QUARTER  → emit 0
            // E2: low ≥ HALF     → emit 1, low -= HALF
            // E3: QUARTER ≤ low < HALF → pending_bits++, low -= QUARTER
            while (range <= QUARTER) {
                if (low < QUARTER) {
                    append_bit_and_pending(0, pending_bits, output_reg, output_reg_len, bytestream_vec);
                } else if (low >= HALF) {
                    append_bit_and_pending(1, pending_bits, output_reg, output_reg_len, bytestream_vec);
                    low -= HALF;
                } else {
                    pending_bits++;
                    low -= QUARTER;
                }
                low   <<= 1;
                range <<= 1;
            }
        }

        // --- termination ---
        // Finally pick one bit to emit so pending_bits gets flushed
        pending_bits += 1ull;
        if (low < QUARTER) {
            append_bit_and_pending(0, pending_bits, output_reg, output_reg_len, bytestream_vec);
        } else {
            append_bit_and_pending(1, pending_bits, output_reg, output_reg_len, bytestream_vec);
        }

        // flush the remaining bits
        spill_partial_reg_to_vec(output_reg, output_reg_len, bytestream_vec);

        // copy to bytestream
        std::memcpy(bytestream + bytes_sum, bytestream_vec.data(), bytestream_vec.size());

        // record length (bytes)
        lengths[(size_t)c_] = (uint8_t)(bytestream_vec.size());
        bytes_sum += (int)bytestream_vec.size();
    }

    return bytes_sum;
}



/////////////////////////////////////////////////////////////////////////////////////////////////
// ---------------- decode (CPU) ----------------
static inline uint32_t load_u32_be(const uint8_t* p, size_t n_avail) {
    uint32_t b0 = n_avail > 0 ? p[0] : 0;
    uint32_t b1 = n_avail > 1 ? p[1] : 0;
    uint32_t b2 = n_avail > 2 ? p[2] : 0;
    uint32_t b3 = n_avail > 3 ? p[3] : 0;
    return (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;
}

struct BitReader {
    const uint8_t* base = nullptr;
    int len = 0, pos = 0;
    uint8_t byte = 0;
    int bit_idx = 1; // MSB-first (1..8)

    BitReader(const uint8_t* p, int n, int start)
        : base(p), len(n), pos(start) {
        byte = (pos < len) ? base[pos] : 0;
    }
    inline uint32_t next_bit() {
        uint32_t b = (byte >> (8 - bit_idx)) & 1u;
        if (++bit_idx == 9) {
            bit_idx = 1;
            if (++pos < len) byte = base[pos];
            else byte = 0;
        }
        return b;
    }
};

static inline int binsearch_symbol(const uint16_t* cdf_chan, int bins_p1, uint16_t count) {

    int lo = 0, hi = bins_p1 - 2; // max_symbol
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        uint16_t c_lo = cdf_chan[mid];
        uint16_t c_hi = (mid == bins_p1 - 2) ? 0x10000u : cdf_chan[mid + 1];
        if (count < c_lo) hi = mid - 1;
        else if (count >= c_hi) lo = mid + 1;
        else return mid;
    }
    if (lo <= 0) return 0;
    if (lo > bins_p1 - 2) return bins_p1 - 2;
    return lo - 1;
}

void dequantize_cpu_into(
    const int8_t* q,          // [L,T,C]
    const float*  max1,       // [L,T]
    int L, int T, int C,
    int bins,                 // e.g. 32
    __fp16* out               // [L,T,C] to write into
) {
    const int   C_scale_i = bins / 2 - 1;
    const float C_scale   = (float)C_scale_i;
    if (C_scale_i <= 0) { std::fill_n(out, (size_t)L*T*C, (__fp16)0); return; }

    const float invC = 1.0f / C_scale;

    auto idx = [&](int l, int t, int c, int T, int C) -> size_t {
        return (size_t)c + (size_t)C * ((size_t)t + (size_t)T * (size_t)l);
    };

    for (int l = 0; l < L; ++l) {
        for (int t = 0; t < T; ++t) {
            const float mul = max1[(size_t)t + (size_t)T * l] * invC;
            for (int c = 0; c < C; ++c) {
                const int8_t qv = q[idx(l,t,c,T,C)];
                const float  f  = ((float)qv - C_scale) * mul;
                out[idx(l,t,c,T,C)] = (__fp16)f;
            }
        }
    }
}


std::vector<int8_t> decode_ntokens_cpu(
    const uint16_t* cdf,  // [L, C, bins_p1] as int16_t
    int L, int C, int bins_p1,
    const EncodeCPUResult& chunk
) {
    if (L != 1) throw std::runtime_error("decode_ntokens_cpu assumes L=1");
    const int t_chunk = chunk.t_chunk;
    const int bins    = bins_p1 - 1;
    const int max_symbol = bins - 1;

    std::vector<int8_t> out((size_t)L * t_chunk * C);

    if ((int)chunk.lengths.size() != L*C) {
        throw std::runtime_error("lengths size mismatch");
    }
    std::vector<int> offsets(L*C, 0);
    int acc = 0;
    for (int c = 0; c < C; ++c) {
        offsets[c] = acc;
        acc += chunk.lengths[c];
    }
    if (acc != (int)chunk.bytestream.size()) {
        throw std::runtime_error("bytestream total size != sum(lengths)");
    }

    auto idx_out = [&](int l, int t, int c) -> size_t {
        return (size_t)c + (size_t)C * ((size_t)t + (size_t)t_chunk * (size_t)l);
    };
    auto ptr_cdf_chan = [&](int l, int c) -> const uint16_t* {
        return reinterpret_cast<const uint16_t*>(cdf) + (size_t)bins_p1 * ((size_t)c + (size_t)C * (size_t)l);
    };

    for (int c = 0; c < C; ++c) {
        const int len = chunk.lengths[c];
        const int offset = offsets[c];
        const uint8_t* bs = chunk.bytestream.data() + offset;

        uint32_t low = 0, high = 0xFFFFFFFFu;
        uint32_t value = load_u32_be(bs, len >= 4 ? 4 : (len > 0 ? len : 0));
        BitReader br(bs, len, /*start*/4);

        const uint16_t* cdf_chan = ptr_cdf_chan(0, c);

        for (int t = 0; t < t_chunk; ++t) {
            const uint64_t span = (uint64_t)high - (uint64_t)low + 1ull;
            uint16_t count = (uint16_t)((((uint64_t)value - (uint64_t)low + 1ull) * 0x10000ull - 1ull) / span);
            int sym = binsearch_symbol(cdf_chan, bins_p1, count);
            out[idx_out(0, t, c)] = (int8_t)sym;

            if (t == t_chunk - 1) break;

            const uint32_t c_lo = cdf_chan[sym];
            const uint32_t c_hi = (sym == max_symbol) ? 0x10000u : cdf_chan[sym + 1];
            high = (uint32_t)((uint64_t)(low - 1u) + ((span * (uint64_t)c_hi) >> 16));
            low  = (uint32_t)((uint64_t) low       + ((span * (uint64_t)c_lo) >> 16));

            while (true) {
                if (low >= 0x80000000u || high < 0x80000000u) {
                    low <<= 1; high <<= 1; high |= 1u;
                    value = (value << 1) | br.next_bit();
                } else if (low >= 0x40000000u && high < 0xC0000000u) {
                    low <<= 1; low &= 0x7FFFFFFFu;
                    high <<= 1; high |= 0x80000001u;
                    value -= 0x40000000u;
                    value = (value << 1) | br.next_bit();
                } else break;
            }
        }
    }

    return out;
}


void dequantize_cpu_into_new(
    const int8_t* q,          // [L,T,C]
    const float*  max1,       // [L,T]
    int T, int C,
    int bins,                 // e.g. 32
    __fp16* out               // [L,T,C] to write into
) {
    const int   C_scale_i = bins / 2 - 1;
    const float C_scale   = (float)C_scale_i;
    if (C_scale_i <= 0) { std::fill_n(out, (size_t)T*C, (__fp16)0); return; }

    const float invC = 1.0f / C_scale;

    auto idx = [&](int t, int c, int T, int C) -> size_t {
        return (size_t)c + (size_t)C * ((size_t)t);
    };

    for (int t = 0; t < T; ++t) {
        const float mul = max1[(size_t)t] * invC;
        for (int c = 0; c < C; ++c) {
            const int8_t qv = q[idx(t,c,T,C)];
            const float  f  = ((float)qv - C_scale) * mul;
            out[idx(t,c,T,C)] = (__fp16)f;
        }
    }
}

void dequantize_cpu_into_new_neon(
    const int8_t* __restrict q,   // [T,C]
    const float*  __restrict max1,// [T]
    int T, int C,
    int bins,
    __fp16* __restrict out        // [T,C]
) {
#if !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    // without fp16 vector arithmetic, fall back to the original
    std::cout << "dequantize_cpu_into_new_neon: no FP16 vector arithmetic, fallback to scalar\n";
    return dequantize_cpu_into_new(q, max1, T, C, bins, out);
#else
    const int C_scale_i = bins / 2 - 1;
    if (C_scale_i <= 0) {
        for (int t = 0; t < T; ++t)
            for (int c = 0; c < C; ++c)
                out[(size_t)t*C + c] = (__fp16)0;
        return;
    }

    const float invC = 1.0f / (float)C_scale_i;
    const int C_vec = (C & ~15); // multiples of 16

    for (int t = 0; t < T; ++t) {
        const __fp16 mul_h = (__fp16)(max1[(size_t)t] * invC);
        const float16x8_t vmul = vdupq_n_f16(mul_h);
        const int8_t*  q_row   = q   + (size_t)t*C;
        __fp16*        o_row   = out + (size_t)t*C;

        int c = 0;
        for (; c < C_vec; c += 16) {
            // load 16×s8
            int8x16_t vq8 = vld1q_s8(q_row + c);

            // s8 -> s16 (low/high)
            int16x8_t vq16_lo = vmovl_s8(vget_low_s8(vq8));
            int16x8_t vq16_hi = vmovl_s8(vget_high_s8(vq8));

            // subtract C_scale_i
            const int16x8_t vshift = vdupq_n_s16((int16_t)C_scale_i);
            vq16_lo = vsubq_s16(vq16_lo, vshift);
            vq16_hi = vsubq_s16(vq16_hi, vshift);

            // s16 -> f16
            float16x8_t vf16_lo = vcvtq_f16_s16(vq16_lo);
            float16x8_t vf16_hi = vcvtq_f16_s16(vq16_hi);

            // scale & store
            vf16_lo = vmulq_f16(vf16_lo, vmul);
            vf16_hi = vmulq_f16(vf16_hi, vmul);
            vst1q_f16(o_row + c,      vf16_lo);
            vst1q_f16(o_row + c + 8,  vf16_hi);
        }
        // tail
        for (; c < C; ++c) {
            const __fp16 f = (__fp16)(((__fp16)q_row[c] - (__fp16)C_scale_i) * mul_h);
            o_row[c] = f;
        }
    }
#endif
}



static inline int symbol_from_cdf_neon(const uint16_t* __restrict cdf_chan,
                                       int bins_p1, uint16_t count) {
    const int bins = bins_p1 - 1;
    int i = 0;
    const uint16x8_t vcount = vdupq_n_u16(count);

    // scan 8 at a time
    for (; i + 8 <= bins; i += 8) {
        const uint16x8_t vlo = vld1q_u16(cdf_chan + i);
        const uint16x8_t vhi = vld1q_u16(cdf_chan + i + 1);

        const uint16x8_t ge_lo = vcgeq_u16(vcount, vlo);
        const uint16x8_t lt_hi = vcltq_u16(vcount, vhi);
        const uint16x8_t ok    = vandq_u16(ge_lo, lt_hi);

        // find the first matching index (safely, in scalar)
        uint16_t mask[8];
        vst1q_u16(mask, ok);
        for (int k = 0; k < 8; ++k) {
            if (mask[k] == 0xFFFFu) return i + k;
        }
    }

    return bins - 1; // safety guard
}

std::vector<int8_t> decode_ntokens_cpu_new(
    const uint16_t* cdf,  // [L, C, bins_p1] as int16_t
    int C, int bins_p1,
    const uint8_t* input, std::vector<uint8_t>& lengths
) {

    const int t_chunk = MAX_TOKENS_PER_CHUNK;
    const int bins    = bins_p1 - 1;
    const int max_symbol = bins - 1;

    std::vector<int8_t> out(t_chunk * C);

    if ((int)lengths.size() != C) {
        throw std::runtime_error("lengths size mismatch");
    }
    std::vector<int> offsets(C, 0);
    int acc = 0;
    for (int c = 0; c < C; ++c) {
        offsets[c] = acc;
        acc += lengths[c];
    }

    for (int c = 0; c < C; ++c) {
        const int len = lengths[c];
        const int offset = offsets[c];
        const uint8_t* bs = input + offset;

        uint32_t low = 0, high = 0xFFFFFFFFu;
        uint32_t value = load_u32_be(bs, len >= 4 ? 4 : (len > 0 ? len : 0));
        BitReader br(bs, len, /*start*/4);

        const uint16_t* cdf_chan = cdf + (size_t)bins_p1 * (size_t)c;

        for (int t = 0; t < t_chunk; ++t) {
            const uint64_t span = (uint64_t)high - (uint64_t)low + 1ull;
            uint16_t count = (uint16_t)((((uint64_t)value - (uint64_t)low + 1ull) * 0x10000ull - 1ull) / span);
            int sym = binsearch_symbol(cdf_chan, bins_p1, count);
            out[t * C + c] = (int8_t)sym;
            // out[c * t_chunk + t] = (int8_t)sym;


            if (t == t_chunk - 1) break;

            const uint32_t c_lo = cdf_chan[sym];
            const uint32_t c_hi = (sym == max_symbol) ? 0x10000u : cdf_chan[sym + 1];
            high = (uint32_t)((uint64_t)(low - 1u) + ((span * (uint64_t)c_hi) >> 16));
            low  = (uint32_t)((uint64_t) low       + ((span * (uint64_t)c_lo) >> 16));

            while (true) {
                if (low >= 0x80000000u || high < 0x80000000u) {
                    low <<= 1; high <<= 1; high |= 1u;
                    value = (value << 1) | br.next_bit();
                } else if (low >= 0x40000000u && high < 0xC0000000u) {
                    low <<= 1; low &= 0x7FFFFFFFu;
                    high <<= 1; high |= 0x80000001u;
                    value -= 0x40000000u;
                    value = (value << 1) | br.next_bit();
                } else break;
            }
        }
    }

    return out;
}

std::vector<int8_t> decode_ntokens_cpu_new_range(
    const uint16_t* cdf,  // [L, C, bins_p1]
    int C, int bins_p1,
    const uint8_t* input, std::vector<uint8_t>& lengths
) {
    constexpr uint32_t Q          = 16;           // Q16
    constexpr uint32_t SCALE      = 1u << Q;      // 0x10000
    constexpr uint32_t FULL       = 0xFFFFFFFFu;  // initial range
    constexpr uint32_t HALF       = 0x80000000u;  // 0.5
    constexpr uint32_t QUARTER    = 0x40000000u;  // 0.25

    const int t_chunk    = MAX_TOKENS_PER_CHUNK;
    const int bins       = bins_p1 - 1;
    const int max_symbol = bins - 1;

    if ((int)lengths.size() != C) {
        throw std::runtime_error("lengths size mismatch");
    }

    std::vector<int8_t> out((size_t)t_chunk * C);

    // per-channel byte offsets
    std::vector<int> offsets(C);
    int acc = 0;
    for (int c = 0; c < C; ++c) {
        offsets[c] = acc;
        acc += lengths[c];
    }

    for (int c = 0; c < C; ++c) {
        const int len = lengths[c];
        const uint8_t* bs = input + offsets[c];

        // switched to (low, range) state
        uint32_t low   = 0u;
        uint32_t range = FULL;

        // load the initial code value (up to 4 bytes)
        uint32_t value = load_u32_be(bs, len >= 4 ? 4 : (len > 0 ? len : 0));
        BitReader br(bs, len, /*start*/4);

        const uint16_t* cdf_chan = cdf + (size_t)bins_p1 * (size_t)c;

        for (int t = 0; t < t_chunk; ++t) {
            // count used for symbol search (Q16 basis)
            // map value in [low, low+range) onto the SCALE scale
            // (keeps the same form as the existing inclusive version)
            const uint64_t num   = ((uint64_t)value - (uint64_t)low + 1ull) * (uint64_t)SCALE - 1ull;
            const uint16_t count = (uint16_t)(num / (uint64_t)range);

            // // find the symbol in the CDF
            const int sym = binsearch_symbol(cdf_chan, bins_p1, count);

            out[(size_t)t * (size_t)C + (size_t)c] = (int8_t)sym;
            

            if (t == t_chunk - 1) break;

            // interval update: [low, low+range) → [low+base, low+base+size)
            const uint32_t c_lo = (uint32_t)cdf_chan[sym];
            const uint32_t c_hi = (sym == max_symbol) ? SCALE
                                                      : (uint32_t)cdf_chan[sym + 1];

            const uint32_t base  = (uint32_t)(((uint64_t)range * c_lo) >> Q);
            const uint32_t size  = (uint32_t)(((uint64_t)range * (c_hi - c_lo)) >> Q);
            low   += base;
            range  = size;       // keep 1 <= range <= 0xFFFFFFFF

            // renorm: grow range + keep value in sync (E1/E2/E3)
            //  - low < QUARTER           : E1
            //  - low >= HALF             : E2 (subtract HALF from both low and value)
            //  - QUARTER <= low < HALF   : E3 (subtract QUARTER from both low and value)
            while (range <= QUARTER) {
                if (low >= HALF) {
                    low   -= HALF;
                    value -= HALF;
                } else if (low >= QUARTER) {
                    low   -= QUARTER;
                    value -= QUARTER;
                }
                low   <<= 1;
                range <<= 1;
                value  = (value << 1) | br.next_bit();
            }
        }
    }

    return out;
}



/////////////////////////////////////////////////////////////////////////////////////////////////


// (low,range) 4-lane update (Q16). For Q15, only Q/SHIFT/SCALE need to change.
static inline void update_range4_neon_rangeQ16(
    const uint32_t low_s[4],
    const uint32_t range_s[4],
    const uint16_t* cdf_chans[4],
    const int sym[4],
    int max_symbol,
    uint32_t out_low[4],
    uint32_t out_range[4]
){
    // constexpr uint32_t SCALE = 0x10000u;

    // // 1) gather
    // uint32_t clo_u32[4], chi_u32[4];
    // for (int i = 0; i < 4; ++i) {
    //     const uint16_t* c = cdf_chans[i];
    //     const int s = sym[i];
    //     clo_u32[i] = (uint32_t)c[s];
    //     chi_u32[i] = (s == max_symbol) ? SCALE : (uint32_t)c[s + 1];
    // }

    // // 2) load
    // const uint32x4_t vlow   = vld1q_u32(low_s);
    // const uint32x4_t vrange = vld1q_u32(range_s);
    // const uint32x4_t vclo   = vld1q_u32(clo_u32);
    // const uint32x4_t vchi   = vld1q_u32(chi_u32);
    // const uint32x4_t vdiff  = vsubq_u32(vchi, vclo); // c_hi - c_lo

    // // 3) base = (range*c_lo)>>16  (multiply lo/hi separately, combine while narrowing)
    // const uint64x2_t base_lo64 = vmull_u32(vget_low_u32(vrange), vget_low_u32(vclo));
    // const uint64x2_t base_hi64 = vmull_high_u32(vrange, vclo);
    // const uint32x2_t base_lo32 = vshrn_n_u64(base_lo64, 16);
    // const uint32x4_t base32    = vshrn_high_n_u64(base_lo32, base_hi64, 16);

    // //     size = (range*(c_hi-c_lo))>>16
    // const uint64x2_t size_lo64 = vmull_u32(vget_low_u32(vrange), vget_low_u32(vdiff));
    // const uint64x2_t size_hi64 = vmull_high_u32(vrange, vdiff);
    // const uint32x2_t size_lo32 = vshrn_n_u64(size_lo64, 16);
    // const uint32x4_t size32    = vshrn_high_n_u64(size_lo32, size_hi64, 16);

    // // 4) low += base ; range = size
    // const uint32x4_t vnewlow   = vaddq_u32(vlow, base32);
    // const uint32x4_t vnewrange = size32;

    // // 5) store
    // vst1q_u32(out_low,   vnewlow);
    // vst1q_u32(out_range, vnewrange);


    constexpr uint32_t SCALE = 0x10000u; // 1<<16
    for (int i = 0; i < 4; ++i) {
        const uint16_t* cdfc = cdf_chans[i];
        const int s = sym[i];

        const uint32_t c_lo = (uint32_t)cdfc[s];
        const uint32_t c_hi = (s == max_symbol) ? SCALE : (uint32_t)cdfc[s + 1];
        const uint32_t rng  = range_s[i];
        const uint32_t lo   = low_s[i];

        // base = floor(range * c_lo / 2^16)
        // size = floor(range * (c_hi - c_lo) / 2^16)
        const uint64_t base64 = ((uint64_t)rng * (uint64_t)c_lo) >> 16;
        const uint64_t size64 = ((uint64_t)rng * (uint64_t)(c_hi - c_lo)) >> 16;

        out_low [i] = (uint32_t)( (uint64_t)lo + base64 );
        out_range[i] = (uint32_t) size64;
    }
}


static inline uint32x4_t read_next_bits_masked(const uint32x4_t mask, BitReader brs[4]) {
    // read a bit for each lane whose mask is 0xFFFFFFFF, otherwise 0
    uint32_t bits[4];
    uint32_t m[4];
    vst1q_u32(m, mask);
    for (int i = 0; i < 4; ++i) {
        bits[i] = (m[i] == 0xFFFFFFFFu) ? (uint32_t)brs[i].next_bit() : 0u;
    }
    return vld1q_u32(bits);
}

std::vector<int8_t> decode_ntokens_cpu_new_simd_range(
    const uint16_t* cdf,  // [C, bins_p1]
    int C, int bins_p1,
    const uint8_t* input, std::vector<uint8_t>& lengths
) {
    // --- constants (Q16). To drop to Q15: SCALE=0x8000, SHIFT=15, QUARTER/HALF unchanged ---
    constexpr uint32_t Q        = 16;
    constexpr uint32_t SCALE    = 1u << Q;       // 0x10000
    constexpr uint32_t FULL     = 0xFFFFFFFFu;   // initial range
    constexpr uint32_t HALF     = 0x80000000u;   // 0.5
    constexpr uint32_t QUARTER  = 0x40000000u;   // 0.25

    const int t_chunk    = MAX_TOKENS_PER_CHUNK;
    const int bins       = bins_p1 - 1;
    const int max_symbol = bins - 1;

    if ((int)lengths.size() != C) throw std::runtime_error("lengths size mismatch");

    std::vector<int8_t> out((size_t)t_chunk * C);

    // per-channel offsets
    std::vector<int> offsets(C);
    int acc = 0;
    for (int c = 0; c < C; ++c) { offsets[c] = acc; acc += lengths[c]; }

    const int C_vec = (C & ~3);
    int c = 0;
    for (; c < C_vec; c += 4) {
        // (low, range, value) state
        uint32_t low  [4] = {0,0,0,0};
        uint32_t range[4] = {FULL,FULL,FULL,FULL};
        uint32_t value[4];

        BitReader brs[4] = {
            BitReader(input + offsets[c+0], lengths[c+0], 4),
            BitReader(input + offsets[c+1], lengths[c+1], 4),
            BitReader(input + offsets[c+2], lengths[c+2], 4),
            BitReader(input + offsets[c+3], lengths[c+3], 4),
        };
        value[0] = load_u32_be(brs[0].base, std::min(4, std::max(0, brs[0].len)));
        value[1] = load_u32_be(brs[1].base, std::min(4, std::max(0, brs[1].len)));
        value[2] = load_u32_be(brs[2].base, std::min(4, std::max(0, brs[2].len)));
        value[3] = load_u32_be(brs[3].base, std::min(4, std::max(0, brs[3].len)));

        const uint16_t* cdf_chans[4] = {
            cdf + (size_t)bins_p1 * (size_t)(c + 0),
            cdf + (size_t)bins_p1 * (size_t)(c + 1),
            cdf + (size_t)bins_p1 * (size_t)(c + 2),
            cdf + (size_t)bins_p1 * (size_t)(c + 3),
        };

        for (int t = 0; t < t_chunk; ++t) {
            int sym[4];

            // compute count + symbol search
            for (int i = 0; i < 4; ++i) {
                // count = ((((value - low) + 1) * SCALE - 1) / range)
                const uint64_t num   = ((uint64_t)value[i] - (uint64_t)low[i] + 1ull) * (uint64_t)SCALE - 1ull;
                const uint16_t count = (uint16_t)(num / (uint64_t)range[i]);

                // sym[i] = symbol_from_cdf_neon(cdf_chans[i], bins_p1, count);
                sym[i] = binsearch_symbol(cdf_chans[i], bins_p1, count);
                out[(size_t)t * C + (size_t)(c + i)] = (int8_t)sym[i];
            }
            if (t == t_chunk - 1) break;

            // range update: [low, low+range) → [low+base, low+base+size)
            uint32_t new_low[4], new_range[4];
            update_range4_neon_rangeQ16(low, range, cdf_chans, sym, /*max_symbol=*/max_symbol,
                                        new_low, new_range);
            for (int i = 0; i < 4; ++i) { low[i] = new_low[i]; range[i] = new_range[i]; }

            // per-lane renorm (grow range)
            for (int i = 0; i < 4; ++i) {
                while (range[i] <= QUARTER) {
                    if (low[i] >= HALF) {
                        low[i]   -= HALF;
                        value[i] -= HALF;
                    } else if (low[i] >= QUARTER) {
                        low[i]   -= QUARTER;
                        value[i] -= QUARTER;
                    }
                    low[i]   <<= 1;
                    range[i] <<= 1;
                    value[i]  = (value[i] << 1) | brs[i].next_bit();
                }
            }
        }
    }

    
    return out;
}

static inline void update_range2_neon_rangeQ16(
    const uint32_t low_s[2],                 // low[2]
    const uint32_t range_s[2],               // range[2]
    const uint16_t* cdf_chans[2],            // per-lane cdf base ptr
    const int sym[2],                        // per-lane symbol
    int max_symbol,
    uint32_t out_low[2],                     // new low[2]
    uint32_t out_range[2]                    // new range[2]
){
    // ===== 1) gather c_lo, c_hi (scalar gather → vector ops) =====
    constexpr uint32_t SCALE = 0x10000u;     // 1<<16
    uint32_t clo_u32[2], chi_u32[2];
    for (int i = 0; i < 2; ++i) {
        const uint16_t* cdfc = cdf_chans[i];
        const int s = sym[i];
        const uint32_t c_lo = (uint32_t)cdfc[s];
        const uint32_t c_hi = (s == max_symbol) ? SCALE : (uint32_t)cdfc[s + 1];
        clo_u32[i] = c_lo;
        chi_u32[i] = c_hi;
    }

    // ===== 2) vector loads =====
    const uint32x2_t vlow   = vld1_u32(low_s);      // low[0..1]
    const uint32x2_t vrange = vld1_u32(range_s);    // range[0..1]
    const uint32x2_t vclo   = vld1_u32(clo_u32);    // c_lo
    const uint32x2_t vchi   = vld1_u32(chi_u32);    // c_hi
    const uint32x2_t vdiff  = vsub_u32(vchi, vclo); // (c_hi - c_lo)

    // ===== 3) 32x32 -> 64 multiply, >> 16 =====
    // base = (range * c_lo) >> 16
    // size = (range * (c_hi - c_lo)) >> 16
    uint64x2_t base64 = vmull_u32(vrange, vclo);
    uint64x2_t size64 = vmull_u32(vrange, vdiff);
    base64 = vshrq_n_u64(base64, 16);
    size64 = vshrq_n_u64(size64, 16);

    // (optional) defensive clamp: prevents size==0 (unnecessary with a well-formed CDF)
    // const uint64x2_t one64 = vdupq_n_u64(1ull);
    // size64 = vmaxq_u64(size64, one64);

    // ===== 4) low += base ; range = size =====
    const uint64x2_t low64 = vaddq_u64(vmovl_u32(vlow), base64);
    const uint32x2_t new_low   = vmovn_u64(low64);
    const uint32x2_t new_range = vmovn_u64(size64);

    vst1_u32(out_low,   new_low);
    vst1_u32(out_range, new_range);
}

std::vector<int8_t> decode_ntokens_cpu_new_range_pairs(
    const uint16_t* cdf,  // [C, bins_p1]
    int C, int bins_p1,
    const uint8_t* input, std::vector<uint8_t>& lengths
) {
    constexpr uint32_t Q        = 16;           // Q16
    constexpr uint32_t SCALE    = 1u << Q;      // 0x10000
    constexpr uint32_t FULL     = 0xFFFFFFFFu;  // initial range
    constexpr uint32_t HALF     = 0x80000000u;  // 0.5
    constexpr uint32_t QUARTER  = 0x40000000u;  // 0.25

    const int t_chunk    = MAX_TOKENS_PER_CHUNK;
    const int bins       = bins_p1 - 1;
    const int max_symbol = bins - 1;

    if ((int)lengths.size() != C) {
        throw std::runtime_error("lengths size mismatch");
    }

    std::vector<int8_t> out((size_t)t_chunk * C);

    // per-channel offsets
    std::vector<int> offsets(C);
    int acc = 0;
    for (int c = 0; c < C; ++c) { offsets[c] = acc; acc += lengths[c]; }

    int c = 0;

    // === process channels in pairs ===
    const int C_pair = (C & ~1);
    for (; c < C_pair; c += 2) {
        // state
        uint32_t low   [2] = {0,0};
        uint32_t range [2] = {FULL, FULL};
        uint32_t value [2];

        BitReader brs[2] = {
            BitReader(input + offsets[c+0], lengths[c+0], 4),
            BitReader(input + offsets[c+1], lengths[c+1], 4),
        };
        value[0] = load_u32_be(brs[0].base, brs[0].len >= 4 ? 4 : (brs[0].len > 0 ? brs[0].len : 0));
        value[1] = load_u32_be(brs[1].base, brs[1].len >= 4 ? 4 : (brs[1].len > 0 ? brs[1].len : 0));

        const uint16_t* cdf_chans[2] = {
            cdf + (size_t)bins_p1 * (size_t)(c + 0),
            cdf + (size_t)bins_p1 * (size_t)(c + 1),
        };

        for (int t = 0; t < t_chunk; ++t) {
            int sym[2];

            // compute count + symbol search (exact division stays scalar)
            for (int i = 0; i < 2; ++i) {
                const uint64_t num   = ((uint64_t)value[i] - (uint64_t)low[i] + 1ull) * (uint64_t)SCALE - 1ull;
                const uint16_t count = (uint16_t)(num / (uint64_t)range[i]);
                sym[i] = binsearch_symbol(cdf_chans[i], bins_p1, count);
                out[(size_t)t * C + (size_t)(c + i)] = (int8_t)sym[i];
            }
            if (t == t_chunk - 1) break;

            // range update: [low, low+range) → [low+base, low+base+size)  (2-lane NEON)
            uint32_t new_low[2], new_range[2];
            update_range2_neon_rangeQ16(low, range, cdf_chans, sym, /*max_symbol=*/max_symbol,
                                        new_low, new_range);
            low[0] = new_low[0]; low[1] = new_low[1];
            range[0] = new_range[0]; range[1] = new_range[1];

            // per-lane renorm (preserves exactness)
            for (int i = 0; i < 2; ++i) {
                while (range[i] <= QUARTER) {
                    if (low[i] >= HALF) {
                        low[i]   -= HALF;
                        value[i] -= HALF;
                    } else if (low[i] >= QUARTER) {
                        low[i]   -= QUARTER;
                        value[i] -= QUARTER;
                    }
                    low[i]   <<= 1;
                    range[i] <<= 1;
                    value[i]  = (value[i] << 1) | brs[i].next_bit();
                }
            }
        }
    }
    return out;
}