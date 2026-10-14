#include "w8a8.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#if GGML_SYCL_DNNL
#include "dequantize.hpp"
#include "gemm.hpp"
struct w8a8_prof {
    const bool enabled = getenv("GGML_W8A8_PROF") != nullptr;
    double ms[5] = {};
    int    calls = 0;
    ~w8a8_prof() {
        if (!enabled || !calls) return;
        static const char * names[5] = { "row_scales", "weights", "acts", "gemm_s8", "out_scale" };
        double total = 0;
        for (double m : ms) total += m;
        fprintf(stderr, "\n[w8a8 prof] %d calls, %.1f ms total\n", calls, total);
        for (int i = 0; i < 5; ++i) {
            fprintf(stderr, "  %-10s %9.1f ms  (%4.1f%%)  %.3f ms/call\n", names[i], ms[i],
                    total > 0 ? 100.0 * ms[i] / total : 0.0, ms[i] / calls);
        }
    }
};
static w8a8_prof g_prof;
template <typename F>
static inline void w8a8_stage(int idx, dpct::queue_ptr stream, F && f) {
    if (!g_prof.enabled) {
        f();
        return;
    }
    stream->wait();
    const auto t0 = std::chrono::steady_clock::now();
    f();
    stream->wait();
    g_prof.ms[idx] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}
// per row: D = max over groups of d, ratio q_g = round(127 * d_g / D); one work-group (16 lanes) per row
static void w8a8_row_scales(ggml_type type, bool reordered, const void * w, float * row_scale, int8_t * ratio,
                            int64_t nrows, int64_t K, dpct::queue_ptr stream) {
    const int64_t   ng   = K / 128;
    const int64_t   nb   = nrows * ng;
    const uint8_t * base = (const uint8_t *) w;
    const bool      ptq  = type == GGML_TYPE_PTQ1_0;
    stream->parallel_for(sycl::nd_range<1>(nrows * 16, 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const int64_t r    = it.get_group(0);
        const int     lane = it.get_local_id(0);
        auto dget = [&](int64_t g) -> float {
            const int64_t ib = r * ng + g;
            if (ptq) {
                return reordered ? (float) ((const sycl::half *) (base + nb * 26))[ib] : (float) ((const block_ptq1_0 *) w)[ib].d;
            }
            return reordered ? (float) ((const sycl::half *) (base + nb * 32))[ib] : (float) ((const block_pq2_0 *) w)[ib].d;
        };
        float m = 0.0f;
        for (int64_t g = lane; g < ng; g += 16) {
            m = sycl::fmax(m, sycl::fabs(dget(g)));
        }
        m = sycl::reduce_over_group(it.get_sub_group(), m, sycl::maximum<float>());
        for (int64_t g = lane; g < ng; g += 16) {
            ratio[r * ng + g] = m > 0 ? (int8_t) sycl::round(127.0f * dget(g) / m) : 0;
        }
        if (lane == 0) {
            row_scale[r] = m / 127.0f;
        }
    });
}

// int8 weights = (code value) * ratio of its group; 8 elements per work item
static void w8a8_weights(ggml_type type, bool reordered, const void * w, const int8_t * ratio, int8_t * w8,
                         int64_t nrows, int64_t K, dpct::queue_ptr stream) {
    const int64_t   nb   = nrows * K / 128;
    const uint8_t * base = (const uint8_t *) w;
    const bool      ptq  = type == GGML_TYPE_PTQ1_0;
    stream->parallel_for(sycl::range<1>(nb * 16), [=](sycl::id<1> id) {
        const int64_t ib = id[0] / 16;
        const int     c  = (int) (id[0] % 16);
        int           v[8];
        if (ptq) {
            const uint8_t * qs;
            const uint8_t * qh;
            if (reordered) {
                qs = base + ib * 24;
                qh = base + nb * 24 + ib * 2;
            } else {
                qs = ((const block_ptq1_0 *) w)[ib].qs;
                qh = ((const block_ptq1_0 *) w)[ib].qh;
            }
            constexpr uint32_t pow3[5] = { 1, 3, 9, 27, 81 };
            if (c < 15) {
                // 8 source bytes in one load: qs[0..7], qs[8..15] (c < 10) or qs[16..23]
                const uint32_t p   = pow3[c < 10 ? c / 2 : c - 10];
                const int      off = c < 10 ? (c % 2) * 8 : 16;
                const uint64_t src = reordered ? *(const uint64_t *) (qs + off) : 0;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const uint32_t byte = reordered ? (uint32_t) (src >> (8 * i)) & 0xFF : qs[off + i];
                    v[i] = (int) ((((byte * p) & 0xFF) * 3) >> 8) - 1;
                }
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) v[i] = (int) ((((qh[i % 2] * pow3[i / 2]) & 0xFF) * 3) >> 8) - 1;
            }
        } else {
            const uint8_t * qs  = reordered ? base + ib * 32 : ((const block_pq2_0 *) w)[ib].qs;
            const uint32_t  two = reordered ? *(const uint16_t *) (qs + 2 * c) : (uint32_t) qs[2 * c] | ((uint32_t) qs[2 * c + 1] << 8);
#pragma unroll
            for (int i = 0; i < 8; ++i) v[i] = (int) ((two >> (2 * i)) & 3) - 1;
        }
        const int r8    = ratio[ib];
        uint64_t  out   = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) out |= (uint64_t) (uint8_t) (int8_t) (v[i] * r8) << (8 * i);
        ((uint64_t *) w8)[id[0]] = out;
    });
}

// per token: s = max|x| / 127, x8 = round(x / s); one 256-item work-group per token row, 16-byte loads
static void w8a8_acts(const float * x, int8_t * x8, float * xs, int64_t ntok, int64_t K, dpct::queue_ptr stream) {
    constexpr int WG = 256;
    GGML_ASSERT(K % 4 == 0);
    stream->parallel_for(sycl::nd_range<1>(ntok * WG, WG), [=](sycl::nd_item<1> it) {
        const int64_t        t    = it.get_group(0);
        const int            lid  = it.get_local_id(0);
        const int64_t        K4   = K / 4;
        const sycl::float4 * row4 = (const sycl::float4 *) (x + t * K);
        float                m    = 0.0f;
        for (int64_t k4 = lid; k4 < K4; k4 += WG) {
            const sycl::float4 v = sycl::fabs(row4[k4]);
            m = sycl::fmax(m, sycl::fmax(sycl::fmax(v.x(), v.y()), sycl::fmax(v.z(), v.w())));
        }
        m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
        const float inv = m > 0 ? 127.0f / m : 0.0f;
        sycl::char4 * out4 = (sycl::char4 *) (x8 + t * K);
        for (int64_t k4 = lid; k4 < K4; k4 += WG) {
            const sycl::float4 v = sycl::round(row4[k4] * inv);
            out4[k4] = sycl::char4((int8_t) v.x(), (int8_t) v.y(), (int8_t) v.z(), (int8_t) v.w());
        }
        if (lid == 0) {
            xs[t] = m / 127.0f;
        }
    });
}

bool ggml_sycl_w8a8_mul_mat(ggml_backend_sycl_context & ctx, ggml_type type, bool reordered, const void * w,
                            const float * x, float * dst, int64_t nrows, int64_t ncols_x, int64_t K,
                            dpct::queue_ptr stream) {
    if ((type != GGML_TYPE_PQ2_0 && type != GGML_TYPE_PTQ1_0) || K % 128 != 0) {
        return false;
    }

    ggml_sycl_pool_alloc<int8_t> ratio (ctx.pool(), nrows * K / 128);
    ggml_sycl_pool_alloc<float>  wscale(ctx.pool(), nrows);
    ggml_sycl_pool_alloc<float>  xscale(ctx.pool(), ncols_x);
    ggml_sycl_pool_alloc<int8_t> w8    (ctx.pool(), nrows * K);
    ggml_sycl_pool_alloc<int8_t> x8    (ctx.pool(), ncols_x * K);

    int8_t * ratio_p = ratio.get();
    float *  ws      = wscale.get();
    float *  xs      = xscale.get();
    int8_t * w8_p    = w8.get();
    int8_t * x8_p    = x8.get();

    w8a8_stage(0, stream, [&] { w8a8_row_scales(type, reordered, w, ws, ratio_p, nrows, K, stream); });
    w8a8_stage(1, stream, [&] { w8a8_weights(type, reordered, w, ratio_p, w8_p, nrows, K, stream); });
    w8a8_stage(2, stream, [&] { w8a8_acts(x, x8_p, xs, ncols_x, K, stream); });
    w8a8_stage(3, stream, [&] {
        DnnlGemmWrapper::gemm_s8(ctx, (int) nrows, (int) ncols_x, (int) K, x8_p, w8_p, dst, stream);
    });

    // dst[t][r] *= xs[t] * ws[r]; 2D range avoids the emulated 64-bit div/mod
    w8a8_stage(4, stream, [&] {
        const int64_t N = nrows;
        stream->parallel_for(sycl::range<2>((size_t) ncols_x, (size_t) nrows), [=](sycl::id<2> id) {
            const size_t t = id[0], r = id[1];
            dst[t * N + r] *= xs[t] * ws[r];
        });
    });

    if (g_prof.enabled) g_prof.calls++;
    return true;   // pool buffers return to the pool at scope exit; the queue is in-order
}

// ARC-LAB: the same int8 GEMM from the TernSYCL 2-bit layout (PTQ1_0 / PQ2_0 repacked in place by ggml_sycl_t2_repack):
// codes [K/16][N] u32 (16 two-bit two's-complement weights per word, k = 16 w + j at bits 2 j), scales [K/128][N] fp16.
// Large prompt batches: oneDNN's int8 GEMM beat the TernSYCL GemmMT for them before T2 went live (pp512 1013 vs ~900).
bool ggml_sycl_w8a8_mul_mat_t2(ggml_backend_sycl_context & ctx, const void * w, const float * x, int64_t x_stride,
                               float * dst, int64_t nrows, int64_t ntok, int64_t K, dpct::queue_ptr stream) {
    if (K % 128 != 0 || nrows % 4 != 0 || x_stride != K) {
        return false;   // caller falls through to the XMX path
    }
    const int64_t    N  = nrows, ng = K / 128;
    const uint32_t * B  = (const uint32_t *) w;
    const uint16_t * SB = (const uint16_t *) ((const char *) w + (size_t) (K / 16) * N * 4);

    ggml_sycl_pool_alloc<int8_t> ratio (ctx.pool(), N * ng);
    ggml_sycl_pool_alloc<float>  wscale(ctx.pool(), N);
    ggml_sycl_pool_alloc<int8_t> w8    (ctx.pool(), N * K);
    ggml_sycl_pool_alloc<int8_t> x8    (ctx.pool(), ntok * K);
    ggml_sycl_pool_alloc<float>  xscale(ctx.pool(), ntok);
    int8_t * ratio_p = ratio.get();
    float *  ws_p    = wscale.get();
    int8_t * w8_p    = w8.get();
    int8_t * x8_p    = x8.get();
    float *  xs_p    = xscale.get();

    // stage 0: per row D = max group scale, ratio = round(127 d_g / D); ratio stored [g][N] (coalesced reads later)
    w8a8_stage(0, stream, [&] {
        stream->parallel_for(sycl::nd_range<1>(N * 16, 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const int64_t r    = it.get_group(0);
            const int     lane = it.get_local_id(0);
            auto dget = [&](int64_t g) -> float { return (float) sycl::bit_cast<sycl::half>(SB[g * N + r]); };
            float m = 0.0f;
            for (int64_t g = lane; g < ng; g += 16) {
                m = sycl::fmax(m, sycl::fabs(dget(g)));
            }
            m = sycl::reduce_over_group(it.get_sub_group(), m, sycl::maximum<float>());
            for (int64_t g = lane; g < ng; g += 16) {
                ratio_p[g * N + r] = m > 0 ? (int8_t) sycl::round(127.0f * dget(g) / m) : 0;
            }
            if (lane == 0) {
                ws_p[r] = m / 127.0f;
            }
        });
    });

    // stage 1: expand 2-bit codes to int8, stored K-major [K][N]. Each item handles one 16-weight word for
    // 4 consecutive rows and writes aligned u32s, so neighbouring items write neighbouring memory.
    w8a8_stage(1, stream, [&] {
        stream->parallel_for(sycl::range<2>((size_t) (K / 16), (size_t) (N / 4)), [=](sycl::id<2> id) {
            const int64_t kw = id[0], n0 = (int64_t) id[1] * 4;
            uint32_t c[4];
            int      r8[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                c[i]  = B[kw * N + n0 + i];
                r8[i] = ratio_p[(kw / 8) * N + n0 + i];   // 8 words of 16 weights = one 128-group
            }
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                uint32_t pk = 0;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int v = ((int) ((c[i] >> (2 * j)) << 30)) >> 30;   // sign-extend the 2-bit code
                    pk |= (uint32_t) (uint8_t) (int8_t) (v * r8[i]) << (8 * i);
                }
                *(uint32_t *) (w8_p + (kw * 16 + j) * N + n0) = pk;
            }
        });
    });

    // stage 2: per-token int8 activations
    w8a8_stage(2, stream, [&] { w8a8_acts(x, x8_p, xs_p, ntok, K, stream); });

    // stage 3: int8 GEMM, weights are [K][N]
    w8a8_stage(3, stream, [&] {
        DnnlGemmWrapper::gemm_s8(ctx, (int) N, (int) ntok, (int) K, x8_p, w8_p, dst, stream, /*w_kn=*/true);
    });

    // stage 4: dst[t][r] *= xs[t] * ws[r]; 2D range avoids the 64-bit div/mod
    w8a8_stage(4, stream, [&] {
        const float * xs = xscale.get();
        stream->parallel_for(sycl::range<2>((size_t) ntok, (size_t) N), [=](sycl::id<2> id) {
            dst[id[0] * N + id[1]] *= xs[id[0]] * ws_p[id[1]];
});
    });

    if (g_prof.enabled) {
        g_prof.calls++;
    }
    return true;
}

#else

bool ggml_sycl_w8a8_mul_mat(ggml_backend_sycl_context &, ggml_type, bool, const void *, const float *, float *, int64_t,
                            int64_t, int64_t, dpct::queue_ptr) {
    return false;
}

bool ggml_sycl_w8a8_mul_mat_t2(ggml_backend_sycl_context &, const void *, const float *, int64_t, float *, int64_t, int64_t,
                               int64_t, dpct::queue_ptr) {
    return false;
}

#endif
