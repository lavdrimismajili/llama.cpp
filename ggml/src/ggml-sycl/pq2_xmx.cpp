#include "pq2_xmx.hpp"
#include "dequantize.hpp"

#if defined(__INTEL_LLVM_COMPILER) && !defined(GGML_SYCL_NO_PQ2_XMX)
#include <sycl/ext/intel/esimd.hpp>
#include <algorithm>

        //#define PQ2_V3_A_CM  //build 2: chunk-major A slab + 128 B A loads only
//        #define PQ2_V3_B_PRE //build 3: pre-transposed, pre-converted B tile only
//        both defines         //build 4
namespace {

namespace esimd = sycl::ext::intel::esimd;
namespace xmx   = sycl::ext::intel::esimd::xmx;

constexpr int PQ2_XMX_QS_BYTES = QK_PQ2_0 / 4;
constexpr int PQ2_XMX_WG       = 16;

constexpr int PQ2_XMX_TARGET_THREADS         = 4096;
constexpr int PQ2_XMX_PREFILL_TARGET_THREADS = 512;
constexpr int PQ2_XMX_SMALL_TARGET_THREADS   = 8192;

static_assert(QK_PQ2_0 == 128 && sizeof(block_pq2_0) == 34, "PQ2_0 layout changed");
static_assert(QK_PTQ1_0 == QK_PQ2_0, "PTQ1_0 expands block for block into PQ2_0 codes");

template <int N>
ESIMD_INLINE esimd::simd<uint32_t, N> pq2_codes_to_s2(esimd::simd<uint32_t, N> x) {
    constexpr uint32_t H = 0xAAAAAAAAu;
    constexpr uint32_t L = 0x55555555u;
    return ((x | H) - L) ^ (~x & H);
}

// ---------------------------------------------------------------------------
// Large-batch path. USE_SLM=true when S==1 and n_tiles_n % PQ2_XMX_WG == 0.
// ---------------------------------------------------------------------------
template <int DPAS_N, int MR, int NR, int S, bool USE_SLM>
ESIMD_INLINE void pq2_xmx_thread(const uint32_t * wq, const uint16_t * wd, const uint32_t * a8, const float * as,
                                 float * dst, int K, int nrows, int ncols, int nrows_dst, int n_tiles_n,
                                 int tile, int ks, int lid) {
    using namespace esimd;
    constexpr int TM = 8 * MR;
    constexpr int TN = DPAS_N * NR;

    if constexpr (S > 1) {
        slm_init<S * MR * NR * 8 * DPAS_N * sizeof(float)>();
    } else if constexpr (USE_SLM) {
        slm_init<TM * QK_PQ2_0>();
    }

    const int m0 = (tile / n_tiles_n) * TM;
    const int n0 = (tile % n_tiles_n) * TN;
    const int nb = K / QK_PQ2_0;
    const int b0 = (int) ((int64_t) ks * nb / S);
    const int b1 = (int) ((int64_t) (ks + 1) * nb / S);

    simd<uint32_t, DPAS_N> d_off[NR];
#pragma unroll
    for (int g = 0; g < NR; ++g) {
        simd<uint32_t, DPAS_N> r(n0 + DPAS_N * g, 1);
        r.merge(simd<uint32_t, DPAS_N>(nrows - 1), r >= (uint32_t) nrows);
        d_off[g] = r * (uint32_t) (nb * sizeof(uint16_t));
    }
    simd<uint32_t, 8> s_off[MR];
#pragma unroll
    for (int s = 0; s < MR; ++s) {
        simd<uint32_t, 8> m(m0 + 8 * s, 1);
        m.merge(simd<uint32_t, 8>(ncols - 1), m >= (uint32_t) ncols);
        s_off[s] = m * (uint32_t) (nb * sizeof(float));
    }

    simd<float, 8 * DPAS_N> acc[MR][NR];
#pragma unroll
    for (int s = 0; s < MR; ++s)
#pragma unroll
        for (int g = 0; g < NR; ++g)
            acc[s][g] = 0.0f;

    constexpr int ROWS_PER_WI = (TM + PQ2_XMX_WG - 1) / PQ2_XMX_WG;

    for (int b = b0; b < b1; ++b) {
        simd<uint32_t, 8 * DPAS_N> w[NR];
        simd<float, DPAS_N>        dw[NR];
#pragma unroll
        for (int g = 0; g < NR; ++g) {
            simd<uint32_t, 8 * DPAS_N> w_raw;
#pragma unroll
            for (int n = 0; n < DPAS_N; ++n) {
                int r = n0 + DPAS_N * g + n;
                if (r >= nrows) r = nrows - 1;
                w_raw.template select<8, 1>(n * 8) = block_load<uint32_t, 8>(
                    (const uint32_t *)((const uint8_t *)wq + (size_t)r * (nb * 32) + b * 32));
            }
            simd<uint32_t, 8 * DPAS_N> w_tr;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                w_tr.template select<DPAS_N, 1>(j * DPAS_N) =
                    w_raw.template select<DPAS_N, DPAS_N>(j);
            }
            w[g] = pq2_codes_to_s2<8 * DPAS_N>(w_tr);

            simd<uint16_t, DPAS_N>   dbits = gather<uint16_t, DPAS_N>(wd, d_off[g] + (uint32_t)(b * sizeof(uint16_t)));
            simd<sycl::half, DPAS_N> dh    = dbits.template bit_cast_view<sycl::half>();
            dw[g] = convert<float>(dh);
        }

        if constexpr (USE_SLM) {
#pragma unroll
            for (int rr = 0; rr < ROWS_PER_WI; ++rr) {
                const int row_idx = lid * ROWS_PER_WI + rr;
                if (row_idx < TM) {
                    int r = m0 + row_idx;
                    if (r >= ncols) r = ncols - 1;
                    const uint8_t * src_row = (const uint8_t *) a8 + (size_t) r * K + b * QK_PQ2_0;
#pragma unroll
                    for (int cc = 0; cc < QK_PQ2_0 / 32; ++cc) {
                        simd<uint32_t, 8> chunk = block_load<uint32_t, 8>(
                            (const uint32_t *)(src_row + cc * 32));
                        slm_block_store<uint32_t, 8>(row_idx * QK_PQ2_0 + cc * 32, chunk);
                    }
                }
            }
            barrier();
        }

#pragma unroll
        for (int s = 0; s < MR; ++s) {
            simd<int, 8 * DPAS_N> ci[NR];
#pragma unroll
            for (int g = 0; g < NR; ++g) ci[g] = 0;

#pragma unroll
            for (int c = 0; c < QK_PQ2_0 / 32; ++c) {
                simd<uint32_t, 64> ad;
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    if constexpr (USE_SLM) {
                        ad.template select<8, 1>(t * 8) = slm_block_load<uint32_t, 8>(
                            (8 * s + t) * QK_PQ2_0 + c * 32);
                    } else {
                        int r = m0 + 8 * s + t;
                        if (r >= ncols) r = ncols - 1;
                        ad.template select<8, 1>(t * 8) = block_load<uint32_t, 8>(
                            (const uint32_t *)((const uint8_t *)a8 + (size_t)r * K + b * 128 + c * 32));
                    }
                }
                simd<signed char, 256> am = ad.template bit_cast_view<signed char>();
#pragma unroll
                for (int g = 0; g < NR; ++g) {
                    simd<uint32_t, 2 * DPAS_N>    bd = w[g].template select<2 * DPAS_N, 1>(2 * DPAS_N * c);
                    simd<signed char, 8 * DPAS_N> bm = bd.template bit_cast_view<signed char>();
                    ci[g] = xmx::dpas<8, 8, int, int, signed char, signed char, xmx::dpas_argument_type::s2,
                                      xmx::dpas_argument_type::s8>(ci[g], bm, am);
                }
            }

            const simd<float, 8> da = gather<float, 8>(as, s_off[s] + (uint32_t) (b * sizeof(float)));
#pragma unroll
            for (int g = 0; g < NR; ++g) {
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    const simd<int, DPAS_N> cit = ci[g].template select<DPAS_N, 1>(DPAS_N * t);
                    acc[s][g].template select<DPAS_N, 1>(DPAS_N * t) += convert<float>(cit) * (dw[g] * da[t]);
                }
            }
        }

        if constexpr (USE_SLM) {
            barrier();
        }
    }

    if constexpr (S > 1) {
#pragma unroll
        for (int s = 0; s < MR; ++s) {
#pragma unroll
            for (int g = 0; g < NR; ++g) {
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                    const uint32_t off = (uint32_t) ((((ks * MR + s) * NR + g) * (8 * DPAS_N) + DPAS_N * q) * sizeof(float));
                    slm_block_store<float, DPAS_N>(off, acc[s][g].template select<DPAS_N, 1>(DPAS_N * q));
                }
            }
        }
        barrier();
        if (ks != 0) {
            return;
        }
        for (int o = 1; o < S; ++o) {
#pragma unroll
            for (int s = 0; s < MR; ++s) {
#pragma unroll
                for (int g = 0; g < NR; ++g) {
#pragma unroll
                    for (int q = 0; q < 8; ++q) {
                        const uint32_t off = (uint32_t) ((((o * MR + s) * NR + g) * (8 * DPAS_N) + DPAS_N * q) * sizeof(float));
                        acc[s][g].template select<DPAS_N, 1>(DPAS_N * q) += slm_block_load<float, DPAS_N>(off);
                    }
                }
            }
        }
    }

    const simd<uint32_t, DPAS_N> lane(0, 1);
#pragma unroll
    for (int s = 0; s < MR; ++s) {
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int m = m0 + 8 * s + t;
            if (m >= ncols) continue;
            float * drow = dst + (size_t) m * nrows_dst;
#pragma unroll
            for (int g = 0; g < NR; ++g) {
                const simd<uint32_t, DPAS_N> n  = lane + (uint32_t) (n0 + DPAS_N * g);
                const simd_mask<DPAS_N>      ok = n < (uint32_t) nrows;
                scatter<float, DPAS_N>(drow, n * (uint32_t) sizeof(float), acc[s][g].template select<DPAS_N, 1>(DPAS_N * t), ok);
            }
        }
    }
}

template <int DPAS_N, int MR, int NR, int S>
static void launch_pq2_xmx(const uint32_t * wq, const uint16_t * wd, const uint32_t * a8, const float * as,
                           float * dst, int K, int nrows, int ncols, int nrows_dst, dpct::queue_ptr stream) {
    constexpr int TM = 8 * MR;
    constexpr int TN = DPAS_N * NR;

    const int n_tiles_m = (ncols + TM - 1) / TM;
    const int n_tiles_n = (nrows + TN - 1) / TN;
    const int n_tiles   = n_tiles_m * n_tiles_n;

    const bool use_slm = (S == 1) && (n_tiles_n % PQ2_XMX_WG == 0);

    stream->submit([&](sycl::handler & h) {
        if constexpr (S > 1) {
            const sycl::nd_range<1> nd{ sycl::range<1>((size_t) n_tiles * S), sycl::range<1>(S) };
            h.parallel_for(nd, [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                pq2_xmx_thread<DPAS_N, MR, NR, S, false>(wq, wd, a8, as, dst, K, nrows, ncols, nrows_dst, n_tiles_n,
                                                         (int) it.get_group(0), (int) it.get_local_id(0),
                                                         (int) it.get_local_id(0));
            });
        } else {
            const size_t global = (size_t) ((n_tiles + PQ2_XMX_WG - 1) / PQ2_XMX_WG) * PQ2_XMX_WG;
            const sycl::nd_range<1> nd{ sycl::range<1>(global), sycl::range<1>(PQ2_XMX_WG) };
            if (use_slm) {
                h.parallel_for(nd, [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                    pq2_xmx_thread<DPAS_N, MR, NR, 1, true>(wq, wd, a8, as, dst, K, nrows, ncols, nrows_dst, n_tiles_n,
                                                            (int) it.get_global_id(0), 0, (int) it.get_local_id(0));
                });
            } else {
                h.parallel_for(nd, [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                    pq2_xmx_thread<DPAS_N, MR, NR, 1, false>(wq, wd, a8, as, dst, K, nrows, ncols, nrows_dst, n_tiles_n,
                                                             (int) it.get_global_id(0), 0, (int) it.get_local_id(0));
                });
            }
        }
    });
}

template <int DPAS_N, int MR, int NR>
static void launch_pq2_xmx_split(const uint32_t * wq, const uint16_t * wd, const uint32_t * a8, const float * as,
                                 float * dst, int K, int nrows, int ncols, int nrows_dst, dpct::queue_ptr stream) {
    const int nb      = K / QK_PQ2_0;
    const int n_tiles = ((ncols + 8 * MR - 1) / (8 * MR)) * ((nrows + DPAS_N * NR - 1) / (DPAS_N * NR));
    const int target    = MR >= 4 ? PQ2_XMX_PREFILL_TARGET_THREADS : PQ2_XMX_TARGET_THREADS;
    const int max_split = MR >= 4 ? 2 : 16;
    int       split     = 1;
    while (split < max_split && n_tiles * split < target && nb >= 4 * split) {
        split *= 2;
    }
    switch (split) {
        case 1:  launch_pq2_xmx<DPAS_N, MR, NR, 1>(wq, wd, a8, as, dst, K, nrows, ncols, nrows_dst, stream);  break;
        case 2:  launch_pq2_xmx<DPAS_N, MR, NR, 2>(wq, wd, a8, as, dst, K, nrows, ncols, nrows_dst, stream);  break;
        case 4:  launch_pq2_xmx<DPAS_N, MR, NR, 4>(wq, wd, a8, as, dst, K, nrows, ncols, nrows_dst, stream);  break;
        case 8:  launch_pq2_xmx<DPAS_N, MR, NR, 8>(wq, wd, a8, as, dst, K, nrows, ncols, nrows_dst, stream);  break;
        default: launch_pq2_xmx<DPAS_N, MR, NR, 16>(wq, wd, a8, as, dst, K, nrows, ncols, nrows_dst, stream); break;
    }
}
template <int DPAS_N, int MR, int NR, int MSPLIT, int NSPLIT>
ESIMD_INLINE void pq2_xmx_thread_block_old(const uint32_t * wq, const uint16_t * wd,
                                           const uint32_t * a8, const float * as,
                                           float * dst, int K, int nrows, int ncols, int nrows_dst,
                                           int m0_base, int n0_base, int lid) {
    using namespace esimd;
    constexpr int TM = 8 * MR;
    constexpr int TN = DPAS_N * NR;
    constexpr int A_SLAB_BYTES = MSPLIT * TM * QK_PQ2_0;
    constexpr int B_SLAB_BYTES = NSPLIT * TN * 32;
    constexpr int TOTAL_SLM    = A_SLAB_BYTES + B_SLAB_BYTES;
    constexpr int WG           = MSPLIT * NSPLIT;
 
    slm_init<TOTAL_SLM>();
 
    const int m_sub = lid / NSPLIT;
    const int n_sub = lid % NSPLIT;
 
    const int m0 = m0_base + m_sub * TM;
    const int n0 = n0_base + n_sub * TN;
    const int nb = K / QK_PQ2_0;
 
    simd<uint32_t, DPAS_N> d_off[NR];
#pragma unroll
    for (int g = 0; g < NR; ++g) {
        simd<uint32_t, DPAS_N> r(n0 + DPAS_N * g, 1);
        r.merge(simd<uint32_t, DPAS_N>(nrows - 1), r >= (uint32_t) nrows);
        d_off[g] = r * (uint32_t) (nb * sizeof(uint16_t));
    }
    simd<uint32_t, 8> s_off[MR];
#pragma unroll
    for (int s = 0; s < MR; ++s) {
        simd<uint32_t, 8> m(m0 + 8 * s, 1);
        m.merge(simd<uint32_t, 8>(ncols - 1), m >= (uint32_t) ncols);
        s_off[s] = m * (uint32_t) (nb * sizeof(float));
    }
 
    simd<float, 8 * DPAS_N> acc[MR][NR];
#pragma unroll
    for (int s = 0; s < MR; ++s)
#pragma unroll
        for (int g = 0; g < NR; ++g)
            acc[s][g] = 0.0f;
 
    constexpr int A_ROWS_TOTAL  = MSPLIT * TM;
    constexpr int B_ROWS_TOTAL  = NSPLIT * TN;
    constexpr int A_ROWS_PER_WI = (A_ROWS_TOTAL + WG - 1) / WG;
    constexpr int B_ROWS_PER_WI = (B_ROWS_TOTAL + WG - 1) / WG;
 
    for (int b = 0; b < nb; ++b) {
        // ---- Cooperative SLM staging for this K-block ----
#pragma unroll
        for (int rr = 0; rr < A_ROWS_PER_WI; ++rr) {
            const int row_idx = lid * A_ROWS_PER_WI + rr;
            if (row_idx < A_ROWS_TOTAL) {
                int r_act = m0_base + row_idx;
                if (r_act >= ncols) r_act = ncols - 1;
#pragma unroll
                for (int cc = 0; cc < QK_PQ2_0 / 32; ++cc) {
                    simd<uint32_t, 8> chunk = block_load<uint32_t, 8>(
                        (const uint32_t *)((const uint8_t *)a8 + (size_t) r_act * K + b * QK_PQ2_0 + cc * 32));
                    slm_block_store<uint32_t, 8>(row_idx * QK_PQ2_0 + cc * 32, chunk);
                }
            }
        }
#pragma unroll
        for (int rr = 0; rr < B_ROWS_PER_WI; ++rr) {
            const int col_idx = lid * B_ROWS_PER_WI + rr;
            if (col_idx < B_ROWS_TOTAL) {
                int r_w = n0_base + col_idx;
                if (r_w >= nrows) r_w = nrows - 1;
                simd<uint32_t, 8> chunk = block_load<uint32_t, 8>(
                    (const uint32_t *)((const uint8_t *)wq + (size_t) r_w * (nb * 32) + b * 32));
                slm_block_store<uint32_t, 8>(A_SLAB_BYTES + col_idx * 32, chunk);
            }
        }
        barrier();
 
        // ---- Per-WI compute ----
        simd<uint32_t, 8 * DPAS_N> w[NR];
        simd<float, DPAS_N>        dw[NR];
#pragma unroll
        for (int g = 0; g < NR; ++g) {
            simd<uint32_t, 8 * DPAS_N> w_raw;
#pragma unroll
            for (int n = 0; n < DPAS_N; ++n) {
                const int col_in_tile = DPAS_N * g + n;
                const int slm_addr = A_SLAB_BYTES + (n_sub * TN + col_in_tile) * 32;
                w_raw.template select<8, 1>(n * 8) = slm_block_load<uint32_t, 8>(slm_addr);
            }
            simd<uint32_t, 8 * DPAS_N> w_tr;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                w_tr.template select<DPAS_N, 1>(j * DPAS_N) =
                    w_raw.template select<DPAS_N, DPAS_N>(j);
            }
            w[g] = pq2_codes_to_s2<8 * DPAS_N>(w_tr);
 
            simd<uint16_t, DPAS_N>   dbits = gather<uint16_t, DPAS_N>(wd, d_off[g] + (uint32_t)(b * sizeof(uint16_t)));
            simd<sycl::half, DPAS_N> dh    = dbits.template bit_cast_view<sycl::half>();
            dw[g] = convert<float>(dh);
        }
 
#pragma unroll
        for (int s = 0; s < MR; ++s) {
            simd<int, 8 * DPAS_N> ci[NR];
#pragma unroll
            for (int g = 0; g < NR; ++g) ci[g] = 0;
 
#pragma unroll
            for (int c = 0; c < QK_PQ2_0 / 32; ++c) {
                simd<uint32_t, 64> ad;
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    const int row_in_tile = 8 * s + t;
                    const int slm_addr = (m_sub * TM + row_in_tile) * QK_PQ2_0 + c * 32;
                    ad.template select<8, 1>(t * 8) = slm_block_load<uint32_t, 8>(slm_addr);
                }
                simd<signed char, 256> am = ad.template bit_cast_view<signed char>();
#pragma unroll
                for (int g = 0; g < NR; ++g) {
                    simd<uint32_t, 2 * DPAS_N> bd = w[g].template select<2 * DPAS_N, 1>(2 * DPAS_N * c);
                    simd<signed char, 8 * DPAS_N> bm = bd.template bit_cast_view<signed char>();
                    ci[g] = xmx::dpas<8, 8, int, int, signed char, signed char,
                                      xmx::dpas_argument_type::s2, xmx::dpas_argument_type::s8>(ci[g], bm, am);
                }
            }
 
            const simd<float, 8> da = gather<float, 8>(as, s_off[s] + (uint32_t)(b * sizeof(float)));
#pragma unroll
            for (int g = 0; g < NR; ++g) {
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    const simd<int, DPAS_N> cit = ci[g].template select<DPAS_N, 1>(DPAS_N * t);
                    acc[s][g].template select<DPAS_N, 1>(DPAS_N * t) += convert<float>(cit) * (dw[g] * da[t]);
                }
            }
        }
        barrier();
    }
 
    const simd<uint32_t, DPAS_N> lane(0, 1);
#pragma unroll
    for (int s = 0; s < MR; ++s) {
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int m = m0 + 8 * s + t;
            if (m >= ncols) continue;
            float * drow = dst + (size_t) m * nrows_dst;
#pragma unroll
            for (int g = 0; g < NR; ++g) {
                const simd<uint32_t, DPAS_N> n  = lane + (uint32_t) (n0 + DPAS_N * g);
                const simd_mask<DPAS_N>      ok = n < (uint32_t) nrows;
                scatter<float, DPAS_N>(drow, n * (uint32_t) sizeof(float),
                                       acc[s][g].template select<DPAS_N, 1>(DPAS_N * t), ok);
            }
        }
    }
}
// ---------------------------------------------------------------------------
// Block-layout kernel. Work-group of MSPLIT*NSPLIT WIs covers a
// (MSPLIT*TM) x (NSPLIT*TN) super-tile. Activations and weights staged into
// SLM once per K-block, then each WI reads its own slice.
// ---------------------------------------------------------------------------
template <int DPAS_N, int MR, int NR, int MSPLIT, int NSPLIT>
ESIMD_INLINE void pq2_xmx_thread_block(const uint32_t * wq, const uint16_t * wd,
                                       const uint32_t * a8, const float * as,
                                       float * dst, int K, int nrows, int ncols, int nrows_dst,
                                       int m0_base, int n0_base, int lid) {
    using namespace esimd;
    constexpr int TM = 8 * MR;
    constexpr int TN = DPAS_N * NR;
    constexpr int A_SLAB_BYTES = MSPLIT * TM * QK_PQ2_0;
    constexpr int B_SLAB_BYTES = NSPLIT * TN * 32;
    constexpr int TOTAL_SLM    = A_SLAB_BYTES + B_SLAB_BYTES;
    constexpr int WG           = MSPLIT * NSPLIT;
 
    constexpr int A_ROWS_TOTAL  = MSPLIT * TM;
    constexpr int B_ROWS_TOTAL  = NSPLIT * TN;
    constexpr int A_ROWS_PER_WI = (A_ROWS_TOTAL + WG - 1) / WG;
    constexpr int B_ROWS_PER_WI = (B_ROWS_TOTAL + WG - 1) / WG;
 
    // switch-only constants
    constexpr int A_CHUNK_BYTES = A_ROWS_TOTAL * 32;           // PQ2_V3_A_CM
    constexpr int WTILE_DW      = 8 * DPAS_N;                  // PQ2_V3_B_PRE
    constexpr int WTILE_BYTES   = WTILE_DW * 4;
    constexpr int B_GROUPS      = NSPLIT * NR;
    constexpr int B_GRP_PER_WI  = (B_GROUPS + WG - 1) / WG;
    constexpr int LDW           = 32;                          // dwords per 128 B SLM message
    static_assert(WTILE_DW % LDW == 0, "B tile must be a multiple of 128 B");
 
    slm_init<TOTAL_SLM>();
 
    const int m_sub = lid / NSPLIT;
    const int n_sub = lid % NSPLIT;
 
    const int m0 = m0_base + m_sub * TM;
    const int n0 = n0_base + n_sub * TN;
    const int nb = K / QK_PQ2_0;
 
    simd<uint32_t, DPAS_N> d_off[NR];
#pragma unroll
    for (int g = 0; g < NR; ++g) {
        simd<uint32_t, DPAS_N> r(n0 + DPAS_N * g, 1);
        r.merge(simd<uint32_t, DPAS_N>(nrows - 1), r >= (uint32_t) nrows);
        d_off[g] = r * (uint32_t) (nb * sizeof(uint16_t));
    }
    simd<uint32_t, 8> s_off[MR];
#pragma unroll
    for (int s = 0; s < MR; ++s) {
        simd<uint32_t, 8> m(m0 + 8 * s, 1);
        m.merge(simd<uint32_t, 8>(ncols - 1), m >= (uint32_t) ncols);
        s_off[s] = m * (uint32_t) (nb * sizeof(float));
    }
 
    simd<float, 8 * DPAS_N> acc[MR][NR];
#pragma unroll
    for (int s = 0; s < MR; ++s)
#pragma unroll
        for (int g = 0; g < NR; ++g)
            acc[s][g] = 0.0f;
 
    for (int b = 0; b < nb; ++b) {
        // ---- Cooperative SLM staging for this K-block: A ----
#pragma unroll
        for (int rr = 0; rr < A_ROWS_PER_WI; ++rr) {
            const int row_idx = lid * A_ROWS_PER_WI + rr;
            if (row_idx < A_ROWS_TOTAL) {
                int r_act = m0_base + row_idx;
                if (r_act >= ncols) r_act = ncols - 1;
#ifdef PQ2_V3_A_CM
                simd<uint32_t, 32> v = block_load<uint32_t, 32>(
                    (const uint32_t *)((const uint8_t *)a8 + (size_t) r_act * K + (size_t) b * QK_PQ2_0));
#pragma unroll
                for (int cc = 0; cc < QK_PQ2_0 / 32; ++cc) {
                    slm_block_store<uint32_t, 8>(cc * A_CHUNK_BYTES + row_idx * 32,
                                                 v.template select<8, 1>(cc * 8));
                }
#else
#pragma unroll
                for (int cc = 0; cc < QK_PQ2_0 / 32; ++cc) {
                    simd<uint32_t, 8> chunk = block_load<uint32_t, 8>(
                        (const uint32_t *)((const uint8_t *)a8 + (size_t) r_act * K + b * QK_PQ2_0 + cc * 32));
                    slm_block_store<uint32_t, 8>(row_idx * QK_PQ2_0 + cc * 32, chunk);
                }
#endif
            }
        }
 
        // ---- B ----
#ifdef PQ2_V3_B_PRE
#pragma unroll
        for (int gg = 0; gg < B_GRP_PER_WI; ++gg) {
            const int G = lid * B_GRP_PER_WI + gg;
            if (G < B_GROUPS) {
                simd<uint32_t, WTILE_DW> w_raw;
#pragma unroll
                for (int n = 0; n < DPAS_N; ++n) {
                    int r_w = n0_base + G * DPAS_N + n;
                    if (r_w >= nrows) r_w = nrows - 1;
                    w_raw.template select<8, 1>(n * 8) = block_load<uint32_t, 8>(
                        (const uint32_t *)((const uint8_t *)wq + (size_t) r_w * (nb * 32) + (size_t) b * 32));
                }
                simd<uint32_t, WTILE_DW> w_tr;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    w_tr.template select<DPAS_N, 1>(j * DPAS_N) = w_raw.template select<DPAS_N, 8>(j);
                }
                w_tr = pq2_codes_to_s2<WTILE_DW>(w_tr);
#pragma unroll
                for (int q = 0; q < WTILE_DW / LDW; ++q) {
                    slm_block_store<uint32_t, LDW>(A_SLAB_BYTES + G * WTILE_BYTES + q * LDW * 4,
                                                   w_tr.template select<LDW, 1>(q * LDW));
                }
            }
        }
#else
#pragma unroll
        for (int rr = 0; rr < B_ROWS_PER_WI; ++rr) {
            const int col_idx = lid * B_ROWS_PER_WI + rr;
            if (col_idx < B_ROWS_TOTAL) {
                int r_w = n0_base + col_idx;
                if (r_w >= nrows) r_w = nrows - 1;
                simd<uint32_t, 8> chunk = block_load<uint32_t, 8>(
                    (const uint32_t *)((const uint8_t *)wq + (size_t) r_w * (nb * 32) + b * 32));
                slm_block_store<uint32_t, 8>(A_SLAB_BYTES + col_idx * 32, chunk);
            }
        }
#endif
        barrier();
 
        // ---- Per-WI compute ----
        simd<uint32_t, 8 * DPAS_N> w[NR];
        simd<float, DPAS_N>        dw[NR];
#pragma unroll
        for (int g = 0; g < NR; ++g) {
#ifdef PQ2_V3_B_PRE
#pragma unroll
            for (int q = 0; q < WTILE_DW / LDW; ++q) {
                w[g].template select<LDW, 1>(q * LDW) = slm_block_load<uint32_t, LDW>(
                    A_SLAB_BYTES + (n_sub * NR + g) * WTILE_BYTES + q * LDW * 4);
            }
#else
            simd<uint32_t, 8 * DPAS_N> w_raw;
#pragma unroll
            for (int n = 0; n < DPAS_N; ++n) {
                const int col_in_tile = DPAS_N * g + n;
                const int slm_addr = A_SLAB_BYTES + (n_sub * TN + col_in_tile) * 32;
                w_raw.template select<8, 1>(n * 8) = slm_block_load<uint32_t, 8>(slm_addr);
            }
            simd<uint32_t, 8 * DPAS_N> w_tr;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                w_tr.template select<DPAS_N, 1>(j * DPAS_N) =
                    w_raw.template select<DPAS_N, DPAS_N>(j);
            }
            w[g] = pq2_codes_to_s2<8 * DPAS_N>(w_tr);
#endif
            simd<uint16_t, DPAS_N>   dbits = gather<uint16_t, DPAS_N>(wd, d_off[g] + (uint32_t)(b * sizeof(uint16_t)));
            simd<sycl::half, DPAS_N> dh    = dbits.template bit_cast_view<sycl::half>();
            dw[g] = convert<float>(dh);
        }
 
#pragma unroll
        for (int s = 0; s < MR; ++s) {
            simd<int, 8 * DPAS_N> ci[NR];
#pragma unroll
            for (int g = 0; g < NR; ++g) ci[g] = 0;
 
#pragma unroll
            for (int c = 0; c < QK_PQ2_0 / 32; ++c) {
                simd<uint32_t, 64> ad;
#ifdef PQ2_V3_A_CM
                {
                    const int a_base = c * A_CHUNK_BYTES + (m_sub * TM + 8 * s) * 32;
                    ad.template select<32, 1>(0)  = slm_block_load<uint32_t, 32>(a_base);
                    ad.template select<32, 1>(32) = slm_block_load<uint32_t, 32>(a_base + 128);
                }
#else
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    const int row_in_tile = 8 * s + t;
                    const int slm_addr = (m_sub * TM + row_in_tile) * QK_PQ2_0 + c * 32;
                    ad.template select<8, 1>(t * 8) = slm_block_load<uint32_t, 8>(slm_addr);
                }
#endif
                simd<signed char, 256> am = ad.template bit_cast_view<signed char>();
#pragma unroll
                for (int g = 0; g < NR; ++g) {
                    simd<uint32_t, 2 * DPAS_N> bd = w[g].template select<2 * DPAS_N, 1>(2 * DPAS_N * c);
                    simd<signed char, 8 * DPAS_N> bm = bd.template bit_cast_view<signed char>();
                    ci[g] = xmx::dpas<8, 8, int, int, signed char, signed char,
                                      xmx::dpas_argument_type::s2, xmx::dpas_argument_type::s8>(ci[g], bm, am);
                }
            }
 
            const simd<float, 8> da = gather<float, 8>(as, s_off[s] + (uint32_t)(b * sizeof(float)));
#pragma unroll
            for (int g = 0; g < NR; ++g) {
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    const simd<int, DPAS_N> cit = ci[g].template select<DPAS_N, 1>(DPAS_N * t);
                    acc[s][g].template select<DPAS_N, 1>(DPAS_N * t) += convert<float>(cit) * (dw[g] * da[t]);
                }
            }
        }
        barrier();
    }
 
    const simd<uint32_t, DPAS_N> lane(0, 1);
#pragma unroll
    for (int s = 0; s < MR; ++s) {
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int m = m0 + 8 * s + t;
            if (m >= ncols) continue;
            float * drow = dst + (size_t) m * nrows_dst;
#pragma unroll
            for (int g = 0; g < NR; ++g) {
                const simd<uint32_t, DPAS_N> n  = lane + (uint32_t) (n0 + DPAS_N * g);
                const simd_mask<DPAS_N>      ok = n < (uint32_t) nrows;
                scatter<float, DPAS_N>(drow, n * (uint32_t) sizeof(float),
                                       acc[s][g].template select<DPAS_N, 1>(DPAS_N * t), ok);
            }
        }
    }
}

template <int DPAS_N, int MR, int NR, int MSPLIT, int NSPLIT>
static bool launch_pq2_xmx_block(const uint32_t * wq, const uint16_t * wd, const uint32_t * a8,
                                 const float * as, float * dst, int K, int nrows, int ncols,
                                 int nrows_dst, dpct::queue_ptr stream) {
    constexpr int TM = 8 * MR;
    constexpr int TN = DPAS_N * NR;
    constexpr int WG = MSPLIT * NSPLIT;

    const int n_tiles_m = (ncols + TM - 1) / TM;
    const int n_tiles_n = (nrows + TN - 1) / TN;

    if (n_tiles_m % MSPLIT != 0 || n_tiles_n % NSPLIT != 0) {
        return false;
    }
    
    const int n_wg_m = n_tiles_m / MSPLIT;
    const int n_wg_n = n_tiles_n / NSPLIT;
    const int n_wgs  = n_wg_m * n_wg_n;

    stream->submit([&](sycl::handler & h) {
        const sycl::nd_range<1> nd{ sycl::range<1>((size_t) n_wgs * WG), sycl::range<1>(WG) };
        h.parallel_for(nd, [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
            const int wg_id = (int) it.get_group(0);
            const int lid   = (int) it.get_local_id(0);
            const int wg_m  = wg_id / n_wg_n;
            const int wg_n  = wg_id % n_wg_n;
            const int m0_base = wg_m * MSPLIT * TM;
            const int n0_base = wg_n * NSPLIT * TN;
#ifdef PQ2_USE_OLD_BLOCK
            pq2_xmx_thread_block_old<DPAS_N, MR, NR, MSPLIT, NSPLIT>(
                wq, wd, a8, as, dst, K, nrows, ncols, nrows_dst, m0_base, n0_base, lid);
 #else
            pq2_xmx_thread_block<DPAS_N, MR, NR, MSPLIT, NSPLIT>(
                wq, wd, a8, as, dst, K, nrows, ncols, nrows_dst, m0_base, n0_base, lid);
 #endif
        });
    });
    return true;
}

// ---------------------------------------------------------------------------
// Small-batch RC path (ncols <= 8) — unchanged.
// ---------------------------------------------------------------------------
template <int DPAS_N, int RC, int NR, int S>
ESIMD_INLINE void pq2_xmx_thread_rc(const uint32_t * wq, const uint16_t * wd, const uint32_t * a8, const float * as,
                                    float * dst, int K, int nrows, int ncols, int nrows_dst, int n_tiles_n,
                                    int tile, int ks) {
    using namespace esimd;
    static_assert(RC >= 1 && RC <= 8, "RepeatCount must be 1..8");
    constexpr int TN = DPAS_N * NR;

    if constexpr (S > 1) {
        slm_init<S * NR * RC * DPAS_N * sizeof(float)>();
    }

    const int n0 = (tile % n_tiles_n) * TN;
    const int nb = K / QK_PQ2_0;
    const int b0 = (int) ((int64_t) ks * nb / S);
    const int b1 = (int) ((int64_t) (ks + 1) * nb / S);

    simd<uint32_t, DPAS_N> d_off[NR];
#pragma unroll
    for (int g = 0; g < NR; ++g) {
        simd<uint32_t, DPAS_N> r(n0 + DPAS_N * g, 1);
        r.merge(simd<uint32_t, DPAS_N>(nrows - 1), r >= (uint32_t) nrows);
        d_off[g] = r * (uint32_t) (nb * sizeof(uint16_t));
    }
    simd<uint32_t, RC> s_off;
    {
        simd<uint32_t, RC> m(0, 1);
        s_off = m * (uint32_t) (nb * sizeof(float));
    }

    simd<float, RC * DPAS_N> acc[NR];
#pragma unroll
    for (int g = 0; g < NR; ++g) acc[g] = 0.0f;

    for (int b = b0; b < b1; ++b) {
        simd<uint32_t, 8 * DPAS_N> w[NR];
        simd<float, DPAS_N>        dw[NR];
#pragma unroll
        for (int g = 0; g < NR; ++g) {
            simd<uint32_t, 8 * DPAS_N> w_raw;
#pragma unroll
            for (int n = 0; n < DPAS_N; ++n) {
                int r = n0 + DPAS_N * g + n;
                if (r >= nrows) r = nrows - 1;
                w_raw.template select<8, 1>(n * 8) = block_load<uint32_t, 8>(
                    (const uint32_t *)((const uint8_t *)wq + (size_t)r * (nb * 32) + b * 32));
            }
            simd<uint32_t, 8 * DPAS_N> w_tr;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                w_tr.template select<DPAS_N, 1>(j * DPAS_N) =
                    w_raw.template select<DPAS_N, DPAS_N>(j);
            }
            w[g] = pq2_codes_to_s2<8 * DPAS_N>(w_tr);

            simd<uint16_t, DPAS_N>   dbits = gather<uint16_t, DPAS_N>(wd, d_off[g] + (uint32_t)(b * sizeof(uint16_t)));
            simd<sycl::half, DPAS_N> dh    = dbits.template bit_cast_view<sycl::half>();
            dw[g] = convert<float>(dh);
        }

        simd<int, RC * DPAS_N> ci[NR];
#pragma unroll
        for (int g = 0; g < NR; ++g) ci[g] = 0;
#pragma unroll
        for (int c = 0; c < QK_PQ2_0 / 32; ++c) {
            simd<uint32_t, RC * 8> ad;
#pragma unroll
            for (int t = 0; t < RC; ++t) {
                ad.template select<8, 1>(t * 8) = block_load<uint32_t, 8>(
                    (const uint32_t *)((const uint8_t *)a8 + (size_t)t * K + b * 128 + c * 32));
            }
            simd<signed char, RC * 32> am = ad.template bit_cast_view<signed char>();
#pragma unroll
            for (int g = 0; g < NR; ++g) {
                simd<uint32_t, 2 * DPAS_N>    bd = w[g].template select<2 * DPAS_N, 1>(2 * DPAS_N * c);
                simd<signed char, 8 * DPAS_N> bm = bd.template bit_cast_view<signed char>();
                ci[g] = xmx::dpas<8, RC, int, int, signed char, signed char, xmx::dpas_argument_type::s2,
                                  xmx::dpas_argument_type::s8>(ci[g], bm, am);
            }
        }

        const simd<float, RC> da = gather<float, RC>(as, s_off + (uint32_t) (b * sizeof(float)));
#pragma unroll
        for (int g = 0; g < NR; ++g) {
#pragma unroll
            for (int t = 0; t < RC; ++t) {
                const simd<int, DPAS_N> cit = ci[g].template select<DPAS_N, 1>(DPAS_N * t);
                acc[g].template select<DPAS_N, 1>(DPAS_N * t) += convert<float>(cit) * (dw[g] * da[t]);
            }
        }
    }

    if constexpr (S > 1) {
#pragma unroll
        for (int g = 0; g < NR; ++g) {
#pragma unroll
            for (int q = 0; q < RC; ++q) {
                const uint32_t off = (uint32_t) (((ks * NR + g) * RC + q) * DPAS_N * sizeof(float));
                slm_block_store<float, DPAS_N>(off, acc[g].template select<DPAS_N, 1>(DPAS_N * q));
            }
        }
        barrier();
        if (ks != 0) return;
        for (int o = 1; o < S; ++o) {
#pragma unroll
            for (int g = 0; g < NR; ++g) {
#pragma unroll
                for (int q = 0; q < RC; ++q) {
                    const uint32_t off = (uint32_t) (((o * NR + g) * RC + q) * DPAS_N * sizeof(float));
                    acc[g].template select<DPAS_N, 1>(DPAS_N * q) += slm_block_load<float, DPAS_N>(off);
                }
            }
        }
    }

    const simd<uint32_t, DPAS_N> lane(0, 1);
#pragma unroll
    for (int t = 0; t < RC; ++t) {
        if (t >= ncols) continue;
        float * drow = dst + (size_t) t * nrows_dst;
#pragma unroll
        for (int g = 0; g < NR; ++g) {
            const simd<uint32_t, DPAS_N> n  = lane + (uint32_t) (n0 + DPAS_N * g);
            const simd_mask<DPAS_N>      ok = n < (uint32_t) nrows;
            scatter<float, DPAS_N>(drow, n * (uint32_t) sizeof(float), acc[g].template select<DPAS_N, 1>(DPAS_N * t), ok);
        }
    }
}

template <int DPAS_N, int RC, int NR, int S>
static void launch_pq2_xmx_rc(const uint32_t * wq, const uint16_t * wd, const uint32_t * a8, const float * as,
                              float * dst, int K, int nrows, int ncols, int nrows_dst, dpct::queue_ptr stream) {
    constexpr int TN = DPAS_N * NR;
    const int n_tiles_n = (nrows + TN - 1) / TN;
    const int n_tiles   = n_tiles_n;

    stream->submit([&](sycl::handler & h) {
        if constexpr (S > 1) {
            const sycl::nd_range<1> nd{ sycl::range<1>((size_t) n_tiles * S), sycl::range<1>(S) };
            h.parallel_for(nd, [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                pq2_xmx_thread_rc<DPAS_N, RC, NR, S>(wq, wd, a8, as, dst, K, nrows, ncols, nrows_dst, n_tiles_n,
                                                     (int) it.get_group(0), (int) it.get_local_id(0));
            });
        } else {
            const size_t global = (size_t) ((n_tiles + PQ2_XMX_WG - 1) / PQ2_XMX_WG) * PQ2_XMX_WG;
            const sycl::nd_range<1> nd{ sycl::range<1>(global), sycl::range<1>(PQ2_XMX_WG) };
            h.parallel_for(nd, [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                const int tile = (int) it.get_global_id(0);
                if (tile < n_tiles) {
                    pq2_xmx_thread_rc<DPAS_N, RC, NR, 1>(wq, wd, a8, as, dst, K, nrows, ncols, nrows_dst, n_tiles_n, tile, 0);
                }
            });
        }
    });
}

template <int DPAS_N, int RC, int NR>
static void launch_pq2_xmx_rc_split(const uint32_t * wq, const uint16_t * wd, const uint32_t * a8, const float * as,
                                    float * dst, int K, int nrows, int ncols, int nrows_dst, dpct::queue_ptr stream) {
    const int nb      = K / QK_PQ2_0;
    const int TN      = DPAS_N * NR;
    const int n_tiles = (nrows + TN - 1) / TN;
    const int target  = PQ2_XMX_SMALL_TARGET_THREADS;
    int       split   = 1;
    while (split < 16 && n_tiles * split < target && nb >= 4 * split) {
        split *= 2;
    }
    switch (split) {
        case 1:  launch_pq2_xmx_rc<DPAS_N, RC, NR, 1>(wq, wd, a8, as, dst, K, nrows, ncols, nrows_dst, stream);  break;
        case 2:  launch_pq2_xmx_rc<DPAS_N, RC, NR, 2>(wq, wd, a8, as, dst, K, nrows, ncols, nrows_dst, stream);  break;
        case 4:  launch_pq2_xmx_rc<DPAS_N, RC, NR, 4>(wq, wd, a8, as, dst, K, nrows, ncols, nrows_dst, stream);  break;
        case 8:  launch_pq2_xmx_rc<DPAS_N, RC, NR, 8>(wq, wd, a8, as, dst, K, nrows, ncols, nrows_dst, stream);  break;
        default: launch_pq2_xmx_rc<DPAS_N, RC, NR, 16>(wq, wd, a8, as, dst, K, nrows, ncols, nrows_dst, stream); break;
    }
}

template <int DPAS_N>
static bool launch_pq2_xmx_rc_dispatch(const uint32_t * wq, const uint16_t * wd, const uint32_t * a8,
                                       const float * as, float * dst, int K, int nrows, int ncols,
                                       int nrows_dst, dpct::queue_ptr stream) {
    constexpr int NR = 4;
    switch (ncols) {
        case 1: launch_pq2_xmx_rc_split<DPAS_N, 1, NR>(wq, wd, a8, as, dst, K, nrows, ncols, nrows_dst, stream); return true;
        case 2: launch_pq2_xmx_rc_split<DPAS_N, 2, NR>(wq, wd, a8, as, dst, K, nrows, ncols, nrows_dst, stream); return true;
        case 3: launch_pq2_xmx_rc_split<DPAS_N, 3, NR>(wq, wd, a8, as, dst, K, nrows, ncols, nrows_dst, stream); return true;
        case 4: launch_pq2_xmx_rc_split<DPAS_N, 4, NR>(wq, wd, a8, as, dst, K, nrows, ncols, nrows_dst, stream); return true;
        case 5: launch_pq2_xmx_rc_split<DPAS_N, 5, NR>(wq, wd, a8, as, dst, K, nrows, ncols, nrows_dst, stream); return true;
        case 6: launch_pq2_xmx_rc_split<DPAS_N, 6, NR>(wq, wd, a8, as, dst, K, nrows, ncols, nrows_dst, stream); return true;
        case 7: launch_pq2_xmx_rc_split<DPAS_N, 7, NR>(wq, wd, a8, as, dst, K, nrows, ncols, nrows_dst, stream); return true;
        case 8: launch_pq2_xmx_rc_split<DPAS_N, 8, NR>(wq, wd, a8, as, dst, K, nrows, ncols, nrows_dst, stream); return true;
        default: return false;
    }
}

template <int MR, int NR>
static void launch_pq2_xmx_split_dispatch(const uint32_t * wq, const uint16_t * wd, const uint32_t * a8,
                                          const float * as, float * dst, int K, int nrows, int ncols,
                                          int nrows_dst, int dpas_w, dpct::queue_ptr stream) {
    if (dpas_w == 8) {
        launch_pq2_xmx_split<8, MR, NR>(wq, wd, a8, as, dst, K, nrows, ncols, nrows_dst, stream);
    } else {
        launch_pq2_xmx_split<16, MR, NR>(wq, wd, a8, as, dst, K, nrows, ncols, nrows_dst, stream);
    }
}

} // namespace

bool ggml_sycl_pq2_xmx_supports_ne0(int64_t ne0) {
    return ne0 % QK_PQ2_0 == 0 && (ne0 / QK_PQ2_0) * PQ2_XMX_QS_BYTES >= 64;
}

bool ggml_sycl_pq2_xmx_reorder(ggml_tensor * src0, dpct::queue_ptr stream) {
    GGML_ASSERT((src0->type == GGML_TYPE_PQ2_0 || src0->type == GGML_TYPE_PTQ1_0) && ggml_is_contiguous(src0));

    const size_t size = ggml_nbytes(src0);
    const size_t nblk = (size_t) ggml_nelements(src0) / QK_PQ2_0;
    uint8_t *    data = (uint8_t *) src0->data;

    constexpr size_t CHUNK_BYTES = 512ull * 1024 * 1024;
    const size_t tmp_size = std::min(size, CHUNK_BYTES);
    void * tmp = sycl::malloc_device(tmp_size, *stream);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for the PQ2_0 XMX reorder, skipping it\n", __func__, tmp_size);
        return false;
    }

    uint8_t *    qs = data;
    sycl::half * d  = (sycl::half *) (data + nblk * PQ2_XMX_QS_BYTES);

    const size_t bytes_per_block =
        (src0->type == GGML_TYPE_PQ2_0) ? sizeof(block_pq2_0) : sizeof(block_ptq1_0);
    const size_t blocks_per_chunk = std::max<size_t>(1, tmp_size / bytes_per_block);

    for (size_t blk_start = 0; blk_start < nblk; ) {
        const size_t chunk_start  = blk_start;
        const size_t chunk_blocks = std::min(blocks_per_chunk, nblk - chunk_start);
        const size_t chunk_bytes  = chunk_blocks * bytes_per_block;
        const size_t byte_start   = chunk_start * bytes_per_block;

        stream->memcpy(tmp, data + byte_start, chunk_bytes).wait();

        if (src0->type == GGML_TYPE_PQ2_0) {
            stream->parallel_for(sycl::range<1>(chunk_blocks), [=](sycl::id<1> i) {
                const size_t gi = chunk_start + i;
                const block_pq2_0 * x = (const block_pq2_0 *) tmp + i;
#pragma unroll
                for (int j = 0; j < PQ2_XMX_QS_BYTES; ++j) {
                    qs[gi * PQ2_XMX_QS_BYTES + j] = x->qs[j];
                }
                d[gi] = x->d;
            }).wait();
        } else {
            stream->parallel_for(sycl::range<1>(chunk_blocks), [=](sycl::id<1> i) {
                const size_t gi = chunk_start + i;
                const block_ptq1_0 * x = (const block_ptq1_0 *) tmp + i;
                for (int j = 0; j < PQ2_XMX_QS_BYTES; ++j) {
                    uint8_t byte = 0;
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        byte |= (uint8_t) ((ptq1_0_trit(x, 4 * j + k) + 1) << (2 * k));
                    }
                    qs[gi * PQ2_XMX_QS_BYTES + j] = byte;
                }
                d[gi] = x->d;
            }).wait();
        }

        blk_start += chunk_blocks;
    }

    sycl::free(tmp, *stream);
    return true;
}

void ggml_sycl_pq2_xmx_mul_mat(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                               ggml_tensor * dst) {
    GGML_ASSERT((src0->type == GGML_TYPE_PQ2_0 || src0->type == GGML_TYPE_PTQ1_0) && src0->ne[2] == 1 &&
                src0->ne[3] == 1);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && src1->nb[0] == sizeof(float));
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_sycl_pq2_xmx_supports_ne0(src0->ne[0]));

    const int K     = (int) src0->ne[0];
    const int nrows = (int) src0->ne[1];
    const int nb    = K / QK_PQ2_0;
    const int ne11  = (int) src1->ne[1];
    const int ne12  = (int) src1->ne[2];
    const int ncols = (int) (src1->ne[1] * src1->ne[2] * src1->ne[3]);

    const int dpas_w = ggml_sycl_info().devices[ctx.device].dpas_exec_size;
    GGML_ASSERT(dpas_w == 8 || dpas_w == 16);

    dpct::queue_ptr stream = ctx.stream();

    ggml_sycl_pool_alloc<int8_t> a8_alloc(ctx.pool(), (size_t) ncols * K + 64);
    ggml_sycl_pool_alloc<float>  as_alloc(ctx.pool(), (size_t) ncols * nb);
    int8_t * a8 = (int8_t *) GGML_PAD((uintptr_t) a8_alloc.get(), 64);
    float *  as = as_alloc.get();

    {
        const char * src1_d = (const char *) src1->data;
        const size_t nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) ncols * nb * 16), sycl::range<1>(16)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const int grp = (int) it.get_group(0);
                const int j   = grp / nb;
                const int b   = grp % nb;
                const int l   = (int) it.get_local_id(0);
                const int i1  = j % ne11;
                const int i2  = (j / ne11) % ne12;
                const int i3  = j / (ne11 * ne12);

                const float * x = (const float *) (src1_d + i1 * nb11 + i2 * nb12 + i3 * nb13) + b * QK_PQ2_0 + l * 8;
                float v[8];
                float amax = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    v[i] = x[i];
                    amax = sycl::fmax(amax, sycl::fabs(v[i]));
                }
                amax = sycl::reduce_over_group(it.get_sub_group(), amax, sycl::maximum<float>());
                const float d  = amax / 127.0f;
                const float id = d != 0.0f ? 1.0f / d : 0.0f;

                sycl::vec<int8_t, 8> q;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    q[i] = (int8_t) sycl::round(v[i] * id);
                }
                *(sycl::vec<int8_t, 8> *) (a8 + (size_t) j * K + b * QK_PQ2_0 + l * 8) = q;
                if (l == 0) {
                    as[(size_t) j * nb + b] = d;
                }
            });
    }

    const uint32_t * wq = (const uint32_t *) src0->data;
    const uint16_t * wd = (const uint16_t *) ((const uint8_t *) src0->data + (size_t) nrows * nb * PQ2_XMX_QS_BYTES);
    float *          dd = (float *) dst->data;
    const int        nrows_dst = (int) dst->ne[0];

    if (ncols <= 8) {
        bool handled = false;
        if (dpas_w == 8) {
            handled = launch_pq2_xmx_rc_dispatch<8>(wq, wd, (const uint32_t *) a8, as, dd, K, nrows, ncols, nrows_dst, stream);
        } else {
            handled = launch_pq2_xmx_rc_dispatch<16>(wq, wd, (const uint32_t *) a8, as, dd, K, nrows, ncols, nrows_dst, stream);
        }
        if (handled) return;
    }

    if (ncols <= 8) {
        launch_pq2_xmx_split_dispatch<1, 2>(wq, wd, (const uint32_t *) a8, as, dd, K, nrows, ncols, nrows_dst, dpas_w, stream);
    } else if (ncols <= 16) {
        launch_pq2_xmx_split_dispatch<2, 2>(wq, wd, (const uint32_t *) a8, as, dd, K, nrows, ncols, nrows_dst, dpas_w, stream);
    } else {
        // Prefill: try the block layout first (MSPLIT x NSPLIT WIs, shared SLM
        // A- and B-slabs). Falls back to the split kernel if the shapes don't
        // divide evenly.
        constexpr int BMR = 2, BNR = 4, BMSPLIT = 4, BNSPLIT = 4;
        bool handled = false;
        if (dpas_w == 8) {
            handled = launch_pq2_xmx_block<8, BMR, BNR, 8, 8>(
              wq, wd, (const uint32_t *) a8, as, dd, K, nrows, ncols, nrows_dst, stream) ||
          launch_pq2_xmx_block<8, BMR, BNR, 4, 4>(
              wq, wd, (const uint32_t *) a8, as, dd, K, nrows, ncols, nrows_dst, stream);
        } else {
        constexpr int BMR = 2, BNR = 4;
        bool handled = false;
        if (dpas_w == 8) {
            handled = launch_pq2_xmx_block<8, BMR, BNR, 4, 4>(
                          wq, wd, (const uint32_t *) a8, as, dd, K, nrows, ncols, nrows_dst, stream);
        } else {
            handled = launch_pq2_xmx_block<16, BMR, BNR, 4, 8>(
                          wq, wd, (const uint32_t *) a8, as, dd, K, nrows, ncols, nrows_dst, stream) ||
                      launch_pq2_xmx_block<16, BMR, BNR, 4, 4>(
                          wq, wd, (const uint32_t *) a8, as, dd, K, nrows, ncols, nrows_dst, stream);
        }
        if (!handled) {
            launch_pq2_xmx_split_dispatch<2, 4>(wq, wd, (const uint32_t *) a8, as, dd, K, nrows, ncols, nrows_dst, dpas_w, stream);
        }
    }
    }
}

#else

bool ggml_sycl_pq2_xmx_supports_ne0(int64_t) {
    return false;
}

bool ggml_sycl_pq2_xmx_reorder(ggml_tensor *, dpct::queue_ptr) {
    return false;
}

void ggml_sycl_pq2_xmx_mul_mat(ggml_backend_sycl_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *) {
    GGML_ABORT("PQ2_0 XMX path is not built in");
}

#endif // __INTEL_LLVM_COMPILER && !GGML_SYCL_NO_PQ2_XMX