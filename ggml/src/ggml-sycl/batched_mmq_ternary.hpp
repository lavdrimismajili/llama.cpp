// batched_mmq_ternary.hpp
#pragma once
#include "vecdotq.hpp"

static __dpct_inline__ void unpack_ptq1_0_row(const block_ptq1_0 * bq, int q[32]) {
    int idx = 0;
#pragma unroll
    for (int g = 0; g < 4; ++g) {
        const uint32_t packed = get_int_from_uint8_aligned(bq->qs, g);
        uint32_t v_lo = (packed & 0x000000FF) | ((packed & 0x0000FF00) << 8);
        uint32_t v_hi = ((packed >> 16) & 0x000000FF) | ((packed & 0xFF000000) >> 8);
#pragma unroll
        for (int t = 0; t < 5; ++t) {
            const uint32_t w_lo = v_lo * 3;
            const uint32_t w_hi = v_hi * 3;
            v_lo = w_lo & 0x00FF00FF;
            v_hi = w_hi & 0x00FF00FF;
            const uint32_t perm = ((w_lo >> 8) & 0x000000FF) |
                                  ((w_lo >> 16) & 0x0000FF00) |
                                  ((w_hi << 8)  & 0x00FF0000) |
                                  (w_hi         & 0xFF000000);
            q[idx++] = byte_sub_4(perm, 0x01010101);
        }
    }
#pragma unroll
    for (int g = 0; g < 2; ++g) {
        const uint32_t packed = get_int_from_uint8_aligned(bq->qs + 16, g);
        uint32_t v_lo = (packed & 0x000000FF) | ((packed & 0x0000FF00) << 8);
        uint32_t v_hi = ((packed >> 16) & 0x000000FF) | ((packed & 0xFF000000) >> 8);
#pragma unroll
        for (int t = 0; t < 5; ++t) {
            const uint32_t w_lo = v_lo * 3;
            const uint32_t w_hi = v_hi * 3;
            v_lo = w_lo & 0x00FF00FF;
            v_hi = w_hi & 0x00FF00FF;
            const uint32_t perm = ((w_lo >> 8) & 0x000000FF) |
                                  ((w_lo >> 16) & 0x0000FF00) |
                                  ((w_hi << 8)  & 0x00FF0000) |
                                  (w_hi         & 0xFF000000);
            q[idx++] = byte_sub_4(perm, 0x01010101);
        }
    }
    uint32_t v = (uint32_t) bq->qh[0] | ((uint32_t) bq->qh[1] << 16);
#pragma unroll
    for (int t = 0; t < 4; t += 2) {
        const uint32_t w0 = v * 3;
        v = w0 & 0x00FF00FF;
        const uint32_t w1 = v * 3;
        v = w1 & 0x00FF00FF;
        const uint32_t perm = ((w0 >> 8) & 0x000000FF) |
                              ((w0 >> 16) & 0x0000FF00) |
                              ((w1 << 8)  & 0x00FF0000) |
                              (w1         & 0xFF000000);
        q[idx++] = byte_sub_4(perm, 0x01010101);
    }
}

static __dpct_inline__ void ptq1_0_ymap(int sub[32], int yidx[32]) {
    int idx = 0;
#pragma unroll
    for (int g = 0; g < 4; ++g) {
#pragma unroll
        for (int t = 0; t < 5; ++t) {
            const int e = t * 16 + 4 * g;
            sub[idx]  = e >> 5;
            yidx[idx] = (e & 31) >> 2;
            ++idx;
        }
    }
#pragma unroll
    for (int g = 0; g < 2; ++g) {
#pragma unroll
        for (int t = 0; t < 5; ++t) {
            const int e = 80 + t * 8 + 4 * g;
            sub[idx]  = e >> 5;
            yidx[idx] = (e & 31) >> 2;
            ++idx;
        }
    }
#pragma unroll
    for (int t = 0; t < 2; ++t) {
        sub[idx]  = 3;
        yidx[idx] = 6 + t;
        ++idx;
    }
}

static __dpct_inline__ void unpack_pq2_0_row(const block_pq2_0 * bq, int q[32]) {
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        q[i] = (int) unpack_2bit_to_byte_lanes((uint32_t) bq->qs[i]);
    }
}

template <bool IsPTQ1_0>
static void mul_mat_batched_ternary_kernel(const void * __restrict__ vx, const void * __restrict__ vy,
                                           float * __restrict__ dst, const int nrows, const int ncols_dst,
                                           const int n_kblocks, const int64_t row_stride,
                                           const int stride_col_y, const int stride_col_dst,
                                           const sycl::nd_item<1> & item) {
    const int row = (int) item.get_global_id(0);
    if (row >= nrows) {
        return;
    }

    constexpr int SUB_PER_BLK = QK_PTQ1_0 / QK8_1;   // 128/32 = 4

    int sub[32];
    int yidx[32];
    if constexpr (IsPTQ1_0) {
        ptq1_0_ymap(sub, yidx);
    } else {
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            sub[i]  = i / 8;
            yidx[i] = i & 7;
        }
    }

    const char * vx_row = (const char *) vx + (size_t) row * (size_t) row_stride;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    int col = 0;

    // ---- 4-column main loop ----
    for (; col + 4 <= ncols_dst; col += 4) {
        float acc[4] = {0.f, 0.f, 0.f, 0.f};

        for (int kb = 0; kb < n_kblocks; ++kb) {
            int q[32];
            float row_scale;

            if constexpr (IsPTQ1_0) {
                const block_ptq1_0 * bq = (const block_ptq1_0 *) vx_row + kb;
                unpack_ptq1_0_row(bq, q);
                row_scale = (float) bq->d;
            } else {
                const block_pq2_0 * bq = (const block_pq2_0 *) vx_row + kb;
                unpack_pq2_0_row(bq, q);
                row_scale = bq->d;
            }

            int sumi[4][4] = {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}};

#pragma unroll
            for (int k = 0; k < 32; ++k) {
                const int s  = sub[k];
                const int yi = yidx[k];
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const block_q8_1 * yc =
                        y + (col + c) * stride_col_y + kb * SUB_PER_BLK + s;
                    const int u = get_int_from_int8_aligned(yc->qs, yi);
                    sumi[c][s] = dpct::dp4a(q[k], u, sumi[c][s]);
                }
            }

#pragma unroll
            for (int c = 0; c < 4; ++c) {
                float partial = 0.f;
                if constexpr (IsPTQ1_0) {
#pragma unroll
                    for (int s = 0; s < 4; ++s) {
                        const block_q8_1 * yc =
                            y + (col + c) * stride_col_y + kb * SUB_PER_BLK + s;
                        partial += (float) yc->ds[0] * (float) sumi[c][s];
                    }
                } else {
#pragma unroll
                    for (int s = 0; s < 4; ++s) {
                        const block_q8_1 * yc =
                            y + (col + c) * stride_col_y + kb * SUB_PER_BLK + s;
                        const sycl::float2 ds8f =
                            yc->ds.convert<float, sycl::rounding_mode::automatic>();
                        partial += sumi[c][s] * ds8f.x() - 0.25f * ds8f.y();
                    }
                }
                acc[c] += row_scale * partial;
            }
        }

#pragma unroll
        for (int c = 0; c < 4; ++c) {
            dst[(col + c) * stride_col_dst + row] = acc[c];
        }
    }

    // ---- tail ----
    for (; col < ncols_dst; ++col) {
        float acc = 0.f;
        for (int kb = 0; kb < n_kblocks; ++kb) {
            int q[32];
            float row_scale;
            if constexpr (IsPTQ1_0) {
                const block_ptq1_0 * bq = (const block_ptq1_0 *) vx_row + kb;
                unpack_ptq1_0_row(bq, q);
                row_scale = (float) bq->d;
            } else {
                const block_pq2_0 * bq = (const block_pq2_0 *) vx_row + kb;
                unpack_pq2_0_row(bq, q);
                row_scale = bq->d;
            }

            int sumi[4] = {0,0,0,0};
#pragma unroll
            for (int k = 0; k < 32; ++k) {
                const int s  = sub[k];
                const int yi = yidx[k];
                const block_q8_1 * yc =
                    y + col * stride_col_y + kb * SUB_PER_BLK + s;
                const int u = get_int_from_int8_aligned(yc->qs, yi);
                sumi[s] = dpct::dp4a(q[k], u, sumi[s]);
            }

            float partial = 0.f;
            if constexpr (IsPTQ1_0) {
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    const block_q8_1 * yc =
                        y + col * stride_col_y + kb * SUB_PER_BLK + s;
                    partial += (float) yc->ds[0] * (float) sumi[s];
                }
            } else {
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    const block_q8_1 * yc =
                        y + col * stride_col_y + kb * SUB_PER_BLK + s;
                    const sycl::float2 ds8f =
                        yc->ds.convert<float, sycl::rounding_mode::automatic>();
                    partial += sumi[s] * ds8f.x() - 0.25f * ds8f.y();
                }
            }
            acc += row_scale * partial;
        }
        dst[col * stride_col_dst + row] = acc;
    }
}

static void ggml_sycl_op_mul_mat_batched_ternary(ggml_backend_sycl_context & ctx, const ggml_tensor * src0,
                                                  const ggml_tensor * src1, ggml_tensor * dst,
                                                  const char * src0_dd_i, const float * /*src1_ddf_i*/,
                                                  const char * src1_ddq_i, float * dst_dd_i,
                                                  const int64_t row_low, const int64_t row_high,
                                                  const int64_t src1_ncols, const int64_t src1_padded_col_size,
                                                  const dpct::queue_ptr & stream) {
    const int nrows = (int) (row_high - row_low);
    // n_kblocks from the weight tensor shape, not from the y tensor size —
    // robust to any padding the backend applies to src1.
    const int n_kblocks = (int) (src0->ne[0] / QK_PTQ1_0);
    // Actual row stride from the tensor, not from n_kblocks * sizeof(block) —
    // robust to any alignment padding between rows.
    const int64_t row_stride = src0->nb[1];
    const int stride_col_y = (int) (src1_padded_col_size / QK8_1);
    const int stride_col_dst = (int) dst->ne[0];

    constexpr int WG = 64;
    const size_t global = (size_t) ((nrows + WG - 1) / WG) * WG;
    const sycl::nd_range<1> nd{ sycl::range<1>(global), sycl::range<1>(WG) };

    if (src0->type == GGML_TYPE_PTQ1_0) {
        stream->parallel_for(nd, [=](sycl::nd_item<1> item) {
            mul_mat_batched_ternary_kernel<true>(src0_dd_i, src1_ddq_i, dst_dd_i, nrows, (int) src1_ncols,
                                                 n_kblocks, row_stride, stride_col_y, stride_col_dst, item);
        });
    } else {
        stream->parallel_for(nd, [=](sycl::nd_item<1> item) {
            mul_mat_batched_ternary_kernel<false>(src0_dd_i, src1_ddq_i, dst_dd_i, nrows, (int) src1_ncols,
                                                  n_kblocks, row_stride, stride_col_y, stride_col_dst, item);
        });
    }
}