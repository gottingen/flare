// Copyright 2023 The EA Authors.
// part of Elastic AI Search
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include <fly/array.h>
#include <fly/dim4.hpp>
#include <fly/signal.h>
#include "symbol_manager.hpp"

fly_err fly_approx1(fly_array *yo, const fly_array yi, const fly_array xo,
                  const fly_interp_type method, const float offGrid) {
    CHECK_ARRAYS(yo, yi, xo);
    CALL(fly_approx1, yo, yi, xo, method, offGrid);
}

fly_err fly_approx1_v2(fly_array *yo, const fly_array yi, const fly_array xo,
                     const fly_interp_type method, const float offGrid) {
    CHECK_ARRAYS(yo, yi, xo);
    CALL(fly_approx1_v2, yo, yi, xo, method, offGrid);
}

fly_err fly_approx2(fly_array *zo, const fly_array zi, const fly_array xo,
                  const fly_array yo, const fly_interp_type method,
                  const float offGrid) {
    CHECK_ARRAYS(zo, zi, xo, yo);
    CALL(fly_approx2, zo, zi, xo, yo, method, offGrid);
}

fly_err fly_approx2_v2(fly_array *zo, const fly_array zi, const fly_array xo,
                     const fly_array yo, const fly_interp_type method,
                     const float offGrid) {
    CHECK_ARRAYS(zo, zi, xo, yo);
    CALL(fly_approx2_v2, zo, zi, xo, yo, method, offGrid);
}

fly_err fly_approx1_uniform(fly_array *yo, const fly_array yi, const fly_array xo,
                          const int xdim, const double xi_beg,
                          const double xi_step, const fly_interp_type method,
                          const float offGrid) {
    CHECK_ARRAYS(yo, yi, xo);
    CALL(fly_approx1_uniform, yo, yi, xo, xdim, xi_beg, xi_step, method,
         offGrid);
}

fly_err fly_approx1_uniform_v2(fly_array *yo, const fly_array yi, const fly_array xo,
                             const int xdim, const double xi_beg,
                             const double xi_step, const fly_interp_type method,
                             const float offGrid) {
    CHECK_ARRAYS(yo, yi, xo);
    CALL(fly_approx1_uniform_v2, yo, yi, xo, xdim, xi_beg, xi_step, method,
         offGrid);
}

fly_err fly_approx2_uniform(fly_array *zo, const fly_array zi, const fly_array xo,
                          const int xdim, const double xi_beg,
                          const double xi_step, const fly_array yo,
                          const int ydim, const double yi_beg,
                          const double yi_step, const fly_interp_type method,
                          const float offGrid) {
    CHECK_ARRAYS(zo, zi, xo, yo);
    CALL(fly_approx2_uniform, zo, zi, xo, xdim, xi_beg, xi_step, yo, ydim,
         yi_beg, yi_step, method, offGrid);
}

fly_err fly_approx2_uniform_v2(fly_array *zo, const fly_array zi, const fly_array xo,
                             const int xdim, const double xi_beg,
                             const double xi_step, const fly_array yo,
                             const int ydim, const double yi_beg,
                             const double yi_step, const fly_interp_type method,
                             const float offGrid) {
    CHECK_ARRAYS(zo, zi, xo, yo);
    CALL(fly_approx2_uniform_v2, zo, zi, xo, xdim, xi_beg, xi_step, yo, ydim,
         yi_beg, yi_step, method, offGrid);
}

fly_err fly_set_fft_plan_cache_size(size_t cache_size) {
    CALL(fly_set_fft_plan_cache_size, cache_size);
}

#define FFT_HAPI_DEF(fly_func)                               \
    fly_err fly_func(fly_array in, const double norm_factor) { \
        CHECK_ARRAYS(in);                                   \
        CALL(fly_func, in, norm_factor);                     \
    }

FFT_HAPI_DEF(fly_fft_inplace)
FFT_HAPI_DEF(fly_fft2_inplace)
FFT_HAPI_DEF(fly_fft3_inplace)
FFT_HAPI_DEF(fly_ifft_inplace)
FFT_HAPI_DEF(fly_ifft2_inplace)
FFT_HAPI_DEF(fly_ifft3_inplace)

fly_err fly_fft(fly_array *out, const fly_array in, const double norm_factor,
              const dim_t odim0) {
    CHECK_ARRAYS(in);
    CALL(fly_fft, out, in, norm_factor, odim0);
}

fly_err fly_fft2(fly_array *out, const fly_array in, const double norm_factor,
               const dim_t odim0, const dim_t odim1) {
    CHECK_ARRAYS(in);
    CALL(fly_fft2, out, in, norm_factor, odim0, odim1);
}

fly_err fly_fft3(fly_array *out, const fly_array in, const double norm_factor,
               const dim_t odim0, const dim_t odim1, const dim_t odim2) {
    CHECK_ARRAYS(in);
    CALL(fly_fft3, out, in, norm_factor, odim0, odim1, odim2);
}

fly_err fly_ifft(fly_array *out, const fly_array in, const double norm_factor,
               const dim_t odim0) {
    CHECK_ARRAYS(in);
    CALL(fly_ifft, out, in, norm_factor, odim0);
}

fly_err fly_ifft2(fly_array *out, const fly_array in, const double norm_factor,
                const dim_t odim0, const dim_t odim1) {
    CHECK_ARRAYS(in);
    CALL(fly_ifft2, out, in, norm_factor, odim0, odim1);
}

fly_err fly_ifft3(fly_array *out, const fly_array in, const double norm_factor,
                const dim_t odim0, const dim_t odim1, const dim_t odim2) {
    CHECK_ARRAYS(in);
    CALL(fly_ifft3, out, in, norm_factor, odim0, odim1, odim2);
}

fly_err fly_fft_r2c(fly_array *out, const fly_array in, const double norm_factor,
                  const dim_t pad0) {
    CHECK_ARRAYS(in);
    CALL(fly_fft_r2c, out, in, norm_factor, pad0);
}

fly_err fly_fft2_r2c(fly_array *out, const fly_array in, const double norm_factor,
                   const dim_t pad0, const dim_t pad1) {
    CHECK_ARRAYS(in);
    CALL(fly_fft2_r2c, out, in, norm_factor, pad0, pad1);
}

fly_err fly_fft3_r2c(fly_array *out, const fly_array in, const double norm_factor,
                   const dim_t pad0, const dim_t pad1, const dim_t pad2) {
    CHECK_ARRAYS(in);
    CALL(fly_fft3_r2c, out, in, norm_factor, pad0, pad1, pad2);
}

#define FFTC2R_HAPI_DEF(fly_func)                                               \
    fly_err fly_func(fly_array *out, const fly_array in, const double norm_factor, \
                   const bool is_odd) {                                        \
        CHECK_ARRAYS(in);                                                      \
        CALL(fly_func, out, in, norm_factor, is_odd);                           \
    }

FFTC2R_HAPI_DEF(fly_fft_c2r)
FFTC2R_HAPI_DEF(fly_fft2_c2r)
FFTC2R_HAPI_DEF(fly_fft3_c2r)

#define CONV_HAPI_DEF(fly_func)                                     \
    fly_err fly_func(fly_array *out, const fly_array signal,           \
                   const fly_array filter, const fly_conv_mode mode, \
                   fly_conv_domain domain) {                        \
        CHECK_ARRAYS(signal, filter);                              \
        CALL(fly_func, out, signal, filter, mode, domain);          \
    }

CONV_HAPI_DEF(fly_convolve1)
CONV_HAPI_DEF(fly_convolve2)
CONV_HAPI_DEF(fly_convolve3)

fly_err fly_convolve2_nn(fly_array *out, const fly_array signal,
                       const fly_array filter, const unsigned stride_dims,
                       const dim_t *strides, const unsigned padding_dims,
                       const dim_t *paddings, const unsigned dilation_dims,
                       const dim_t *dilations) {
    CHECK_ARRAYS(signal, filter);
    CALL(fly_convolve2_nn, out, signal, filter, stride_dims, strides,
         padding_dims, paddings, dilation_dims, dilations);
}

fly_err fly_convolve2_gradient_nn(
    fly_array *out, const fly_array incoming_gradient,
    const fly_array original_signal, const fly_array original_filter,
    const fly_array convolved_output, const unsigned stride_dims,
    const dim_t *strides, const unsigned padding_dims, const dim_t *paddings,
    const unsigned dilation_dims, const dim_t *dilations,
    fly_conv_gradient_type grad_type) {
    CHECK_ARRAYS(incoming_gradient, original_signal, original_filter,
                 convolved_output);
    CALL(fly_convolve2_gradient_nn, out, incoming_gradient, original_signal,
         original_filter, convolved_output, stride_dims, strides, padding_dims,
         paddings, dilation_dims, dilations, grad_type);
}

#define FFT_CONV_HAPI_DEF(fly_func)                                   \
    fly_err fly_func(fly_array *out, const fly_array signal,             \
                   const fly_array filter, const fly_conv_mode mode) { \
        CHECK_ARRAYS(signal, filter);                                \
        CALL(fly_func, out, signal, filter, mode);                    \
    }

FFT_CONV_HAPI_DEF(fly_fft_convolve1)
FFT_CONV_HAPI_DEF(fly_fft_convolve2)
FFT_CONV_HAPI_DEF(fly_fft_convolve3)

fly_err fly_convolve2_sep(fly_array *out, const fly_array col_filter,
                        const fly_array row_filter, const fly_array signal,
                        const fly_conv_mode mode) {
    CHECK_ARRAYS(col_filter, row_filter, signal);
    CALL(fly_convolve2_sep, out, col_filter, row_filter, signal, mode);
}

fly_err fly_fir(fly_array *y, const fly_array b, const fly_array x) {
    CHECK_ARRAYS(b, x);
    CALL(fly_fir, y, b, x);
}

fly_err fly_iir(fly_array *y, const fly_array b, const fly_array a,
              const fly_array x) {
    CHECK_ARRAYS(b, a, x);
    CALL(fly_iir, y, b, a, x);
}

fly_err fly_medfilt(fly_array *out, const fly_array in, const dim_t wind_length,
                  const dim_t wind_width, const fly_border_type edge_pad) {
    CHECK_ARRAYS(in);
    CALL(fly_medfilt, out, in, wind_length, wind_width, edge_pad);
}

fly_err fly_medfilt1(fly_array *out, const fly_array in, const dim_t wind_width,
                   const fly_border_type edge_pad) {
    CHECK_ARRAYS(in);
    CALL(fly_medfilt1, out, in, wind_width, edge_pad);
}

fly_err fly_medfilt2(fly_array *out, const fly_array in, const dim_t wind_length,
                   const dim_t wind_width, const fly_border_type edge_pad) {
    CHECK_ARRAYS(in);
    CALL(fly_medfilt2, out, in, wind_length, wind_width, edge_pad);
}
