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
#include <fly/defines.h>
#include <fly/image.h>
#include "symbol_manager.hpp"

fly_err fly_gradient(fly_array *dx, fly_array *dy, const fly_array in) {
    CHECK_ARRAYS(in);
    CALL(fly_gradient, dx, dy, in);
}

fly_err fly_load_image(fly_array *out, const char *filename, const bool isColor) {
    CALL(fly_load_image, out, filename, isColor);
}

fly_err fly_save_image(const char *filename, const fly_array in) {
    CHECK_ARRAYS(in);
    CALL(fly_save_image, filename, in);
}

fly_err fly_load_image_memory(fly_array *out, const void *ptr) {
    CALL(fly_load_image_memory, out, ptr);
}

fly_err fly_save_image_memory(void **ptr, const fly_array in,
                            const fly_image_format format) {
    CHECK_ARRAYS(in);
    CALL(fly_save_image_memory, ptr, in, format);
}

fly_err fly_delete_image_memory(void *ptr) { CALL(fly_delete_image_memory, ptr); }

fly_err fly_load_image_native(fly_array *out, const char *filename) {
    CALL(fly_load_image_native, out, filename);
}

fly_err fly_save_image_native(const char *filename, const fly_array in) {
    CHECK_ARRAYS(in);
    CALL(fly_save_image_native, filename, in);
}

fly_err fly_is_image_io_available(bool *out) {
    CALL(fly_is_image_io_available, out);
}

fly_err fly_resize(fly_array *out, const fly_array in, const dim_t odim0,
                 const dim_t odim1, const fly_interp_type method) {
    CHECK_ARRAYS(in);
    CALL(fly_resize, out, in, odim0, odim1, method);
}

fly_err fly_transform(fly_array *out, const fly_array in, const fly_array transform,
                    const dim_t odim0, const dim_t odim1,
                    const fly_interp_type method, const bool inverse) {
    CHECK_ARRAYS(in, transform);
    CALL(fly_transform, out, in, transform, odim0, odim1, method, inverse);
}

fly_err fly_transform_v2(fly_array *out, const fly_array in,
                       const fly_array transform, const dim_t odim0,
                       const dim_t odim1, const fly_interp_type method,
                       const bool inverse) {
    CHECK_ARRAYS(out, in, transform);
    CALL(fly_transform_v2, out, in, transform, odim0, odim1, method, inverse);
}

fly_err fly_transform_coordinates(fly_array *out, const fly_array tf,
                                const float d0, const float d1) {
    CHECK_ARRAYS(tf);
    CALL(fly_transform_coordinates, out, tf, d0, d1);
}

fly_err fly_rotate(fly_array *out, const fly_array in, const float theta,
                 const bool crop, const fly_interp_type method) {
    CHECK_ARRAYS(in);
    CALL(fly_rotate, out, in, theta, crop, method);
}

fly_err fly_translate(fly_array *out, const fly_array in, const float trans0,
                    const float trans1, const dim_t odim0, const dim_t odim1,
                    const fly_interp_type method) {
    CHECK_ARRAYS(in);
    CALL(fly_translate, out, in, trans0, trans1, odim0, odim1, method);
}

fly_err fly_scale(fly_array *out, const fly_array in, const float scale0,
                const float scale1, const dim_t odim0, const dim_t odim1,
                const fly_interp_type method) {
    CHECK_ARRAYS(in);
    CALL(fly_scale, out, in, scale0, scale1, odim0, odim1, method);
}

fly_err fly_skew(fly_array *out, const fly_array in, const float skew0,
               const float skew1, const dim_t odim0, const dim_t odim1,
               const fly_interp_type method, const bool inverse) {
    CHECK_ARRAYS(in);
    CALL(fly_skew, out, in, skew0, skew1, odim0, odim1, method, inverse);
}

fly_err fly_histogram(fly_array *out, const fly_array in, const unsigned nbins,
                    const double minval, const double maxval) {
    CHECK_ARRAYS(in);
    CALL(fly_histogram, out, in, nbins, minval, maxval);
}

fly_err fly_dilate(fly_array *out, const fly_array in, const fly_array mask) {
    CHECK_ARRAYS(in, mask);
    CALL(fly_dilate, out, in, mask);
}

fly_err fly_dilate3(fly_array *out, const fly_array in, const fly_array mask) {
    CHECK_ARRAYS(in, mask);
    CALL(fly_dilate3, out, in, mask);
}

fly_err fly_erode(fly_array *out, const fly_array in, const fly_array mask) {
    CHECK_ARRAYS(in, mask);
    CALL(fly_erode, out, in, mask);
}

fly_err fly_erode3(fly_array *out, const fly_array in, const fly_array mask) {
    CHECK_ARRAYS(in, mask);
    CALL(fly_erode3, out, in, mask);
}

fly_err fly_bilateral(fly_array *out, const fly_array in, const float spatial_sigma,
                    const float chromatic_sigma, const bool isColor) {
    CHECK_ARRAYS(in);
    CALL(fly_bilateral, out, in, spatial_sigma, chromatic_sigma, isColor);
}

fly_err fly_mean_shift(fly_array *out, const fly_array in,
                     const float spatial_sigma, const float chromatic_sigma,
                     const unsigned iter, const bool is_color) {
    CHECK_ARRAYS(in);
    CALL(fly_mean_shift, out, in, spatial_sigma, chromatic_sigma, iter,
         is_color);
}

fly_err fly_minfilt(fly_array *out, const fly_array in, const dim_t wind_length,
                  const dim_t wind_width, const fly_border_type edge_pad) {
    CHECK_ARRAYS(in);
    CALL(fly_minfilt, out, in, wind_length, wind_width, edge_pad);
}

fly_err fly_maxfilt(fly_array *out, const fly_array in, const dim_t wind_length,
                  const dim_t wind_width, const fly_border_type edge_pad) {
    CHECK_ARRAYS(in);
    CALL(fly_maxfilt, out, in, wind_length, wind_width, edge_pad);
}

fly_err fly_regions(fly_array *out, const fly_array in,
                  const fly_connectivity connectivity, const fly_dtype ty) {
    CHECK_ARRAYS(in);
    CALL(fly_regions, out, in, connectivity, ty);
}

fly_err fly_sobel_operator(fly_array *dx, fly_array *dy, const fly_array img,
                         const unsigned ker_size) {
    CHECK_ARRAYS(img);
    CALL(fly_sobel_operator, dx, dy, img, ker_size);
}

fly_err fly_rgb2gray(fly_array *out, const fly_array in, const float rPercent,
                   const float gPercent, const float bPercent) {
    CHECK_ARRAYS(in);
    CALL(fly_rgb2gray, out, in, rPercent, gPercent, bPercent);
}

fly_err fly_gray2rgb(fly_array *out, const fly_array in, const float rFactor,
                   const float gFactor, const float bFactor) {
    CHECK_ARRAYS(in);
    CALL(fly_gray2rgb, out, in, rFactor, gFactor, bFactor);
}

fly_err fly_hist_equal(fly_array *out, const fly_array in, const fly_array hist) {
    CHECK_ARRAYS(in, hist);
    CALL(fly_hist_equal, out, in, hist);
}

fly_err fly_gaussian_kernel(fly_array *out, const int rows, const int cols,
                          const double sigma_r, const double sigma_c) {
    CALL(fly_gaussian_kernel, out, rows, cols, sigma_r, sigma_c);
}

fly_err fly_hsv2rgb(fly_array *out, const fly_array in) {
    CHECK_ARRAYS(in);
    CALL(fly_hsv2rgb, out, in);
}

fly_err fly_rgb2hsv(fly_array *out, const fly_array in) {
    CHECK_ARRAYS(in);
    CALL(fly_rgb2hsv, out, in);
}

fly_err fly_color_space(fly_array *out, const fly_array image, const fly_cspace_t to,
                      const fly_cspace_t from) {
    CHECK_ARRAYS(image);
    CALL(fly_color_space, out, image, to, from);
}

fly_err fly_unwrap(fly_array *out, const fly_array in, const dim_t wx,
                 const dim_t wy, const dim_t sx, const dim_t sy, const dim_t px,
                 const dim_t py, const bool is_column) {
    CHECK_ARRAYS(in);
    CALL(fly_unwrap, out, in, wx, wy, sx, sy, px, py, is_column);
}

fly_err fly_wrap(fly_array *out, const fly_array in, const dim_t ox, const dim_t oy,
               const dim_t wx, const dim_t wy, const dim_t sx, const dim_t sy,
               const dim_t px, const dim_t py, const bool is_column) {
    CHECK_ARRAYS(in);
    CALL(fly_wrap, out, in, ox, oy, wx, wy, sx, sy, px, py, is_column);
}

fly_err fly_wrap_v2(fly_array *out, const fly_array in, const dim_t ox,
                  const dim_t oy, const dim_t wx, const dim_t wy,
                  const dim_t sx, const dim_t sy, const dim_t px,
                  const dim_t py, const bool is_column) {
    CHECK_ARRAYS(out, in);
    CALL(fly_wrap_v2, out, in, ox, oy, wx, wy, sx, sy, px, py, is_column);
}

fly_err fly_sat(fly_array *out, const fly_array in) {
    CHECK_ARRAYS(in);
    CALL(fly_sat, out, in);
}

fly_err fly_ycbcr2rgb(fly_array *out, const fly_array in,
                    const fly_ycc_std standard) {
    CHECK_ARRAYS(in);
    CALL(fly_ycbcr2rgb, out, in, standard);
}

fly_err fly_rgb2ycbcr(fly_array *out, const fly_array in,
                    const fly_ycc_std standard) {
    CHECK_ARRAYS(in);
    CALL(fly_rgb2ycbcr, out, in, standard);
}

fly_err fly_canny(fly_array *out, const fly_array in, const fly_canny_threshold ct,
                const float t1, const float t2, const unsigned sw,
                const bool isf) {
    CHECK_ARRAYS(in);
    CALL(fly_canny, out, in, ct, t1, t2, sw, isf);
}

fly_err fly_anisotropic_diffusion(fly_array *out, const fly_array in,
                                const float dt, const float K,
                                const unsigned iterations,
                                const fly_flux_function fftype,
                                const fly_diffusion_eq eq) {
    CHECK_ARRAYS(in);
    CALL(fly_anisotropic_diffusion, out, in, dt, K, iterations, fftype, eq);
}

fly_err fly_iterative_deconv(fly_array *out, const fly_array in, const fly_array ker,
                           const unsigned iterations, const float relax_factor,
                           const fly_iterative_deconv_algo algo) {
    CHECK_ARRAYS(in, ker);
    CALL(fly_iterative_deconv, out, in, ker, iterations, relax_factor, algo);
}

fly_err fly_inverse_deconv(fly_array *out, const fly_array in, const fly_array psf,
                         const float gamma, const fly_inverse_deconv_algo algo) {
    CHECK_ARRAYS(in, psf);
    CALL(fly_inverse_deconv, out, in, psf, gamma, algo);
}

fly_err fly_confidence_cc(fly_array *out, const fly_array in, const fly_array seedx,
                        const fly_array seedy, const unsigned radius,
                        const unsigned multiplier, const int iter,
                        const double segmented_value) {
    CHECK_ARRAYS(in, seedx, seedy);
    CALL(fly_confidence_cc, out, in, seedx, seedy, radius, multiplier, iter,
         segmented_value);
}
