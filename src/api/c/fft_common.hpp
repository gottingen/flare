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

#include <copy.hpp>
#include <fft.hpp>
#include <handle.hpp>

void computePaddedDims(fly::dim4 &pdims, const fly::dim4 &idims, const dim_t npad,
                       dim_t const *const pad);

template<typename inType, typename outType>
detail::Array<outType> fft(const detail::Array<inType> input,
                           const double norm_factor, const dim_t npad,
                           const dim_t *const pad, const int rank,
                           const bool direction) {
    using fly::dim4;
    using detail::fft_inplace;
    using detail::reshape;
    using detail::scalar;

    dim4 pdims(1);
    computePaddedDims(pdims, input.dims(), npad, pad);
    auto res = reshape(input, pdims, scalar<outType>(0));

    fft_inplace<outType>(res, rank, direction);
    if (norm_factor != 1.0) multiply_inplace(res, norm_factor);

    return res;
}

template<typename inType, typename outType>
detail::Array<outType> fft_r2c(const detail::Array<inType> input,
                               const double norm_factor, const dim_t npad,
                               const dim_t *const pad, const int rank) {
    using fly::dim4;
    using detail::Array;
    using detail::fft_r2c;
    using detail::multiply_inplace;
    using detail::reshape;
    using detail::scalar;

    const dim4 &idims = input.dims();

    bool is_pad = false;
    for (int i = 0; i < npad; i++) { is_pad |= (pad[i] != idims[i]); }

    Array<inType> tmp = input;

    if (is_pad) {
        dim4 pdims(1);
        computePaddedDims(pdims, input.dims(), npad, pad);
        tmp = reshape(input, pdims, scalar<inType>(0));
    }

    auto res = fft_r2c<outType, inType>(tmp, rank);
    if (norm_factor != 1.0) multiply_inplace(res, norm_factor);

    return res;
}

template<typename inType, typename outType>
detail::Array<outType> fft_c2r(const detail::Array<inType> input,
                               const double norm_factor, const fly::dim4 &odims,
                               const int rank) {
    using detail::Array;
    using detail::fft_c2r;
    using detail::multiply_inplace;

    Array<outType> output = fft_c2r<outType, inType>(input, odims, rank);

    if (norm_factor != 1) {
        // Normalize input because tmp was not normalized
        multiply_inplace(output, norm_factor);
    }

    return output;
}
