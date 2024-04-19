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

#include <fly/algorithm.h>
#include <fly/array.h>
#include <fly/dim4.hpp>
#include <fly/statistics.h>
#include "common.hpp"
#include "error.hpp"
#include <fly/half.hpp>
#ifdef FLY_CUDA
#include <cuda_fp16.h>
#include <traits.hpp>
#endif

namespace fly {

array mean(const array& in, const dim_t dim) {
    fly_array temp = 0;
    FLY_THROW(fly_mean(&temp, in.get(), getFNSD(dim, in.dims())));
    return array(temp);
}

array mean(const array& in, const array& weights, const dim_t dim) {
    fly_array temp = 0;
    FLY_THROW(fly_mean_weighted(&temp, in.get(), weights.get(),
                              getFNSD(dim, in.dims())));
    return array(temp);
}

#define INSTANTIATE_MEAN(T)                                                  \
    template<>                                                               \
    FLY_API T mean(const array& in) {                                          \
        double ret_val;                                                      \
        FLY_THROW(fly_mean_all(&ret_val, NULL, in.get()));                     \
        return cast<T>(ret_val);                                             \
    }                                                                        \
    template<>                                                               \
    FLY_API T mean(const array& in, const array& wts) {                        \
        double ret_val;                                                      \
        FLY_THROW(fly_mean_all_weighted(&ret_val, NULL, in.get(), wts.get())); \
        return cast<T>(ret_val);                                             \
    }

template<>
FLY_API fly_cfloat mean(const array& in) {
    double real, imag;
    FLY_THROW(fly_mean_all(&real, &imag, in.get()));
    return {static_cast<float>(real), static_cast<float>(imag)};
}

template<>
FLY_API fly_cdouble mean(const array& in) {
    double real, imag;
    FLY_THROW(fly_mean_all(&real, &imag, in.get()));
    return {real, imag};
}

template<>
FLY_API fly_cfloat mean(const array& in, const array& weights) {
    double real, imag;
    FLY_THROW(fly_mean_all_weighted(&real, &imag, in.get(), weights.get()));
    return {static_cast<float>(real), static_cast<float>(imag)};
}

template<>
FLY_API fly_cdouble mean(const array& in, const array& weights) {
    double real, imag;
    FLY_THROW(fly_mean_all_weighted(&real, &imag, in.get(), weights.get()));
    return {real, imag};
}

INSTANTIATE_MEAN(float);
INSTANTIATE_MEAN(double);
INSTANTIATE_MEAN(int);
INSTANTIATE_MEAN(unsigned int);
INSTANTIATE_MEAN(char);
INSTANTIATE_MEAN(unsigned char);
INSTANTIATE_MEAN(long long);
INSTANTIATE_MEAN(unsigned long long);
INSTANTIATE_MEAN(short);
INSTANTIATE_MEAN(unsigned short);
INSTANTIATE_MEAN(fly_half);
INSTANTIATE_MEAN(half_float::half);  // Add support for public API
#ifdef FLY_CUDA
INSTANTIATE_MEAN(__half);
#endif

#undef INSTANTIATE_MEAN

}  // namespace fly
