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
#include <fly/statistics.h>
#include "common.hpp"
#include "error.hpp"
#include <fly/half.hpp>
#ifdef FLY_CUDA
#include <cuda_fp16.h>
#include <traits.hpp>
#endif

namespace fly {

array var(const array& in, const bool isbiased, const dim_t dim) {
    const fly_var_bias bias =
        (isbiased ? FLY_VARIANCE_SAMPLE : FLY_VARIANCE_POPULATION);
    return var(in, bias, dim);
}

array var(const array& in, const fly_var_bias bias, const dim_t dim) {
    fly_array temp = 0;
    FLY_THROW(fly_var_v2(&temp, in.get(), bias, getFNSD(dim, in.dims())));
    return array(temp);
}

array var(const array& in, const array& weights, const dim_t dim) {
    fly_array temp = 0;
    FLY_THROW(fly_var_weighted(&temp, in.get(), weights.get(),
                             getFNSD(dim, in.dims())));
    return array(temp);
}

#define INSTANTIATE_VAR(T)                                                 \
    template<>                                                             \
    FLY_API T var(const array& in, const fly_var_bias bias) {                 \
        double ret_val;                                                    \
        FLY_THROW(fly_var_all_v2(&ret_val, NULL, in.get(), bias));           \
        return cast<T>(ret_val);                                           \
    }                                                                      \
    template<>                                                             \
    FLY_API T var(const array& in, const bool isbiased) {                    \
        const fly_var_bias bias =                                           \
            (isbiased ? FLY_VARIANCE_SAMPLE : FLY_VARIANCE_POPULATION);      \
        return var<T>(in, bias);                                           \
    }                                                                      \
                                                                           \
    template<>                                                             \
    FLY_API T var(const array& in, const array& weights) {                   \
        double ret_val;                                                    \
        FLY_THROW(                                                          \
            fly_var_all_weighted(&ret_val, NULL, in.get(), weights.get())); \
        return cast<T>(ret_val);                                           \
    }

template<>
FLY_API fly_cfloat var(const array& in, const fly_var_bias bias) {
    double real, imag;
    FLY_THROW(fly_var_all_v2(&real, &imag, in.get(), bias));
    return {static_cast<float>(real), static_cast<float>(imag)};
}

template<>
FLY_API fly_cdouble var(const array& in, const fly_var_bias bias) {
    double real, imag;
    FLY_THROW(fly_var_all_v2(&real, &imag, in.get(), bias));
    return {real, imag};
}

template<>
FLY_API fly_cfloat var(const array& in, const bool isbiased) {
    const fly_var_bias bias =
        (isbiased ? FLY_VARIANCE_SAMPLE : FLY_VARIANCE_POPULATION);
    return var<fly_cfloat>(in, bias);
}

template<>
FLY_API fly_cdouble var(const array& in, const bool isbiased) {
    const fly_var_bias bias =
        (isbiased ? FLY_VARIANCE_SAMPLE : FLY_VARIANCE_POPULATION);
    return var<fly_cdouble>(in, bias);
}

template<>
FLY_API fly_cfloat var(const array& in, const array& weights) {
    double real, imag;
    FLY_THROW(fly_var_all_weighted(&real, &imag, in.get(), weights.get()));
    return {static_cast<float>(real), static_cast<float>(imag)};
}

template<>
FLY_API fly_cdouble var(const array& in, const array& weights) {
    double real, imag;
    FLY_THROW(fly_var_all_weighted(&real, &imag, in.get(), weights.get()));
    return {real, imag};
}

INSTANTIATE_VAR(float);
INSTANTIATE_VAR(double);
INSTANTIATE_VAR(int);
INSTANTIATE_VAR(unsigned int);
INSTANTIATE_VAR(long long);
INSTANTIATE_VAR(unsigned long long);
INSTANTIATE_VAR(short);
INSTANTIATE_VAR(unsigned short);
INSTANTIATE_VAR(char);
INSTANTIATE_VAR(unsigned char);
INSTANTIATE_VAR(fly_half);
INSTANTIATE_VAR(half_float::half);
#ifdef FLY_CUDA
INSTANTIATE_VAR(__half);
#endif

#undef INSTANTIATE_VAR

}  // namespace fly
