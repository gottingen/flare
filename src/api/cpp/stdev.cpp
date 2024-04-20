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

namespace fly {

#define INSTANTIATE_STDEV(T)                                       \
    template<>                                                     \
    FLY_API T stdev(const array& in, const fly_var_bias bias) {       \
        double ret_val;                                            \
        FLY_THROW(fly_stdev_all_v2(&ret_val, NULL, in.get(), bias)); \
        return (T)ret_val;                                         \
    }                                                              \
    template<>                                                     \
    FLY_API T stdev(const array& in) {                               \
        return stdev<T>(in, FLY_VARIANCE_POPULATION);               \
    }

template<>
FLY_API fly_cfloat stdev(const array& in, const fly_var_bias bias) {
    double real, imag;
    FLY_THROW(fly_stdev_all_v2(&real, &imag, in.get(), bias));
    return {static_cast<float>(real), static_cast<float>(imag)};
}

template<>
FLY_API fly_cdouble stdev(const array& in, const fly_var_bias bias) {
    double real, imag;
    FLY_THROW(fly_stdev_all_v2(&real, &imag, in.get(), bias));
    return {real, imag};
}

template<>
FLY_API fly_cfloat stdev(const array& in) {
    return stdev<fly_cfloat>(in, FLY_VARIANCE_POPULATION);
}

template<>
FLY_API fly_cdouble stdev(const array& in) {
    return stdev<fly_cdouble>(in, FLY_VARIANCE_POPULATION);
}

INSTANTIATE_STDEV(float);
INSTANTIATE_STDEV(double);
INSTANTIATE_STDEV(int);
INSTANTIATE_STDEV(unsigned int);
INSTANTIATE_STDEV(long long);
INSTANTIATE_STDEV(unsigned long long);
INSTANTIATE_STDEV(short);
INSTANTIATE_STDEV(unsigned short);
INSTANTIATE_STDEV(char);
INSTANTIATE_STDEV(unsigned char);

#undef INSTANTIATE_STDEV

array stdev(const array& in, const fly_var_bias bias, const dim_t dim) {
    fly_array temp = 0;
    FLY_THROW(fly_stdev_v2(&temp, in.get(), bias, getFNSD(dim, in.dims())));
    return array(temp);
}

array stdev(const array& in, const dim_t dim) {
    return stdev(in, FLY_VARIANCE_POPULATION, dim);
}

}  // namespace fly
