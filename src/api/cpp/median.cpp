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
#include <fly/statistics.h>
#include "common.hpp"
#include "error.hpp"

namespace fly {

#define INSTANTIATE_MEDIAN(T)                              \
    template<>                                             \
    FLY_API T median(const array& in) {                      \
        double ret_val;                                    \
        FLY_THROW(fly_median_all(&ret_val, NULL, in.get())); \
        return (T)ret_val;                                 \
    }

INSTANTIATE_MEDIAN(float);
INSTANTIATE_MEDIAN(double);
INSTANTIATE_MEDIAN(int);
INSTANTIATE_MEDIAN(unsigned int);
INSTANTIATE_MEDIAN(char);
INSTANTIATE_MEDIAN(unsigned char);
INSTANTIATE_MEDIAN(long long);
INSTANTIATE_MEDIAN(unsigned long long);
INSTANTIATE_MEDIAN(short);
INSTANTIATE_MEDIAN(unsigned short);

#undef INSTANTIATE_MEDIAN

array median(const array& in, const dim_t dim) {
    fly_array temp = 0;
    FLY_THROW(fly_median(&temp, in.get(), getFNSD(dim, in.dims())));
    return array(temp);
}

}  // namespace fly
