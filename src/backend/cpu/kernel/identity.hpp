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

#pragma once
#include <Param.hpp>
#include <math.hpp>

namespace flare {
namespace cpu {
namespace kernel {

template<typename T>
void identity(Param<T> out) {
    T *ptr                  = out.get();
    const fly::dim4 out_dims = out.dims();

    for (dim_t k = 0; k < out_dims[2] * out_dims[3]; k++) {
        for (dim_t j = 0; j < out_dims[1]; j++) {
            for (dim_t i = 0; i < out_dims[0]; i++) {
                ptr[j * out_dims[0] + i] =
                    (i == j) ? scalar<T>(1) : scalar<T>(0);
            }
        }
        ptr += out_dims[0] * out_dims[1];
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace flare
