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
#include <fly/compatible.h>
#include <fly/dim4.hpp>
#include <fly/image.h>
#include "error.hpp"

namespace fly {
array gaussianKernel(const int rows, const int cols, const double sig_r,
                     const double sig_c) {
    fly_array res;
    FLY_THROW(fly_gaussian_kernel(&res, rows, cols, sig_r, sig_c));
    return array(res);
}

// Compatible function
array gaussiankernel(const int rows, const int cols, const double sig_r,
                     const double sig_c) {
    return gaussianKernel(rows, cols, sig_r, sig_c);
}

}  // namespace fly
