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

#include <Array.hpp>

namespace flare {
namespace cuda {

template<typename T, typename accT>
Array<T> convolve(Array<T> const &signal, Array<accT> const &filter,
                  FLY_BATCH_KIND kind, const int rank, const bool expand);

template<typename T, typename accT>
Array<T> convolve2(Array<T> const &signal, Array<accT> const &c_filter,
                   Array<accT> const &r_filter, const bool expand);

template<typename T>
Array<T> convolve2(Array<T> const &signal, Array<T> const &filter,
                   const dim4 stride, const dim4 padding, const dim4 dilation);

template<typename T>
Array<T> conv2DataGradient(const Array<T> &incoming_gradient,
                           const Array<T> &original_signal,
                           const Array<T> &original_filter,
                           const Array<T> &convolved_output, fly::dim4 stride,
                           fly::dim4 padding, fly::dim4 dilation);

template<typename T>
Array<T> conv2FilterGradient(const Array<T> &incoming_gradient,
                             const Array<T> &original_signal,
                             const Array<T> &original_filter,
                             const Array<T> &convolved_output, fly::dim4 stride,
                             fly::dim4 padding, fly::dim4 dilation);
}  // namespace cuda
}  // namespace flare
