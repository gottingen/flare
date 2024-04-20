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
namespace cpu {
template<typename Ti, typename Tw, typename To>
Array<To> mean(const Array<Ti>& in, const int dim);

template<typename T, typename Tw>
Array<T> mean(const Array<T>& in, const Array<Tw>& wt, const int dim);

template<typename T, typename Tw>
T mean(const Array<T>& in, const Array<Tw>& wts);

template<typename Ti, typename Tw, typename To>
To mean(const Array<Ti>& in);
}  // namespace cpu
}  // namespace flare
