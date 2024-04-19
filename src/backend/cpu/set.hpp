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
#include <Array.hpp>

namespace flare {
namespace cpu {
template<typename T>
Array<T> setUnique(const Array<T> &in, const bool is_sorted);

template<typename T>
Array<T> setUnion(const Array<T> &first, const Array<T> &second,
                  const bool is_unique);

template<typename T>
Array<T> setIntersect(const Array<T> &first, const Array<T> &second,
                      const bool is_unique);
}  // namespace cpu
}  // namespace flare
