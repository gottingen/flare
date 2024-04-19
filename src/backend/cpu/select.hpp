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
void select(Array<T> &out, const Array<char> &cond, const Array<T> &a,
            const Array<T> &b);

template<typename T, bool flip>
void select_scalar(Array<T> &out, const Array<char> &cond, const Array<T> &a,
                   const T &b);

template<typename T>
Array<T> createSelectNode(const Array<char> &cond, const Array<T> &a,
                          const Array<T> &b, const fly::dim4 &odims) {
    Array<T> out = createEmptyArray<T>(odims);
    select(out, cond, a, b);
    return out;
}

template<typename T, bool flip>
Array<T> createSelectNode(const Array<char> &cond, const Array<T> &a,
                          const T &b, const fly::dim4 &odims) {
    Array<T> out = createEmptyArray<T>(odims);
    select_scalar<T, flip>(out, cond, a, b);
    return out;
}
}  // namespace cpu
}  // namespace flare
