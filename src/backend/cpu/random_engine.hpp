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
#include <backend.hpp>
#include <fly/defines.h>

namespace flare {
namespace cpu {
void initMersenneState(Array<uint> &state, const uintl seed,
                       const Array<uint> &tbl);

template<typename T>
Array<T> uniformDistribution(const fly::dim4 &dims,
                             const fly_random_engine_type type,
                             const unsigned long long seed,
                             unsigned long long &counter);

template<typename T>
Array<T> normalDistribution(const fly::dim4 &dims,
                            const fly_random_engine_type type,
                            const unsigned long long seed,
                            unsigned long long &counter);

template<typename T>
Array<T> uniformDistribution(const fly::dim4 &dims, Array<uint> pos,
                             Array<uint> sh1, Array<uint> sh2, uint mask,
                             Array<uint> recursion_table,
                             Array<uint> temper_table, Array<uint> state);

template<typename T>
Array<T> normalDistribution(const fly::dim4 &dims, Array<uint> pos,
                            Array<uint> sh1, Array<uint> sh2, uint mask,
                            Array<uint> recursion_table,
                            Array<uint> temper_table, Array<uint> state);
}  // namespace cpu
}  // namespace flare
