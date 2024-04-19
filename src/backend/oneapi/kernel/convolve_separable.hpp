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

namespace flare {
namespace oneapi {
namespace kernel {

// below shared MAX_*_LEN's are calculated based on
// a maximum shared memory configuration of 48KB per block
// considering complex types as well
constexpr int MAX_SCONV_FILTER_LEN = 31;

template<typename T, typename accT>
void convSep(Param<T> out, const Param<T> sig, const Param<accT> filt,
             const int cDim, const bool expand);

}  // namespace kernel
}  // namespace oneapi
}  // namespace flare
