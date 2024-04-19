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
#include <common/defines.hpp>
#include <optypes.hpp>

template<fly_op_t T>
static const char *binOpName() {
    return "ADD_OP";
}

template<>
inline const char *binOpName<fly_add_t>() {
    return "ADD_OP";
}
template<>
inline const char *binOpName<fly_mul_t>() {
    return "MUL_OP";
}
template<>
inline const char *binOpName<fly_and_t>() {
    return "AND_OP";
}
template<>
inline const char *binOpName<fly_or_t>() {
    return "OR_OP";
}
template<>
inline const char *binOpName<fly_min_t>() {
    return "MIN_OP";
}
template<>
inline const char *binOpName<fly_max_t>() {
    return "MAX_OP";
}
template<>
inline const char *binOpName<fly_notzero_t>() {
    return "NOTZERO_OP";
}
