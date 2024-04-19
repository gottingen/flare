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
#include <common/SparseArray.hpp>
#include <optypes.hpp>
#include <sparse.hpp>

namespace flare {
namespace cpu {
// These two functions cannot be overloaded by return type.
// So have to give them separate names.
template<typename T, fly_op_t op>
Array<T> arithOpD(const common::SparseArray<T> &lhs, const Array<T> &rhs,
                  const bool reverse = false);

template<typename T, fly_op_t op>
common::SparseArray<T> arithOp(const common::SparseArray<T> &lhs,
                               const Array<T> &rhs, const bool reverse = false);

template<typename T, fly_op_t op>
common::SparseArray<T> arithOp(const common::SparseArray<T> &lhs,
                               const common::SparseArray<T> &rhs);
}  // namespace cpu
}  // namespace flare
