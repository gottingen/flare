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

namespace flare {
namespace cpu {
template<typename T, fly_storage stype>
common::SparseArray<T> sparseConvertDenseToStorage(const Array<T> &in);

template<typename T, fly_storage stype>
Array<T> sparseConvertStorageToDense(const common::SparseArray<T> &in);

template<typename T, fly_storage dest, fly_storage src>
common::SparseArray<T> sparseConvertStorageToStorage(
    const common::SparseArray<T> &in);
}  // namespace cpu
}  // namespace flare
