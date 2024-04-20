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
namespace common {

class SparseArrayBase;
template<typename T>
class SparseArray;

////////////////////////////////////////////////////////////////////////////
// Friend functions for Sparse Array Creation
////////////////////////////////////////////////////////////////////////////
template<typename T>
SparseArray<T> createEmptySparseArray(const fly::dim4 &_dims, dim_t _nNZ,
                                      const fly::storage _storage);

template<typename T>
SparseArray<T> createHostDataSparseArray(const fly::dim4 &_dims, const dim_t nNZ,
                                         const T *const _values,
                                         const int *const _rowIdx,
                                         const int *const _colIdx,
                                         const fly::storage _storage);

template<typename T>
SparseArray<T> createDeviceDataSparseArray(const fly::dim4 &_dims,
                                           const dim_t nNZ, T *const _values,
                                           int *const _rowIdx,
                                           int *const _colIdx,
                                           const fly::storage _storage,
                                           const bool _copy = false);

template<typename T>
SparseArray<T> createArrayDataSparseArray(const fly::dim4 &_dims,
                                          const detail::Array<T> &_values,
                                          const detail::Array<int> &_rowIdx,
                                          const detail::Array<int> &_colIdx,
                                          const fly::storage _storage,
                                          const bool _copy = false);

template<typename T>
SparseArray<T> *initSparseArray();

template<typename T>
void destroySparseArray(SparseArray<T> *sparse);

/// Performs a deep copy of the \p input array.
///
/// \param[in] other    The sparse array that is to be copied
/// \returns A deep copy of the input sparse array
template<typename T>
SparseArray<T> copySparseArray(const SparseArray<T> &other);

}  // namespace common
}  // namespace flare
