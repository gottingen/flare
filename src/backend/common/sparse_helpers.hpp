/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

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
