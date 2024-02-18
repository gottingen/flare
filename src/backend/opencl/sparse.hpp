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
#include <common/SparseArray.hpp>

namespace flare {
namespace opencl {

template<typename T, fly_storage stype>
common::SparseArray<T> sparseConvertDenseToStorage(const Array<T> &in);

template<typename T, fly_storage stype>
Array<T> sparseConvertStorageToDense(const common::SparseArray<T> &in);

template<typename T, fly_storage dest, fly_storage src>
common::SparseArray<T> sparseConvertStorageToStorage(
    const common::SparseArray<T> &in);

}  // namespace opencl
}  // namespace flare
