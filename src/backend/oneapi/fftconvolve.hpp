/*******************************************************
 * Copyright (c) 2022, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>

namespace flare {
namespace oneapi {
template<typename T>
Array<T> fftconvolve(Array<T> const& signal, Array<T> const& filter,
                     const bool expand, FLY_BATCH_KIND kind, const int rank);
}  // namespace oneapi
}  // namespace flare
