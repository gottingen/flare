/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <kernel/range.hpp>
#include <range.hpp>

#include <Array.hpp>
#include <common/half.hpp>
#include <err_opencl.hpp>
#include <math.hpp>
#include <stdexcept>

using flare::common::half;

namespace flare {
namespace opencl {
template<typename T>
Array<T> range(const dim4& dim, const int seq_dim) {
    // Set dimension along which the sequence should be
    // Other dimensions are simply tiled
    int _seq_dim = seq_dim;
    if (seq_dim < 0) {
        _seq_dim = 0;  // column wise sequence
    }

    if (_seq_dim < 0 || _seq_dim > 3) {
        FLY_ERROR("Invalid rep selection", FLY_ERR_ARG);
    }

    Array<T> out = createEmptyArray<T>(dim);
    kernel::range<T>(out, _seq_dim);

    return out;
}

#define INSTANTIATE(T) \
    template Array<T> range<T>(const fly::dim4& dims, const int seq_dims);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(half)
}  // namespace opencl
}  // namespace flare
