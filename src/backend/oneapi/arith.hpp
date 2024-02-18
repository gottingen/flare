/*******************************************************
 * Copyright (c) 2022, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Array.hpp>
#include <common/jit/BinaryNode.hpp>
#include <optypes.hpp>
#include <fly/dim4.hpp>

namespace flare {
namespace oneapi {

template<typename T, fly_op_t op>
Array<T> arithOp(const Array<T> &&lhs, const Array<T> &&rhs,
                 const fly::dim4 &odims) {
    return common::createBinaryNode<T, T, op>(lhs, rhs, odims);
}

template<typename T, fly_op_t op>
Array<T> arithOp(const Array<T> &lhs, const Array<T> &rhs,
                 const fly::dim4 &odims) {
    return common::createBinaryNode<T, T, op>(lhs, rhs, odims);
}
}  // namespace oneapi
}  // namespace flare
