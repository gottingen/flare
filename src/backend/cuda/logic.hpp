/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <common/jit/BinaryNode.hpp>
#include <fly/dim4.hpp>

namespace flare {
namespace cuda {
template<typename T, fly_op_t op>
Array<char> logicOp(const Array<T> &lhs, const Array<T> &rhs,
                    const fly::dim4 &odims) {
    return common::createBinaryNode<char, T, op>(lhs, rhs, odims);
}

template<typename T, fly_op_t op>
Array<T> bitOp(const Array<T> &lhs, const Array<T> &rhs,
               const fly::dim4 &odims) {
    return common::createBinaryNode<T, T, op>(lhs, rhs, odims);
}
}  // namespace cuda
}  // namespace flare
