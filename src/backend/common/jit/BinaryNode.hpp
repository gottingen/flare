/*******************************************************
 * Copyright (c) 2018, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/jit/NaryNode.hpp>

#include <cmath>

namespace flare {
namespace common {
class BinaryNode : public NaryNode {
   public:
    BinaryNode(const fly::dtype type, const char *op_str, common::Node_ptr lhs,
               common::Node_ptr rhs, fly_op_t op)
        : NaryNode(type, op_str, 2, {{lhs, rhs}}, op,
                   std::max(lhs->getHeight(), rhs->getHeight()) + 1) {}
};

template<typename To, typename Ti, fly_op_t op>
detail::Array<To> createBinaryNode(const detail::Array<Ti> &lhs,
                                   const detail::Array<Ti> &rhs,
                                   const fly::dim4 &odims);

}  // namespace common
}  // namespace flare
