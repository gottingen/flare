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
#include <cast.hpp>
#include <common/Logger.hpp>
#include <memory>

#ifdef FLY_CPU
#include <jit/UnaryNode.hpp>
#endif

namespace flare {
namespace common {
/// This function determines if consecutive cast operations should be
/// removed from a JIT AST.
///
/// This function returns true if consecutive cast operations in the JIT AST
/// should be removed. Multiple cast operations are removed when going from
/// a smaller type to a larger type and back again OR if the conversion is
/// between two floating point types including complex types.
///
///                  Cast operations that will be removed
///                        outer -> inner -> outer
///
///                                inner cast
///           f32  f64  c32  c64  s32  u32   u8   b8  s64  u64  s16  u16  f16
///     f32    x    x    x    x                                            x
///     f64    x    x    x    x                                            x
///  o  c32    x    x    x    x                                            x
///  u  c64    x    x    x    x                                            x
///  t  s32    x    x    x    x    x    x              x    x              x
///  e  u32    x    x    x    x    x    x              x    x              x
///  r   u8    x    x    x    x    x    x    x    x    x    x    x    x    x
///      b8    x    x    x    x    x    x    x    x    x    x    x    x    x
///  c  s64    x    x    x    x                        x    x              x
///  a  u64    x    x    x    x                        x    x              x
///  s  s16    x    x    x    x    x    x              x    x    x    x    x
///  t  u16    x    x    x    x    x    x              x    x    x    x    x
///     f16    x    x    x    x                                            x
///
/// \param[in] outer The type of the second cast and the child of the
///            previous cast
/// \param[in] inner  The type of the first cast
///
/// \returns True if the inner cast operation should be removed
constexpr bool canOptimizeCast(fly::dtype outer, fly::dtype inner) {
    if (isFloating(outer)) {
        if (isFloating(inner)) { return true; }
    } else {
        if (isFloating(inner)) { return true; }
        if (dtypeSize(inner) >= dtypeSize(outer)) { return true; }
    }

    return false;
}

#ifdef FLY_CPU
template<typename To, typename Ti>
struct CastWrapper {
    static clog::logger *getLogger() noexcept {
        static std::shared_ptr<clog::logger> logger =
            common::loggerFactory("ast");
        return logger.get();
    }

    detail::Array<To> operator()(const detail::Array<Ti> &in) {
        using detail::jit::UnaryNode;

        common::Node_ptr in_node = in.getNode();
        constexpr fly::dtype to_dtype =
            static_cast<fly::dtype>(fly::dtype_traits<To>::fly_type);
        constexpr fly::dtype in_dtype =
            static_cast<fly::dtype>(fly::dtype_traits<Ti>::fly_type);

        if (canOptimizeCast(to_dtype, in_dtype)) {
            // JIT optimization in the cast of multiple sequential casts that
            // become idempotent - check to see if the previous operation was
            // also a cast
            // TODO: handle arbitrarily long chains of casts
            auto in_node_unary =
                std::dynamic_pointer_cast<UnaryNode<To, Ti, fly_cast_t>>(
                    in_node);

            if (in_node_unary && in_node_unary->getOp() == fly_cast_t) {
                // child child's output type is the input type of the child
                FLY_TRACE("Cast optimiztion performed by removing cast to {}",
                         fly::dtype_traits<Ti>::getName());
                auto in_child_node = in_node_unary->getChildren()[0];
                if (in_child_node->getType() == to_dtype) {
                    // ignore the input node and simply connect a noop node from
                    // the child's child to produce this op's output
                    return detail::createNodeArray<To>(in.dims(),
                                                       in_child_node);
                }
            }
        }

        auto node = std::make_shared<UnaryNode<To, Ti, fly_cast_t>>(in_node);

        return detail::createNodeArray<To>(in.dims(), move(node));
    }
};
#else

template<typename To, typename Ti>
struct CastWrapper {
    static clog::logger *getLogger() noexcept {
        static std::shared_ptr<clog::logger> logger =
            common::loggerFactory("ast");
        return logger.get();
    }

    detail::Array<To> operator()(const detail::Array<Ti> &in) {
        using flare::common::UnaryNode;
        detail::CastOp<To, Ti> cop;
        common::Node_ptr in_node = in.getNode();
        constexpr fly::dtype to_dtype =
            static_cast<fly::dtype>(fly::dtype_traits<To>::fly_type);
        constexpr fly::dtype in_dtype =
            static_cast<fly::dtype>(fly::dtype_traits<Ti>::fly_type);

        if (canOptimizeCast(to_dtype, in_dtype)) {
            // JIT optimization in the cast of multiple sequential casts that
            // become idempotent - check to see if the previous operation was
            // also a cast
            // TODO: handle arbitrarily long chains of casts
            auto in_node_unary =
                std::dynamic_pointer_cast<common::UnaryNode>(in_node);

            if (in_node_unary && in_node_unary->getOp() == fly_cast_t) {
                // child child's output type is the input type of the child
                FLY_TRACE("Cast optimiztion performed by removing cast to {}",
                         fly::dtype_traits<Ti>::getName());
                auto in_child_node = in_node_unary->getChildren()[0];
                if (in_child_node->getType() == to_dtype) {
                    // ignore the input node and simply connect a noop node from
                    // the child's child to produce this op's output
                    return detail::createNodeArray<To>(in.dims(),
                                                       in_child_node);
                }
            }
        }

        common::UnaryNode *node =
            new common::UnaryNode(to_dtype, cop.name(), in_node, fly_cast_t);
        return detail::createNodeArray<To>(in.dims(), common::Node_ptr(node));
    }
};

#endif

template<typename T>
struct CastWrapper<T, T> {
    detail::Array<T> operator()(const detail::Array<T> &in);
};

template<typename To, typename Ti>
auto cast(detail::Array<Ti> &&in)
    -> std::enable_if_t<std::is_same<Ti, To>::value, detail::Array<To>> {
    return std::move(in);
}

template<typename To, typename Ti>
auto cast(const detail::Array<Ti> &in)
    -> std::enable_if_t<std::is_same<Ti, To>::value, detail::Array<To>> {
    return in;
}

template<typename To, typename Ti>
auto cast(const detail::Array<Ti> &in)
    -> std::enable_if_t<std::is_same<Ti, To>::value == false,
                        detail::Array<To>> {
    CastWrapper<To, Ti> cast_op;
    return cast_op(in);
}

}  // namespace common
}  // namespace flare
