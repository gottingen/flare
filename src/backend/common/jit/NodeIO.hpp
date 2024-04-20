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
#include <common/jit/Node.hpp>
#include <collie/strings/format.h>

#include <common/util.hpp>

template<>
struct fmt::formatter<fly::dtype> : fmt::formatter<char> {
    template<typename FormatContext>
    auto format(const fly::dtype& p, FormatContext& ctx) -> decltype(ctx.out()) {
        format_to(ctx.out(), "{}", flare::common::getName(p));
        return ctx.out();
    }
};

template<>
struct fmt::formatter<flare::common::Node> {
    // Presentation format: 'p' - pointer, 't' - type.
    // char presentation;
    bool pointer;
    bool type;
    bool children;
    bool op;

    // Parses format specifications of the form ['f' | 'e'].
    constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        auto it = ctx.begin(), end = ctx.end();

        if (it == end || *it == '}') {
            pointer = type = children = op = true;
            return it;
        }

        while (it != end && *it != '}') {
            switch (*it) {
                case 'p': pointer = true; break;
                case 't': type = true; break;
                case 'c': children = true; break;
                case 'o': op = true; break;
                default: throw format_error("invalid format");
            }
            ++it;
        }

        // Return an iterator past the end of the parsed range:
        return it;
    }

    // Formats the point p using the parsed format specification (presentation)
    // stored in this formatter.
    template<typename FormatContext>
    auto format(const flare::common::Node& node, FormatContext& ctx)
        -> decltype(ctx.out()) {
        // ctx.out() is an output iterator to write to.

        format_to(ctx.out(), "{{");
        if (pointer) format_to(ctx.out(), "{} ", (void*)&node);
        if (op) {
            if (isBuffer(node)) {
                format_to(ctx.out(), "buffer ");
            } else if (isScalar(node)) {
                format_to(ctx.out(), "scalar ",
                          flare::common::toString(node.getOp()));
            } else {
                format_to(ctx.out(), "{} ",
                          flare::common::toString(node.getOp()));
            }
        }
        if (type) format_to(ctx.out(), "{} ", node.getType());
        if (children) {
            int count;
            for (count = 0; count < flare::common::Node::kMaxChildren &&
                            node.m_children[count].get() != nullptr;
                 count++) {}
            if (count > 0) {
                format_to(ctx.out(), "children: {{ ");
                for (int i = 0; i < count; i++) {
                    format_to(ctx.out(), "{} ", *(node.m_children[i].get()));
                }
                format_to(ctx.out(), "\b}} ");
            }
        }
        format_to(ctx.out(), "\b}}");

        return ctx.out();
    }
};
