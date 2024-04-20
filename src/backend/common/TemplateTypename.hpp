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

#include <common/TemplateArg.hpp>
#include <traits.hpp>

#include <string>

template<typename T>
struct TemplateTypename {
    operator TemplateArg() const noexcept {
        return {std::string(fly::dtype_traits<T>::getName())};
    }
    operator std::string() const noexcept {
        return {std::string(fly::dtype_traits<T>::getName())};
    }
};

#define SPECIALIZE(TYPE, NAME)                                  \
    template<>                                                  \
    struct TemplateTypename<TYPE> {                             \
        operator TemplateArg() const noexcept {                 \
            return TemplateArg(std::string(#NAME));             \
        }                                                       \
        operator std::string() const noexcept { return #NAME; } \
    }

SPECIALIZE(unsigned char, detail::uchar);
SPECIALIZE(unsigned int, detail::uint);
SPECIALIZE(unsigned short, detail::ushort);
SPECIALIZE(long long, long long);
SPECIALIZE(unsigned long long, unsigned long long);

#undef SPECIALIZE
