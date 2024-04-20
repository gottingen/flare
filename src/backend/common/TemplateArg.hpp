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

#include <common/util.hpp>

#include <array>
#include <string>
#include <utility>

template<typename T>
struct TemplateTypename;

struct TemplateArg {
    std::string _tparam;

    TemplateArg(std::string str) : _tparam(std::move(str)) {}

    template<typename T>
    constexpr TemplateArg(TemplateTypename<T> arg) noexcept : _tparam(arg) {}

    template<typename T>
    constexpr TemplateArg(T value) noexcept
        : _tparam(flare::common::toString(value)) {}
};

template<typename... Targs>
std::array<TemplateArg, sizeof...(Targs)> TemplateArgs(Targs &&...args) {
    return std::array<TemplateArg, sizeof...(Targs)>{
        std::forward<Targs>(args)...};
}

#define DefineKey(arg) " -D " #arg
#define DefineValue(arg) " -D " #arg "=" + flare::common::toString(arg)
#define DefineKeyValue(key, arg) \
    " -D " #key "=" + flare::common::toString(arg)
#define DefineKeyFromStr(arg) " -D " + std::string(arg)
