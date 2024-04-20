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
#include <common/jit/BufferNodeBase.hpp>
#include "../Param.hpp"

namespace flare {
namespace cuda {
namespace jit {
template<typename T>
using BufferNode = common::BufferNodeBase<std::shared_ptr<T>, Param<T>>;
}  // namespace jit
}  // namespace cuda

namespace common {

template<typename DataType, typename ParamType>
bool BufferNodeBase<DataType, ParamType>::operator==(
    const BufferNodeBase<DataType, ParamType> &other) const noexcept {
    // clang-format off
    return m_data.get() == other.m_data.get() &&
           m_bytes == other.m_bytes &&
           m_param.ptr == other.m_param.ptr;
    // clang-format on
}

}  // namespace common

}  // namespace flare
