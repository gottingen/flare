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

#if defined(FLY_USE_NEW_CUSPARSE_API)
// CUDA Toolkit 10.0 or later

#include <common/unique_handle.hpp>
#include <cudaDataType.hpp>
#include <cusparse.hpp>

#include <utility>

namespace flare {
namespace cuda {

template<typename T>
auto cusparseDescriptor(const common::SparseArray<T> &in) {
    auto dims = in.dims();

    return common::make_handle<cusparseSpMatDescr_t>(in);
}

template<typename T>
auto denVecDescriptor(const Array<T> &in) {
    return common::make_handle<cusparseDnVecDescr_t>(
        in.elements(), (void *)(in.get()), getType<T>());
}

template<typename T>
auto denMatDescriptor(const Array<T> &in) {
    auto dims    = in.dims();
    auto strides = in.strides();
    return common::make_handle<cusparseDnMatDescr_t>(
        dims[0], dims[1], strides[1], (void *)in.get(), getType<T>(),
        CUSPARSE_ORDER_COL);
}

}  // namespace cuda
}  // namespace flare

#endif
