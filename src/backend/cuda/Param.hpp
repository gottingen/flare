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

#include <backend.hpp>
#include <types.hpp>
#include <fly/defines.h>

namespace flare {
namespace cuda {

template<typename T>
class Param {
   public:
    dim_t dims[4];
    dim_t strides[4];
    T *ptr;

    __DH__ Param() noexcept : dims(), strides(), ptr(nullptr) {}

    __DH__
    Param(T *iptr, const dim_t *idims, const dim_t *istrides) noexcept
        : dims{idims[0], idims[1], idims[2], idims[3]}
        , strides{istrides[0], istrides[1], istrides[2], istrides[3]}
        , ptr(iptr) {}

    __DH__ size_t elements() const noexcept {
        return dims[0] * dims[1] * dims[2] * dims[3];
    }

    dim_t *dims_ptr() { return dims; }
    dim_t *strides_ptr() { return strides; }

    Param(const Param<T> &other) noexcept               = default;
    Param(Param<T> &&other) noexcept                    = default;
    Param<T> &operator=(const Param<T> &other) noexcept = default;
    Param<T> &operator=(Param<T> &&other) noexcept      = default;
};

template<typename T>
Param<T> flat(Param<T> in) {
    in.dims[0] = in.elements();
    in.dims[1] = 1;
    in.dims[2] = 1;
    in.dims[3] = 1;
    return in;
}

template<typename T>
class CParam {
   public:
    dim_t dims[4];
    dim_t strides[4];
    const T *ptr;

    __DH__ CParam(const T *iptr, const dim_t *idims, const dim_t *istrides)
        : dims{idims[0], idims[1], idims[2], idims[3]}
        , strides{istrides[0], istrides[1], istrides[2], istrides[3]}
        , ptr(iptr) {}

    __DH__ CParam(Param<T> &in)
        : dims{in.dims[0], in.dims[1], in.dims[2], in.dims[3]}
        , strides{in.strides[0], in.strides[1], in.strides[2], in.strides[3]}
        , ptr(in.ptr) {}

    __DH__ size_t elements() const noexcept {
        return dims[0] * dims[1] * dims[2] * dims[3];
    }

    CParam(const CParam<T> &other) noexcept               = default;
    CParam(CParam<T> &&other) noexcept                    = default;
    CParam<T> &operator=(const CParam<T> &other) noexcept = default;
    CParam<T> &operator=(CParam<T> &&other) noexcept      = default;
};

}  // namespace cuda
}  // namespace flare
