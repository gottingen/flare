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
#include <fly/defines.h>
#include <fly/dim4.hpp>

namespace flare {
namespace cpu {

/// \brief Constant parameter object who's memory cannot be modified. Params
///        represent the view of the memory in the kernel object. They do not
///        own the memory.
template<typename T>
class CParam {
   private:
    const T *m_ptr;
    fly::dim4 m_dims;
    fly::dim4 m_strides;

   public:
    CParam(const T *iptr, const fly::dim4 &idims,
           const fly::dim4 &istrides) noexcept
        : m_ptr(iptr) {
        for (int i = 0; i < 4; i++) {
            m_dims[i]    = idims[i];
            m_strides[i] = istrides[i];
        }
    }

    /// \brief returns the pointer to the memory
    constexpr const T *get() const noexcept { return m_ptr; }

    /// Gets the shape/dimension of the memory
    fly::dim4 dims() const noexcept { return m_dims; }

    /// Gets the stride of the memory
    fly::dim4 strides() const noexcept { return m_strides; }

    /// Returns the size of a particular dimension
    ///
    /// \param[in] i The dimension
    constexpr dim_t dims(int i) const noexcept { return m_dims[i]; }

    /// Returns the stride of a particular dimension
    ///
    /// \param[in] i The dimension
    constexpr dim_t strides(int i) const noexcept { return m_strides[i]; }

    constexpr CParam()                                 = delete;
    constexpr CParam(const CParam &other)              = default;
    constexpr CParam(CParam &&other)                   = default;
    CParam<T> &operator=(CParam &&other) noexcept      = default;
    CParam<T> &operator=(const CParam &other) noexcept = default;
    ~CParam()                                          = default;
};

/// \brief Parameter object usually passed into kernels. Params
///        represent the view of the memory in the kernel object. They do not
///        own the memory.
template<typename T>
class Param {
   private:
    T *m_ptr;
    fly::dim4 m_dims;
    fly::dim4 m_strides;

   public:
    /// Creates an empty Param object pointing to null
    Param() noexcept : m_ptr(nullptr) {}

    /// Creates an new Param object given a pointer, dimension and strides
    Param(T *iptr, const fly::dim4 &idims, const fly::dim4 &istrides) noexcept
        : m_ptr(iptr) {
        for (int i = 0; i < 4; i++) {
            m_dims[i]    = idims[i];
            m_strides[i] = istrides[i];
        }
    }

    /// returns the pointer to the object
    T *get() noexcept { return m_ptr; }

    /// Param to CParam implicit conversion operator
    constexpr operator CParam<T>() const noexcept {
        return CParam<T>(const_cast<T *>(m_ptr), m_dims, m_strides);
    }

    /// Gets the shape/dimension of the memory
    fly::dim4 dims() const noexcept { return m_dims; }

    /// Gets the stride of the memory
    fly::dim4 strides() const noexcept { return m_strides; }

    /// Returns the size of a particular dimension
    ///
    /// \param[in] i The dimension
    constexpr dim_t dims(int i) const noexcept { return m_dims[i]; }

    /// Returns the stride of a particular dimension
    ///
    /// \param[in] i The dimension
    constexpr dim_t strides(int i) const noexcept { return m_strides[i]; }

    ~Param()                                         = default;
    constexpr Param(const Param &other)              = default;
    constexpr Param(Param &&other)                   = default;
    Param<T> &operator=(Param &&other) noexcept      = default;
    Param<T> &operator=(const Param &other) noexcept = default;
};

template<typename T>
class Array;

// These functions are needed to convert Array<T> to Param<T> when queueing up
// functions. This is fine becacuse we only have 1 compute queue. This ensures
// there's no race conditions.

/// \brief Converts Array<T> to Param<T> or CParam<T> based on the constness
///        of the Array<T> object. If called on anything else, the object is
///        returned unchanged.
///
/// \param[in] val The value to convert to Param<T>
template<typename T>
const T &toParam(const T &val) noexcept {
    return val;
}

/// \brief Converts Array<T> to Param<T> or CParam<T> based on the constness
///        of the Array<T> object. If called on anything else, the object is
///        returned unchanged.
///
/// \param[in] val The value to convert to Param<T>
template<typename T>
Param<T> toParam(Array<T> &val) noexcept {
    return Param<T>(val.get(), val.dims(), val.strides());
}

/// \brief Converts Array<T> to Param<T> or CParam<T> based on the constness
///        of the Array<T> object. If called on anything else, the object is
///        returned unchanged.
///
/// \param[in] val The value to convert to Param<T>
template<typename T>
CParam<T> toParam(const Array<T> &val) noexcept {
    return CParam<T>(val.get(), val.dims(), val.strides());
}

}  // namespace cpu
}  // namespace flare
