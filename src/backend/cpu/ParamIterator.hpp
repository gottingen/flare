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
#include <Param.hpp>
#include <fly/dim4.hpp>

#include <array>
#include <cstddef>
#include <iterator>
#include <vector>

namespace flare {
namespace cpu {

/// A Param iterator that iterates through a Param object
template<typename T>
class ParamIterator {
   public:
    using difference_type   = ptrdiff_t;
    using value_type        = T;
    using pointer           = T*;
    using reference         = T&;
    using const_reference   = const T&;
    using iterator_category = std::forward_iterator_tag;

    /// Creates a sentinel iterator. This is equivalent to the end iterator
    ParamIterator() noexcept
        : ptr(nullptr)
        , dims(1)
        , stride(1)
        , dim_index{dims[0], dims[1], dims[2], dims[3]} {}

    /// ParamIterator Constructor
    ParamIterator(cpu::Param<T>& in) noexcept
        : ptr(in.get())
        , dims(in.dims())
        , stride(calcIteratorStrides(dims, in.strides()))
        , dim_index{in.dims()[0], in.dims()[1], in.dims()[2], in.dims()[3]} {}

    ParamIterator(cpu::CParam<typename std::remove_const<T>::type>& in) noexcept
        : ptr(const_cast<pointer>(in.get()))
        , dims(in.dims())
        , stride(calcIteratorStrides(dims, in.strides()))
        , dim_index{in.dims()[0], in.dims()[1], in.dims()[2], in.dims()[3]} {}

    /// The equality operator
    bool operator==(const ParamIterator& other) const noexcept {
        return ptr == other.ptr;
    }

    /// The inequality operator
    bool operator!=(const ParamIterator& other) const noexcept {
        return ptr != other.ptr;
    }

    /// Advances the iterator, pre increment operator
    ParamIterator& operator++() noexcept {
        for (int i = 0; i < FLY_MAX_DIMS; i++) {
            dim_index[i]--;
            ptr += stride[i];
            if (dim_index[i]) { return *this; }
            dim_index[i] = dims[i];
        }
        ptr = nullptr;
        return *this;
    }

    /// Advances the iterator by count elements
    ParamIterator& operator+=(std::size_t count) noexcept {
        while (count-- > 0) { operator++(); }
        return *this;
    }

    reference operator*() noexcept { return *ptr; }

    const_reference operator*() const noexcept { return *ptr; }

    const pointer operator->() const noexcept { return ptr; }

    ParamIterator(const ParamIterator<T>& other) = default;
    ParamIterator(ParamIterator<T>&& other)      = default;
    ~ParamIterator() noexcept                    = default;
    ParamIterator<T>& operator=(const ParamIterator<T>& other) noexcept =
        default;
    ParamIterator<T>& operator=(ParamIterator<T>&& other) noexcept = default;

   private:
    T* ptr;

    // The dimension of the array
    const fly::dim4 dims;

    // The iterator's stride
    const fly::dim4 stride;

    // NOTE: This is not really the true coordinate of the iteration. It's
    // values will go down as you move through the array.
    std::array<dim_t, FLY_MAX_DIMS> dim_index;

    /// Calculates the iterator offsets.
    ///
    /// These are different from the original offsets because they define
    /// the stride from the end of the last element in the previous dimension
    /// to the first element on the next dimension.
    static dim4 calcIteratorStrides(const dim4& dims,
                                    const dim4& stride) noexcept {
        return dim4(stride[0], stride[1] - (stride[0] * dims[0]),
                    stride[2] - (stride[1] * dims[1]),
                    stride[3] - (stride[2] * dims[2]));
    }
};

template<typename T>
ParamIterator<T> begin(Param<T>& param) {
    return ParamIterator<T>(param);
}

template<typename T>
ParamIterator<T> end(Param<T>& param) {
    return ParamIterator<T>();
}

template<typename T>
ParamIterator<const T> begin(CParam<T>& param) {
    return ParamIterator<const T>(param);
}

template<typename T>
ParamIterator<const T> end(CParam<T>& param) {
    return ParamIterator<const T>();
}

}  // namespace cpu
}  // namespace flare
