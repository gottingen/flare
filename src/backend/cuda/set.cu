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

#include <Array.hpp>
#include <copy.hpp>
#include <debug_cuda.hpp>
#include <set.hpp>
#include <sort.hpp>
#include <thrust_utils.hpp>
#include <fly/dim4.hpp>

#include <thrust/device_ptr.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <algorithm>

namespace flare {
namespace cuda {
using fly::dim4;

template<typename T>
Array<T> setUnique(const Array<T> &in, const bool is_sorted) {
    Array<T> out = copyArray<T>(in);

    thrust::device_ptr<T> out_ptr = thrust::device_pointer_cast<T>(out.get());
    thrust::device_ptr<T> out_ptr_end = out_ptr + out.elements();

    if (!is_sorted) THRUST_SELECT(thrust::sort, out_ptr, out_ptr_end);
    thrust::device_ptr<T> out_ptr_last;
    THRUST_SELECT_OUT(out_ptr_last, thrust::unique, out_ptr, out_ptr_end);

    out.resetDims(dim4(thrust::distance(out_ptr, out_ptr_last)));
    return out;
}

template<typename T>
Array<T> setUnion(const Array<T> &first, const Array<T> &second,
                  const bool is_unique) {
    Array<T> unique_first  = first;
    Array<T> unique_second = second;

    if (!is_unique) {
        unique_first  = setUnique(first, false);
        unique_second = setUnique(second, false);
    }

    dim_t out_size = unique_first.elements() + unique_second.elements();
    Array<T> out   = createEmptyArray<T>(dim4(out_size));

    thrust::device_ptr<T> first_ptr =
        thrust::device_pointer_cast<T>(unique_first.get());
    thrust::device_ptr<T> first_ptr_end = first_ptr + unique_first.elements();

    thrust::device_ptr<T> second_ptr =
        thrust::device_pointer_cast<T>(unique_second.get());
    thrust::device_ptr<T> second_ptr_end =
        second_ptr + unique_second.elements();

    thrust::device_ptr<T> out_ptr = thrust::device_pointer_cast<T>(out.get());

    thrust::device_ptr<T> out_ptr_last;
    THRUST_SELECT_OUT(out_ptr_last, thrust::set_union, first_ptr, first_ptr_end,
                      second_ptr, second_ptr_end, out_ptr);

    out.resetDims(dim4(thrust::distance(out_ptr, out_ptr_last)));

    return out;
}

template<typename T>
Array<T> setIntersect(const Array<T> &first, const Array<T> &second,
                      const bool is_unique) {
    Array<T> unique_first  = first;
    Array<T> unique_second = second;

    if (!is_unique) {
        unique_first  = setUnique(first, false);
        unique_second = setUnique(second, false);
    }

    dim_t out_size =
        std::max(unique_first.elements(), unique_second.elements());
    Array<T> out = createEmptyArray<T>(dim4(out_size));

    thrust::device_ptr<T> first_ptr =
        thrust::device_pointer_cast<T>(unique_first.get());
    thrust::device_ptr<T> first_ptr_end = first_ptr + unique_first.elements();

    thrust::device_ptr<T> second_ptr =
        thrust::device_pointer_cast<T>(unique_second.get());
    thrust::device_ptr<T> second_ptr_end =
        second_ptr + unique_second.elements();

    thrust::device_ptr<T> out_ptr = thrust::device_pointer_cast<T>(out.get());

    thrust::device_ptr<T> out_ptr_last;
    THRUST_SELECT_OUT(out_ptr_last, thrust::set_intersection, first_ptr,
                      first_ptr_end, second_ptr, second_ptr_end, out_ptr);

    out.resetDims(dim4(thrust::distance(out_ptr, out_ptr_last)));

    return out;
}

#define INSTANTIATE(T)                                                        \
    template Array<T> setUnique<T>(const Array<T> &in, const bool is_sorted); \
    template Array<T> setUnion<T>(                                            \
        const Array<T> &first, const Array<T> &second, const bool is_unique); \
    template Array<T> setIntersect<T>(                                        \
        const Array<T> &first, const Array<T> &second, const bool is_unique);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(char)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(intl)
INSTANTIATE(uintl)
}  // namespace cuda
}  // namespace flare
