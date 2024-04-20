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
#include <err_cpu.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <set.hpp>
#include <sort.hpp>
#include <fly/dim4.hpp>
#include <algorithm>
#include <complex>
#include <vector>

namespace flare {
namespace cpu {

using fly::dim4;
using std::distance;
using std::set_intersection;
using std::set_union;
using std::unique;

template<typename T>
Array<T> setUnique(const Array<T> &in, const bool is_sorted) {
    Array<T> out = createEmptyArray<T>(fly::dim4());
    if (is_sorted) {
        out = copyArray<T>(in);
    } else {
        out = sort<T>(in, 0, true);
    }

    // Need to sync old jobs since we need to
    // operator on pointers directly in std::unique
    getQueue().sync();

    T *ptr    = out.get();
    T *last   = unique(ptr, ptr + in.elements());
    auto dist = static_cast<dim_t>(distance(ptr, last));

    dim4 dims(dist, 1, 1, 1);
    out.resetDims(dims);
    return out;
}

template<typename T>
Array<T> setUnion(const Array<T> &first, const Array<T> &second,
                  const bool is_unique) {
    Array<T> uFirst  = first;
    Array<T> uSecond = second;

    if (!is_unique) {
        // FIXME: Perhaps copy + unique would do ?
        uFirst  = setUnique(first, false);
        uSecond = setUnique(second, false);
    }

    dim_t first_elements  = uFirst.elements();
    dim_t second_elements = uSecond.elements();
    dim_t elements        = first_elements + second_elements;

    Array<T> out = createEmptyArray<T>(fly::dim4(elements));

    T *ptr  = out.get();
    T *last = set_union(uFirst.get(), uFirst.get() + first_elements,
                        uSecond.get(), uSecond.get() + second_elements, ptr);

    auto dist = static_cast<dim_t>(distance(ptr, last));
    dim4 dims(dist, 1, 1, 1);
    out.resetDims(dims);

    return out;
}

template<typename T>
Array<T> setIntersect(const Array<T> &first, const Array<T> &second,
                      const bool is_unique) {
    Array<T> uFirst  = first;
    Array<T> uSecond = second;

    if (!is_unique) {
        uFirst  = setUnique(first, false);
        uSecond = setUnique(second, false);
    }

    dim_t first_elements  = uFirst.elements();
    dim_t second_elements = uSecond.elements();
    dim_t elements        = std::max(first_elements, second_elements);

    Array<T> out = createEmptyArray<T>(fly::dim4(elements));

    T *ptr = out.get();
    T *last =
        set_intersection(uFirst.get(), uFirst.get() + first_elements,
                         uSecond.get(), uSecond.get() + second_elements, ptr);

    auto dist = static_cast<dim_t>(distance(ptr, last));
    dim4 dims(dist, 1, 1, 1);
    out.resetDims(dims);

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

}  // namespace cpu
}  // namespace flare
