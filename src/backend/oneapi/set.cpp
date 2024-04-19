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
// oneDPL headers should be included before standard headers
#define ONEDPL_USE_PREDEFINED_POLICIES 0
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>

#include <Array.hpp>
#include <common/deprecated.hpp>
#include <copy.hpp>
#include <err_oneapi.hpp>
#include <set.hpp>
#include <sort.hpp>
#include <fly/dim4.hpp>

namespace flare {
namespace oneapi {
using fly::dim4;

using std::conditional;
using std::is_same;
template<typename T>
using ltype_t = typename conditional<is_same<T, intl>::value, cl_long, T>::type;

template<typename T>
using type_t =
    typename conditional<is_same<T, uintl>::value, cl_ulong, ltype_t<T>>::type;

template<typename T>
Array<T> setUnique(const Array<T> &in, const bool is_sorted) {
    auto dpl_policy = ::oneapi::dpl::execution::make_device_policy(getQueue());

    Array<T> out = copyArray<T>(in);

    auto out_begin = ::oneapi::dpl::begin(*out.get());
    auto out_end   = out_begin + out.elements();

    if (!is_sorted) {
        std::sort(dpl_policy, out_begin, out_end,
                  [](auto lhs, auto rhs) { return lhs < rhs; });
    }

    out_end = std::unique(dpl_policy, out_begin, out_end);

    out.resetDims(dim4(std::distance(out_begin, out_end), 1, 1, 1));

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

    size_t out_size = unique_first.elements() + unique_second.elements();
    Array<T> out    = createEmptyArray<T>(dim4(out_size, 1, 1, 1));

    auto dpl_policy = ::oneapi::dpl::execution::make_device_policy(getQueue());

    auto first_begin = ::oneapi::dpl::begin(*unique_first.get());
    auto first_end   = first_begin + unique_first.elements();

    auto second_begin = ::oneapi::dpl::begin(*unique_second.get());
    auto second_end   = second_begin + unique_second.elements();

    auto out_begin = ::oneapi::dpl::begin(*out.get());

    auto out_end = std::set_union(dpl_policy, first_begin, first_end,
                                  second_begin, second_end, out_begin);
    out.resetDims(dim4(std::distance(out_begin, out_end), 1, 1, 1));
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

    size_t out_size =
        std::max(unique_first.elements(), unique_second.elements());
    Array<T> out = createEmptyArray<T>(dim4(out_size, 1, 1, 1));

    auto dpl_policy = ::oneapi::dpl::execution::make_device_policy(getQueue());

    auto first_begin = ::oneapi::dpl::begin(*unique_first.get());
    auto first_end   = first_begin + unique_first.elements();

    auto second_begin = ::oneapi::dpl::begin(*unique_second.get());
    auto second_end   = second_begin + unique_second.elements();

    auto out_begin = ::oneapi::dpl::begin(*out.get());

    auto out_end = std::set_intersection(dpl_policy, first_begin, first_end,
                                         second_begin, second_end, out_begin);
    out.resetDims(dim4(std::distance(out_begin, out_end), 1, 1, 1));
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
}  // namespace oneapi
}  // namespace flare
