// Copyright 2023 The Elastic-AI Authors.
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

#ifndef FLARE_CORE_COMMON_EXTENTS_H_
#define FLARE_CORE_COMMON_EXTENTS_H_

#include <cstddef>
#include <type_traits>
#include <flare/core/defines.h>

namespace flare {
namespace experimental {

constexpr ptrdiff_t dynamic_extent = -1;

template <ptrdiff_t... ExtentSpecs>
struct Extents {
  /* TODO @enhancement flesh this out more */
};

template <class Exts, ptrdiff_t NewExtent>
struct PrependExtent;

template <ptrdiff_t... Exts, ptrdiff_t NewExtent>
struct PrependExtent<Extents<Exts...>, NewExtent> {
  using type = Extents<NewExtent, Exts...>;
};

template <class Exts, ptrdiff_t NewExtent>
struct AppendExtent;

template <ptrdiff_t... Exts, ptrdiff_t NewExtent>
struct AppendExtent<Extents<Exts...>, NewExtent> {
  using type = Extents<Exts..., NewExtent>;
};

}  // end namespace experimental

namespace detail {

namespace _parse_view_extents_impl {

template <class T>
struct _all_remaining_extents_dynamic : std::true_type {};

template <class T>
struct _all_remaining_extents_dynamic<T*> : _all_remaining_extents_dynamic<T> {
};

template <class T, unsigned N>
struct _all_remaining_extents_dynamic<T[N]> : std::false_type {};

template <class T, class Result, class = void>
struct _parse_impl {
  using type = Result;
};

// We have to treat the case of int**[x] specially, since it *doesn't* go
// backwards
template <class T, ptrdiff_t... ExtentSpec>
struct _parse_impl<T*, flare::experimental::Extents<ExtentSpec...>,
                   std::enable_if_t<_all_remaining_extents_dynamic<T>::value>>
    : _parse_impl<T, flare::experimental::Extents<
                         flare::experimental::dynamic_extent, ExtentSpec...>> {
};

// int*(*[x])[y] should still work also (meaning int[][x][][y])
template <class T, ptrdiff_t... ExtentSpec>
struct _parse_impl<
    T*, flare::experimental::Extents<ExtentSpec...>,
    std::enable_if_t<!_all_remaining_extents_dynamic<T>::value>> {
  using _next = flare::experimental::AppendExtent<
      typename _parse_impl<T, flare::experimental::Extents<ExtentSpec...>,
                           void>::type,
      flare::experimental::dynamic_extent>;
  using type = typename _next::type;
};

template <class T, ptrdiff_t... ExtentSpec, unsigned N>
struct _parse_impl<T[N], flare::experimental::Extents<ExtentSpec...>, void>
    : _parse_impl<
          T, flare::experimental::Extents<ExtentSpec...,
                                           ptrdiff_t(N)>  // TODO @pedantic this
                                                          // could be a
                                                          // narrowing cast
          > {};

}  // end namespace _parse_view_extents_impl

template <class DataType>
struct ParseViewExtents {
  using type = typename _parse_view_extents_impl ::_parse_impl<
      DataType, flare::experimental::Extents<>>::type;
};

template <class ValueType, ptrdiff_t Ext>
struct ApplyExtent {
  using type = ValueType[Ext];
};

template <class ValueType>
struct ApplyExtent<ValueType, flare::experimental::dynamic_extent> {
  using type = ValueType*;
};

template <class ValueType, unsigned N, ptrdiff_t Ext>
struct ApplyExtent<ValueType[N], Ext> {
  using type = typename ApplyExtent<ValueType, Ext>::type[N];
};

template <class ValueType, ptrdiff_t Ext>
struct ApplyExtent<ValueType*, Ext> {
  using type = ValueType * [Ext];
};

template <class ValueType>
struct ApplyExtent<ValueType*, flare::experimental::dynamic_extent> {
  using type =
      typename ApplyExtent<ValueType,
                           flare::experimental::dynamic_extent>::type*;
};

template <class ValueType, unsigned N>
struct ApplyExtent<ValueType[N], flare::experimental::dynamic_extent> {
  using type =
      typename ApplyExtent<ValueType,
                           flare::experimental::dynamic_extent>::type[N];
};

}  // end namespace detail

}  // end namespace flare

#endif  // FLARE_CORE_COMMON_EXTENTS_H_
