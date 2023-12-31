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

#ifndef FLARE_CORE_EXP_INTEROP_H_
#define FLARE_CORE_EXP_INTEROP_H_

#include <flare/core_fwd.h>
#include <flare/core/memory/layout.h>
#include <flare/core/memory/memory_traits.h>
#include <flare/core/tensor/view.h>
#include <flare/core/common/utilities.h>
#include <type_traits>

namespace flare {
namespace detail {

// ------------------------------------------------------------------ //
//  this is used to convert
//      flare::Device<ExecSpace, MemSpace> to MemSpace
//
template <typename Tp>
struct device_memory_space {
  using type = Tp;
};

template <typename ExecT, typename MemT>
struct device_memory_space<flare::Device<ExecT, MemT>> {
  using type = MemT;
};

template <typename Tp>
using device_memory_space_t = typename device_memory_space<Tp>::type;

// ------------------------------------------------------------------ //
//  this is the impl version which takes a view and converts to python
//  view type
//
template <typename, typename...>
struct python_view_type_impl;

template <template <typename...> class ViewT, typename ValueT,
          typename... Types>
struct python_view_type_impl<ViewT<ValueT>, type_list<Types...>> {
  using type = ViewT<ValueT, device_memory_space_t<Types>...>;
};

template <template <typename...> class ViewT, typename ValueT,
          typename... Types>
struct python_view_type_impl<ViewT<ValueT, Types...>>
    : python_view_type_impl<ViewT<ValueT>,
                            filter_type_list_t<is_default_memory_trait,
                                               type_list<Types...>, false>> {};

template <typename... T>
using python_view_type_impl_t = typename python_view_type_impl<T...>::type;

}  // namespace detail
}  // namespace flare

namespace flare {

template <typename DataType, class... Properties>
class DynRankView;

namespace detail {

// Duplicate from the header file for DynRankView to avoid core depending on
// containers.
template <class>
struct is_dyn_rank_view_dup : public std::false_type {};

template <class D, class... P>
struct is_dyn_rank_view_dup<flare::DynRankView<D, P...>>
    : public std::true_type {};

}  // namespace detail

namespace experimental {

// ------------------------------------------------------------------ //
//  this is used to extract the uniform type of a view
//
template <typename ViewT>
struct python_view_type {
  static_assert(
      flare::is_view<std::decay_t<ViewT>>::value ||
          flare::detail::is_dyn_rank_view_dup<std::decay_t<ViewT>>::value,
      "Error! python_view_type only supports flare::View and "
      "flare::DynRankView");

  using type =
      flare::detail::python_view_type_impl_t<typename ViewT::array_type>;
};

template <typename ViewT>
using python_view_type_t = typename python_view_type<ViewT>::type;

template <typename Tp>
auto as_python_type(Tp&& _v) {
  using cast_type = python_view_type_t<Tp>;
  return static_cast<cast_type>(std::forward<Tp>(_v));
}
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_CORE_EXP_INTEROP_H_
