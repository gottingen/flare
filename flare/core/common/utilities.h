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

#ifndef FLARE_CORE_COMMON_UTILITIES_H_
#define FLARE_CORE_COMMON_UTILITIES_H_

#include <flare/core/defines.h>
#include <cstdint>
#include <type_traits>
#include <initializer_list>  // in-order comma operator fold emulation
#include <utility>           // integer_sequence and friends

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace flare {
namespace detail {

// same as std::integral_constant but with __host__ __device__ annotations on
// the implicit conversion function and the call operator
template <class T, T v>
struct integral_constant {
  using value_type         = T;
  using type               = integral_constant<T, v>;
  static constexpr T value = v;
  FLARE_FUNCTION constexpr operator value_type() const noexcept {
    return value;
  }
  FLARE_FUNCTION constexpr value_type operator()() const noexcept {
    return value;
  }
};

//==============================================================================

template <typename... Is>
struct always_true : std::true_type {};

//==============================================================================

#if defined(__cpp_lib_type_identity)
// since C++20
using std::type_identity;
using std::type_identity_t;
#else
template <typename T>
struct type_identity {
  using type = T;
};

template <typename T>
using type_identity_t = typename type_identity<T>::type;
#endif

#if defined(__cpp_lib_remove_cvref)
// since C++20
using std::remove_cvref;
using std::remove_cvref_t;
#else
template <class T>
struct remove_cvref {
  using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

template <class T>
using remove_cvref_t = typename remove_cvref<T>::type;
#endif

// same as C++23 std::to_underlying but with __host__ __device__ annotations
template <typename E>
FLARE_FUNCTION constexpr std::underlying_type_t<E> to_underlying(
    E e) noexcept {
  return static_cast<std::underlying_type_t<E>>(e);
}

#if defined(__cpp_lib_is_scoped_enum)
// since C++23
using std::is_scoped_enum;
using std::is_scoped_enum_v;
#else
template <typename E, bool = std::is_enum_v<E>>
struct is_scoped_enum_impl : std::false_type {};

template <typename E>
struct is_scoped_enum_impl<E, true>
    : std::bool_constant<!std::is_convertible_v<E, std::underlying_type_t<E>>> {
};

template <typename E>
struct is_scoped_enum : is_scoped_enum_impl<E>::type {};

template <typename E>
inline constexpr bool is_scoped_enum_v = is_scoped_enum<E>::value;
#endif



template <class Type, template <class...> class Template, class Enable = void>
struct is_specialization_of : std::false_type {};

template <template <class...> class Template, class... Args>
struct is_specialization_of<Template<Args...>, Template> : std::true_type {};


// An intentionally uninstantiateable type_list for metaprogramming purposes
template <class...>
struct type_list;



// Currently linear complexity; if we use this a lot, maybe make it better?

template <class Entry, class InList, class OutList>
struct _type_list_remove_first_impl;

template <class Entry, class T, class... Ts, class... OutTs>
struct _type_list_remove_first_impl<Entry, type_list<T, Ts...>,
                                    type_list<OutTs...>>
    : _type_list_remove_first_impl<Entry, type_list<Ts...>,
                                   type_list<OutTs..., T>> {};

template <class Entry, class... Ts, class... OutTs>
struct _type_list_remove_first_impl<Entry, type_list<Entry, Ts...>,
                                    type_list<OutTs...>>
    : _type_list_remove_first_impl<Entry, type_list<>,
                                   type_list<OutTs..., Ts...>> {};

template <class Entry, class... OutTs>
struct _type_list_remove_first_impl<Entry, type_list<>, type_list<OutTs...>>
    : type_identity<type_list<OutTs...>> {};

template <class Entry, class List>
struct type_list_remove_first
    : _type_list_remove_first_impl<Entry, List, type_list<>> {};



template <template <class> class UnaryPred, class List>
struct type_list_any;

template <template <class> class UnaryPred, class... Ts>
struct type_list_any<UnaryPred, type_list<Ts...>>
    : std::bool_constant<(UnaryPred<Ts>::value || ...)> {};

//  concat_type_list combines types in multiple type_lists

// forward declaration
template <typename... T>
struct concat_type_list;

// alias
template <typename... T>
using concat_type_list_t = typename concat_type_list<T...>::type;

// final instantiation
template <typename... T>
struct concat_type_list<type_list<T...>> {
  using type = type_list<T...>;
};

// combine consecutive type_lists
template <typename... T, typename... U, typename... Tail>
struct concat_type_list<type_list<T...>, type_list<U...>, Tail...>
    : concat_type_list<type_list<T..., U...>, Tail...> {};

//  filter_type_list generates type-list of types which satisfy
//  PredicateT<T>::value == ValueT

template <template <typename> class PredicateT, typename TypeListT,
          bool ValueT = true>
struct filter_type_list;

template <template <typename> class PredicateT, typename... T, bool ValueT>
struct filter_type_list<PredicateT, type_list<T...>, ValueT> {
  using type =
      concat_type_list_t<std::conditional_t<PredicateT<T>::value == ValueT,
                                            type_list<T>, type_list<>>...>;
};

template <template <typename> class PredicateT, typename T, bool ValueT = true>
using filter_type_list_t =
    typename filter_type_list<PredicateT, T, ValueT>::type;


//==============================================================================
// The weird !sizeof(F*) to express false is to make the
// expression dependent on the type of F, and thus only applicable
// at instantiation and not first-pass semantic analysis of the
// template definition.
template <typename T>
constexpr bool dependent_false_v = !sizeof(T*);
//==============================================================================

}  // namespace detail
}  // namespace flare

#endif  // FLARE_CORE_COMMON_UTILITIES_H_
