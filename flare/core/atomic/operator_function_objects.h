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


#ifndef FLARE_CORE_ATOMIC_OPERATOR_FUNCTION_OBJECTS_H_
#define FLARE_CORE_ATOMIC_OPERATOR_FUNCTION_OBJECTS_H_

#include <flare/core/defines.h>
#include <type_traits>

// Function objects that represent common arithmetic and logical
// Combination operands to be used in a compare-and-exchange based atomic operation
namespace flare {
namespace detail {

template <class Scalar1, class Scalar2>
struct max_operator {
    FLARE_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) {
    return (val1 > val2 ? val1 : val2);
  }
    FLARE_FORCEINLINE_FUNCTION
  static constexpr bool check_early_exit(Scalar1 const& val1, Scalar2 const& val2) {
    return val1 > val2;
  }
};

template <class Scalar1, class Scalar2>
struct min_operator {
  FLARE_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) {
    return (val1 < val2 ? val1 : val2);
  }
  FLARE_FORCEINLINE_FUNCTION
  static constexpr bool check_early_exit(Scalar1 const& val1, Scalar2 const& val2) {
    return val1 < val2;
  }
};

template <class Op, class Scalar1, class Scalar2, class = bool>
struct may_exit_early : std::false_type {};

// This exit early optimization causes weird compiler errors with MSVC 2019
#ifndef FLARE_HAVE_MSVC_ATOMICS
template <class Op, class Scalar1, class Scalar2>
struct may_exit_early<Op,
                      Scalar1,
                      Scalar2,
                      decltype(Op::check_early_exit(std::declval<Scalar1 const&>(),
                                                    std::declval<Scalar2 const&>()))>
    : std::true_type {};
#endif

template <class Op, class Scalar1, class Scalar2>
constexpr FLARE_FUNCTION
    std::enable_if_t<may_exit_early<Op, Scalar1, Scalar2>::value, bool>
    check_early_exit(Op const&, Scalar1 const& val1, Scalar2 const& val2) {
  return Op::check_early_exit(val1, val2);
}

template <class Op, class Scalar1, class Scalar2>
constexpr FLARE_FUNCTION
    std::enable_if_t<!may_exit_early<Op, Scalar1, Scalar2>::value, bool>
    check_early_exit(Op const&, Scalar1 const&, Scalar2 const&) {
  return false;
}

template <class Scalar1, class Scalar2>
struct add_operator {
  FLARE_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) { return val1 + val2; }
};

template <class Scalar1, class Scalar2>
struct sub_operator {
  FLARE_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) { return val1 - val2; }
};

template <class Scalar1, class Scalar2>
struct mul_operator {
  FLARE_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) { return val1 * val2; }
};

template <class Scalar1, class Scalar2>
struct div_operator {
  FLARE_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) { return val1 / val2; }
};

template <class Scalar1, class Scalar2>
struct mod_operator {
  FLARE_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) { return val1 % val2; }
};

template <class Scalar1, class Scalar2>
struct and_operator {
  FLARE_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) { return val1 & val2; }
};

template <class Scalar1, class Scalar2>
struct or_operator {
  FLARE_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) { return val1 | val2; }
};

template <class Scalar1, class Scalar2>
struct xor_operator {
  FLARE_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) { return val1 ^ val2; }
};

template <class Scalar1, class Scalar2>
struct nand_operator {
  FLARE_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) {
    return ~(val1 & val2);
  }
};

template <class Scalar1, class Scalar2>
struct lshift_operator {
  FLARE_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) {
    return val1 << val2;
  }
};

template <class Scalar1, class Scalar2>
struct rshift_operator {
  FLARE_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) {
    return val1 >> val2;
  }
};

template <class Scalar1, class Scalar2>
struct inc_mod_operator {
  FLARE_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) {
    return ((val1 >= val2) ? Scalar1(0) : val1 + Scalar1(1));
  }
};

template <class Scalar1, class Scalar2>
struct dec_mod_operator {
  FLARE_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) {
    return (((val1 == Scalar1(0)) | (val1 > val2)) ? val2 : (val1 - Scalar1(1)));
  }
};

template <class Scalar1, class Scalar2>
struct store_operator {
  FLARE_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1&, const Scalar2& val2) { return val2; }
};

template <class Scalar1, class Scalar2>
struct load_operator {
  FLARE_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2&) { return val1; }
};

}  // namespace detail
}  // namespace flare

#endif  // FLARE_CORE_ATOMIC_OPERATOR_FUNCTION_OBJECTS_H_
