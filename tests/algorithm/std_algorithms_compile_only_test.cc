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

#include <flare/algorithm.h>

namespace Test {
namespace stdalgos {
namespace compileonly {

template <class ValueType>
struct TrivialUnaryFunctor {
  FLARE_INLINE_FUNCTION
  ValueType operator()(const ValueType a) const { return a; }
};

template <class ValueType>
struct TrivialBinaryFunctor {
  FLARE_INLINE_FUNCTION
  ValueType operator()(const ValueType &a, const ValueType &b) const {
    return (a + b);
  }
};

template <class ValueType>
struct TrivialUnaryPredicate {
  FLARE_INLINE_FUNCTION
  bool operator()(const ValueType val) const {
    (void)val;
    return true;
  }
};

template <class ValueType>
struct TrivialBinaryPredicate {
  FLARE_INLINE_FUNCTION
  bool operator()(const ValueType val, const ValueType val2) const {
    (void)val;
    (void)val2;
    return true;
  }
};

template <class ValueType>
struct TimesTwoFunctor {
  FLARE_INLINE_FUNCTION
  void operator()(ValueType &val) const { val *= (ValueType)2; }
};

template <class ValueType>
struct TrivialComparator {
  FLARE_INLINE_FUNCTION
  bool operator()(const ValueType &a, const ValueType &b) const {
    return a > b;
  }
};

template <class ValueType>
struct TrivialGenerator {
  FLARE_INLINE_FUNCTION
  ValueType operator()() const { return ValueType{}; }
};

template <class ValueType>
struct TrivialReduceJoinFunctor {
  FLARE_FUNCTION
  ValueType operator()(const ValueType &a, const ValueType &b) const {
    return a + b;
  }
};

template <class ValueType>
struct TrivialTransformReduceUnaryTransformer {
  FLARE_FUNCTION
  ValueType operator()(const ValueType &a) const { return a; }
};

template <class ValueType>
struct TrivialTransformReduceBinaryTransformer {
  FLARE_FUNCTION
  ValueType operator()(const ValueType &a, const ValueType &b) const {
    return (a * b);
  }
};

namespace KE = flare::experimental;

struct TestStruct {
  // put all code here and don't call from main
  // so that even if one runs the executable,
  // nothing is run anyway

  using count_type      = std::size_t;
  using T               = double;
  flare::View<T *> in1 = flare::View<T *>("in1", 10);
  flare::View<T *> in2 = flare::View<T *>("in2", 10);
  flare::View<T *> in3 = flare::View<T *>("in3", 10);
  flare::DefaultExecutionSpace exe_space;
  std::string const label = "trivial";

//
// just iterators
//
#define TEST_ALGO_MACRO_B1E1(ALGO)                                \
  (void)KE::ALGO(exe_space, /*--*/ KE::begin(in1), KE::end(in1)); \
  (void)KE::ALGO(label, exe_space, KE::begin(in1), KE::end(in1));

#define TEST_ALGO_MACRO_B1E1B2(ALGO)                             \
  (void)KE::ALGO(exe_space, /*--*/ KE::begin(in1), KE::end(in1), \
                 KE::begin(in2));                                \
  (void)KE::ALGO(label, exe_space, KE::begin(in1), KE::end(in1), \
                 KE::begin(in2));

#define TEST_ALGO_MACRO_B1E1B2E2(ALGO)                           \
  (void)KE::ALGO(exe_space, /*--*/ KE::begin(in1), KE::end(in1), \
                 KE::begin(in2), KE::end(in2));                  \
  (void)KE::ALGO(label, exe_space, KE::begin(in1), KE::end(in1), \
                 KE::begin(in2), KE::end(in2));

#define TEST_ALGO_MACRO_B1E1E2(ALGO)                             \
  (void)KE::ALGO(exe_space, /*--*/ KE::begin(in1), KE::end(in1), \
                 KE::end(in2));                                  \
  (void)KE::ALGO(label, exe_space, KE::begin(in1), KE::end(in1), KE::end(in2));

#define TEST_ALGO_MACRO_B1E1E2B3(ALGO)                                         \
  (void)KE::ALGO(exe_space, /*--*/ KE::begin(in1), KE::end(in1), KE::end(in2), \
                 KE::begin(in3));                                              \
  (void)KE::ALGO(label, exe_space, KE::begin(in1), KE::end(in1), KE::end(in2), \
                 KE::begin(in3));

#define TEST_ALGO_MACRO_B1E1E1B2(ALGO)                                         \
  (void)KE::ALGO(exe_space, /*--*/ KE::begin(in1), KE::end(in1), KE::end(in1), \
                 KE::begin(in2));                                              \
  (void)KE::ALGO(label, exe_space, KE::begin(in1), KE::end(in1), KE::end(in1), \
                 KE::begin(in2));

//
// iterators and params
//
#define TEST_ALGO_MACRO_B1_VARIAD(ALGO, ...)                     \
  (void)KE::ALGO(exe_space, /*--*/ KE::begin(in1), __VA_ARGS__); \
  (void)KE::ALGO(label, exe_space, KE::begin(in1), __VA_ARGS__);

#define TEST_ALGO_MACRO_B1E1_VARIAD(ALGO, ...)                                 \
  (void)KE::ALGO(exe_space, /*--*/ KE::begin(in1), KE::end(in1), __VA_ARGS__); \
  (void)KE::ALGO(label, exe_space, KE::begin(in1), KE::end(in1), __VA_ARGS__);

#define TEST_ALGO_MACRO_B1E1B2_VARIAD(ALGO, ...)                 \
  (void)KE::ALGO(exe_space, /*--*/ KE::begin(in1), KE::end(in1), \
                 KE::begin(in2), __VA_ARGS__);                   \
  (void)KE::ALGO(label, exe_space, KE::begin(in1), KE::end(in1), \
                 KE::begin(in2), __VA_ARGS__);

#define TEST_ALGO_MACRO_B1_ARG_B2(ALGO, ARG)                             \
  (void)KE::ALGO(exe_space, /*--*/ KE::begin(in1), ARG, KE::begin(in2)); \
  (void)KE::ALGO(label, exe_space, KE::begin(in1), ARG, KE::begin(in2));

#define TEST_ALGO_MACRO_B1E1B2B3_VARIAD(ALGO, ...)               \
  (void)KE::ALGO(exe_space, /*--*/ KE::begin(in1), KE::end(in1), \
                 KE::begin(in2), KE::begin(in3), __VA_ARGS__);   \
  (void)KE::ALGO(label, exe_space, KE::begin(in1), KE::end(in1), \
                 KE::begin(in2), KE::begin(in3), __VA_ARGS__);

#define TEST_ALGO_MACRO_B1E1B2E2_VARIAD(ALGO, ARG)               \
  (void)KE::ALGO(exe_space, /*--*/ KE::begin(in1), KE::end(in1), \
                 KE::begin(in2), KE::end(in2), ARG);             \
  (void)KE::ALGO(label, exe_space, KE::begin(in1), KE::end(in1), \
                 KE::begin(in2), KE::end(in2), ARG);

//
// views only
//
#define TEST_ALGO_MACRO_V1(ALGO)         \
  (void)KE::ALGO(exe_space, /*--*/ in1); \
  (void)KE::ALGO(label, exe_space, in1);

#define TEST_ALGO_MACRO_V1V2(ALGO)            \
  (void)KE::ALGO(exe_space, /*--*/ in1, in2); \
  (void)KE::ALGO(label, exe_space, in1, in2);

#define TEST_ALGO_MACRO_V1V2V3(ALGO)               \
  (void)KE::ALGO(exe_space, /*--*/ in1, in2, in3); \
  (void)KE::ALGO(label, exe_space, in1, in2, in3);

//
// views and params
//
#define TEST_ALGO_MACRO_V1_VARIAD(ALGO, ...)          \
  (void)KE::ALGO(exe_space, /*--*/ in1, __VA_ARGS__); \
  (void)KE::ALGO(label, exe_space, in1, __VA_ARGS__);

#define TEST_ALGO_MACRO_V1V2_VARIAD(ALGO, ...)             \
  (void)KE::ALGO(exe_space, /*--*/ in1, in2, __VA_ARGS__); \
  (void)KE::ALGO(label, exe_space, in1, in2, __VA_ARGS__);

#define TEST_ALGO_MACRO_V1V2V3_VARIAD(ALGO, ...)                \
  (void)KE::ALGO(exe_space, /*--*/ in1, in2, in3, __VA_ARGS__); \
  (void)KE::ALGO(label, exe_space, in1, in2, in3, __VA_ARGS__);

#define TEST_ALGO_MACRO_V1_ARG_V2(ALGO, ARG)       \
  (void)KE::ALGO(exe_space, /*--*/ in1, ARG, in2); \
  (void)KE::ALGO(label, exe_space, in1, ARG, in2);

  void non_modifying_seq_ops() {
    TEST_ALGO_MACRO_B1E1_VARIAD(find, T{});
    TEST_ALGO_MACRO_V1_VARIAD(find, T{});

    TEST_ALGO_MACRO_B1E1_VARIAD(find_if, TrivialUnaryPredicate<T>());
    TEST_ALGO_MACRO_V1_VARIAD(find_if, TrivialUnaryPredicate<T>());

    TEST_ALGO_MACRO_B1E1_VARIAD(find_if_not, TrivialUnaryPredicate<T>());
    TEST_ALGO_MACRO_V1_VARIAD(find_if_not, TrivialUnaryPredicate<T>());

    TEST_ALGO_MACRO_B1E1_VARIAD(for_each, TimesTwoFunctor<T>());
    TEST_ALGO_MACRO_V1_VARIAD(for_each, TimesTwoFunctor<T>());

    TEST_ALGO_MACRO_B1_VARIAD(for_each_n, count_type{}, TimesTwoFunctor<T>());
    TEST_ALGO_MACRO_V1_VARIAD(for_each_n, count_type{}, TimesTwoFunctor<T>());

    TEST_ALGO_MACRO_B1E1_VARIAD(count_if, TrivialUnaryPredicate<T>());
    TEST_ALGO_MACRO_V1_VARIAD(count_if, TrivialUnaryPredicate<T>());

    TEST_ALGO_MACRO_B1E1_VARIAD(count, T{});
    TEST_ALGO_MACRO_V1_VARIAD(count, T{});

    TEST_ALGO_MACRO_B1E1B2E2(mismatch);
    TEST_ALGO_MACRO_B1E1B2E2_VARIAD(mismatch, TrivialBinaryPredicate<T>());
    TEST_ALGO_MACRO_V1V2(mismatch);
    TEST_ALGO_MACRO_V1V2_VARIAD(mismatch, TrivialBinaryPredicate<T>());

    TEST_ALGO_MACRO_B1E1_VARIAD(all_of, TrivialUnaryPredicate<T>());
    TEST_ALGO_MACRO_V1_VARIAD(all_of, TrivialUnaryPredicate<T>());

    TEST_ALGO_MACRO_B1E1_VARIAD(any_of, TrivialUnaryPredicate<T>());
    TEST_ALGO_MACRO_V1_VARIAD(any_of, TrivialUnaryPredicate<T>());

    TEST_ALGO_MACRO_B1E1_VARIAD(none_of, TrivialUnaryPredicate<T>());
    TEST_ALGO_MACRO_V1_VARIAD(none_of, TrivialUnaryPredicate<T>());

    TEST_ALGO_MACRO_B1E1B2(equal);
    TEST_ALGO_MACRO_B1E1B2_VARIAD(equal, TrivialBinaryPredicate<T>());
    TEST_ALGO_MACRO_V1V2(equal);
    TEST_ALGO_MACRO_V1V2_VARIAD(equal, TrivialBinaryPredicate<T>());
    TEST_ALGO_MACRO_B1E1B2E2(equal);
    TEST_ALGO_MACRO_B1E1B2E2_VARIAD(equal, TrivialBinaryPredicate<T>());

    TEST_ALGO_MACRO_B1E1B2E2(lexicographical_compare);
    TEST_ALGO_MACRO_B1E1B2E2_VARIAD(lexicographical_compare,
                                    TrivialComparator<T>());
    TEST_ALGO_MACRO_V1V2(lexicographical_compare);
    TEST_ALGO_MACRO_V1V2_VARIAD(lexicographical_compare,
                                TrivialComparator<T>());

    TEST_ALGO_MACRO_B1E1(adjacent_find);
    TEST_ALGO_MACRO_V1(adjacent_find);
    TEST_ALGO_MACRO_B1E1_VARIAD(adjacent_find, TrivialBinaryFunctor<T>());
    TEST_ALGO_MACRO_V1_VARIAD(adjacent_find, TrivialBinaryFunctor<T>());

    TEST_ALGO_MACRO_B1E1B2E2(search);
    TEST_ALGO_MACRO_V1V2(search);
    TEST_ALGO_MACRO_B1E1B2E2_VARIAD(search, TrivialBinaryFunctor<T>());
    TEST_ALGO_MACRO_V1V2_VARIAD(search, TrivialBinaryFunctor<T>());

    TEST_ALGO_MACRO_B1E1B2E2(find_first_of);
    TEST_ALGO_MACRO_V1V2(find_first_of);
    TEST_ALGO_MACRO_B1E1B2E2_VARIAD(find_first_of, TrivialBinaryFunctor<T>());
    TEST_ALGO_MACRO_V1V2_VARIAD(find_first_of, TrivialBinaryFunctor<T>());

    TEST_ALGO_MACRO_B1E1_VARIAD(search_n, count_type{}, T{});
    TEST_ALGO_MACRO_V1_VARIAD(search_n, count_type{}, T{});
    TEST_ALGO_MACRO_B1E1_VARIAD(search_n, count_type{}, T{},
                                TrivialBinaryPredicate<T>());
    TEST_ALGO_MACRO_V1_VARIAD(search_n, count_type{}, T{},
                              TrivialBinaryPredicate<T>());

    TEST_ALGO_MACRO_B1E1B2E2(find_end);
    TEST_ALGO_MACRO_V1V2(find_end);
    TEST_ALGO_MACRO_B1E1B2E2_VARIAD(find_end, TrivialBinaryFunctor<T>());
    TEST_ALGO_MACRO_V1V2_VARIAD(find_end, TrivialBinaryFunctor<T>());
  }

  void modifying_seq_ops() {
    TEST_ALGO_MACRO_B1E1B2_VARIAD(replace_copy, T{}, T{});
    TEST_ALGO_MACRO_V1V2_VARIAD(replace_copy, T{}, T{});

    TEST_ALGO_MACRO_B1E1B2_VARIAD(replace_copy_if, TrivialUnaryPredicate<T>(),
                                  T{});
    TEST_ALGO_MACRO_V1V2_VARIAD(replace_copy_if, TrivialUnaryPredicate<T>(),
                                T{});

    TEST_ALGO_MACRO_B1E1_VARIAD(replace, T{}, T{});
    TEST_ALGO_MACRO_V1_VARIAD(replace, T{}, T{});

    TEST_ALGO_MACRO_B1E1_VARIAD(replace_if, TrivialUnaryPredicate<T>(), T{});
    TEST_ALGO_MACRO_V1_VARIAD(replace_if, TrivialUnaryPredicate<T>(), T{});

    TEST_ALGO_MACRO_B1E1B2(copy);
    TEST_ALGO_MACRO_V1V2(copy);

    TEST_ALGO_MACRO_B1_ARG_B2(copy_n, count_type{});
    TEST_ALGO_MACRO_V1_ARG_V2(copy_n, count_type{});

    TEST_ALGO_MACRO_B1E1B2(copy_backward);
    TEST_ALGO_MACRO_V1V2(copy_backward);

    TEST_ALGO_MACRO_B1E1B2_VARIAD(copy_if, TrivialUnaryPredicate<T>());
    TEST_ALGO_MACRO_V1V2_VARIAD(copy_if, TrivialUnaryPredicate<T>());

    TEST_ALGO_MACRO_B1E1_VARIAD(fill, T{});
    TEST_ALGO_MACRO_V1_VARIAD(fill, T{});

    TEST_ALGO_MACRO_B1_VARIAD(fill_n, count_type{}, T{});
    TEST_ALGO_MACRO_V1_VARIAD(fill_n, count_type{}, T{});

    TEST_ALGO_MACRO_B1E1B2_VARIAD(transform, TrivialUnaryFunctor<T>{});
    TEST_ALGO_MACRO_V1V2_VARIAD(transform, TrivialUnaryFunctor<T>{});

    TEST_ALGO_MACRO_B1E1B2_VARIAD(transform, TrivialUnaryFunctor<T>{});
    TEST_ALGO_MACRO_B1E1B2B3_VARIAD(transform, TrivialBinaryFunctor<T>{});
    TEST_ALGO_MACRO_V1V2_VARIAD(transform, TrivialUnaryFunctor<T>{});
    TEST_ALGO_MACRO_V1V2V3_VARIAD(transform, TrivialBinaryFunctor<T>{});

    TEST_ALGO_MACRO_B1E1_VARIAD(generate, TrivialGenerator<T>{});
    TEST_ALGO_MACRO_V1_VARIAD(generate, TrivialGenerator<T>{});

    TEST_ALGO_MACRO_B1_VARIAD(generate_n, count_type{}, TrivialGenerator<T>{});
    TEST_ALGO_MACRO_V1_VARIAD(generate_n, count_type{}, TrivialGenerator<T>{});

    TEST_ALGO_MACRO_B1E1B2(reverse_copy);
    TEST_ALGO_MACRO_V1V2(reverse_copy);

    TEST_ALGO_MACRO_B1E1(reverse);
    TEST_ALGO_MACRO_V1(reverse);

    TEST_ALGO_MACRO_B1E1B2(move);
    TEST_ALGO_MACRO_V1V2(move);

    TEST_ALGO_MACRO_B1E1E2(move_backward);
    TEST_ALGO_MACRO_V1V2(move_backward);

    TEST_ALGO_MACRO_B1E1B2(swap_ranges);
    TEST_ALGO_MACRO_V1V2(swap_ranges);

    TEST_ALGO_MACRO_B1E1(unique);
    TEST_ALGO_MACRO_V1(unique);
    TEST_ALGO_MACRO_B1E1_VARIAD(unique, TrivialBinaryPredicate<T>{});
    TEST_ALGO_MACRO_V1_VARIAD(unique, TrivialBinaryPredicate<T>{});

    TEST_ALGO_MACRO_B1E1B2(unique_copy);
    TEST_ALGO_MACRO_V1V2(unique_copy);
    TEST_ALGO_MACRO_B1E1B2_VARIAD(unique_copy, TrivialBinaryPredicate<T>{});
    TEST_ALGO_MACRO_V1V2_VARIAD(unique_copy, TrivialBinaryPredicate<T>{});

    TEST_ALGO_MACRO_B1E1E2(rotate);
    TEST_ALGO_MACRO_V1_VARIAD(rotate, count_type{});

    TEST_ALGO_MACRO_B1E1E1B2(rotate_copy);
    TEST_ALGO_MACRO_V1_ARG_V2(rotate_copy, count_type{});

    TEST_ALGO_MACRO_B1E1_VARIAD(remove_if, TrivialUnaryPredicate<T>{});
    TEST_ALGO_MACRO_V1_VARIAD(remove_if, TrivialUnaryPredicate<T>{});

    TEST_ALGO_MACRO_B1E1_VARIAD(remove, T{});
    TEST_ALGO_MACRO_V1_VARIAD(remove, T{});

    TEST_ALGO_MACRO_B1E1B2_VARIAD(remove_copy, T{});
    TEST_ALGO_MACRO_V1V2_VARIAD(remove_copy, T{});

    TEST_ALGO_MACRO_B1E1B2_VARIAD(remove_copy_if, TrivialUnaryPredicate<T>());
    TEST_ALGO_MACRO_V1V2_VARIAD(remove_copy_if, TrivialUnaryPredicate<T>());

    TEST_ALGO_MACRO_B1E1_VARIAD(shift_left, count_type{});
    TEST_ALGO_MACRO_V1_VARIAD(shift_left, count_type{});

    TEST_ALGO_MACRO_B1E1_VARIAD(shift_right, count_type{});
    TEST_ALGO_MACRO_V1_VARIAD(shift_right, count_type{});
  }

  void sorting_ops() {
    TEST_ALGO_MACRO_B1E1(is_sorted_until);
    TEST_ALGO_MACRO_V1(is_sorted_until);

    TEST_ALGO_MACRO_B1E1_VARIAD(is_sorted_until, TrivialComparator<T>());
    TEST_ALGO_MACRO_V1_VARIAD(is_sorted_until, TrivialComparator<T>());

    TEST_ALGO_MACRO_B1E1(is_sorted);
    TEST_ALGO_MACRO_V1(is_sorted);

    TEST_ALGO_MACRO_B1E1_VARIAD(is_sorted, TrivialComparator<T>());
    TEST_ALGO_MACRO_V1_VARIAD(is_sorted, TrivialComparator<T>());
  }

  void minmax_ops() {
    TEST_ALGO_MACRO_B1E1(min_element);
    TEST_ALGO_MACRO_V1(min_element);
    TEST_ALGO_MACRO_B1E1(max_element);
    TEST_ALGO_MACRO_V1(max_element);
    TEST_ALGO_MACRO_B1E1(minmax_element);
    TEST_ALGO_MACRO_V1(minmax_element);

    TEST_ALGO_MACRO_B1E1_VARIAD(min_element, TrivialComparator<T>());
    TEST_ALGO_MACRO_V1_VARIAD(min_element, TrivialComparator<T>());
    TEST_ALGO_MACRO_B1E1_VARIAD(max_element, TrivialComparator<T>());
    TEST_ALGO_MACRO_V1_VARIAD(max_element, TrivialComparator<T>());
    TEST_ALGO_MACRO_B1E1_VARIAD(minmax_element, TrivialComparator<T>());
    TEST_ALGO_MACRO_V1_VARIAD(minmax_element, TrivialComparator<T>());
  }

  void partitionig_ops() {
    TEST_ALGO_MACRO_B1E1_VARIAD(is_partitioned, TrivialUnaryPredicate<T>());
    TEST_ALGO_MACRO_V1_VARIAD(is_partitioned, TrivialUnaryPredicate<T>());

    TEST_ALGO_MACRO_B1E1B2B3_VARIAD(partition_copy, TrivialUnaryPredicate<T>());
    TEST_ALGO_MACRO_V1V2V3_VARIAD(partition_copy, TrivialUnaryPredicate<T>());

    TEST_ALGO_MACRO_B1E1_VARIAD(partition_point, TrivialUnaryPredicate<T>());
    TEST_ALGO_MACRO_V1_VARIAD(partition_point, TrivialUnaryPredicate<T>());
  }

  void numeric() {
    TEST_ALGO_MACRO_B1E1B2(adjacent_difference);
    TEST_ALGO_MACRO_B1E1B2_VARIAD(adjacent_difference,
                                  TrivialBinaryFunctor<T>());
    TEST_ALGO_MACRO_V1V2(adjacent_difference);
    TEST_ALGO_MACRO_V1V2_VARIAD(adjacent_difference, TrivialBinaryFunctor<T>());

    TEST_ALGO_MACRO_B1E1B2_VARIAD(exclusive_scan, T{});
    TEST_ALGO_MACRO_V1V2_VARIAD(exclusive_scan, T{});
    TEST_ALGO_MACRO_B1E1B2_VARIAD(exclusive_scan, T{},
                                  TrivialBinaryFunctor<T>());
    TEST_ALGO_MACRO_V1V2_VARIAD(exclusive_scan, T{}, TrivialBinaryFunctor<T>());

    TEST_ALGO_MACRO_B1E1B2_VARIAD(transform_exclusive_scan, T{},
                                  TrivialBinaryFunctor<T>(),
                                  TrivialUnaryFunctor<T>());
    TEST_ALGO_MACRO_V1V2_VARIAD(transform_exclusive_scan, T{},
                                TrivialBinaryFunctor<T>(),
                                TrivialUnaryFunctor<T>());

    TEST_ALGO_MACRO_B1E1B2(inclusive_scan);
    TEST_ALGO_MACRO_V1V2(inclusive_scan);
    TEST_ALGO_MACRO_B1E1B2_VARIAD(inclusive_scan, TrivialBinaryFunctor<T>());
    TEST_ALGO_MACRO_V1V2_VARIAD(inclusive_scan, TrivialBinaryFunctor<T>());
    TEST_ALGO_MACRO_B1E1B2_VARIAD(inclusive_scan, TrivialBinaryFunctor<T>(),
                                  T{});
    TEST_ALGO_MACRO_V1V2_VARIAD(inclusive_scan, TrivialBinaryFunctor<T>(), T{});

    TEST_ALGO_MACRO_B1E1B2_VARIAD(transform_inclusive_scan,
                                  TrivialBinaryFunctor<T>(),
                                  TrivialUnaryFunctor<T>());
    TEST_ALGO_MACRO_V1V2_VARIAD(transform_inclusive_scan,
                                TrivialBinaryFunctor<T>(),
                                TrivialUnaryFunctor<T>());
    TEST_ALGO_MACRO_B1E1B2_VARIAD(transform_inclusive_scan,
                                  TrivialBinaryFunctor<T>(),
                                  TrivialUnaryFunctor<T>(), T{});
    TEST_ALGO_MACRO_V1V2_VARIAD(transform_inclusive_scan,
                                TrivialBinaryFunctor<T>(),
                                TrivialUnaryFunctor<T>(), T{});

    TEST_ALGO_MACRO_B1E1(reduce);
    TEST_ALGO_MACRO_V1(reduce);
    TEST_ALGO_MACRO_B1E1_VARIAD(reduce, T{});
    TEST_ALGO_MACRO_V1_VARIAD(reduce, T{});
    TEST_ALGO_MACRO_B1E1_VARIAD(reduce, T{}, TrivialReduceJoinFunctor<T>());
    TEST_ALGO_MACRO_V1_VARIAD(reduce, T{}, TrivialReduceJoinFunctor<T>());

    TEST_ALGO_MACRO_B1E1B2_VARIAD(transform_reduce, T{});
    TEST_ALGO_MACRO_V1V2_VARIAD(transform_reduce, T{});
    TEST_ALGO_MACRO_B1E1B2_VARIAD(transform_reduce, T{},
                                  TrivialReduceJoinFunctor<T>(),
                                  TrivialTransformReduceBinaryTransformer<T>());
    TEST_ALGO_MACRO_V1V2_VARIAD(transform_reduce, T{},
                                TrivialReduceJoinFunctor<T>(),
                                TrivialTransformReduceBinaryTransformer<T>());

    TEST_ALGO_MACRO_B1E1_VARIAD(transform_reduce, T{},
                                TrivialReduceJoinFunctor<T>(),
                                TrivialTransformReduceUnaryTransformer<T>());
    TEST_ALGO_MACRO_V1_VARIAD(transform_reduce, T{},
                              TrivialReduceJoinFunctor<T>(),
                              TrivialTransformReduceUnaryTransformer<T>());
  }
};

}  // namespace compileonly
}  // namespace stdalgos
}  // namespace Test

int main() { return 0; }
