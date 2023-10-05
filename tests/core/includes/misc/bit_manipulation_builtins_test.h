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

#include <doctest.h>

#include <flare/core.h>

// clang-format off
template<class>
struct type_helper;
#define DEFINE_TYPE_NAME(T) \
template <> struct type_helper<T> { static char const * name() { return #T; } };

DEFINE_TYPE_NAME(unsigned char)

DEFINE_TYPE_NAME(unsigned short)

DEFINE_TYPE_NAME(unsigned int)

DEFINE_TYPE_NAME(unsigned long)

DEFINE_TYPE_NAME(unsigned long long)

DEFINE_TYPE_NAME(char)

DEFINE_TYPE_NAME(short)

DEFINE_TYPE_NAME(int)

DEFINE_TYPE_NAME(long)

DEFINE_TYPE_NAME(long long)

#undef DEFINE_TYPE_NAME
// clang-format on

#define DEFINE_BIT_MANIPULATION_FUNCTION_EVAL(FUNC)   \
  struct BitManipFunction_##FUNC {                    \
    template <class T>                                \
    static FLARE_FUNCTION auto eval_constexpr(T x) { \
      return flare::FUNC(x);                         \
    }                                                 \
    template <class T>                                \
    static FLARE_FUNCTION auto eval_builtin(T x) {   \
      return flare::experimental::FUNC##_builtin(x); \
    }                                                 \
    static char const* name() { return #FUNC; }       \
  }

DEFINE_BIT_MANIPULATION_FUNCTION_EVAL(countl_zero);

DEFINE_BIT_MANIPULATION_FUNCTION_EVAL(countl_one);

DEFINE_BIT_MANIPULATION_FUNCTION_EVAL(countr_zero);

DEFINE_BIT_MANIPULATION_FUNCTION_EVAL(countr_one);

DEFINE_BIT_MANIPULATION_FUNCTION_EVAL(popcount);

DEFINE_BIT_MANIPULATION_FUNCTION_EVAL(has_single_bit);

DEFINE_BIT_MANIPULATION_FUNCTION_EVAL(bit_ceil);

DEFINE_BIT_MANIPULATION_FUNCTION_EVAL(bit_floor);

DEFINE_BIT_MANIPULATION_FUNCTION_EVAL(bit_width);

#undef DEFINE_BIT_MANIPULATION_FUNCTION_EVAL

template<class Space, class Func, class Arg, std::size_t N>
struct TestBitManipFunction {
    Arg val_[N];

    TestBitManipFunction(const Arg (&val)[N]) {
        std::copy(val, val + N, val_);
        run();
    }

    void run() const {
        int errors = 0;
        flare::parallel_reduce(flare::RangePolicy<Space>(0, N), *this, errors);
        REQUIRE_EQ(errors, 0);
    }

    FLARE_FUNCTION void operator()(int i, int &e) const {
        if (Func::eval_builtin(val_[i]) != Func::eval_constexpr(val_[i])) {
            ++e;
            flare::printf("value at %x which is %d was expected to be %d\n",
                          (unsigned) val_[i], (int) Func::eval_builtin(val_[i]),
                          (int) Func::eval_constexpr(val_[i]));
        }
    }
};

template<class Space, class... Func, class Arg, std::size_t N>
void do_test_bit_manip_function(const Arg (&x)[N]) {
    (void) std::initializer_list<int>{
            (TestBitManipFunction<Space, Func, Arg, N>(x), 0)...};
}

#define TEST_BIT_MANIP_FUNCTION(FUNC) \
  do_test_bit_manip_function<TEST_EXECSPACE, BitManipFunction_##FUNC>

template<class UInt>
void test_bit_manip_countl_zero() {
    using flare::experimental::countl_zero_builtin;
    static_assert(noexcept(countl_zero_builtin(UInt())));
    static_assert(std::is_same_v<decltype(countl_zero_builtin(UInt())), int>);
    constexpr auto max = flare::experimental::finite_max_v<UInt>;
    TEST_BIT_MANIP_FUNCTION(countl_zero)
            ({
                     UInt(0),
                     UInt(1),
                     UInt(2),
                     UInt(3),
                     UInt(4),
                     UInt(5),
                     UInt(6),
                     UInt(7),
                     UInt(8),
                     UInt(9),
                     UInt(127),
                     UInt(128),
                     UInt(max),
             });
}

TEST_CASE("TEST_CATEGORY, bit_manip_countl_zero") {
    test_bit_manip_countl_zero<unsigned char>();
    test_bit_manip_countl_zero<unsigned short>();
    test_bit_manip_countl_zero<unsigned int>();
    test_bit_manip_countl_zero<unsigned long>();
    test_bit_manip_countl_zero<unsigned long long>();
}

template<class UInt>
void test_bit_manip_countl_one() {
    using flare::experimental::countl_one_builtin;
    static_assert(noexcept(countl_one_builtin(UInt())));
    static_assert(std::is_same_v<decltype(countl_one_builtin(UInt())), int>);
    constexpr auto dig = flare::experimental::digits_v<UInt>;
    constexpr auto max = flare::experimental::finite_max_v<UInt>;
    TEST_BIT_MANIP_FUNCTION(countl_one)
            ({
                     // clang-format off
                     UInt(0),
                     UInt(1),
                     UInt(2),
                     UInt(3),
                     UInt(4),
                     UInt(5),
                     UInt(6),
                     UInt(7),
                     UInt(8),
                     UInt(9),
                     UInt(100),
                     UInt(127),
                     UInt(128),
                     UInt(max),
                     UInt(max - 1),
                     UInt(max - 2),
                     UInt(max - 3),
                     UInt(max - 4),
                     UInt(max - 5),
                     UInt(max - 6),
                     UInt(max - 7),
                     UInt(max - 8),
                     UInt(max - 9),
                     UInt(max - 126),
                     UInt(max - 127),
                     UInt(max - 128),
                     UInt(UInt(1) << (dig - 1)),
                     UInt(UInt(3) << (dig - 2)),
                     UInt(UInt(7) << (dig - 3)),
                     UInt(UInt(255) << (dig - 8)),
                     // clang-format on
             });
}

TEST_CASE("TEST_CATEGORY, bit_manip_countl_one") {
    test_bit_manip_countl_one<unsigned char>();
    test_bit_manip_countl_one<unsigned short>();
    test_bit_manip_countl_one<unsigned int>();
    test_bit_manip_countl_one<unsigned long>();
    test_bit_manip_countl_one<unsigned long long>();
}

template<class UInt>
void test_bit_manip_countr_zero() {
    using flare::experimental::countr_zero_builtin;
    static_assert(noexcept(countr_zero_builtin(UInt())));
    static_assert(std::is_same_v<decltype(countr_zero_builtin(UInt())), int>);
    constexpr auto max = flare::experimental::finite_max_v<UInt>;
    TEST_BIT_MANIP_FUNCTION(countr_zero)
            ({
                     UInt(0),
                     UInt(1),
                     UInt(2),
                     UInt(3),
                     UInt(4),
                     UInt(5),
                     UInt(6),
                     UInt(7),
                     UInt(8),
                     UInt(9),
                     UInt(126),
                     UInt(127),
                     UInt(128),
                     UInt(129),
                     UInt(130),
                     UInt(max),
             });
}

TEST_CASE("TEST_CATEGORY, bit_manip_countr_zero") {
    test_bit_manip_countr_zero<unsigned char>();
    test_bit_manip_countr_zero<unsigned short>();
    test_bit_manip_countr_zero<unsigned int>();
    test_bit_manip_countr_zero<unsigned long>();
    test_bit_manip_countr_zero<unsigned long long>();
}

template<class UInt>
void test_bit_manip_countr_one() {
    using flare::experimental::countr_one_builtin;
    static_assert(noexcept(countr_one_builtin(UInt())));
    static_assert(std::is_same_v<decltype(countr_one_builtin(UInt())), int>);
    constexpr auto max = flare::experimental::finite_max_v<UInt>;
    TEST_BIT_MANIP_FUNCTION(countr_one)
            ({
                     UInt(0),
                     UInt(1),
                     UInt(2),
                     UInt(3),
                     UInt(4),
                     UInt(5),
                     UInt(6),
                     UInt(7),
                     UInt(8),
                     UInt(9),
                     UInt(126),
                     UInt(127),
                     UInt(128),
                     UInt(max - 1),
                     UInt(max),
             });
}

TEST_CASE("TEST_CATEGORY, bit_manip_countr_one") {
    test_bit_manip_countr_one<unsigned char>();
    test_bit_manip_countr_one<unsigned short>();
    test_bit_manip_countr_one<unsigned int>();
    test_bit_manip_countr_one<unsigned long>();
    test_bit_manip_countr_one<unsigned long long>();
}

template<class UInt>
void test_bit_manip_popcount() {
    using flare::experimental::popcount_builtin;
    static_assert(noexcept(popcount_builtin(UInt())));
    static_assert(std::is_same_v<decltype(popcount_builtin(UInt())), int>);
    constexpr auto max = flare::experimental::finite_max_v<UInt>;
    TEST_BIT_MANIP_FUNCTION(popcount)
            ({
                     UInt(0),
                     UInt(1),
                     UInt(2),
                     UInt(3),
                     UInt(4),
                     UInt(5),
                     UInt(6),
                     UInt(7),
                     UInt(8),
                     UInt(9),
                     UInt(127),
                     UInt(max),
                     UInt(max - 1),
             });
}

TEST_CASE("TEST_CATEGORY, bit_manip_popcount") {
    test_bit_manip_popcount<unsigned char>();
    test_bit_manip_popcount<unsigned short>();
    test_bit_manip_popcount<unsigned int>();
    test_bit_manip_popcount<unsigned long>();
    test_bit_manip_popcount<unsigned long long>();
}

template<class UInt>
void test_bit_manip_has_single_bit() {
    using flare::experimental::has_single_bit_builtin;
    static_assert(noexcept(has_single_bit_builtin(UInt())));
    static_assert(std::is_same_v<decltype(has_single_bit_builtin(UInt())), bool>);
    constexpr auto max = flare::experimental::finite_max_v<UInt>;
    constexpr UInt one = 1;
    TEST_BIT_MANIP_FUNCTION(has_single_bit)
            ({
                     // clang-format off
                     UInt(0),
                     UInt(1),
                     UInt(2),
                     UInt(3),
                     UInt(4),
                     UInt(5),
                     UInt(6),
                     UInt(7),
                     UInt(8),
                     UInt(9),
                     UInt(max),
                     UInt(one << 0),
                     UInt(one << 1),
                     UInt(one << 2),
                     UInt(one << 3),
                     UInt(one << 4),
                     UInt(one << 5),
                     UInt(one << 6),
                     UInt(one << 7),
                     // clang-format on
             });
}

TEST_CASE("TEST_CATEGORY, bit_manip_has_single_bit") {
    test_bit_manip_has_single_bit<unsigned char>();
    test_bit_manip_has_single_bit<unsigned short>();
    test_bit_manip_has_single_bit<unsigned int>();
    test_bit_manip_has_single_bit<unsigned long>();
    test_bit_manip_has_single_bit<unsigned long long>();
}

template<class UInt>
void test_bit_manip_bit_floor() {
    using flare::experimental::bit_floor_builtin;
    static_assert(noexcept(bit_floor_builtin(UInt())));
    static_assert(std::is_same_v<decltype(bit_floor_builtin(UInt())), UInt>);
    constexpr auto max = flare::experimental::finite_max_v<UInt>;
    TEST_BIT_MANIP_FUNCTION(bit_floor)
            ({
                     UInt(0),
                     UInt(1),
                     UInt(2),
                     UInt(3),
                     UInt(4),
                     UInt(5),
                     UInt(6),
                     UInt(7),
                     UInt(8),
                     UInt(9),
                     UInt(125),
                     UInt(126),
                     UInt(127),
                     UInt(128),
                     UInt(129),
                     UInt(max),
             });
}

TEST_CASE("TEST_CATEGORY, bit_manip_bit_floor") {
    test_bit_manip_bit_floor<unsigned char>();
    test_bit_manip_bit_floor<unsigned short>();
    test_bit_manip_bit_floor<unsigned int>();
    test_bit_manip_bit_floor<unsigned long>();
    test_bit_manip_bit_floor<unsigned long long>();
}

template<class UInt>
void test_bit_manip_bit_ceil() {
    using flare::experimental::bit_ceil_builtin;
    static_assert(noexcept(bit_ceil_builtin(UInt())));
    static_assert(std::is_same_v<decltype(bit_ceil_builtin(UInt())), UInt>);
    TEST_BIT_MANIP_FUNCTION(bit_ceil)
            ({
                     // clang-format off
                     UInt(0),
                     UInt(1),
                     UInt(2),
                     UInt(3),
                     UInt(4),
                     UInt(5),
                     UInt(6),
                     UInt(7),
                     UInt(8),
                     UInt(9),
                     UInt(60),
                     UInt(61),
                     UInt(62),
                     UInt(63),
                     UInt(64),
                     UInt(65),
                     UInt(66),
                     UInt(67),
                     UInt(68),
                     UInt(69),
                     // clang-format on
             });
}

TEST_CASE("TEST_CATEGORY, bit_manip_bit_ceil") {
    test_bit_manip_bit_ceil<unsigned char>();
    test_bit_manip_bit_ceil<unsigned short>();
    test_bit_manip_bit_ceil<unsigned int>();
    test_bit_manip_bit_ceil<unsigned long>();
    test_bit_manip_bit_ceil<unsigned long long>();
}

template<class UInt>
void test_bit_manip_bit_width() {
    using flare::experimental::bit_width_builtin;
    static_assert(noexcept(bit_width_builtin(UInt())));
    static_assert(std::is_same_v<decltype(bit_width_builtin(UInt())), UInt>);
    constexpr auto max = flare::experimental::finite_max_v<UInt>;
    TEST_BIT_MANIP_FUNCTION(bit_width)
            ({
                     UInt(0),
                     UInt(1),
                     UInt(2),
                     UInt(3),
                     UInt(4),
                     UInt(5),
                     UInt(6),
                     UInt(7),
                     UInt(8),
                     UInt(9),
                     UInt(max - 1),
                     UInt(max),
             });
}

TEST_CASE("TEST_CATEGORY, bit_manip_bit_width") {
    test_bit_manip_bit_width<unsigned char>();
    test_bit_manip_bit_width<unsigned short>();
    test_bit_manip_bit_width<unsigned int>();
    test_bit_manip_bit_width<unsigned long>();
    test_bit_manip_bit_width<unsigned long long>();
}

#undef TEST_BIT_MANIP_FUNCTION

#define DEFINE_BIT_ROTATE_FUNCTION_EVAL(FUNC)                \
  struct BitRotateFunction_##FUNC {                          \
    template <class T>                                       \
    static FLARE_FUNCTION auto eval_constexpr(T x, int s) { \
      return flare::FUNC(x, s);                             \
    }                                                        \
    template <class T>                                       \
    static FLARE_FUNCTION auto eval_builtin(T x, int s) {   \
      return flare::experimental::FUNC##_builtin(x, s);     \
    }                                                        \
    static char const* name() { return #FUNC; }              \
  }

DEFINE_BIT_ROTATE_FUNCTION_EVAL(rotl);

DEFINE_BIT_ROTATE_FUNCTION_EVAL(rotr);

#undef DEFINE_BIT_ROTATE_FUNCTION_EVAL

template<class T>
struct P {
    using type = T;
    T x;
    int s;
};

template<class Space, class Func, class Arg, std::size_t N>
struct TestBitRotateFunction {
    Arg val_[N];

    TestBitRotateFunction(const Arg (&val)[N]) {
        std::copy(val, val + N, val_);
        run();
    }

    void run() const {
        int errors = 0;
        flare::parallel_reduce(flare::RangePolicy<Space>(0, N), *this, errors);
        REQUIRE_EQ(errors, 0);
    }

    FLARE_FUNCTION void operator()(int i, int &e) const {
        if (Func::eval_builtin(val_[i].x, val_[i].s) !=
            Func::eval_constexpr(val_[i].x, val_[i].s)) {
            ++e;
            flare::printf(
                    "value at %x rotated by %d which is %x was expected to be %x\n",
                    (unsigned) val_[i].x, val_[i].s,
                    (unsigned) Func::eval_builtin(val_[i].x, val_[i].s),
                    (unsigned) Func::eval_constexpr(val_[i].x, val_[i].s));
        }
    }
};

template<class Space, class... Func, class Arg, std::size_t N>
void do_test_bit_rotate_function(const Arg (&x)[N]) {
    (void) std::initializer_list<int>{
            (TestBitRotateFunction<Space, Func, Arg, N>(x), 0)...};
}

#define TEST_BIT_ROTATE_FUNCTION(FUNC) \
  do_test_bit_rotate_function<TEST_EXECSPACE, BitRotateFunction_##FUNC>

template<class UInt>
void test_bit_manip_rotl() {
    using flare::experimental::rotl_builtin;
    static_assert(noexcept(rotl_builtin(UInt(), 0)));
    static_assert(std::is_same_v<decltype(rotl_builtin(UInt(), 0)), UInt>);
    constexpr auto dig = flare::experimental::digits_v<UInt>;
    constexpr auto max = flare::experimental::finite_max_v<UInt>;
    TEST_BIT_ROTATE_FUNCTION(rotl)
            ({
                     // clang-format off
                     P<UInt>{UInt(0), 0},
                     P<UInt>{UInt(0), 1},
                     P<UInt>{UInt(0), 4},
                     P<UInt>{UInt(0), 8},
                     P<UInt>{max, 0},
                     P<UInt>{max, 1},
                     P<UInt>{max, 4},
                     P<UInt>{max, 8},
                     P<UInt>{UInt(1), 0},
                     P<UInt>{UInt(1), 1},
                     P<UInt>{UInt(1), 4},
                     P<UInt>{UInt(1), dig},
                     P<UInt>{UInt(7), dig},
                     P<UInt>{UInt(6), dig - 1},
                     P<UInt>{UInt(3), 6},
                     P<UInt>{UInt(max - 1), 0},
                     P<UInt>{UInt(max - 1), 1},
                     P<UInt>{UInt(max - 1), 2},
                     P<UInt>{UInt(max - 1), 3},
                     P<UInt>{UInt(max - 1), 4},
                     P<UInt>{UInt(max - 1), 5},
                     P<UInt>{UInt(max - 1), 6},
                     P<UInt>{UInt(max - 1), 7},
                     P<UInt>{UInt(1), 0},
                     P<UInt>{UInt(1), 1},
                     P<UInt>{UInt(1), 2},
                     P<UInt>{UInt(1), 3},
                     P<UInt>{UInt(1), 4},
                     P<UInt>{UInt(1), 5},
                     P<UInt>{UInt(1), 6},
                     P<UInt>{UInt(1), 7},
                     // clang-format on
             });
}

TEST_CASE("TEST_CATEGORY, bit_manip_rotl") {
    test_bit_manip_rotl<unsigned char>();
    test_bit_manip_rotl<unsigned short>();
    test_bit_manip_rotl<unsigned int>();
    test_bit_manip_rotl<unsigned long>();
    test_bit_manip_rotl<unsigned long long>();
}

template<class UInt>
void test_bit_manip_rotr() {
    using flare::rotr;
    using flare::experimental::rotr_builtin;
    static_assert(noexcept(rotr_builtin(UInt(), 0)));
    static_assert(std::is_same_v<decltype(rotr_builtin(UInt(), 0)), UInt>);
    constexpr auto dig = flare::experimental::digits_v<UInt>;
    constexpr auto max = flare::experimental::finite_max_v<UInt>;
    TEST_BIT_ROTATE_FUNCTION(rotr)
            ({
                     // clang-format off
                     P<UInt>{UInt(0), 0},
                     P<UInt>{UInt(0), 1},
                     P<UInt>{UInt(0), 4},
                     P<UInt>{UInt(0), 8},
                     P<UInt>{max, 0},
                     P<UInt>{max, 1},
                     P<UInt>{max, 4},
                     P<UInt>{max, 8},
                     P<UInt>{UInt(128), 0},
                     P<UInt>{UInt(128), 1},
                     P<UInt>{UInt(128), 4},
                     P<UInt>{UInt(1), dig},
                     P<UInt>{UInt(7), dig},
                     P<UInt>{UInt(6), dig - 1},
                     P<UInt>{UInt(36), dig - 2},
                     P<UInt>{UInt(max - 1), 0},
                     P<UInt>{UInt(max - 1), 1},
                     P<UInt>{UInt(max - 1), 2},
                     P<UInt>{UInt(max - 1), 3},
                     P<UInt>{UInt(max - 1), 4},
                     P<UInt>{UInt(max - 1), 5},
                     P<UInt>{UInt(max - 1), 6},
                     P<UInt>{UInt(max - 1), 7},
                     P<UInt>{UInt(128), 0},
                     P<UInt>{UInt(128), 1},
                     P<UInt>{UInt(128), 2},
                     P<UInt>{UInt(128), 3},
                     P<UInt>{UInt(128), 4},
                     P<UInt>{UInt(128), 5},
                     P<UInt>{UInt(128), 6},
                     P<UInt>{UInt(128), 0},
                     // clang-format on
             });
}

TEST_CASE("TEST_CATEGORY, bit_manip_rotr") {
    test_bit_manip_rotr<unsigned char>();
    test_bit_manip_rotr<unsigned short>();
    test_bit_manip_rotr<unsigned int>();
    test_bit_manip_rotr<unsigned long>();
    test_bit_manip_rotr<unsigned long long>();
}

#undef TEST_BIT_ROTATE_FUNCTION

template<class Space, class T>
struct TestByteswapFunction {
    TestByteswapFunction() { run(); }

    void run() const {
        int errors = 0;
        flare::parallel_reduce(flare::RangePolicy<Space>(0, 1), *this, errors);
        REQUIRE_EQ(errors, 0);
    }

    FLARE_FUNCTION void operator()(int, int &e) const {
        T value;
        T expected;
        switch (sizeof(T)) {
            case 1:
                value = static_cast<T>(0x12);
                expected = static_cast<T>(0x12);
                break;
            case 2:
                value = static_cast<T>(0x1234);
                expected = static_cast<T>(0x3412);
                break;
            case 4:
                value = static_cast<T>(0x60AF8503);
                expected = static_cast<T>(0x0385AF60);
                break;
            case 8:
                value = static_cast<T>(0xABCDFE9477936406);
                expected = static_cast<T>(0x0664937794FECDAB);
                break;
            default:
                flare::abort("logic error");
        }
        using flare::experimental::byteswap_builtin;
        if (byteswap_builtin(value) != expected) {
            ++e;
            flare::printf("value at %llx which is %llx was expected to be %llx\n",
                          (unsigned long long) value,
                          (unsigned long long) byteswap_builtin(value),
                          (unsigned long long) expected);
        }
    }
};

template<class Integral>
void test_bit_manip_byteswap() {
    using flare::rotr;
    using flare::experimental::byteswap_builtin;
    static_assert(noexcept(byteswap_builtin(Integral())));
    static_assert(
            std::is_same_v<decltype(byteswap_builtin(Integral())), Integral>);
    TestByteswapFunction<TEST_EXECSPACE, Integral>();
}

TEST_CASE("TEST_CATEGORY, bit_manip_byeswap") {
    test_bit_manip_byteswap<char>();
    test_bit_manip_byteswap<unsigned char>();
    test_bit_manip_byteswap<short>();
    test_bit_manip_byteswap<unsigned short>();
    test_bit_manip_byteswap<int>();
    test_bit_manip_byteswap<unsigned int>();
    test_bit_manip_byteswap<long>();
    test_bit_manip_byteswap<unsigned long>();
    test_bit_manip_byteswap<long long>();
    test_bit_manip_byteswap<unsigned long long>();
}

// CUDA doesn't provide memcmp
FLARE_FUNCTION int my_memcmp(void const *lhs, void const *rhs, size_t count) {
    auto u1 = static_cast<unsigned char const *>(lhs);
    auto u2 = static_cast<unsigned char const *>(rhs);
    while (count-- != 0) {
        if (*u1 != *u2) {
            return (*u1 < *u2) ? -1 : +1;
        }
        ++u1;
        ++u2;
    }
    return 0;
}

template<class Space>
struct TestBitCastFunction {
    TestBitCastFunction() { run(); }

    void run() const {
        int errors = 0;
        flare::parallel_reduce(flare::RangePolicy<Space>(0, 1), *this, errors);
        REQUIRE_EQ(errors, 0);
    }

    template<typename To, typename From>
#if defined(FLARE_COMPILER_GNU) && (900 <= FLARE_COMPILER_GNU) && \
    (FLARE_COMPILER_GNU < 930)
    // workaround compiler bug seen in GCC 9.0.1 and GCC 9.2.0
    FLARE_FUNCTION bool check(const From& from) const
#else
    static FLARE_FUNCTION bool check(const From &from)
#endif
    {
        using flare::experimental::bit_cast_builtin;
        return bit_cast_builtin<From>(bit_cast_builtin<To>(from)) == from;
    }

    FLARE_FUNCTION void operator()(int, int &e) const {
        using flare::bit_cast;
        if (bit_cast<int>(123) != 123) {
            ++e;
            FLARE_IMPL_DO_NOT_USE_PRINTF("failed check #1\n");
        }
        if (bit_cast<int>(123u) != 123) {
            ++e;
            FLARE_IMPL_DO_NOT_USE_PRINTF("failed check #2\n");
        }
        if (bit_cast<int>(~0u) != ~0) {
            ++e;
            FLARE_IMPL_DO_NOT_USE_PRINTF("failed check #3\n");
        }
        if constexpr (sizeof(int) == sizeof(float)) {
            if (!check<int>(12.34f)) {
                ++e;
                FLARE_IMPL_DO_NOT_USE_PRINTF("failed check #4\n");
            }
        }
        if constexpr (sizeof(unsigned long long) == sizeof(double)) {
            if (!check<unsigned long long>(123.456)) {
                ++e;
                FLARE_IMPL_DO_NOT_USE_PRINTF("failed check #5\n");
            }
        }

#if defined(FLARE_ENABLE_CUDA) && \
    defined(FLARE_COMPILER_NVHPC)  // FIXME_NVHPC 23.7
        if constexpr (std::is_same_v<Space, flare::Cuda>) {
          return;
        }
#endif
        struct S {
            int i;

            FLARE_FUNCTION bool operator==(const char *s) const {
                return my_memcmp(&i, s, sizeof(i)) == 0;
            }
        };
        char arr[sizeof(int)];
        char arr2[sizeof(int)];
        for (size_t i = 0; i < sizeof(int); ++i) {
            arr[i] = i + 1;
            arr2[i] = (i + 1) * -(i % 2);
        }
        if (!(bit_cast<S>(arr) == arr)) {
            ++e;
            FLARE_IMPL_DO_NOT_USE_PRINTF("failed check #6\n");
        }
        if (!(bit_cast<S>(arr2) == arr2)) {
            ++e;
            FLARE_IMPL_DO_NOT_USE_PRINTF("failed check #7\n");
        }
    }
};

TEST_CASE("TEST_CATEGORY, bit_manip_bit_cast") {
    TestBitCastFunction<TEST_EXECSPACE>();
}
