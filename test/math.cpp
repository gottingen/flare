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
#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <fly/arith.h>
#include <fly/data.h>
#include <fly/device.h>
#include <fly/exception.h>
#include <fly/random.h>

#include <complex>

// This makes the macros cleaner
using fly::array;
using fly::dim4;
using fly::dtype_traits;
using fly::exception;
using fly::randu;
using half_float::half;
using std::abs;
using std::endl;
using std::vector;

const int num        = 10000;
const float hlf_err  = 1e-2;
const float flt_err  = 1e-3;
const double dbl_err = 1e-6;

typedef std::complex<float> complex_float;
typedef std::complex<double> complex_double;

template<typename T>
T sigmoid(T in) {
    return T(1.0 / (1.0 + std::exp(-in)));
}

template<typename T>
T rsqrt(T in) {
    return T(1.0 / sqrt(in));
}

#define MATH_TEST(T, func, err, lo, hi)                                        \
    TEST(MathTests, Test_##func##_##T) {                                       \
        try {                                                                  \
            SUPPORTED_TYPE_CHECK(T);                                           \
            fly_dtype ty = (fly_dtype)dtype_traits<T>::fly_type;                  \
            array a     = (hi - lo) * randu(num, ty) + lo + err;               \
            a           = a.as(ty);                                            \
            eval(a);                                                           \
            array b = func(a);                                                 \
            vector<T> h_a(a.elements());                                       \
            a.host(&h_a[0]);                                                   \
            for (size_t i = 0; i < h_a.size(); i++) { h_a[i] = func(h_a[i]); } \
                                                                               \
            ASSERT_VEC_ARRAY_NEAR(h_a, dim4(h_a.size()), b, err);              \
        } catch (exception & ex) { FAIL() << ex.what(); }                      \
    }

#define MATH_TESTS_HALF(func) MATH_TEST(half, func, hlf_err, 0.05f, 0.95f)
#define MATH_TESTS_FLOAT(func) MATH_TEST(float, func, flt_err, 0.05f, 0.95f)
#define MATH_TESTS_DOUBLE(func) MATH_TEST(double, func, dbl_err, 0.05, 0.95)

#define MATH_TESTS_CFLOAT(func) \
    MATH_TEST(complex_float, func, flt_err, 0.05f, 0.95f)
#define MATH_TESTS_CDOUBLE(func) \
    MATH_TEST(complex_double, func, dbl_err, 0.05, 0.95)

#define MATH_TESTS_REAL(func) \
    MATH_TESTS_HALF(func)     \
    MATH_TESTS_FLOAT(func)    \
    MATH_TESTS_DOUBLE(func)

#define MATH_TESTS_CPLX(func) \
    MATH_TESTS_CFLOAT(func)   \
    MATH_TESTS_CDOUBLE(func)

#define MATH_TESTS_ALL(func) \
    MATH_TESTS_REAL(func)    \
    MATH_TESTS_CPLX(func)

#define MATH_TESTS_LIMITS_REAL(func, lo, hi) \
    MATH_TEST(half, func, hlf_err, lo, hi)   \
    MATH_TEST(float, func, flt_err, lo, hi)  \
    MATH_TEST(double, func, dbl_err, lo, hi)

#define MATH_TESTS_LIMITS_CPLX(func, lo, hi)        \
    MATH_TEST(complex_float, func, flt_err, lo, hi) \
    MATH_TEST(complex_double, func, dbl_err, lo, hi)

MATH_TESTS_ALL(sin)
MATH_TESTS_ALL(cos)
MATH_TESTS_ALL(tan)

MATH_TESTS_REAL(asin)
MATH_TESTS_REAL(acos)
MATH_TESTS_REAL(atan)

MATH_TESTS_ALL(sinh)
MATH_TESTS_ALL(cosh)
MATH_TESTS_ALL(tanh)

MATH_TESTS_ALL(sqrt)
MATH_TESTS_ALL(exp)
MATH_TESTS_ALL(log)
MATH_TESTS_REAL(log10)
MATH_TESTS_REAL(log2)
MATH_TESTS_REAL(rsqrt)

MATH_TESTS_REAL(sigmoid)

MATH_TESTS_LIMITS_REAL(abs, -10, 10)
MATH_TESTS_LIMITS_REAL(ceil, -10, 10)
MATH_TESTS_LIMITS_REAL(floor, -10, 10)

#if __cplusplus > 199711L || _MSC_VER >= 1800
MATH_TESTS_CPLX(asin)
MATH_TESTS_CPLX(acos)
MATH_TESTS_CPLX(atan)

MATH_TESTS_ALL(asinh)
MATH_TESTS_ALL(atanh)
MATH_TESTS_LIMITS_REAL(acosh, 1, 5)
MATH_TESTS_LIMITS_CPLX(acosh, 1, 5)
MATH_TESTS_LIMITS_REAL(round, -10, 10)
MATH_TESTS_REAL(cbrt)
MATH_TESTS_REAL(expm1)
MATH_TESTS_REAL(log1p)
MATH_TESTS_REAL(erf)
MATH_TESTS_REAL(erfc)
#endif

TEST(MathTests, Not) {
    array a  = randu(5, 5, b8);
    array b  = !a;
    char *ha = a.host<char>();
    char *hb = b.host<char>();

    for (int i = 0; i < a.elements(); i++) { ASSERT_EQ(ha[i] ^ hb[i], true); }

    fly_free_host(ha);
    fly_free_host(hb);
}
