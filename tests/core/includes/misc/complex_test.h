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

#include <flare/core.h>
#include <cstdio>
#include <sstream>
#include <doctest.h>

namespace Test {

// Test construction and assignment

template <class ExecSpace>
struct TestComplexConstruction {
  flare::View<flare::complex<double> *, ExecSpace> d_results;
  typename flare::View<flare::complex<double> *, ExecSpace>::HostMirror
      h_results;

  void testit() {
    d_results = flare::View<flare::complex<double> *, ExecSpace>(
        "TestComplexConstruction", 10);
    h_results = flare::create_mirror_view(d_results);

    flare::parallel_for(flare::RangePolicy<ExecSpace>(0, 1), *this);
    flare::fence();
    flare::deep_copy(h_results, d_results);

      REQUIRE_EQ(h_results(0).real(), 1.5);
    REQUIRE_EQ(h_results(0).imag(), 2.5);
    REQUIRE_EQ(h_results(1).real(), 1.5);
    REQUIRE_EQ(h_results(1).imag(), 2.5);
    REQUIRE_EQ(h_results(2).real(), 0.0);
    REQUIRE_EQ(h_results(2).imag(), 0.0);
    REQUIRE_EQ(h_results(3).real(), 3.5);
    REQUIRE_EQ(h_results(3).imag(), 0.0);
    REQUIRE_EQ(h_results(4).real(), 4.5);
    REQUIRE_EQ(h_results(4).imag(), 5.5);
    REQUIRE_EQ(h_results(5).real(), 1.5);
    REQUIRE_EQ(h_results(5).imag(), 2.5);
    REQUIRE_EQ(h_results(6).real(), 4.5);
    REQUIRE_EQ(h_results(6).imag(), 5.5);
    REQUIRE_EQ(h_results(7).real(), 7.5);
    REQUIRE_EQ(h_results(7).imag(), 0.0);
    REQUIRE_EQ(h_results(8).real(), double(8));
    REQUIRE_EQ(h_results(8).imag(), 0.0);

    // Copy construction conversion between
    // flare::complex and std::complex doesn't compile
    flare::complex<double> a(1.5, 2.5), b(3.25, 5.25), r_kk;
    std::complex<double> sa(a), sb(3.25, 5.25), r;
    r    = a;
    r_kk = a;
    REQUIRE_EQ(r.real(), r_kk.real());
    REQUIRE_EQ(r.imag(), r_kk.imag());
    r    = sb * a;
    r_kk = b * a;
    REQUIRE_EQ(r.real(), r_kk.real());
    REQUIRE_EQ(r.imag(), r_kk.imag());
    r    = sa;
    r_kk = a;
    REQUIRE_EQ(r.real(), r_kk.real());
    REQUIRE_EQ(r.imag(), r_kk.imag());
  }

  FLARE_INLINE_FUNCTION
  void operator()(const int & /*i*/) const {
    flare::complex<double> a(1.5, 2.5);
    d_results(0) = a;
    flare::complex<double> b(a);
    d_results(1)              = b;
    flare::complex<double> c = flare::complex<double>();
    d_results(2)              = c;
    flare::complex<double> d(3.5);
    d_results(3) = d;
    flare::complex<double> a_v(4.5, 5.5);
    d_results(4) = a_v;
    flare::complex<double> b_v(a);
    d_results(5) = b_v;
    flare::complex<double> e(a_v);
    d_results(6) = e;

    d_results(7) = double(7.5);
    d_results(8) = int(8);
  }
};

TEST_CASE("TEST_CATEGORY, complex_construction") {
  TestComplexConstruction<TEST_EXECSPACE> test;
  test.testit();
}

// Test Math FUnction

template <class ExecSpace>
struct TestComplexBasicMath {
  flare::View<flare::complex<double> *, ExecSpace> d_results;
  typename flare::View<flare::complex<double> *, ExecSpace>::HostMirror
      h_results;

  void testit() {
    d_results = flare::View<flare::complex<double> *, ExecSpace>(
        "TestComplexBasicMath", 24);
    h_results = flare::create_mirror_view(d_results);

    flare::parallel_for(flare::RangePolicy<ExecSpace>(0, 1), *this);
    flare::fence();
    flare::deep_copy(h_results, d_results);

    std::complex<double> a(1.5, 2.5);
    std::complex<double> b(3.25, 5.75);
    std::complex<double> d(1.0, 2.0);
    double c = 9.3;
    int e    = 2;

    std::complex<double> r;
    r = a + b;
    REQUIRE_EQ(h_results(0).real(), r.real());
    REQUIRE_EQ(h_results(0).imag(), r.imag());
    r = a - b;
    REQUIRE_EQ(h_results(1).real(), r.real());
    REQUIRE_EQ(h_results(1).imag(), r.imag());
    r = a * b;
    REQUIRE_EQ(h_results(2).real(), r.real());
    REQUIRE_EQ(h_results(2).imag(), r.imag());
    r = a / b;
    REQUIRE_EQ(h_results(3).real(), r.real());
    REQUIRE_EQ(h_results(3).imag(), r.imag());
    r = d + a;
    REQUIRE_EQ(h_results(4).real(), r.real());
    REQUIRE_EQ(h_results(4).imag(), r.imag());
    r = d - a;
    REQUIRE_EQ(h_results(5).real(), r.real());
    REQUIRE_EQ(h_results(5).imag(), r.imag());
    r = d * a;
    REQUIRE_EQ(h_results(6).real(), r.real());
    REQUIRE_EQ(h_results(6).imag(), r.imag());
    r = d / a;
    REQUIRE_EQ(h_results(7).real(), r.real());
    REQUIRE_EQ(h_results(7).imag(), r.imag());
    r = a + c;
    REQUIRE_EQ(h_results(8).real(), r.real());
    REQUIRE_EQ(h_results(8).imag(), r.imag());
    r = a - c;
    REQUIRE_EQ(h_results(9).real(), r.real());
    REQUIRE_EQ(h_results(9).imag(), r.imag());
    r = a * c;
    REQUIRE_EQ(h_results(10).real(), r.real());
    REQUIRE_EQ(h_results(10).imag(), r.imag());
    r = a / c;
    REQUIRE_EQ(h_results(11).real(), r.real());
    REQUIRE_EQ(h_results(11).imag(), r.imag());
    r = d + c;
    REQUIRE_EQ(h_results(12).real(), r.real());
    REQUIRE_EQ(h_results(12).imag(), r.imag());
    r = d - c;
    REQUIRE_EQ(h_results(13).real(), r.real());
    REQUIRE_EQ(h_results(13).imag(), r.imag());
    r = d * c;
    REQUIRE_EQ(h_results(14).real(), r.real());
    REQUIRE_EQ(h_results(14).imag(), r.imag());
    r = d / c;
    REQUIRE_EQ(h_results(15).real(), r.real());
    REQUIRE_EQ(h_results(15).imag(), r.imag());
    r = c + a;
    REQUIRE_EQ(h_results(16).real(), r.real());
    REQUIRE_EQ(h_results(16).imag(), r.imag());
    r = c - a;
    REQUIRE_EQ(h_results(17).real(), r.real());
    REQUIRE_EQ(h_results(17).imag(), r.imag());
    r = c * a;
    REQUIRE_EQ(h_results(18).real(), r.real());
    REQUIRE_EQ(h_results(18).imag(), r.imag());
    r = c / a;
    REQUIRE_EQ(h_results(19).real(), r.real());
    REQUIRE_EQ(h_results(19).imag(), r.imag());

    r = a;
    /* r = a+e; */ REQUIRE_EQ(h_results(20).real(), r.real() + e);
    REQUIRE_EQ(h_results(20).imag(), r.imag());
    /* r = a-e; */ REQUIRE_EQ(h_results(21).real(), r.real() - e);
    REQUIRE_EQ(h_results(21).imag(), r.imag());
    /* r = a*e; */ REQUIRE_EQ(h_results(22).real(), r.real() * e);
    REQUIRE_EQ(h_results(22).imag(), r.imag() * e);
    /* r = a/e; */ REQUIRE_EQ(h_results(23).real(), r.real() / 2);
    REQUIRE_EQ(h_results(23).imag(), r.imag() / e);
  }

  FLARE_INLINE_FUNCTION
  void operator()(const int & /*i*/) const {
    flare::complex<double> a(1.5, 2.5);
    flare::complex<double> b(3.25, 5.75);
    // Basic math complex / complex
    d_results(0) = a + b;
    d_results(1) = a - b;
    d_results(2) = a * b;
    d_results(3) = a / b;
    d_results(4).real(1.0);
    d_results(4).imag(2.0);
    d_results(4) += a;
    d_results(5) = flare::complex<double>(1.0, 2.0);
    d_results(5) -= a;
    d_results(6) = flare::complex<double>(1.0, 2.0);
    d_results(6) *= a;
    d_results(7) = flare::complex<double>(1.0, 2.0);
    d_results(7) /= a;

    // Basic math complex / scalar
    double c      = 9.3;
    d_results(8)  = a + c;
    d_results(9)  = a - c;
    d_results(10) = a * c;
    d_results(11) = a / c;
    d_results(12).real(1.0);
    d_results(12).imag(2.0);
    d_results(12) += c;
    d_results(13) = flare::complex<double>(1.0, 2.0);
    d_results(13) -= c;
    d_results(14) = flare::complex<double>(1.0, 2.0);
    d_results(14) *= c;
    d_results(15) = flare::complex<double>(1.0, 2.0);
    d_results(15) /= c;

    // Basic math scalar / complex
    d_results(16) = c + a;
    d_results(17) = c - a;
    d_results(18) = c * a;
    d_results(19) = c / a;

    int e         = 2;
    d_results(20) = a + e;
    d_results(21) = a - e;
    d_results(22) = a * e;
    d_results(23) = a / e;
  }
};

TEST_CASE("TEST_CATEGORY, complex_basic_math") {
  TestComplexBasicMath<TEST_EXECSPACE> test;
  test.testit();
}

template <class ExecSpace>
struct TestComplexSpecialFunctions {
  flare::View<flare::complex<double> *, ExecSpace> d_results;
  typename flare::View<flare::complex<double> *, ExecSpace>::HostMirror
      h_results;

  void testit() {
    d_results = flare::View<flare::complex<double> *, ExecSpace>(
        "TestComplexSpecialFunctions", 20);
    h_results = flare::create_mirror_view(d_results);

    flare::parallel_for(flare::RangePolicy<ExecSpace>(0, 1), *this);
    flare::fence();
    flare::deep_copy(h_results, d_results);

    std::complex<double> a(1.5, 2.5);
    double c = 9.3;

    std::complex<double> r;
    r = a;
    REQUIRE_EQ(h_results(0).real(), r.real());
    REQUIRE_EQ(h_results(0).imag(), r.imag());
    r = std::sqrt(a);
    REQUIRE_EQ(h_results(1).real(), r.real());
    REQUIRE_EQ(h_results(1).imag(), r.imag());
    r = std::pow(a, c);
    REQUIRE_EQ(h_results(2).real(), r.real());
    REQUIRE_EQ(h_results(2).imag(), r.imag());
    r = std::abs(a);
    REQUIRE_EQ(h_results(3).real(), r.real());
    REQUIRE_EQ(h_results(3).imag(), r.imag());
    r = std::exp(a);
    REQUIRE_EQ(h_results(4).real(), r.real());
    REQUIRE_EQ(h_results(4).imag(), r.imag());
    r = flare::exp(a);
    REQUIRE_EQ(h_results(4).real(), r.real());
    REQUIRE_EQ(h_results(4).imag(), r.imag());
    r = std::log(a);
    REQUIRE_EQ(h_results(5).real(), r.real());
    REQUIRE_EQ(h_results(5).imag(), r.imag());
    r = std::sin(a);
    REQUIRE_EQ(h_results(6).real(), r.real());
    REQUIRE_EQ(h_results(6).imag(), r.imag());
    r = std::cos(a);
    REQUIRE_EQ(h_results(7).real(), r.real());
    REQUIRE_EQ(h_results(7).imag(), r.imag());
    r = std::tan(a);
    REQUIRE_EQ(h_results(8).real(), r.real());
    REQUIRE_EQ(h_results(8).imag(), r.imag());
    r = std::sinh(a);
    REQUIRE_EQ(h_results(9).real(), r.real());
    REQUIRE_EQ(h_results(9).imag(), r.imag());
    r = std::cosh(a);
    REQUIRE_EQ(h_results(10).real(), r.real());
    REQUIRE_EQ(h_results(10).imag(), r.imag());
    r = std::tanh(a);
    REQUIRE_EQ(h_results(11).real(), r.real());
    REQUIRE_EQ(h_results(11).imag(), r.imag());
    r = std::asinh(a);
    REQUIRE_EQ(h_results(12).real(), r.real());
    REQUIRE_EQ(h_results(12).imag(), r.imag());
    r = std::acosh(a);
    REQUIRE_EQ(h_results(13).real(), r.real());
    REQUIRE_EQ(h_results(13).imag(), r.imag());
    // atanh
    // Work around a bug in gcc 5.3.1 where the compiler cannot compute atanh
    r = {0.163481616851666003, 1.27679502502111284};
    REQUIRE_EQ(h_results(14).real(), r.real());
    REQUIRE_EQ(h_results(14).imag(), r.imag());
    r = std::asin(a);
    REQUIRE_EQ(h_results(15).real(), r.real());
    REQUIRE_EQ(h_results(15).imag(), r.imag());
    r = std::acos(a);
    REQUIRE_EQ(h_results(16).real(), r.real());
    REQUIRE_EQ(h_results(16).imag(), r.imag());
    // atan
    // Work around a bug in gcc 5.3.1 where the compiler cannot compute atan
    r = {1.380543138238714, 0.2925178131625636};
    REQUIRE_EQ(h_results(17).real(), r.real());
    REQUIRE_EQ(h_results(17).imag(), r.imag());
    // log10
    r = std::log10(a);
    REQUIRE_EQ(h_results(18).real(), r.real());
    REQUIRE_EQ(h_results(18).imag(), r.imag());
  }

  FLARE_INLINE_FUNCTION
  void operator()(const int & /*i*/) const {
    flare::complex<double> a(1.5, 2.5);
    flare::complex<double> b(3.25, 5.75);
    double c = 9.3;

    d_results(0)  = flare::complex<double>(flare::real(a), flare::imag(a));
    d_results(1)  = flare::sqrt(a);
    d_results(2)  = flare::pow(a, c);
    d_results(3)  = flare::abs(a);
    d_results(4)  = flare::exp(a);
    d_results(5)  = flare::log(a);
    d_results(6)  = flare::sin(a);
    d_results(7)  = flare::cos(a);
    d_results(8)  = flare::tan(a);
    d_results(9)  = flare::sinh(a);
    d_results(10) = flare::cosh(a);
    d_results(11) = flare::tanh(a);
    d_results(12) = flare::asinh(a);
    d_results(13) = flare::acosh(a);
    d_results(14) = flare::atanh(a);
    d_results(15) = flare::asin(a);
    d_results(16) = flare::acos(a);
    d_results(17) = flare::atan(a);
    d_results(18) = flare::log10(a);
  }
};

void testComplexIO() {
  flare::complex<double> z = {3.14, 1.41};
  std::stringstream ss;
  ss << z;
  REQUIRE_EQ(ss.str(), "(3.14,1.41)");

  ss.str("1 (2) (3,4)");
  ss.clear();
  ss >> z;
  REQUIRE_EQ(z, (flare::complex<double>{1, 0}));
  ss >> z;
  REQUIRE_EQ(z, (flare::complex<double>{2, 0}));
  ss >> z;
  REQUIRE_EQ(z, (flare::complex<double>{3, 4}));
}

TEST_CASE("TEST_CATEGORY, complex_special_funtions") {
  TestComplexSpecialFunctions<TEST_EXECSPACE> test;
  test.testit();
}

TEST_CASE("TEST_CATEGORY, complex_io") { testComplexIO(); }

TEST_CASE("TEST_CATEGORY, complex_trivially_copyable") {
  // flare::complex<RealType> is trivially copyable when RealType is
  // trivially copyable
  using RealType = double;
  // clang claims compatibility with gcc 4.2.1 but all versions tested know
  // about std::is_trivially_copyable.
  REQUIRE((std::is_trivially_copyable<flare::complex<RealType>>::value || !std::is_trivially_copyable<RealType>::value));
}

template <class ExecSpace>
struct TestBugPowAndLogComplex {
  flare::View<flare::complex<double> *, ExecSpace> d_pow;
  flare::View<flare::complex<double> *, ExecSpace> d_log;
  TestBugPowAndLogComplex() : d_pow("pow", 2), d_log("log", 2) { test(); }
  void test() {
    flare::parallel_for(flare::RangePolicy<ExecSpace>(0, 1), *this);
    auto h_pow =
        flare::create_mirror_view_and_copy(flare::HostSpace(), d_pow);
    REQUIRE_EQ(h_pow(0).real(), 18);
    REQUIRE_EQ(h_pow(0).imag(), 26);
    REQUIRE_EQ(h_pow(1).real(), -18);
    REQUIRE_EQ(h_pow(1).imag(), 26);
    auto h_log =
        flare::create_mirror_view_and_copy(flare::HostSpace(), d_log);
    REQUIRE_EQ(h_log(0).real(), 1.151292546497023);
    REQUIRE_EQ(h_log(0).imag(), 0.3217505543966422);
    REQUIRE_EQ(h_log(1).real(), 1.151292546497023);
    REQUIRE_EQ(h_log(1).imag(), 2.819842099193151);
  }
  FLARE_FUNCTION void operator()(int) const {
    d_pow(0) = flare::pow(flare::complex<double>(+3., 1.), 3.);
    d_pow(1) = flare::pow(flare::complex<double>(-3., 1.), 3.);
    d_log(0) = flare::log(flare::complex<double>(+3., 1.));
    d_log(1) = flare::log(flare::complex<double>(-3., 1.));
  }
};

TEST_CASE("TEST_CATEGORY, complex_issue_3865") {
  TestBugPowAndLogComplex<TEST_EXECSPACE>();
}

TEST_CASE("TEST_CATEGORY, complex_operations_arithmetic_types_overloads") {
  static_assert(flare::real(1) == 1.);
  static_assert(flare::real(2.f) == 2.f);
  static_assert(flare::real(3.) == 3.);
  static_assert(flare::real(4.l) == 4.l);
  static_assert((std::is_same<decltype(flare::real(1)), double>::value));
  static_assert((std::is_same<decltype(flare::real(2.f)), float>::value));
  static_assert((std::is_same<decltype(flare::real(3.)), double>::value));
  static_assert(
      (std::is_same<decltype(flare::real(4.l)), long double>::value));

  static_assert(flare::imag(1) == 0.);
  static_assert(flare::imag(2.f) == 0.f);
  static_assert(flare::imag(3.) == 0.);
  static_assert(flare::imag(4.l) == 0.l);
  static_assert((std::is_same<decltype(flare::imag(1)), double>::value));
  static_assert((std::is_same<decltype(flare::imag(2.f)), float>::value));
  static_assert((std::is_same<decltype(flare::imag(3.)), double>::value));
  static_assert(
      (std::is_same<decltype(flare::real(4.l)), long double>::value));

  // FIXME in principle could be checked at compile time too
  REQUIRE_EQ(flare::conj(1), flare::complex<double>(1));
  REQUIRE_EQ(flare::conj(2.f), flare::complex<float>(2.f));
  REQUIRE_EQ(flare::conj(3.), flare::complex<double>(3.));
// long double has size 12 but flare::complex requires 2*sizeof(T) to be a
// power of two.
#ifndef FLARE_IMPL_32BIT
  REQUIRE_EQ(flare::conj(4.l), flare::complex<long double>(4.l));
  static_assert((
      std::is_same<decltype(flare::conj(1)), flare::complex<double>>::value));
#endif
  static_assert((std::is_same<decltype(flare::conj(2.f)),
                              flare::complex<float>>::value));
  static_assert((std::is_same<decltype(flare::conj(3.)),
                              flare::complex<double>>::value));
  static_assert((std::is_same<decltype(flare::conj(4.l)),
                              flare::complex<long double>>::value));
}

}  // namespace Test
