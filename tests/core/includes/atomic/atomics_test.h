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
#include <doctest.h>

namespace TestAtomic {

// Struct for testing arbitrary size atomics.

template <int N>
struct SuperScalar {
  double val[N];

  FLARE_INLINE_FUNCTION
  SuperScalar() {
    for (int i = 0; i < N; i++) {
      val[i] = 0.0;
    }
  }

  FLARE_INLINE_FUNCTION
  SuperScalar(const SuperScalar& src) {
    for (int i = 0; i < N; i++) {
      val[i] = src.val[i];
    }
  }

  FLARE_INLINE_FUNCTION
  SuperScalar(const volatile SuperScalar& src) {
    for (int i = 0; i < N; i++) {
      val[i] = src.val[i];
    }
  }

  FLARE_INLINE_FUNCTION
  SuperScalar& operator=(const SuperScalar& src) {
    for (int i = 0; i < N; i++) {
      val[i] = src.val[i];
    }
    return *this;
  }

  FLARE_INLINE_FUNCTION
  SuperScalar& operator=(const volatile SuperScalar& src) {
    for (int i = 0; i < N; i++) {
      val[i] = src.val[i];
    }
    return *this;
  }

  FLARE_INLINE_FUNCTION
  void operator=(const SuperScalar& src) volatile {
    for (int i = 0; i < N; i++) {
      val[i] = src.val[i];
    }
  }

  FLARE_INLINE_FUNCTION
  SuperScalar operator+(const SuperScalar& src) const {
    SuperScalar tmp = *this;
    for (int i = 0; i < N; i++) {
      tmp.val[i] += src.val[i];
    }
    return tmp;
  }

  FLARE_INLINE_FUNCTION
  SuperScalar& operator+=(const double& src) {
    for (int i = 0; i < N; i++) {
      val[i] += 1.0 * (i + 1) * src;
    }
    return *this;
  }

  FLARE_INLINE_FUNCTION
  SuperScalar& operator+=(const SuperScalar& src) {
    for (int i = 0; i < N; i++) {
      val[i] += src.val[i];
    }
    return *this;
  }

  FLARE_INLINE_FUNCTION
  bool operator==(const SuperScalar& src) const {
    bool compare = true;
    for (int i = 0; i < N; i++) {
      compare = compare && (val[i] == src.val[i]);
    }
    return compare;
  }

  FLARE_INLINE_FUNCTION
  bool operator!=(const SuperScalar& src) const {
    bool compare = true;
    for (int i = 0; i < N; i++) {
      compare = compare && (val[i] == src.val[i]);
    }
    return !compare;
  }

  FLARE_INLINE_FUNCTION
  SuperScalar(const double& src) {
    for (int i = 0; i < N; i++) {
      val[i] = 1.0 * (i + 1) * src;
    }
  }
};

template <int N>
std::ostream& operator<<(std::ostream& os, const SuperScalar<N>& dt) {
  os << "{ ";
  for (int i = 0; i < N - 1; i++) {
    os << dt.val[i] << ", ";
  }
  os << dt.val[N - 1] << "}";

  return os;
}

template <class T, class DEVICE_TYPE>
struct ZeroFunctor {
  using execution_space = DEVICE_TYPE;
  using type            = typename flare::View<T, execution_space>;
  using h_type          = typename flare::View<T, execution_space>::HostMirror;

  type data;

  FLARE_INLINE_FUNCTION
  void operator()(int) const { data() = 0; }
};

//---------------------------------------------------
//--------------atomic_fetch_add---------------------
//---------------------------------------------------

template <class T, class DEVICE_TYPE>
struct AddFunctor {
  using execution_space = DEVICE_TYPE;
  using type            = flare::View<T, execution_space>;

  type data;

  FLARE_INLINE_FUNCTION
  void operator()(int) const { flare::atomic_fetch_add(&data(), (T)1); }
};

template <class T, class DEVICE_TYPE>
struct AddFunctorReduce {
  using execution_space = DEVICE_TYPE;
  using type            = flare::View<T, execution_space>;

  type data;

  FLARE_INLINE_FUNCTION
  void operator()(int, int&) const { flare::atomic_fetch_add(&data(), (T)1); }
};

template <class T, class execution_space>
T AddLoop(int loop) {
  struct ZeroFunctor<T, execution_space> f_zero;
  typename ZeroFunctor<T, execution_space>::type data("Data");
  typename ZeroFunctor<T, execution_space>::h_type h_data("HData");

  f_zero.data = data;

  flare::parallel_for(1, f_zero);
  execution_space().fence();

  struct AddFunctor<T, execution_space> f_add;

  f_add.data = data;
  flare::parallel_for(loop, f_add);
  execution_space().fence();

  flare::deep_copy(h_data, data);
  T val = h_data();

  struct AddFunctorReduce<T, execution_space> f_add_red;
  f_add_red.data = data;
  int dummy_result;
  flare::parallel_reduce(loop, f_add_red, dummy_result);
  execution_space().fence();

  return val;
}

template <class T>
T AddLoopSerial(int loop) {
  T* data = new T[1];
  data[0] = 0;

  for (int i = 0; i < loop; i++) {
    *data += (T)1;
  }

  T val = *data;
  delete[] data;

  return val;
}

//------------------------------------------------------
//--------------atomic_compare_exchange-----------------
//------------------------------------------------------

template <class T, class DEVICE_TYPE>
struct CASFunctor {
  using execution_space = DEVICE_TYPE;
  using type            = flare::View<T, execution_space>;

  type data;

  FLARE_INLINE_FUNCTION
  void operator()(int) const {
    T old = data();
    T newval, assumed;

    do {
      assumed = old;
      newval  = assumed + (T)1;
      old     = flare::atomic_compare_exchange(&data(), assumed, newval);
    } while (old != assumed);
  }
};

template <class T, class DEVICE_TYPE>
struct CASFunctorReduce {
  using execution_space = DEVICE_TYPE;
  using type            = flare::View<T, execution_space>;

  type data;

  FLARE_INLINE_FUNCTION
  void operator()(int, int&) const {
    T old = data();
    T newval, assumed;

    do {
      assumed = old;
      newval  = assumed + (T)1;
      old     = flare::atomic_compare_exchange(&data(), assumed, newval);
    } while (old != assumed);
  }
};

template <class T, class execution_space>
T CASLoop(int loop) {
  struct ZeroFunctor<T, execution_space> f_zero;
  typename ZeroFunctor<T, execution_space>::type data("Data");
  typename ZeroFunctor<T, execution_space>::h_type h_data("HData");

  f_zero.data = data;
  flare::parallel_for(1, f_zero);
  execution_space().fence();

  struct CASFunctor<T, execution_space> f_cas;
  f_cas.data = data;
  flare::parallel_for(loop, f_cas);
  execution_space().fence();

  flare::deep_copy(h_data, data);
  T val = h_data();

  struct CASFunctorReduce<T, execution_space> f_cas_red;
  f_cas_red.data = data;
  int dummy_result;
  flare::parallel_reduce(loop, f_cas_red, dummy_result);
  execution_space().fence();

  return val;
}

template <class T>
T CASLoopSerial(int loop) {
  T* data = new T[1];
  data[0] = 0;

  for (int i = 0; i < loop; i++) {
    T assumed;
    T newval;
    T old;

    do {
      assumed = *data;
      newval  = assumed + (T)1;
      old     = *data;
      *data   = newval;
    } while (!(assumed == old));
  }

  T val = *data;
  delete[] data;

  return val;
}

//----------------------------------------------
//--------------atomic_exchange-----------------
//----------------------------------------------

template <class T, class DEVICE_TYPE>
struct ExchFunctor {
  using execution_space = DEVICE_TYPE;
  using type            = flare::View<T, execution_space>;

  type data, data2;

  FLARE_INLINE_FUNCTION
  void operator()(int i) const {
    T old = flare::atomic_exchange(&data(), (T)i);
    flare::atomic_fetch_add(&data2(), old);
  }
};

template <class T, class DEVICE_TYPE>
struct ExchFunctorReduce {
  using execution_space = DEVICE_TYPE;
  using type            = flare::View<T, execution_space>;

  type data, data2;

  FLARE_INLINE_FUNCTION
  void operator()(int i, int&) const {
    T old = flare::atomic_exchange(&data(), (T)i);
    flare::atomic_fetch_add(&data2(), old);
  }
};

template <class T, class execution_space>
T ExchLoop(int loop) {
  struct ZeroFunctor<T, execution_space> f_zero;
  typename ZeroFunctor<T, execution_space>::type data("Data");
  typename ZeroFunctor<T, execution_space>::h_type h_data("HData");

  f_zero.data = data;
  flare::parallel_for(1, f_zero);
  execution_space().fence();

  typename ZeroFunctor<T, execution_space>::type data2("Data");
  typename ZeroFunctor<T, execution_space>::h_type h_data2("HData");

  f_zero.data = data2;
  flare::parallel_for(1, f_zero);
  execution_space().fence();

  struct ExchFunctor<T, execution_space> f_exch;
  f_exch.data  = data;
  f_exch.data2 = data2;
  flare::parallel_for(loop, f_exch);
  execution_space().fence();

  flare::deep_copy(h_data, data);
  flare::deep_copy(h_data2, data2);
  T val = h_data() + h_data2();

  struct ExchFunctorReduce<T, execution_space> f_exch_red;
  f_exch_red.data  = data;
  f_exch_red.data2 = data2;
  int dummy_result;
  flare::parallel_reduce(loop, f_exch_red, dummy_result);
  execution_space().fence();

  return val;
}

template <class T>
T ExchLoopSerial(std::conditional_t<
                 !std::is_same<T, flare::complex<double> >::value, int, void>
                     loop) {
  T* data  = new T[1];
  T* data2 = new T[1];
  data[0]  = 0;
  data2[0] = 0;

  for (int i = 0; i < loop; i++) {
    T old = *data;
    *data = (T)i;
    *data2 += old;
  }

  T val = *data2 + *data;
  delete[] data;
  delete[] data2;

  return val;
}

template <class T>
T ExchLoopSerial(std::conditional_t<
                 std::is_same<T, flare::complex<double> >::value, int, void>
                     loop) {
  T* data  = new T[1];
  T* data2 = new T[1];
  data[0]  = 0;
  data2[0] = 0;

  for (int i = 0; i < loop; i++) {
    T old        = *data;
    data->real() = (static_cast<double>(i));
    data->imag() = 0;
    *data2 += old;
  }

  T val = *data2 + *data;
  delete[] data;
  delete[] data2;

  return val;
}

template <class T, class DeviceType>
T LoopVariant(int loop, int test) {
  switch (test) {
    case 1: return AddLoop<T, DeviceType>(loop);
    case 2: return CASLoop<T, DeviceType>(loop);
    case 3: return ExchLoop<T, DeviceType>(loop);
  }

  return 0;
}

template <class T>
T LoopVariantSerial(int loop, int test) {
  switch (test) {
    case 1: return AddLoopSerial<T>(loop);
    case 2: return CASLoopSerial<T>(loop);
    case 3: return ExchLoopSerial<T>(loop);
  }

  return 0;
}

template <class T, class DeviceType>
bool Loop(int loop, int test) {
  T res       = LoopVariant<T, DeviceType>(loop, test);
  T resSerial = LoopVariantSerial<T>(loop, test);

  bool passed = true;

  if (resSerial != res) {
    passed = false;

    std::cout << "Loop<" << typeid(T).name() << ">( test = " << test
              << " FAILED : " << resSerial << " != " << res << std::endl;
  }

  return passed;
}

}  // namespace TestAtomic

namespace Test {

TEST_CASE("TEST_CATEGORY, atomics") {
  const int loop_count = 1e4;

  REQUIRE((TestAtomic::Loop<int, TEST_EXECSPACE>(loop_count, 1)));
  REQUIRE((TestAtomic::Loop<int, TEST_EXECSPACE>(loop_count, 2)));
  REQUIRE((TestAtomic::Loop<int, TEST_EXECSPACE>(loop_count, 3)));

  REQUIRE((TestAtomic::Loop<unsigned int, TEST_EXECSPACE>(loop_count, 1)));
  REQUIRE((TestAtomic::Loop<unsigned int, TEST_EXECSPACE>(loop_count, 2)));
  REQUIRE((TestAtomic::Loop<unsigned int, TEST_EXECSPACE>(loop_count, 3)));

  REQUIRE((TestAtomic::Loop<long int, TEST_EXECSPACE>(loop_count, 1)));
  REQUIRE((TestAtomic::Loop<long int, TEST_EXECSPACE>(loop_count, 2)));
  REQUIRE((TestAtomic::Loop<long int, TEST_EXECSPACE>(loop_count, 3)));

  REQUIRE(
      (TestAtomic::Loop<unsigned long int, TEST_EXECSPACE>(loop_count, 1)));
  REQUIRE(
      (TestAtomic::Loop<unsigned long int, TEST_EXECSPACE>(loop_count, 2)));
  REQUIRE(
      (TestAtomic::Loop<unsigned long int, TEST_EXECSPACE>(loop_count, 3)));

  REQUIRE((TestAtomic::Loop<long long int, TEST_EXECSPACE>(loop_count, 1)));
  REQUIRE((TestAtomic::Loop<long long int, TEST_EXECSPACE>(loop_count, 2)));
  REQUIRE((TestAtomic::Loop<long long int, TEST_EXECSPACE>(loop_count, 3)));

  REQUIRE((TestAtomic::Loop<double, TEST_EXECSPACE>(loop_count, 1)));
  REQUIRE((TestAtomic::Loop<double, TEST_EXECSPACE>(loop_count, 2)));
  REQUIRE((TestAtomic::Loop<double, TEST_EXECSPACE>(loop_count, 3)));

  REQUIRE((TestAtomic::Loop<float, TEST_EXECSPACE>(100, 1)));
  REQUIRE((TestAtomic::Loop<float, TEST_EXECSPACE>(100, 2)));
  REQUIRE((TestAtomic::Loop<float, TEST_EXECSPACE>(100, 3)));

  REQUIRE((TestAtomic::Loop<flare::complex<float>, TEST_EXECSPACE>(1, 1)));
  REQUIRE((TestAtomic::Loop<flare::complex<float>, TEST_EXECSPACE>(1, 2)));
  REQUIRE((TestAtomic::Loop<flare::complex<float>, TEST_EXECSPACE>(1, 3)));

  REQUIRE(
      (TestAtomic::Loop<flare::complex<float>, TEST_EXECSPACE>(100, 1)));
  REQUIRE(
      (TestAtomic::Loop<flare::complex<float>, TEST_EXECSPACE>(100, 2)));
  REQUIRE(
      (TestAtomic::Loop<flare::complex<float>, TEST_EXECSPACE>(100, 3)));

  REQUIRE(
      (TestAtomic::Loop<flare::complex<double>, TEST_EXECSPACE>(1, 1)));
  REQUIRE(
      (TestAtomic::Loop<flare::complex<double>, TEST_EXECSPACE>(1, 2)));
  REQUIRE(
      (TestAtomic::Loop<flare::complex<double>, TEST_EXECSPACE>(1, 3)));

  REQUIRE(
      (TestAtomic::Loop<flare::complex<double>, TEST_EXECSPACE>(100, 1)));
  REQUIRE(
      (TestAtomic::Loop<flare::complex<double>, TEST_EXECSPACE>(100, 2)));
  REQUIRE(
      (TestAtomic::Loop<flare::complex<double>, TEST_EXECSPACE>(100, 3)));

// WORKAROUND MSVC
#ifndef _WIN32
  REQUIRE(
      (TestAtomic::Loop<TestAtomic::SuperScalar<4>, TEST_EXECSPACE>(100, 1)));
  REQUIRE(
      (TestAtomic::Loop<TestAtomic::SuperScalar<4>, TEST_EXECSPACE>(100, 2)));
  REQUIRE(
      (TestAtomic::Loop<TestAtomic::SuperScalar<4>, TEST_EXECSPACE>(100, 3)));
#endif
}

// see https://github.com/trilinos/Trilinos/pull/11506
struct TpetraUseCase {
  template <class Scalar>
  struct AbsMaxHelper {
    Scalar value;

    FLARE_FUNCTION AbsMaxHelper& operator+=(AbsMaxHelper const& rhs) {
      Scalar lhs_abs_value = flare::abs(value);
      Scalar rhs_abs_value = flare::abs(rhs.value);
      value = lhs_abs_value > rhs_abs_value ? lhs_abs_value : rhs_abs_value;
      return *this;
    }

    FLARE_FUNCTION AbsMaxHelper operator+(AbsMaxHelper const& rhs) const {
      AbsMaxHelper ret = *this;
      ret += rhs;
      return ret;
    }
  };

  using T = int;
  flare::View<T, TEST_EXECSPACE> d_{"lbl"};
  FLARE_FUNCTION void operator()(int i) const {
    // 0, -1, 2, -3, ...
    auto v_i = static_cast<T>(i);
    if (i % 2 == 1) v_i = -v_i;
    flare::atomic_add(reinterpret_cast<AbsMaxHelper<T>*>(&d_()),
                       AbsMaxHelper<T>{v_i});
  }

  TpetraUseCase() {
    flare::parallel_for(flare::RangePolicy<TEST_EXECSPACE>(0, 10), *this);
  }

  void check() {
    T v;
    flare::deep_copy(v, d_);
    REQUIRE_EQ(v, 9);
  }
};

TEST_CASE("TEST_CATEGORY, atomics_tpetra_max_abs") { TpetraUseCase().check(); }

}  // namespace Test
