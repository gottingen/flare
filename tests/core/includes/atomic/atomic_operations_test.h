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

namespace TestAtomicOperations {

//-----------------------------------------------
//--------------zero_functor---------------------
//-----------------------------------------------

template <class T, class DEVICE_TYPE>
struct ZeroFunctor {
  using execution_space = DEVICE_TYPE;
  using type            = typename flare::View<T, execution_space>;
  using h_type          = typename flare::View<T, execution_space>::HostMirror;

  type data;

  FLARE_INLINE_FUNCTION
  void operator()(int) const { data() = 0; }
};

//-----------------------------------------------
//--------------init_functor---------------------
//-----------------------------------------------

template <class T, class DEVICE_TYPE>
struct InitFunctor {
  using execution_space = DEVICE_TYPE;
  using type            = typename flare::View<T, execution_space>;
  using h_type          = typename flare::View<T, execution_space>::HostMirror;

  type data;
  T init_value;

  FLARE_INLINE_FUNCTION
  void operator()(int) const { data() = init_value; }

  InitFunctor(T _init_value) : init_value(_init_value) {}
};

//---------------------------------------------------
//--------------atomic_load/store/assign---------------------
//---------------------------------------------------
template <class T, class DEVICE_TYPE>
struct LoadStoreFunctor {
  using execution_space = DEVICE_TYPE;
  using type            = flare::View<T, execution_space>;

  type data;
  T i0;
  T i1;

  FLARE_INLINE_FUNCTION
  void operator()(int) const {
    T old = flare::atomic_load(&data());
    if (old != i0)
      flare::abort("flare Atomic Load didn't get the right value");
    flare::atomic_store(&data(), i1);
    flare::atomic_assign(&data(), old);
  }
  LoadStoreFunctor(T _i0, T _i1) : i0(_i0), i1(_i1) {}
};

template <class T, class DeviceType>
bool LoadStoreAtomicTest(T i0, T i1) {
  using execution_space = typename DeviceType::execution_space;
  struct InitFunctor<T, execution_space> f_init(i0);
  typename InitFunctor<T, execution_space>::type data("Data");
  typename InitFunctor<T, execution_space>::h_type h_data("HData");

  f_init.data = data;
  flare::parallel_for(1, f_init);
  execution_space().fence();

  struct LoadStoreFunctor<T, execution_space> f(i0, i1);

  f.data = data;
  flare::parallel_for(1, f);

  flare::deep_copy(h_data, data);

  return h_data() == i0;
}

//---------------------------------------------------
//--------------atomic_fetch_max---------------------
//---------------------------------------------------

template <class T, class DEVICE_TYPE>
struct MaxFunctor {
  using execution_space = DEVICE_TYPE;
  using type            = flare::View<T, execution_space>;

  type data;
  T i0;
  T i1;

  FLARE_INLINE_FUNCTION
  void operator()(int) const {
    // flare::atomic_fetch_max( &data(), (T) 1 );
    flare::atomic_fetch_max(&data(), (T)i1);
  }
  MaxFunctor(T _i0, T _i1) : i0(_i0), i1(_i1) {}
};

template <class T, class execution_space>
T MaxAtomic(T i0, T i1) {
  struct InitFunctor<T, execution_space> f_init(i0);
  typename InitFunctor<T, execution_space>::type data("Data");
  typename InitFunctor<T, execution_space>::h_type h_data("HData");

  f_init.data = data;
  flare::parallel_for(1, f_init);
  execution_space().fence();

  struct MaxFunctor<T, execution_space> f(i0, i1);

  f.data = data;
  flare::parallel_for(1, f);
  execution_space().fence();

  flare::deep_copy(h_data, data);
  T val = h_data();

  return val;
}

template <class T>
T MaxAtomicCheck(T i0, T i1) {
  T* data = new T[1];
  data[0] = 0;

  *data = (i0 > i1 ? i0 : i1);

  T val = *data;
  delete[] data;

  return val;
}

template <class T, class DeviceType>
bool MaxAtomicTest(T i0, T i1) {
  T res       = MaxAtomic<T, DeviceType>(i0, i1);
  T resSerial = MaxAtomicCheck<T>(i0, i1);

  bool passed = true;

  if (resSerial != res) {
    passed = false;

    std::cout << "Loop<" << typeid(T).name() << ">( test = MaxAtomicTest"
              << " FAILED : " << resSerial << " != " << res << std::endl;
  }

  return passed;
}

//---------------------------------------------------
//--------------atomic_fetch_min---------------------
//---------------------------------------------------

template <class T, class DEVICE_TYPE>
struct MinFunctor {
  using execution_space = DEVICE_TYPE;
  using type            = flare::View<T, execution_space>;

  type data;
  T i0;
  T i1;

  FLARE_INLINE_FUNCTION
  void operator()(int) const { flare::atomic_fetch_min(&data(), (T)i1); }

  MinFunctor(T _i0, T _i1) : i0(_i0), i1(_i1) {}
};

template <class T, class execution_space>
T MinAtomic(T i0, T i1) {
  struct InitFunctor<T, execution_space> f_init(i0);
  typename InitFunctor<T, execution_space>::type data("Data");
  typename InitFunctor<T, execution_space>::h_type h_data("HData");

  f_init.data = data;
  flare::parallel_for(1, f_init);
  execution_space().fence();

  struct MinFunctor<T, execution_space> f(i0, i1);

  f.data = data;
  flare::parallel_for(1, f);
  execution_space().fence();

  flare::deep_copy(h_data, data);
  T val = h_data();

  return val;
}

template <class T>
T MinAtomicCheck(T i0, T i1) {
  T* data = new T[1];
  data[0] = 0;

  *data = (i0 < i1 ? i0 : i1);

  T val = *data;
  delete[] data;

  return val;
}

template <class T, class DeviceType>
bool MinAtomicTest(T i0, T i1) {
  T res       = MinAtomic<T, DeviceType>(i0, i1);
  T resSerial = MinAtomicCheck<T>(i0, i1);

  bool passed = true;

  if (resSerial != res) {
    passed = false;

    std::cout << "Loop<" << typeid(T).name() << ">( test = MinAtomicTest"
              << " FAILED : " << resSerial << " != " << res << std::endl;
  }

  return passed;
}

//---------------------------------------------------
//--------------atomic_increment---------------------
//---------------------------------------------------

template <class T, class DEVICE_TYPE>
struct IncFunctor {
  using execution_space = DEVICE_TYPE;
  using type            = flare::View<T, execution_space>;

  type data;
  T i0;

  FLARE_INLINE_FUNCTION
  void operator()(int) const { flare::atomic_increment(&data()); }

  IncFunctor(T _i0) : i0(_i0) {}
};

template <class T, class execution_space>
T IncAtomic(T i0) {
  struct InitFunctor<T, execution_space> f_init(i0);
  typename InitFunctor<T, execution_space>::type data("Data");
  typename InitFunctor<T, execution_space>::h_type h_data("HData");

  f_init.data = data;
  flare::parallel_for(1, f_init);
  execution_space().fence();

  struct IncFunctor<T, execution_space> f(i0);

  f.data = data;
  flare::parallel_for(1, f);
  execution_space().fence();

  flare::deep_copy(h_data, data);
  T val = h_data();

  return val;
}

template <class T>
T IncAtomicCheck(T i0) {
  T* data = new T[1];
  data[0] = 0;

  *data = i0 + 1;

  T val = *data;
  delete[] data;

  return val;
}

template <class T, class DeviceType>
bool IncAtomicTest(T i0) {
  T res       = IncAtomic<T, DeviceType>(i0);
  T resSerial = IncAtomicCheck<T>(i0);

  bool passed = true;

  if (resSerial != res) {
    passed = false;

    std::cout << "Loop<" << typeid(T).name() << ">( test = IncAtomicTest"
              << " FAILED : " << resSerial << " != " << res << std::endl;
  }

  return passed;
}

//---------------------------------------------------
//-------------atomic_wrapping_increment-------------
//---------------------------------------------------

template <class T, class DEVICE_TYPE>
struct WrappingIncFunctor {
  using execution_space = DEVICE_TYPE;
  using type            = flare::View<T, execution_space>;

  type data;
  T i0;
  T i1;

  FLARE_INLINE_FUNCTION
  void operator()(int) const {
    flare::atomic_fetch_inc_mod(&data(), (T)i1, flare::MemoryOrderRelaxed(),
                                flare::MemoryScopeDevice());
  }

  WrappingIncFunctor(T _i0, T _i1) : i0(_i0), i1(_i1) {}
};

template <class T, class execution_space>
T WrappingIncAtomic(T i0, T i1) {
  struct InitFunctor<T, execution_space> f_init(i0);
  typename InitFunctor<T, execution_space>::type data("Data");
  typename InitFunctor<T, execution_space>::h_type h_data("HData");

  f_init.data = data;
  flare::parallel_for(1, f_init);
  execution_space().fence();

  struct WrappingIncFunctor<T, execution_space> f(i0, i1);

  f.data = data;
  flare::parallel_for(1, f);
  execution_space().fence();

  flare::deep_copy(h_data, data);
  T val = h_data();

  return val;
}

template <class T>
T WrappingIncAtomicCheck(T i0, T i1) {
  T* data = new T[1];
  data[0] = 0;

  // Wraps to 0 when i0 >= i1
  *data = ((i0 >= i1) ? (T)0 : i0 + (T)1);

  T val = *data;
  delete[] data;

  return val;
}

template <class T, class DeviceType>
bool WrappingIncAtomicTest(T i0, T i1) {
  T res       = WrappingIncAtomic<T, DeviceType>(i0, i1);
  T resSerial = WrappingIncAtomicCheck<T>(i0, i1);

  bool passed = true;

  if (resSerial != res) {
    passed = false;

    std::cout << "Loop<" << typeid(T).name()
              << ">( test = WrappingIncAtomicTest"
              << " FAILED : " << resSerial << " != " << res << std::endl;
  }

  return passed;
}

//---------------------------------------------------
//--------------atomic_decrement---------------------
//---------------------------------------------------

template <class T, class DEVICE_TYPE>
struct DecFunctor {
  using execution_space = DEVICE_TYPE;
  using type            = flare::View<T, execution_space>;

  type data;
  T i0;

  FLARE_INLINE_FUNCTION
  void operator()(int) const { flare::atomic_decrement(&data()); }

  DecFunctor(T _i0) : i0(_i0) {}
};

template <class T, class execution_space>
T DecAtomic(T i0) {
  struct InitFunctor<T, execution_space> f_init(i0);
  typename InitFunctor<T, execution_space>::type data("Data");
  typename InitFunctor<T, execution_space>::h_type h_data("HData");

  f_init.data = data;
  flare::parallel_for(1, f_init);
  execution_space().fence();

  struct DecFunctor<T, execution_space> f(i0);

  f.data = data;
  flare::parallel_for(1, f);
  execution_space().fence();

  flare::deep_copy(h_data, data);
  T val = h_data();

  return val;
}

template <class T>
T DecAtomicCheck(T i0) {
  T* data = new T[1];
  data[0] = 0;

  *data = i0 - 1;

  T val = *data;
  delete[] data;

  return val;
}

template <class T, class DeviceType>
bool DecAtomicTest(T i0) {
  T res       = DecAtomic<T, DeviceType>(i0);
  T resSerial = DecAtomicCheck<T>(i0);

  bool passed = true;

  if (resSerial != res) {
    passed = false;

    std::cout << "Loop<" << typeid(T).name() << ">( test = DecAtomicTest"
              << " FAILED : " << resSerial << " != " << res << std::endl;
  }

  return passed;
}

//---------------------------------------------------
//-------------atomic_wrapping_decrement-------------
//---------------------------------------------------

template <class T, class DEVICE_TYPE>
struct WrappingDecFunctor {
  using execution_space = DEVICE_TYPE;
  using type            = flare::View<T, execution_space>;

  type data;
  T i0;
  T i1;

  FLARE_INLINE_FUNCTION
  void operator()(int) const {
    flare::atomic_fetch_dec_mod(&data(), (T)i1, flare::MemoryOrderRelaxed(),
                                flare::MemoryScopeDevice());
  }

  WrappingDecFunctor(T _i0, T _i1) : i0(_i0), i1(_i1) {}
};

template <class T, class execution_space>
T WrappingDecAtomic(T i0, T i1) {
  struct InitFunctor<T, execution_space> f_init(i0);
  typename InitFunctor<T, execution_space>::type data("Data");
  typename InitFunctor<T, execution_space>::h_type h_data("HData");

  f_init.data = data;
  flare::parallel_for(1, f_init);
  execution_space().fence();

  struct WrappingDecFunctor<T, execution_space> f(i0, i1);

  f.data = data;
  flare::parallel_for(1, f);
  execution_space().fence();

  flare::deep_copy(h_data, data);
  T val = h_data();

  return val;
}

template <class T>
T WrappingDecAtomicCheck(T i0, T i1) {
  T* data = new T[1];
  data[0] = 0;

  // Wraps to i1 when i0 <= 0
  // i0 should never be negative
  *data = ((i0 <= (T)0) ? i1 : i0 - (T)1);

  T val = *data;
  delete[] data;

  return val;
}

template <class T, class DeviceType>
bool WrappingDecAtomicTest(T i0, T i1) {
  T res       = WrappingDecAtomic<T, DeviceType>(i0, i1);
  T resSerial = WrappingDecAtomicCheck<T>(i0, i1);

  bool passed = true;

  if (resSerial != res) {
    passed = false;

    std::cout << "Loop<" << typeid(T).name()
              << ">( test = WrappingDecAtomicTest"
              << " FAILED : " << resSerial << " != " << res << std::endl;
  }

  return passed;
}

//---------------------------------------------------
//--------------atomic_fetch_mul---------------------
//---------------------------------------------------

template <class T, class DEVICE_TYPE>
struct MulFunctor {
  using execution_space = DEVICE_TYPE;
  using type            = flare::View<T, execution_space>;

  type data;
  T i0;
  T i1;

  FLARE_INLINE_FUNCTION
  void operator()(int) const { flare::atomic_fetch_mul(&data(), (T)i1); }

  MulFunctor(T _i0, T _i1) : i0(_i0), i1(_i1) {}
};

template <class T, class execution_space>
T MulAtomic(T i0, T i1) {
  struct InitFunctor<T, execution_space> f_init(i0);
  typename InitFunctor<T, execution_space>::type data("Data");
  typename InitFunctor<T, execution_space>::h_type h_data("HData");

  f_init.data = data;
  flare::parallel_for(1, f_init);
  execution_space().fence();

  struct MulFunctor<T, execution_space> f(i0, i1);

  f.data = data;
  flare::parallel_for(1, f);
  execution_space().fence();

  flare::deep_copy(h_data, data);
  T val = h_data();

  return val;
}

template <class T>
T MulAtomicCheck(T i0, T i1) {
  T* data = new T[1];
  data[0] = 0;

  *data = i0 * i1;

  T val = *data;
  delete[] data;

  return val;
}

template <class T, class DeviceType>
bool MulAtomicTest(T i0, T i1) {
  T res       = MulAtomic<T, DeviceType>(i0, i1);
  T resSerial = MulAtomicCheck<T>(i0, i1);

  bool passed = true;

  if (resSerial != res) {
    passed = false;

    std::cout << "Loop<" << typeid(T).name() << ">( test = MulAtomicTest"
              << " FAILED : " << resSerial << " != " << res << std::endl;
  }

  return passed;
}

//---------------------------------------------------
//--------------atomic_fetch_div---------------------
//---------------------------------------------------

template <class T, class DEVICE_TYPE>
struct DivFunctor {
  using execution_space = DEVICE_TYPE;
  using type            = flare::View<T, execution_space>;

  type data;
  T i0;
  T i1;

  FLARE_INLINE_FUNCTION
  void operator()(int) const { flare::atomic_fetch_div(&data(), (T)i1); }

  DivFunctor(T _i0, T _i1) : i0(_i0), i1(_i1) {}
};

template <class T, class execution_space>
T DivAtomic(T i0, T i1) {
  struct InitFunctor<T, execution_space> f_init(i0);
  typename InitFunctor<T, execution_space>::type data("Data");
  typename InitFunctor<T, execution_space>::h_type h_data("HData");

  f_init.data = data;
  flare::parallel_for(1, f_init);
  execution_space().fence();

  struct DivFunctor<T, execution_space> f(i0, i1);

  f.data = data;
  flare::parallel_for(1, f);
  execution_space().fence();

  flare::deep_copy(h_data, data);
  T val = h_data();

  return val;
}

template <class T>
T DivAtomicCheck(T i0, T i1) {
  T* data = new T[1];
  data[0] = 0;

  *data = i0 / i1;

  T val = *data;
  delete[] data;

  return val;
}

template <class T, class DeviceType>
bool DivAtomicTest(T i0, T i1) {
  T res       = DivAtomic<T, DeviceType>(i0, i1);
  T resSerial = DivAtomicCheck<T>(i0, i1);

  bool passed = true;

  using flare::abs;
  if (abs((resSerial - res) * 1.) > 1e-5) {
    passed = false;

    std::cout << "Loop<" << typeid(T).name() << ">( test = DivAtomicTest"
              << " FAILED : " << resSerial << " != " << res << std::endl;
  }

  return passed;
}

//---------------------------------------------------
//--------------atomic_fetch_mod---------------------
//---------------------------------------------------

template <class T, class DEVICE_TYPE>
struct ModFunctor {
  using execution_space = DEVICE_TYPE;
  using type            = flare::View<T, execution_space>;

  type data;
  T i0;
  T i1;

  FLARE_INLINE_FUNCTION
  void operator()(int) const { flare::atomic_fetch_mod(&data(), (T)i1); }

  ModFunctor(T _i0, T _i1) : i0(_i0), i1(_i1) {}
};

template <class T, class execution_space>
T ModAtomic(T i0, T i1) {
  struct InitFunctor<T, execution_space> f_init(i0);
  typename InitFunctor<T, execution_space>::type data("Data");
  typename InitFunctor<T, execution_space>::h_type h_data("HData");

  f_init.data = data;
  flare::parallel_for(1, f_init);
  execution_space().fence();

  struct ModFunctor<T, execution_space> f(i0, i1);

  f.data = data;
  flare::parallel_for(1, f);
  execution_space().fence();

  flare::deep_copy(h_data, data);
  T val = h_data();

  return val;
}

template <class T>
T ModAtomicCheck(T i0, T i1) {
  T* data = new T[1];
  data[0] = 0;

  *data = i0 % i1;

  T val = *data;
  delete[] data;

  return val;
}

template <class T, class DeviceType>
bool ModAtomicTest(T i0, T i1) {
  T res       = ModAtomic<T, DeviceType>(i0, i1);
  T resSerial = ModAtomicCheck<T>(i0, i1);

  bool passed = true;

  if (resSerial != res) {
    passed = false;

    std::cout << "Loop<" << typeid(T).name() << ">( test = ModAtomicTest"
              << " FAILED : " << resSerial << " != " << res << std::endl;
  }

  return passed;
}

//---------------------------------------------------
//--------------atomic_fetch_and---------------------
//---------------------------------------------------

template <class T, class DEVICE_TYPE>
struct AndFunctor {
  using execution_space = DEVICE_TYPE;
  using type            = flare::View<T, execution_space>;

  type data;
  T i0;
  T i1;

  FLARE_INLINE_FUNCTION
  void operator()(int) const {
    T result = flare::atomic_fetch_and(&data(), (T)i1);
    flare::atomic_and(&data(), result);
  }

  AndFunctor(T _i0, T _i1) : i0(_i0), i1(_i1) {}
};

template <class T, class execution_space>
T AndAtomic(T i0, T i1) {
  struct InitFunctor<T, execution_space> f_init(i0);
  typename InitFunctor<T, execution_space>::type data("Data");
  typename InitFunctor<T, execution_space>::h_type h_data("HData");

  f_init.data = data;
  flare::parallel_for(1, f_init);
  execution_space().fence();

  struct AndFunctor<T, execution_space> f(i0, i1);

  f.data = data;
  flare::parallel_for(1, f);
  execution_space().fence();

  flare::deep_copy(h_data, data);
  T val = h_data();

  return val;
}

template <class T>
T AndAtomicCheck(T i0, T i1) {
  T* data = new T[1];
  data[0] = 0;

  *data = i0 & i1;

  T val = *data;
  delete[] data;

  return val;
}

template <class T, class DeviceType>
bool AndAtomicTest(T i0, T i1) {
  T res       = AndAtomic<T, DeviceType>(i0, i1);
  T resSerial = AndAtomicCheck<T>(i0, i1);

  bool passed = true;

  if (resSerial != res) {
    passed = false;

    std::cout << "Loop<" << typeid(T).name() << ">( test = AndAtomicTest"
              << " FAILED : " << resSerial << " != " << res << std::endl;
  }

  return passed;
}

//---------------------------------------------------
//--------------atomic_fetch_or----------------------
//---------------------------------------------------

template <class T, class DEVICE_TYPE>
struct OrFunctor {
  using execution_space = DEVICE_TYPE;
  using type            = flare::View<T, execution_space>;

  type data;
  T i0;
  T i1;

  FLARE_INLINE_FUNCTION
  void operator()(int) const {
    T result = flare::atomic_fetch_or(&data(), (T)i1);
    flare::atomic_or(&data(), result);
  }

  OrFunctor(T _i0, T _i1) : i0(_i0), i1(_i1) {}
};

template <class T, class execution_space>
T OrAtomic(T i0, T i1) {
  struct InitFunctor<T, execution_space> f_init(i0);
  typename InitFunctor<T, execution_space>::type data("Data");
  typename InitFunctor<T, execution_space>::h_type h_data("HData");

  f_init.data = data;
  flare::parallel_for(1, f_init);
  execution_space().fence();

  struct OrFunctor<T, execution_space> f(i0, i1);

  f.data = data;
  flare::parallel_for(1, f);
  execution_space().fence();

  flare::deep_copy(h_data, data);
  T val = h_data();

  return val;
}

template <class T>
T OrAtomicCheck(T i0, T i1) {
  T* data = new T[1];
  data[0] = 0;

  *data = i0 | i1;

  T val = *data;
  delete[] data;

  return val;
}

template <class T, class DeviceType>
bool OrAtomicTest(T i0, T i1) {
  T res       = OrAtomic<T, DeviceType>(i0, i1);
  T resSerial = OrAtomicCheck<T>(i0, i1);

  bool passed = true;

  if (resSerial != res) {
    passed = false;

    std::cout << "Loop<" << typeid(T).name() << ">( test = OrAtomicTest"
              << " FAILED : " << resSerial << " != " << res << std::endl;
  }

  return passed;
}

//---------------------------------------------------
//--------------atomic_fetch_xor---------------------
//---------------------------------------------------

template <class T, class DEVICE_TYPE>
struct XorFunctor {
  using execution_space = DEVICE_TYPE;
  using type            = flare::View<T, execution_space>;

  type data;
  T i0;
  T i1;

  FLARE_INLINE_FUNCTION
  void operator()(int) const { flare::atomic_fetch_xor(&data(), (T)i1); }

  XorFunctor(T _i0, T _i1) : i0(_i0), i1(_i1) {}
};

template <class T, class execution_space>
T XorAtomic(T i0, T i1) {
  struct InitFunctor<T, execution_space> f_init(i0);
  typename InitFunctor<T, execution_space>::type data("Data");
  typename InitFunctor<T, execution_space>::h_type h_data("HData");

  f_init.data = data;
  flare::parallel_for(1, f_init);
  execution_space().fence();

  struct XorFunctor<T, execution_space> f(i0, i1);

  f.data = data;
  flare::parallel_for(1, f);
  execution_space().fence();

  flare::deep_copy(h_data, data);
  T val = h_data();

  return val;
}

template <class T>
T XorAtomicCheck(T i0, T i1) {
  T* data = new T[1];
  data[0] = 0;

  *data = i0 ^ i1;

  T val = *data;
  delete[] data;

  return val;
}

template <class T, class DeviceType>
bool XorAtomicTest(T i0, T i1) {
  T res       = XorAtomic<T, DeviceType>(i0, i1);
  T resSerial = XorAtomicCheck<T>(i0, i1);

  bool passed = true;

  if (resSerial != res) {
    passed = false;

    std::cout << "Loop<" << typeid(T).name() << ">( test = XorAtomicTest"
              << " FAILED : " << resSerial << " != " << res << std::endl;
  }

  return passed;
}

//---------------------------------------------------
//--------------atomic_fetch_lshift---------------------
//---------------------------------------------------

template <class T, class DEVICE_TYPE>
struct LShiftFunctor {
  using execution_space = DEVICE_TYPE;
  using type            = flare::View<T, execution_space>;

  type data;
  T i0;
  T i1;

  FLARE_INLINE_FUNCTION
  void operator()(int) const { flare::atomic_fetch_lshift(&data(), (T)i1); }

  LShiftFunctor(T _i0, T _i1) : i0(_i0), i1(_i1) {}
};

template <class T, class execution_space>
T LShiftAtomic(T i0, T i1) {
  struct InitFunctor<T, execution_space> f_init(i0);
  typename InitFunctor<T, execution_space>::type data("Data");
  typename InitFunctor<T, execution_space>::h_type h_data("HData");

  f_init.data = data;
  flare::parallel_for(1, f_init);
  execution_space().fence();

  struct LShiftFunctor<T, execution_space> f(i0, i1);

  f.data = data;
  flare::parallel_for(1, f);
  execution_space().fence();

  flare::deep_copy(h_data, data);
  T val = h_data();

  return val;
}

template <class T>
T LShiftAtomicCheck(T i0, T i1) {
  T* data = new T[1];
  data[0] = 0;

  *data = i0 << i1;

  T val = *data;
  delete[] data;

  return val;
}

template <class T, class DeviceType>
bool LShiftAtomicTest(T i0, T i1) {
  T res       = LShiftAtomic<T, DeviceType>(i0, i1);
  T resSerial = LShiftAtomicCheck<T>(i0, i1);

  bool passed = true;

  if (resSerial != res) {
    passed = false;

    std::cout << "Loop<" << typeid(T).name() << ">( test = LShiftAtomicTest"
              << " FAILED : " << resSerial << " != " << res << std::endl;
  }

  return passed;
}

//---------------------------------------------------
//--------------atomic_fetch_rshift---------------------
//---------------------------------------------------

template <class T, class DEVICE_TYPE>
struct RShiftFunctor {
  using execution_space = DEVICE_TYPE;
  using type            = flare::View<T, execution_space>;

  type data;
  T i0;
  T i1;

  FLARE_INLINE_FUNCTION
  void operator()(int) const { flare::atomic_fetch_rshift(&data(), (T)i1); }

  RShiftFunctor(T _i0, T _i1) : i0(_i0), i1(_i1) {}
};

template <class T, class execution_space>
T RShiftAtomic(T i0, T i1) {
  struct InitFunctor<T, execution_space> f_init(i0);
  typename InitFunctor<T, execution_space>::type data("Data");
  typename InitFunctor<T, execution_space>::h_type h_data("HData");

  f_init.data = data;
  flare::parallel_for(1, f_init);
  execution_space().fence();

  struct RShiftFunctor<T, execution_space> f(i0, i1);

  f.data = data;
  flare::parallel_for(1, f);
  execution_space().fence();

  flare::deep_copy(h_data, data);
  T val = h_data();

  return val;
}

template <class T>
T RShiftAtomicCheck(T i0, T i1) {
  T* data = new T[1];
  data[0] = 0;

  *data = i0 >> i1;

  T val = *data;
  delete[] data;

  return val;
}

template <class T, class DeviceType>
bool RShiftAtomicTest(T i0, T i1) {
  T res       = RShiftAtomic<T, DeviceType>(i0, i1);
  T resSerial = RShiftAtomicCheck<T>(i0, i1);

  bool passed = true;

  if (resSerial != res) {
    passed = false;

    std::cout << "Loop<" << typeid(T).name() << ">( test = RShiftAtomicTest"
              << " FAILED : " << resSerial << " != " << res << std::endl;
  }

  return passed;
}

//---------------------------------------------------
//--------------atomic_test_control------------------
//---------------------------------------------------

template <class T, class DeviceType>
bool AtomicOperationsTestIntegralType(int i0, int i1, int test) {
  switch (test) {
    case 1: return MaxAtomicTest<T, DeviceType>((T)i0, (T)i1);
    case 2: return MinAtomicTest<T, DeviceType>((T)i0, (T)i1);
    case 3: return MulAtomicTest<T, DeviceType>((T)i0, (T)i1);
    case 4: return DivAtomicTest<T, DeviceType>((T)i0, (T)i1);
    case 5: return ModAtomicTest<T, DeviceType>((T)i0, (T)i1);
    case 6: return AndAtomicTest<T, DeviceType>((T)i0, (T)i1);
    case 7: return OrAtomicTest<T, DeviceType>((T)i0, (T)i1);
    case 8: return XorAtomicTest<T, DeviceType>((T)i0, (T)i1);
    case 9: return LShiftAtomicTest<T, DeviceType>((T)i0, (T)i1);
    case 10: return RShiftAtomicTest<T, DeviceType>((T)i0, (T)i1);
    case 11: return IncAtomicTest<T, DeviceType>((T)i0);
    case 12: return DecAtomicTest<T, DeviceType>((T)i0);
    case 13: return LoadStoreAtomicTest<T, DeviceType>((T)i0, (T)i1);
  }

  return 0;
}

template <class T, class DeviceType>
bool AtomicOperationsTestUnsignedIntegralType(int i0, int i1, int test) {
  switch (test) {
    case 1: return WrappingIncAtomicTest<T, DeviceType>((T)i0, (T)i1);
    case 2: return WrappingDecAtomicTest<T, DeviceType>((T)i0, (T)i1);
  }

  return 0;
}

template <class T, class DeviceType>
bool AtomicOperationsTestNonIntegralType(int i0, int i1, int test) {
  switch (test) {
    case 1: return MaxAtomicTest<T, DeviceType>((T)i0, (T)i1);
    case 2: return MinAtomicTest<T, DeviceType>((T)i0, (T)i1);
    case 3: return MulAtomicTest<T, DeviceType>((T)i0, (T)i1);
    case 4: return DivAtomicTest<T, DeviceType>((T)i0, (T)i1);
    case 5: return LoadStoreAtomicTest<T, DeviceType>((T)i0, (T)i1);
  }

  return 0;
}

}  // namespace TestAtomicOperations
