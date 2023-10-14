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
namespace TestAtomicTensors {

//-------------------------------------------------
//-----------atomic tensor api tests-----------------
//-------------------------------------------------

template <class T, class... P>
size_t allocation_count(const flare::Tensor<T, P...>& tensor) {
  const size_t card  = tensor.size();
  const size_t alloc = tensor.span();

  const int memory_span = flare::Tensor<int*>::required_allocation_size(100);

  return (card <= alloc && memory_span == 400) ? alloc : 0;
}

template <class DataType, class DeviceType,
          unsigned Rank = flare::TensorTraits<DataType>::rank>
struct TestTensorOperator_LeftAndRight;

template <class DataType, class DeviceType>
struct TestTensorOperator_LeftAndRight<DataType, DeviceType, 1> {
  using execution_space = typename DeviceType::execution_space;
  using memory_space    = typename DeviceType::memory_space;
  using size_type       = typename execution_space::size_type;

  using value_type = int;

  FLARE_INLINE_FUNCTION
  static void join(value_type& update, const value_type& input) {
    update |= input;
  }

  FLARE_INLINE_FUNCTION
  static void init(value_type& update) { update = 0; }

  using left_tensor = flare::Tensor<DataType, flare::LayoutLeft, execution_space,
                                 flare::MemoryTraits<flare::Atomic> >;

  using right_tensor =
      flare::Tensor<DataType, flare::LayoutRight, execution_space,
                   flare::MemoryTraits<flare::Atomic> >;

  using stride_tensor =
      flare::Tensor<DataType, flare::LayoutStride, execution_space,
                   flare::MemoryTraits<flare::Atomic> >;

  left_tensor left;
  right_tensor right;
  stride_tensor left_stride;
  stride_tensor right_stride;
  int64_t left_alloc;
  int64_t right_alloc;

  TestTensorOperator_LeftAndRight()
      : left("left"),
        right("right"),
        left_stride(left),
        right_stride(right),
        left_alloc(allocation_count(left)),
        right_alloc(allocation_count(right)) {}

  static void testit() {
    TestTensorOperator_LeftAndRight driver;

    int error_flag = 0;

    flare::parallel_reduce(1, driver, error_flag);

    REQUIRE_EQ(error_flag, 0);
  }

  FLARE_INLINE_FUNCTION
  void operator()(const size_type, value_type& update) const {
    for (unsigned i0 = 0; i0 < unsigned(left.extent(0)); ++i0) {
      // Below checks that values match, but unable to check the references.
      // Should this be able to be checked?
      if (left(i0) != left.access(i0, 0, 0, 0, 0, 0, 0, 0)) {
        update |= 3;
      }
      if (right(i0) != right.access(i0, 0, 0, 0, 0, 0, 0, 0)) {
        update |= 3;
      }
      if (left(i0) != left_stride(i0)) {
        update |= 4;
      }
      if (right(i0) != right_stride(i0)) {
        update |= 8;
      }
      /*
            if ( &left( i0 )  != &left( i0, 0, 0, 0, 0, 0, 0, 0 ) )  { update |=
         3; } if ( &right( i0 ) != &right( i0, 0, 0, 0, 0, 0, 0, 0 ) ) { update
         |= 3; } if ( &left( i0 )  != &left_stride( i0 ) ) { update |= 4; } if (
         &right( i0 ) != &right_stride( i0 ) ) { update |= 8; }
      */
    }
  }
};

template <typename T, class DeviceType>
class TestAtomicTensorAPI {
 public:
  using device = DeviceType;

  enum { N0 = 1000, N1 = 3, N2 = 5, N3 = 7 };

  using DTensor0           = flare::Tensor<T, device>;
  using DTensor1           = flare::Tensor<T*, device>;
  using DTensor2           = flare::Tensor<T * [N1], device>;
  using DTensor3           = flare::Tensor<T * [N1][N2], device>;
  using DTensor4           = flare::Tensor<T * [N1][N2][N3], device>;
  using const_DTensor4     = flare::Tensor<const T * [N1][N2][N3], device>;
  using DTensor4_unmanaged = flare::Tensor<T****, device, flare::MemoryUnmanaged>;
  using host             = typename DTensor0::host_mirror_space;

  using aTensor0 = flare::Tensor<T, device, flare::MemoryTraits<flare::Atomic> >;
  using aTensor1 =
      flare::Tensor<T*, device, flare::MemoryTraits<flare::Atomic> >;
  using aTensor2 =
      flare::Tensor<T * [N1], device, flare::MemoryTraits<flare::Atomic> >;
  using aTensor3 =
      flare::Tensor<T * [N1][N2], device, flare::MemoryTraits<flare::Atomic> >;
  using aTensor4       = flare::Tensor<T * [N1][N2][N3], device,
                              flare::MemoryTraits<flare::Atomic> >;
  using const_aTensor4 = flare::Tensor<const T * [N1][N2][N3], device,
                                    flare::MemoryTraits<flare::Atomic> >;

  using aTensor4_unmanaged =
      flare::Tensor<T****, device,
                   flare::MemoryTraits<flare::Unmanaged | flare::Atomic> >;

  using host_atomic = typename aTensor0::host_mirror_space;

  TestAtomicTensorAPI() {
    TestTensorOperator_LeftAndRight<int[2], device>::testit();
    run_test_rank0();
    run_test_rank4();
    run_test_const();
  }

  static void run_test_rank0() {
    DTensor0 dx, dy;
    aTensor0 ax, ay, az;

    dx = DTensor0("dx");
    dy = DTensor0("dy");
    REQUIRE_EQ(dx.use_count(), 1);
    REQUIRE_EQ(dy.use_count(), 1);

    ax = dx;
    ay = dy;
    REQUIRE_EQ(dx.use_count(), 2);
    REQUIRE_EQ(dy.use_count(), 2);
    REQUIRE_EQ(dx.use_count(), ax.use_count());

    az = ax;
    REQUIRE_EQ(dx.use_count(), 3);
    REQUIRE_EQ(ax.use_count(), 3);
    REQUIRE_EQ(az.use_count(), 3);
    REQUIRE_EQ(az.use_count(), ax.use_count());
  }

  static void run_test_rank4() {
    DTensor4 dx, dy;
    aTensor4 ax, ay, az;

    dx = DTensor4("dx", N0);
    dy = DTensor4("dy", N0);
    REQUIRE_EQ(dx.use_count(), 1);
    REQUIRE_EQ(dy.use_count(), 1);

    ax = dx;
    ay = dy;
    REQUIRE_EQ(dx.use_count(), 2);
    REQUIRE_EQ(dy.use_count(), 2);
    REQUIRE_EQ(dx.use_count(), ax.use_count());

    DTensor4_unmanaged unmanaged_dx = dx;
    REQUIRE_EQ(dx.use_count(), 2);

    az = ax;
    REQUIRE_EQ(dx.use_count(), 3);
    REQUIRE_EQ(ax.use_count(), 3);
    REQUIRE_EQ(az.use_count(), 3);
    REQUIRE_EQ(az.use_count(), ax.use_count());

    aTensor4_unmanaged unmanaged_ax = ax;
    REQUIRE_EQ(ax.use_count(), 3);

    aTensor4_unmanaged unmanaged_ax_from_ptr_dx = aTensor4_unmanaged(
        dx.data(), dx.extent(0), dx.extent(1), dx.extent(2), dx.extent(3));
    REQUIRE_EQ(ax.use_count(), 3);

    const_aTensor4 const_ax = ax;
    REQUIRE_EQ(ax.use_count(), 4);
    REQUIRE_EQ(const_ax.use_count(), ax.use_count());

    REQUIRE_NE(ax.data(), nullptr);
    REQUIRE_NE(const_ax.data(), nullptr);  // referenceable ptr
    REQUIRE_NE(unmanaged_ax.data(), nullptr);
    REQUIRE_NE(unmanaged_ax_from_ptr_dx.data(), nullptr);
    REQUIRE_NE(ay.data(), nullptr);
    //    REQUIRE_NE( ax, ay );
    //    Above test results in following runtime error from gtest:
    //    Expected: (ax) != (ay), actual: 32-byte object <30-01 D0-A0 D8-7F
    //    00-00 00-31 44-0C 01-00 00-00 E8-03 00-00 00-00 00-00 69-00 00-00
    //    00-00 00-00> vs 32-byte object <80-01 D0-A0 D8-7F 00-00 00-A1 4A-0C
    //    01-00 00-00 E8-03 00-00 00-00 00-00 69-00 00-00 00-00 00-00>

    REQUIRE_EQ(ax.extent(0), unsigned(N0));
    REQUIRE_EQ(ax.extent(1), unsigned(N1));
    REQUIRE_EQ(ax.extent(2), unsigned(N2));
    REQUIRE_EQ(ax.extent(3), unsigned(N3));

    REQUIRE_EQ(ay.extent(0), unsigned(N0));
    REQUIRE_EQ(ay.extent(1), unsigned(N1));
    REQUIRE_EQ(ay.extent(2), unsigned(N2));
    REQUIRE_EQ(ay.extent(3), unsigned(N3));

    REQUIRE_EQ(unmanaged_ax_from_ptr_dx.span(),
              unsigned(N0) * unsigned(N1) * unsigned(N2) * unsigned(N3));
  }

  using DataType = T[2];

  static void check_auto_conversion_to_const(
      const flare::Tensor<const DataType, device,
                         flare::MemoryTraits<flare::Atomic> >& arg_const,
      const flare::Tensor<const DataType, device,
                         flare::MemoryTraits<flare::Atomic> >& arg) {
    REQUIRE_EQ(arg_const, arg);
  }

  static void run_test_const() {
    using typeX =
        flare::Tensor<DataType, device, flare::MemoryTraits<flare::Atomic> >;
    using const_typeX = flare::Tensor<const DataType, device,
                                     flare::MemoryTraits<flare::Atomic> >;

    typeX x("X");
    const_typeX xc = x;

    // REQUIRE_EQ( xc ,  x ); // const xc is referenceable, non-const x is not
    // REQUIRE_EQ( x ,  xc );

    check_auto_conversion_to_const(x, xc);
  }
};

//---------------------------------------------------
//-----------initialization functors-----------------
//---------------------------------------------------

template <class T, class execution_space>
struct InitFunctor_Seq {
  using tensor_type = flare::Tensor<T*, execution_space>;

  tensor_type input;
  const int64_t length;

  InitFunctor_Seq(tensor_type& input_, const int64_t length_)
      : input(input_), length(length_) {}

  FLARE_INLINE_FUNCTION
  void operator()(const int64_t i) const {
    if (i < length) {
      input(i) = (T)i;
    }
  }
};

template <class T, class execution_space>
struct InitFunctor_ModTimes {
  using tensor_type = flare::Tensor<T*, execution_space>;

  tensor_type input;
  const int64_t length;
  const int64_t remainder;

  InitFunctor_ModTimes(tensor_type& input_, const int64_t length_,
                       const int64_t remainder_)
      : input(input_), length(length_), remainder(remainder_) {}

  FLARE_INLINE_FUNCTION
  void operator()(const int64_t i) const {
    if (i < length) {
      if (i % (remainder + 1) == remainder) {
        input(i) = (T)2;
      } else {
        input(i) = (T)1;
      }
    }
  }
};

template <class T, class execution_space>
struct InitFunctor_ModShift {
  using tensor_type = flare::Tensor<T*, execution_space>;

  tensor_type input;
  const int64_t length;
  const int64_t remainder;

  InitFunctor_ModShift(tensor_type& input_, const int64_t length_,
                       const int64_t remainder_)
      : input(input_), length(length_), remainder(remainder_) {}

  FLARE_INLINE_FUNCTION
  void operator()(const int64_t i) const {
    if (i < length) {
      if (i % (remainder + 1) == remainder) {
        input(i) = 1;
      }
    }
  }
};

//---------------------------------------------------
//-----------atomic tensor plus-equal------------------
//---------------------------------------------------

template <class T, class execution_space>
struct PlusEqualAtomicTensorFunctor {
  using atomic_tensor_type =
      flare::Tensor<T*, execution_space, flare::MemoryTraits<flare::Atomic> >;
  using tensor_type = flare::Tensor<T*, execution_space>;

  tensor_type input;
  atomic_tensor_type even_odd_result;
  const int64_t length;

  // Wrap the result tensor in an atomic tensor, use this for operator
  PlusEqualAtomicTensorFunctor(const tensor_type& input_,
                             tensor_type& even_odd_result_, const int64_t length_)
      : input(input_), even_odd_result(even_odd_result_), length(length_) {}

  FLARE_INLINE_FUNCTION
  void operator()(const int64_t i) const {
    if (i < length) {
      if (i % 2 == 0) {
        even_odd_result(0) += input(i);
      } else {
        even_odd_result(1) += input(i);
      }
    }
  }
};

template <class T, class execution_space>
T PlusEqualAtomicTensor(const int64_t input_length) {
  using tensor_type      = flare::Tensor<T*, execution_space>;
  using host_tensor_type = typename tensor_type::HostMirror;

  const int64_t length = input_length;

  tensor_type input("input_tensor", length);
  tensor_type result_tensor("result_tensor", 2);

  InitFunctor_Seq<T, execution_space> init_f(input, length);
  flare::parallel_for(flare::RangePolicy<execution_space>(0, length), init_f);

  PlusEqualAtomicTensorFunctor<T, execution_space> functor(input, result_tensor,
                                                         length);
  flare::parallel_for(flare::RangePolicy<execution_space>(0, length),
                       functor);
  flare::fence();

  host_tensor_type h_result_tensor = flare::create_mirror_tensor(result_tensor);
  flare::deep_copy(h_result_tensor, result_tensor);

  return (T)(h_result_tensor(0) + h_result_tensor(1));
}

template <class T>
T PlusEqualAtomicTensorCheck(const int64_t input_length) {
  const int64_t N = input_length;
  T result[2];

  if (N % 2 == 0) {
    const int64_t half_sum_end = (N / 2) - 1;
    const int64_t full_sum_end = N - 1;
    result[0] = half_sum_end * (half_sum_end + 1) / 2;  // Even sum.
    result[1] =
        (full_sum_end * (full_sum_end + 1) / 2) - result[0];  // Odd sum.
  } else {
    const int64_t half_sum_end = (T)(N / 2);
    const int64_t full_sum_end = N - 2;
    result[0] = half_sum_end * (half_sum_end - 1) / 2;  // Even sum.
    result[1] =
        (full_sum_end * (full_sum_end - 1) / 2) - result[0];  // Odd sum.
  }

  return (T)(result[0] + result[1]);
}

template <class T, class DeviceType>
bool PlusEqualAtomicTensorTest(int64_t input_length) {
  T res       = PlusEqualAtomicTensor<T, DeviceType>(input_length);
  T resSerial = PlusEqualAtomicTensorCheck<T>(input_length);

  bool passed = true;

  if (resSerial != res) {
    passed = false;

    std::cout << "Loop<" << typeid(T).name()
              << ">( test = PlusEqualAtomicTensorTest"
              << " FAILED : " << resSerial << " != " << res << std::endl;
  }

  return passed;
}

//---------------------------------------------------
//-----------atomic tensor minus-equal-----------------
//---------------------------------------------------

template <class T, class execution_space>
struct MinusEqualAtomicTensorFunctor {
  using atomic_tensor_type =
      flare::Tensor<T*, execution_space, flare::MemoryTraits<flare::Atomic> >;
  using tensor_type = flare::Tensor<T*, execution_space>;

  tensor_type input;
  atomic_tensor_type even_odd_result;
  const int64_t length;

  // Wrap the result tensor in an atomic tensor, use this for operator.
  MinusEqualAtomicTensorFunctor(const tensor_type& input_,
                              tensor_type& even_odd_result_,
                              const int64_t length_)
      : input(input_), even_odd_result(even_odd_result_), length(length_) {}

  FLARE_INLINE_FUNCTION
  void operator()(const int64_t i) const {
    if (i < length) {
      if (i % 2 == 0) {
        even_odd_result(0) -= input(i);
      } else {
        even_odd_result(1) -= input(i);
      }
    }
  }
};

template <class T, class execution_space>
T MinusEqualAtomicTensor(const int64_t input_length) {
  using tensor_type      = flare::Tensor<T*, execution_space>;
  using host_tensor_type = typename tensor_type::HostMirror;

  const int64_t length = input_length;

  tensor_type input("input_tensor", length);
  tensor_type result_tensor("result_tensor", 2);

  InitFunctor_Seq<T, execution_space> init_f(input, length);
  flare::parallel_for(flare::RangePolicy<execution_space>(0, length), init_f);

  MinusEqualAtomicTensorFunctor<T, execution_space> functor(input, result_tensor,
                                                          length);
  flare::parallel_for(flare::RangePolicy<execution_space>(0, length),
                       functor);
  flare::fence();

  host_tensor_type h_result_tensor = flare::create_mirror_tensor(result_tensor);
  flare::deep_copy(h_result_tensor, result_tensor);

  return (T)(h_result_tensor(0) + h_result_tensor(1));
}

template <class T>
T MinusEqualAtomicTensorCheck(const int64_t input_length) {
  const int64_t N = input_length;
  T result[2];

  if (N % 2 == 0) {
    const int64_t half_sum_end = (N / 2) - 1;
    const int64_t full_sum_end = N - 1;
    result[0] = -1 * (half_sum_end * (half_sum_end + 1) / 2);  // Even sum.
    result[1] =
        -1 * ((full_sum_end * (full_sum_end + 1) / 2) + result[0]);  // Odd sum.
  } else {
    const int64_t half_sum_end = (int64_t)(N / 2);
    const int64_t full_sum_end = N - 2;
    result[0] = -1 * (half_sum_end * (half_sum_end - 1) / 2);  // Even sum.
    result[1] =
        -1 * ((full_sum_end * (full_sum_end - 1) / 2) + result[0]);  // Odd sum.
  }

  return (result[0] + result[1]);
}

template <class T, class DeviceType>
bool MinusEqualAtomicTensorTest(int64_t input_length) {
  T res       = MinusEqualAtomicTensor<T, DeviceType>(input_length);
  T resSerial = MinusEqualAtomicTensorCheck<T>(input_length);

  bool passed = true;

  if (resSerial != res) {
    passed = false;

    std::cout << "Loop<" << typeid(T).name()
              << ">( test = MinusEqualAtomicTensorTest"
              << " FAILED : " << resSerial << " != " << res << std::endl;
  }

  return passed;
}

//---------------------------------------------------
//-----------atomic tensor times-equal-----------------
//---------------------------------------------------

template <class T, class execution_space>
struct TimesEqualAtomicTensorFunctor {
  using atomic_tensor_type =
      flare::Tensor<T*, execution_space, flare::MemoryTraits<flare::Atomic> >;
  using tensor_type = flare::Tensor<T*, execution_space>;

  tensor_type input;
  atomic_tensor_type result;
  const int64_t length;

  // Wrap the result tensor in an atomic tensor, use this for operator
  TimesEqualAtomicTensorFunctor(const tensor_type& input_, tensor_type& result_,
                              const int64_t length_)
      : input(input_), result(result_), length(length_) {}

  FLARE_INLINE_FUNCTION
  void operator()(const int64_t i) const {
    if (i < length && i > 0) {
      result(0) *= (double)input(i);
    }
  }
};

template <class T, class execution_space>
T TimesEqualAtomicTensor(const int64_t input_length, const int64_t remainder) {
  using tensor_type      = flare::Tensor<T*, execution_space>;
  using host_tensor_type = typename tensor_type::HostMirror;

  const int64_t length = input_length;

  tensor_type input("input_tensor", length);
  tensor_type result_tensor("result_tensor", 1);
  deep_copy(result_tensor, 1.0);

  InitFunctor_ModTimes<T, execution_space> init_f(input, length, remainder);
  flare::parallel_for(flare::RangePolicy<execution_space>(0, length), init_f);

  TimesEqualAtomicTensorFunctor<T, execution_space> functor(input, result_tensor,
                                                          length);
  flare::parallel_for(flare::RangePolicy<execution_space>(0, length),
                       functor);
  flare::fence();

  host_tensor_type h_result_tensor = flare::create_mirror_tensor(result_tensor);
  flare::deep_copy(h_result_tensor, result_tensor);

  return (T)(h_result_tensor(0));
}

template <class T>
T TimesEqualAtomicTensorCheck(const int64_t input_length,
                            const int64_t remainder) {
  // Analytical result.
  const int64_t N = input_length;
  T result        = 1.0;

  for (int64_t i = 2; i < N; ++i) {
    if (i % (remainder + 1) == remainder) {
      result *= 2.0;
    } else {
      result *= 1.0;
    }
  }

  return (T)result;
}

template <class T, class DeviceType>
bool TimesEqualAtomicTensorTest(const int64_t input_length) {
  const int64_t remainder = 23;
  T res       = TimesEqualAtomicTensor<T, DeviceType>(input_length, remainder);
  T resSerial = TimesEqualAtomicTensorCheck<T>(input_length, remainder);

  bool passed = true;

  if (resSerial != res) {
    passed = false;

    std::cout << "Loop<" << typeid(T).name()
              << ">( test = TimesEqualAtomicTensorTest"
              << " FAILED : " << resSerial << " != " << res << std::endl;
  }

  return passed;
}

//---------------------------------------------------
//------------atomic tensor div-equal------------------
//---------------------------------------------------

template <class T, class execution_space>
struct DivEqualAtomicTensorFunctor {
  using atomic_tensor_type =
      flare::Tensor<T, execution_space, flare::MemoryTraits<flare::Atomic> >;
  using tensor_type        = flare::Tensor<T*, execution_space>;
  using scalar_tensor_type = flare::Tensor<T, execution_space>;

  tensor_type input;
  atomic_tensor_type result;
  const int64_t length;

  // Wrap the result tensor in an atomic tensor, use this for operator.
  DivEqualAtomicTensorFunctor(const tensor_type& input_, scalar_tensor_type& result_,
                            const int64_t length_)
      : input(input_), result(result_), length(length_) {}

  FLARE_INLINE_FUNCTION
  void operator()(const int64_t i) const {
    if (i < length && i > 0) {
      result() /= (double)(input(i));
    }
  }
};

template <class T, class execution_space>
T DivEqualAtomicTensor(const int64_t input_length, const int64_t remainder) {
  using tensor_type             = flare::Tensor<T*, execution_space>;
  using scalar_tensor_type      = flare::Tensor<T, execution_space>;
  using host_scalar_tensor_type = typename scalar_tensor_type::HostMirror;

  const int64_t length = input_length;

  tensor_type input("input_tensor", length);
  scalar_tensor_type result_tensor("result_tensor");
  flare::deep_copy(result_tensor, 12121212121);

  InitFunctor_ModTimes<T, execution_space> init_f(input, length, remainder);
  flare::parallel_for(flare::RangePolicy<execution_space>(0, length), init_f);

  DivEqualAtomicTensorFunctor<T, execution_space> functor(input, result_tensor,
                                                        length);
  flare::parallel_for(flare::RangePolicy<execution_space>(0, length),
                       functor);
  flare::fence();

  host_scalar_tensor_type h_result_tensor = flare::create_mirror_tensor(result_tensor);
  flare::deep_copy(h_result_tensor, result_tensor);

  return (T)(h_result_tensor());
}

template <class T>
T DivEqualAtomicTensorCheck(const int64_t input_length, const int64_t remainder) {
  const int64_t N = input_length;
  T result        = 12121212121.0;
  for (int64_t i = 2; i < N; ++i) {
    if (i % (remainder + 1) == remainder) {
      result /= 1.0;
    } else {
      result /= 2.0;
    }
  }

  return (T)result;
}

template <class T, class DeviceType>
bool DivEqualAtomicTensorTest(const int64_t input_length) {
  const int64_t remainder = 23;

  T res       = DivEqualAtomicTensor<T, DeviceType>(input_length, remainder);
  T resSerial = DivEqualAtomicTensorCheck<T>(input_length, remainder);

  bool passed = true;

  if (resSerial != res) {
    passed = false;

    std::cout << "Loop<" << typeid(T).name()
              << ">( test = DivEqualAtomicTensorTest"
              << " FAILED : " << resSerial << " != " << res << std::endl;
  }

  return passed;
}

//---------------------------------------------------
//------------atomic tensor mod-equal------------------
//---------------------------------------------------

template <class T, class execution_space>
struct ModEqualAtomicTensorFunctor {
  using atomic_tensor_type =
      flare::Tensor<T, execution_space, flare::MemoryTraits<flare::Atomic> >;
  using tensor_type        = flare::Tensor<T*, execution_space>;
  using scalar_tensor_type = flare::Tensor<T, execution_space>;

  tensor_type input;
  atomic_tensor_type result;
  const int64_t length;

  // Wrap the result tensor in an atomic tensor, use this for operator.
  ModEqualAtomicTensorFunctor(const tensor_type& input_, scalar_tensor_type& result_,
                            const int64_t length_)
      : input(input_), result(result_), length(length_) {}

  FLARE_INLINE_FUNCTION
  void operator()(const int64_t i) const {
    if (i < length && i > 0) {
      result() %= (double)(input(i));
    }
  }
};

template <class T, class execution_space>
T ModEqualAtomicTensor(const int64_t input_length, const int64_t remainder) {
  using tensor_type             = flare::Tensor<T*, execution_space>;
  using scalar_tensor_type      = flare::Tensor<T, execution_space>;
  using host_scalar_tensor_type = typename scalar_tensor_type::HostMirror;

  const int64_t length = input_length;

  tensor_type input("input_tensor", length);
  scalar_tensor_type result_tensor("result_tensor");
  flare::deep_copy(result_tensor, 12121212121);

  InitFunctor_ModTimes<T, execution_space> init_f(input, length, remainder);
  flare::parallel_for(flare::RangePolicy<execution_space>(0, length), init_f);

  ModEqualAtomicTensorFunctor<T, execution_space> functor(input, result_tensor,
                                                        length);
  flare::parallel_for(flare::RangePolicy<execution_space>(0, length),
                       functor);
  flare::fence();

  host_scalar_tensor_type h_result_tensor = flare::create_mirror_tensor(result_tensor);
  flare::deep_copy(h_result_tensor, result_tensor);

  return (T)(h_result_tensor());
}

template <class T>
T ModEqualAtomicTensorCheck(const int64_t input_length, const int64_t remainder) {
  const int64_t N = input_length;
  T result        = 12121212121;
  for (int64_t i = 2; i < N; ++i) {
    if (i % (remainder + 1) == remainder) {
      result %= 1;
    } else {
      result %= 2;
    }
  }

  return (T)result;
}

template <class T, class DeviceType>
bool ModEqualAtomicTensorTest(const int64_t input_length) {
  static_assert(std::is_integral<T>::value,
                "ModEqualAtomicTensor Error: Type must be integral type for this "
                "unit test");

  const int64_t remainder = 23;

  T res       = ModEqualAtomicTensor<T, DeviceType>(input_length, remainder);
  T resSerial = ModEqualAtomicTensorCheck<T>(input_length, remainder);

  bool passed = true;

  if (resSerial != res) {
    passed = false;

    std::cout << "Loop<" << typeid(T).name()
              << ">( test = ModEqualAtomicTensorTest"
              << " FAILED : " << resSerial << " != " << res << std::endl;
  }

  return passed;
}

//---------------------------------------------------
//------------atomic tensor rs-equal------------------
//---------------------------------------------------

template <class T, class execution_space>
struct RSEqualAtomicTensorFunctor {
  using atomic_tensor_type = flare::Tensor<T****, execution_space,
                                        flare::MemoryTraits<flare::Atomic> >;
  using tensor_type        = flare::Tensor<T*, execution_space>;
  using result_tensor_type = flare::Tensor<T****, execution_space>;

  const tensor_type input;
  atomic_tensor_type result;
  const int64_t length;
  const int64_t value;

  // Wrap the result tensor in an atomic tensor, use this for operator.
  RSEqualAtomicTensorFunctor(const tensor_type& input_, result_tensor_type& result_,
                           const int64_t& length_, const int64_t& value_)
      : input(input_), result(result_), length(length_), value(value_) {}

  FLARE_INLINE_FUNCTION
  void operator()(const int64_t i) const {
    if (i < length) {
      if (i % 4 == 0) {
        result(1, 0, 0, 0) >>= input(i);
      } else if (i % 4 == 1) {
        result(0, 1, 0, 0) >>= input(i);
      } else if (i % 4 == 2) {
        result(0, 0, 1, 0) >>= input(i);
      } else if (i % 4 == 3) {
        result(0, 0, 0, 1) >>= input(i);
      }
    }
  }
};

template <class T, class execution_space>
T RSEqualAtomicTensor(const int64_t input_length, const int64_t value,
                    const int64_t remainder) {
  using tensor_type             = flare::Tensor<T*, execution_space>;
  using result_tensor_type      = flare::Tensor<T****, execution_space>;
  using host_scalar_tensor_type = typename result_tensor_type::HostMirror;

  const int64_t length = input_length;

  tensor_type input("input_tensor", length);
  result_tensor_type result_tensor("result_tensor", 2, 2, 2, 2);
  host_scalar_tensor_type h_result_tensor = flare::create_mirror_tensor(result_tensor);
  h_result_tensor(1, 0, 0, 0)           = value;
  h_result_tensor(0, 1, 0, 0)           = value;
  h_result_tensor(0, 0, 1, 0)           = value;
  h_result_tensor(0, 0, 0, 1)           = value;
  flare::deep_copy(result_tensor, h_result_tensor);

  InitFunctor_ModShift<T, execution_space> init_f(input, length, remainder);
  flare::parallel_for(flare::RangePolicy<execution_space>(0, length), init_f);

  RSEqualAtomicTensorFunctor<T, execution_space> functor(input, result_tensor,
                                                       length, value);
  flare::parallel_for(flare::RangePolicy<execution_space>(0, length),
                       functor);
  flare::fence();

  flare::deep_copy(h_result_tensor, result_tensor);

  return (T)(h_result_tensor(1, 0, 0, 0));
}

template <class T>
T RSEqualAtomicTensorCheck(const int64_t input_length, const int64_t value,
                         const int64_t remainder) {
  T result[4];
  result[0] = value;
  result[1] = value;
  result[2] = value;
  result[3] = value;

  T* input = new T[input_length];
  for (int64_t i = 0; i < input_length; ++i) {
    if (i % (remainder + 1) == remainder) {
      input[i] = 1;
    } else {
      input[i] = 0;
    }
  }

  for (int64_t i = 0; i < input_length; ++i) {
    if (i % 4 == 0) {
      result[0] >>= input[i];
    } else if (i % 4 == 1) {
      result[1] >>= input[i];
    } else if (i % 4 == 2) {
      result[2] >>= input[i];
    } else if (i % 4 == 3) {
      result[3] >>= input[i];
    }
  }

  delete[] input;

  return (T)result[0];
}

template <class T, class DeviceType>
bool RSEqualAtomicTensorTest(const int64_t input_length) {
  static_assert(std::is_integral<T>::value,
                "RSEqualAtomicTensorTest: Must be integral type for test");

  const int64_t remainder = 61042;       // prime - 1
  const int64_t value     = 1073741825;  //  2^30+1
  T res = RSEqualAtomicTensor<T, DeviceType>(input_length, value, remainder);
  T resSerial = RSEqualAtomicTensorCheck<T>(input_length, value, remainder);

  bool passed = true;

  if (resSerial != res) {
    passed = false;

    std::cout << "Loop<" << typeid(T).name()
              << ">( test = RSEqualAtomicTensorTest"
              << " FAILED : " << resSerial << " != " << res << std::endl;
  }

  return passed;
}

//---------------------------------------------------
//------------atomic tensor ls-equal------------------
//---------------------------------------------------

template <class T, class execution_space>
struct LSEqualAtomicTensorFunctor {
  using atomic_tensor_type = flare::Tensor<T****, execution_space,
                                        flare::MemoryTraits<flare::Atomic> >;
  using tensor_type        = flare::Tensor<T*, execution_space>;
  using result_tensor_type = flare::Tensor<T****, execution_space>;

  tensor_type input;
  atomic_tensor_type result;
  const int64_t length;
  const int64_t value;

  // Wrap the result tensor in an atomic tensor, use this for operator.
  LSEqualAtomicTensorFunctor(const tensor_type& input_, result_tensor_type& result_,
                           const int64_t& length_, const int64_t& value_)
      : input(input_), result(result_), length(length_), value(value_) {}

  FLARE_INLINE_FUNCTION
  void operator()(const int64_t i) const {
    if (i < length) {
      if (i % 4 == 0) {
        result(1, 0, 0, 0) <<= input(i);
      } else if (i % 4 == 1) {
        result(0, 1, 0, 0) <<= input(i);
      } else if (i % 4 == 2) {
        result(0, 0, 1, 0) <<= input(i);
      } else if (i % 4 == 3) {
        result(0, 0, 0, 1) <<= input(i);
      }
    }
  }
};

template <class T, class execution_space>
T LSEqualAtomicTensor(const int64_t input_length, const int64_t value,
                    const int64_t remainder) {
  using tensor_type             = flare::Tensor<T*, execution_space>;
  using result_tensor_type      = flare::Tensor<T****, execution_space>;
  using host_scalar_tensor_type = typename result_tensor_type::HostMirror;

  const int64_t length = input_length;

  tensor_type input("input_tensor", length);
  result_tensor_type result_tensor("result_tensor", 2, 2, 2, 2);
  host_scalar_tensor_type h_result_tensor = flare::create_mirror_tensor(result_tensor);
  h_result_tensor(1, 0, 0, 0)           = value;
  h_result_tensor(0, 1, 0, 0)           = value;
  h_result_tensor(0, 0, 1, 0)           = value;
  h_result_tensor(0, 0, 0, 1)           = value;
  flare::deep_copy(result_tensor, h_result_tensor);

  InitFunctor_ModShift<T, execution_space> init_f(input, length, remainder);
  flare::parallel_for(flare::RangePolicy<execution_space>(0, length), init_f);

  LSEqualAtomicTensorFunctor<T, execution_space> functor(input, result_tensor,
                                                       length, value);
  flare::parallel_for(flare::RangePolicy<execution_space>(0, length),
                       functor);
  flare::fence();

  flare::deep_copy(h_result_tensor, result_tensor);

  return (T)(h_result_tensor(1, 0, 0, 0));
}

template <class T>
T LSEqualAtomicTensorCheck(const int64_t input_length, const int64_t value,
                         const int64_t remainder) {
  T result[4];
  result[0] = value;
  result[1] = value;
  result[2] = value;
  result[3] = value;

  T* input = new T[input_length];
  for (int64_t i = 0; i < input_length; ++i) {
    if (i % (remainder + 1) == remainder) {
      input[i] = 1;
    } else {
      input[i] = 0;
    }
  }

  for (int64_t i = 0; i < input_length; ++i) {
    if (i % 4 == 0) {
      result[0] <<= input[i];
    } else if (i % 4 == 1) {
      result[1] <<= input[i];
    } else if (i % 4 == 2) {
      result[2] <<= input[i];
    } else if (i % 4 == 3) {
      result[3] <<= input[i];
    }
  }

  delete[] input;

  return (T)result[0];
}

template <class T, class DeviceType>
bool LSEqualAtomicTensorTest(const int64_t input_length) {
  static_assert(std::is_integral<T>::value,
                "LSEqualAtomicTensorTest: Must be integral type for test");

  const int64_t remainder = 61042;  // prime - 1
  const int64_t value     = 1;      //  2^30+1
  T res = LSEqualAtomicTensor<T, DeviceType>(input_length, value, remainder);
  T resSerial = LSEqualAtomicTensorCheck<T>(input_length, value, remainder);

  bool passed = true;

  if (resSerial != res) {
    passed = false;

    std::cout << "Loop<" << typeid(T).name()
              << ">( test = RSEqualAtomicTensorTest"
              << " FAILED : " << resSerial << " != " << res << std::endl;
  }

  return passed;
}

//---------------------------------------------------
//-----------atomic tensor and-equal-----------------
//---------------------------------------------------

template <class T, class execution_space>
struct AndEqualAtomicTensorFunctor {
  using atomic_tensor_type =
      flare::Tensor<T*, execution_space, flare::MemoryTraits<flare::Atomic> >;
  using tensor_type = flare::Tensor<T*, execution_space>;

  tensor_type input;
  atomic_tensor_type even_odd_result;
  const int64_t length;

  // Wrap the result tensor in an atomic tensor, use this for operator.
  AndEqualAtomicTensorFunctor(const tensor_type& input_,
                            tensor_type& even_odd_result_, const int64_t length_)
      : input(input_), even_odd_result(even_odd_result_), length(length_) {}

  FLARE_INLINE_FUNCTION
  void operator()(const int64_t i) const {
    if (i < length) {
      if (i % 2 == 0) {
        even_odd_result(0) &= input(i);
      } else {
        even_odd_result(1) &= input(i);
      }
    }
  }
};

template <class T, class execution_space>
T AndEqualAtomicTensor(const int64_t input_length) {
  using tensor_type      = flare::Tensor<T*, execution_space>;
  using host_tensor_type = typename tensor_type::HostMirror;

  const int64_t length = input_length;

  tensor_type input("input_tensor", length);
  tensor_type result_tensor("result_tensor", 2);
  flare::deep_copy(result_tensor, 1);

  InitFunctor_Seq<T, execution_space> init_f(input, length);
  flare::parallel_for(flare::RangePolicy<execution_space>(0, length), init_f);

  AndEqualAtomicTensorFunctor<T, execution_space> functor(input, result_tensor,
                                                        length);
  flare::parallel_for(flare::RangePolicy<execution_space>(0, length),
                       functor);
  flare::fence();

  host_tensor_type h_result_tensor = flare::create_mirror_tensor(result_tensor);
  flare::deep_copy(h_result_tensor, result_tensor);

  return (T)(h_result_tensor(0));
}

template <class T>
T AndEqualAtomicTensorCheck(const int64_t input_length) {
  const int64_t N = input_length;
  T result[2]     = {1};
  for (int64_t i = 0; i < N; ++i) {
    if (N % 2 == 0) {
      result[0] &= (T)i;
    } else {
      result[1] &= (T)i;
    }
  }

  return (result[0]);
}

template <class T, class DeviceType>
bool AndEqualAtomicTensorTest(int64_t input_length) {
  static_assert(std::is_integral<T>::value,
                "AndEqualAtomicTensorTest: Must be integral type for test");

  T res       = AndEqualAtomicTensor<T, DeviceType>(input_length);
  T resSerial = AndEqualAtomicTensorCheck<T>(input_length);

  bool passed = true;

  if (resSerial != res) {
    passed = false;

    std::cout << "Loop<" << typeid(T).name()
              << ">( test = AndEqualAtomicTensorTest"
              << " FAILED : " << resSerial << " != " << res << std::endl;
  }

  return passed;
}

//---------------------------------------------------
//-----------atomic tensor or-equal-----------------
//---------------------------------------------------

template <class T, class execution_space>
struct OrEqualAtomicTensorFunctor {
  using atomic_tensor_type =
      flare::Tensor<T*, execution_space, flare::MemoryTraits<flare::Atomic> >;
  using tensor_type = flare::Tensor<T*, execution_space>;

  tensor_type input;
  atomic_tensor_type even_odd_result;
  const int64_t length;

  // Wrap the result tensor in an atomic tensor, use this for operator.
  OrEqualAtomicTensorFunctor(const tensor_type& input_, tensor_type& even_odd_result_,
                           const int64_t length_)
      : input(input_), even_odd_result(even_odd_result_), length(length_) {}

  FLARE_INLINE_FUNCTION
  void operator()(const int64_t i) const {
    if (i < length) {
      if (i % 2 == 0) {
        even_odd_result(0) |= input(i);
      } else {
        even_odd_result(1) |= input(i);
      }
    }
  }
};

template <class T, class execution_space>
T OrEqualAtomicTensor(const int64_t input_length) {
  using tensor_type      = flare::Tensor<T*, execution_space>;
  using host_tensor_type = typename tensor_type::HostMirror;

  const int64_t length = input_length;

  tensor_type input("input_tensor", length);
  tensor_type result_tensor("result_tensor", 2);

  InitFunctor_Seq<T, execution_space> init_f(input, length);
  flare::parallel_for(flare::RangePolicy<execution_space>(0, length), init_f);

  OrEqualAtomicTensorFunctor<T, execution_space> functor(input, result_tensor,
                                                       length);
  flare::parallel_for(flare::RangePolicy<execution_space>(0, length),
                       functor);
  flare::fence();

  host_tensor_type h_result_tensor = flare::create_mirror_tensor(result_tensor);
  flare::deep_copy(h_result_tensor, result_tensor);

  return (T)(h_result_tensor(0));
}

template <class T>
T OrEqualAtomicTensorCheck(const int64_t input_length) {
  const int64_t N = input_length;
  T result[2]     = {0};
  for (int64_t i = 0; i < N; ++i) {
    if (i % 2 == 0) {
      result[0] |= (T)i;
    } else {
      result[1] |= (T)i;
    }
  }

  return (T)(result[0]);
}

template <class T, class DeviceType>
bool OrEqualAtomicTensorTest(int64_t input_length) {
  static_assert(std::is_integral<T>::value,
                "OrEqualAtomicTensorTest: Must be integral type for test");

  T res       = OrEqualAtomicTensor<T, DeviceType>(input_length);
  T resSerial = OrEqualAtomicTensorCheck<T>(input_length);

  bool passed = true;

  if (resSerial != res) {
    passed = false;

    std::cout << "Loop<" << typeid(T).name()
              << ">( test = OrEqualAtomicTensorTest"
              << " FAILED : " << resSerial << " != " << res << std::endl;
  }

  return passed;
}

//---------------------------------------------------
//-----------atomic tensor xor-equal-----------------
//---------------------------------------------------

template <class T, class execution_space>
struct XOrEqualAtomicTensorFunctor {
  using atomic_tensor_type =
      flare::Tensor<T*, execution_space, flare::MemoryTraits<flare::Atomic> >;
  using tensor_type = flare::Tensor<T*, execution_space>;

  tensor_type input;
  atomic_tensor_type even_odd_result;
  const int64_t length;

  // Wrap the result tensor in an atomic tensor, use this for operator.
  XOrEqualAtomicTensorFunctor(const tensor_type& input_,
                            tensor_type& even_odd_result_, const int64_t length_)
      : input(input_), even_odd_result(even_odd_result_), length(length_) {}

  FLARE_INLINE_FUNCTION
  void operator()(const int64_t i) const {
    if (i < length) {
      if (i % 2 == 0) {
        even_odd_result(0) ^= input(i);
      } else {
        even_odd_result(1) ^= input(i);
      }
    }
  }
};

template <class T, class execution_space>
T XOrEqualAtomicTensor(const int64_t input_length) {
  using tensor_type      = flare::Tensor<T*, execution_space>;
  using host_tensor_type = typename tensor_type::HostMirror;

  const int64_t length = input_length;

  tensor_type input("input_tensor", length);
  tensor_type result_tensor("result_tensor", 2);

  InitFunctor_Seq<T, execution_space> init_f(input, length);
  flare::parallel_for(flare::RangePolicy<execution_space>(0, length), init_f);

  XOrEqualAtomicTensorFunctor<T, execution_space> functor(input, result_tensor,
                                                        length);
  flare::parallel_for(flare::RangePolicy<execution_space>(0, length),
                       functor);
  flare::fence();

  host_tensor_type h_result_tensor = flare::create_mirror_tensor(result_tensor);
  flare::deep_copy(h_result_tensor, result_tensor);

  return (T)(h_result_tensor(0));
}

template <class T>
T XOrEqualAtomicTensorCheck(const int64_t input_length) {
  const int64_t N = input_length;
  T result[2]     = {0};
  for (int64_t i = 0; i < N; ++i) {
    if (i % 2 == 0) {
      result[0] ^= (T)i;
    } else {
      result[1] ^= (T)i;
    }
  }

  return (T)(result[0]);
}

template <class T, class DeviceType>
bool XOrEqualAtomicTensorTest(int64_t input_length) {
  static_assert(std::is_integral<T>::value,
                "XOrEqualAtomicTensorTest: Must be integral type for test");

  T res       = XOrEqualAtomicTensor<T, DeviceType>(input_length);
  T resSerial = XOrEqualAtomicTensorCheck<T>(input_length);

  bool passed = true;

  if (resSerial != res) {
    passed = false;

    std::cout << "Loop<" << typeid(T).name()
              << ">( test = XOrEqualAtomicTensorTest"
              << " FAILED : " << resSerial << " != " << res << std::endl;
  }

  return passed;
}

// inc/dec?

//---------------------------------------------------
//--------------atomic_test_control------------------
//---------------------------------------------------

template <class T, class DeviceType>
bool AtomicTesnorsTestIntegralType(const int length, int test) {
  static_assert(std::is_integral<T>::value,
                "TestAtomicTensors Error: Non-integral type passed into "
                "IntegralType tests");

  switch (test) {
    case 1: return PlusEqualAtomicTensorTest<T, DeviceType>(length);
    case 2: return MinusEqualAtomicTensorTest<T, DeviceType>(length);
    case 3: return RSEqualAtomicTensorTest<T, DeviceType>(length);
    case 4: return LSEqualAtomicTensorTest<T, DeviceType>(length);
    case 5: return ModEqualAtomicTensorTest<T, DeviceType>(length);
    case 6: return AndEqualAtomicTensorTest<T, DeviceType>(length);
    case 7: return OrEqualAtomicTensorTest<T, DeviceType>(length);
    case 8: return XOrEqualAtomicTensorTest<T, DeviceType>(length);
  }

  return 0;
}

template <class T, class DeviceType>
bool AtomicTensorsTestNonIntegralType(const int length, int test) {
  switch (test) {
    case 1: return PlusEqualAtomicTensorTest<T, DeviceType>(length);
    case 2: return MinusEqualAtomicTensorTest<T, DeviceType>(length);
    case 3: return TimesEqualAtomicTensorTest<T, DeviceType>(length);
    case 4: return DivEqualAtomicTensorTest<T, DeviceType>(length);
  }

  return 0;
}

}  // namespace TestAtomicTensors

namespace Test {

TEST_CASE("TEST_CATEGORY, atomic_tensors_integral") {
  const int64_t length = 1000000;
  {
    // Integral Types.
    REQUIRE(
        (TestAtomicTensors::AtomicTesnorsTestIntegralType<int64_t, TEST_EXECSPACE>(
            length, 1)));
    REQUIRE(
        (TestAtomicTensors::AtomicTesnorsTestIntegralType<int64_t, TEST_EXECSPACE>(
            length, 2)));
    REQUIRE(
        (TestAtomicTensors::AtomicTesnorsTestIntegralType<int64_t, TEST_EXECSPACE>(
            length, 3)));
    REQUIRE(
        (TestAtomicTensors::AtomicTesnorsTestIntegralType<int64_t, TEST_EXECSPACE>(
            length, 4)));
    REQUIRE(
        (TestAtomicTensors::AtomicTesnorsTestIntegralType<int64_t, TEST_EXECSPACE>(
            length, 5)));
    REQUIRE(
        (TestAtomicTensors::AtomicTesnorsTestIntegralType<int64_t, TEST_EXECSPACE>(
            length, 6)));
    REQUIRE(
        (TestAtomicTensors::AtomicTesnorsTestIntegralType<int64_t, TEST_EXECSPACE>(
            length, 7)));
    REQUIRE(
        (TestAtomicTensors::AtomicTesnorsTestIntegralType<int64_t, TEST_EXECSPACE>(
            length, 8)));
  }
}

TEST_CASE("TEST_CATEGORY, atomic_tensors_nonintegral") {
  const int64_t length = 1000000;
  {
    // Non-Integral Types.
    REQUIRE((
        TestAtomicTensors::AtomicTensorsTestNonIntegralType<double, TEST_EXECSPACE>(
            length, 1)));
    REQUIRE((
        TestAtomicTensors::AtomicTensorsTestNonIntegralType<double, TEST_EXECSPACE>(
            length, 2)));
    REQUIRE((
        TestAtomicTensors::AtomicTensorsTestNonIntegralType<double, TEST_EXECSPACE>(
            length, 3)));
    REQUIRE((
        TestAtomicTensors::AtomicTensorsTestNonIntegralType<double, TEST_EXECSPACE>(
            length, 4)));
  }
}

TEST_CASE("TEST_CATEGORY, atomic_tensor_api") {
  TestAtomicTensors::TestAtomicTensorAPI<int, TEST_EXECSPACE>();
}
}  // namespace Test
