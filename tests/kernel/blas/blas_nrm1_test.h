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
//
// Created by jeff on 23-10-8.
//

#ifndef FLARE_BLAS_NRM1_TEST_H
#define FLARE_BLAS_NRM1_TEST_H
#include <flare/kernel/blas/nrm1.h>
#include <kernel/common/test_utility.h>

namespace Test {
    template <class TensorTypeA, class Device>
    void impl_test_nrm1(int N) {
        typedef typename TensorTypeA::value_type ScalarA;
        typedef flare::ArithTraits<ScalarA> AT;
        typedef typename AT::mag_type mag_type;
        typedef flare::ArithTraits<mag_type> MAT;

        tensor_stride_adapter<TensorTypeA> a("a", N);

        flare::Random_XorShift64_Pool<typename Device::execution_space> rand_pool(
                13718);

        ScalarA randStart, randEnd;
        Test::getRandomBounds(10.0, randStart, randEnd);
        flare::fill_random(a.d_tensor, rand_pool, randStart, randEnd);

        flare::deep_copy(a.h_base, a.d_base);

        double eps = (std::is_same<typename flare::ArithTraits<ScalarA>::mag_type,
                float>::value
                      ? 1e-4
                      : 1e-7);

        mag_type expected_result = 0;
        for (int i = 0; i < N; i++) {
            // note: for complex, BLAS asum (aka our nrm1) is _not_
            // the sum of magnitudes - it's the sum of absolute real and imaginary
            // parts. See netlib, MKL, and CUBLAS documentation.
            //
            // This is safe; ArithTraits<T>::imag is 0 if T is real.
            expected_result +=
                    MAT::abs(AT::real(a.h_tensor(i))) + MAT::abs(AT::imag(a.h_tensor(i)));
        }

        mag_type nonconst_result = flare::blas::nrm1(a.d_tensor);
        EXPECT_NEAR_KK(nonconst_result, expected_result, eps * expected_result);

        mag_type const_result = flare::blas::nrm1(a.d_tensor_const);
        EXPECT_NEAR_KK(const_result, expected_result, eps * expected_result);
    }

    template <class TensorTypeA, class Device>
    void impl_test_nrm1_mv(int N, int K) {
        typedef typename TensorTypeA::value_type ScalarA;
        typedef flare::ArithTraits<ScalarA> AT;
        typedef typename AT::mag_type mag_type;
        typedef flare::ArithTraits<mag_type> MAT;

        tensor_stride_adapter<TensorTypeA> a("A", N, K);

        flare::Random_XorShift64_Pool<typename Device::execution_space> rand_pool(
                13718);

        ScalarA randStart, randEnd;
        Test::getRandomBounds(10.0, randStart, randEnd);
        flare::fill_random(a.d_tensor, rand_pool, randStart, randEnd);

        flare::deep_copy(a.h_base, a.d_base);

        double eps = (std::is_same<typename flare::ArithTraits<ScalarA>::mag_type,
                float>::value
                      ? 1e-4
                      : 1e-7);

        flare::Tensor<mag_type*, flare::HostSpace> expected_result("Expected Nrm1",
                                                                   K);
        for (int k = 0; k < K; k++) {
            expected_result(k) = MAT::zero();
            for (int i = 0; i < N; i++) {
                expected_result(k) += MAT::abs(AT::real(a.h_tensor(i, k))) +
                                      MAT::abs(AT::imag(a.h_tensor(i, k)));
            }
        }

        flare::Tensor<mag_type*, flare::HostSpace> r("Nrm1::Result", K);
        flare::Tensor<mag_type*, flare::HostSpace> c_r("Nrm1::ConstResult", K);

        flare::blas::nrm1(r, a.d_tensor);
        flare::blas::nrm1(c_r, a.d_tensor_const);
        flare::fence();
        for (int k = 0; k < K; k++) {
            EXPECT_NEAR_KK(r(k), expected_result(k), eps * expected_result(k));
        }
    }
}  // namespace Test

template <class ScalarA, class Device>
int test_nrm1() {
#if defined(FLARE_TEST_LAYOUTLEFT)
    typedef flare::Tensor<ScalarA*, flare::LayoutLeft, Device> tensor_type_a_ll;
    Test::impl_test_nrm1<tensor_type_a_ll, Device>(0);
    Test::impl_test_nrm1<tensor_type_a_ll, Device>(13);
    Test::impl_test_nrm1<tensor_type_a_ll, Device>(1024);
    Test::impl_test_nrm1<tensor_type_a_ll, Device>(132231);
#endif

#if defined(FLARE_TEST_LAYOUTRIGHT)
    typedef flare::Tensor<ScalarA*, flare::LayoutRight, Device> tensor_type_a_lr;
    Test::impl_test_nrm1<tensor_type_a_lr, Device>(0);
    Test::impl_test_nrm1<tensor_type_a_lr, Device>(13);
    Test::impl_test_nrm1<tensor_type_a_lr, Device>(1024);
    Test::impl_test_nrm1<tensor_type_a_lr, Device>(132231);
#endif

#if defined(FLARE_TEST_ALL_TYPES)
    typedef flare::Tensor<ScalarA*, flare::LayoutStride, Device> tensor_type_a_ls;
    Test::impl_test_nrm1<tensor_type_a_ls, Device>(0);
    Test::impl_test_nrm1<tensor_type_a_ls, Device>(13);
    Test::impl_test_nrm1<tensor_type_a_ls, Device>(1024);
    Test::impl_test_nrm1<tensor_type_a_ls, Device>(132231);
#endif

    return 1;
}

template <class ScalarA, class Device>
int test_nrm1_mv() {
#if defined(FLARE_TEST_LAYOUTLEFT)
    typedef flare::Tensor<ScalarA**, flare::LayoutLeft, Device> tensor_type_a_ll;
    Test::impl_test_nrm1_mv<tensor_type_a_ll, Device>(0, 5);
    Test::impl_test_nrm1_mv<tensor_type_a_ll, Device>(13, 5);
    Test::impl_test_nrm1_mv<tensor_type_a_ll, Device>(1024, 5);
    Test::impl_test_nrm1_mv<tensor_type_a_ll, Device>(789, 1);
    Test::impl_test_nrm1_mv<tensor_type_a_ll, Device>(132231, 5);
#endif

#if defined(FLARE_TEST_LAYOUTRIGHT)
    typedef flare::Tensor<ScalarA**, flare::LayoutRight, Device> tensor_type_a_lr;
    Test::impl_test_nrm1_mv<tensor_type_a_lr, Device>(0, 5);
    Test::impl_test_nrm1_mv<tensor_type_a_lr, Device>(13, 5);
    Test::impl_test_nrm1_mv<tensor_type_a_lr, Device>(1024, 5);
    Test::impl_test_nrm1_mv<tensor_type_a_lr, Device>(789, 1);
    Test::impl_test_nrm1_mv<tensor_type_a_lr, Device>(132231, 5);
#endif

#if defined(FLARE_TEST_ALL_TYPES)
    typedef flare::Tensor<ScalarA**, flare::LayoutStride, Device> tensor_type_a_ls;
    Test::impl_test_nrm1_mv<tensor_type_a_ls, Device>(0, 5);
    Test::impl_test_nrm1_mv<tensor_type_a_ls, Device>(13, 5);
    Test::impl_test_nrm1_mv<tensor_type_a_ls, Device>(1024, 5);
    Test::impl_test_nrm1_mv<tensor_type_a_ls, Device>(789, 1);
    Test::impl_test_nrm1_mv<tensor_type_a_ls, Device>(132231, 5);
#endif

    return 1;
}

#if defined(FLARE_TEST_FLOAT)
TEST_CASE_FIXTURE(TestCategory, "nrm1_float") {
flare::Profiling::pushRegion("flare::blas::Test::nrm1_float");
test_nrm1<float, TestDevice>();
flare::Profiling::popRegion();
}
TEST_CASE_FIXTURE(TestCategory, "nrm1_mv_float") {
flare::Profiling::pushRegion("flare::blas::Test::nrm1_mv_float");
test_nrm1_mv<float, TestDevice>();
flare::Profiling::popRegion();
}
#endif

#if defined(FLARE_TEST_DOUBLE)
TEST_CASE_FIXTURE(TestCategory, "nrm1_double") {
flare::Profiling::pushRegion("flare::blas::Test::nrm1_double");
test_nrm1<double, TestDevice>();
flare::Profiling::popRegion();
}
TEST_CASE_FIXTURE(TestCategory, "nrm1_mv_double") {
flare::Profiling::pushRegion("flare::blas::Test::nrm1_mv_double");
test_nrm1_mv<double, TestDevice>();
flare::Profiling::popRegion();
}
#endif

#if defined(FLARE_TEST_COMPLEX_DOUBLE)
TEST_CASE_FIXTURE(TestCategory, "nrm1_complex_double") {
flare::Profiling::pushRegion("flare::blas::Test::nrm1_complex_double");
test_nrm1<flare::complex<double>, TestDevice>();
flare::Profiling::popRegion();
}
TEST_CASE_FIXTURE(TestCategory, "nrm1_mv_complex_double") {
flare::Profiling::pushRegion("flare::blas::Test::nrm1_mv_complex_double");
test_nrm1_mv<flare::complex<double>, TestDevice>();
flare::Profiling::popRegion();
}
#endif

#if defined(FLARE_TEST_INT)
TEST_CASE_FIXTURE(TestCategory, "nrm1_int") {
flare::Profiling::pushRegion("flare::blas::Test::nrm1_int");
test_nrm1<int, TestDevice>();
flare::Profiling::popRegion();
}
TEST_CASE_FIXTURE(TestCategory, "nrm1_mv_int") {
flare::Profiling::pushRegion("flare::blas::Test::nrm1_mv_int");
test_nrm1_mv<int, TestDevice>();
flare::Profiling::popRegion();
}
#endif


#endif //FLARE_BLAS_NRM1_TEST_H
