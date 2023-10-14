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

#ifndef FLARE_BLAS_SWAP_TEST_H
#define FLARE_BLAS_SWAP_TEST_H

#include <flare/kernel/blas/swap.h>
#include <kernel/common/test_utility.h>

namespace Test {
    namespace Impl {

        template<class VectorType>
        void test_swap(int const vector_length) {
            using vector_type = VectorType;
            using execution_space = typename vector_type::execution_space;
            using scalar_type = typename VectorType::non_const_value_type;
            using mag_type = typename flare::ArithTraits<scalar_type>::mag_type;

            // Note that Xref and Yref need to always be copies of X and Y
            // hence the use of create_mirror instead of create_mirror_tensor.
            vector_type X("X", vector_length), Y("Y", vector_length);
            typename vector_type::HostMirror Xref = flare::create_mirror(Y);
            typename vector_type::HostMirror Yref = flare::create_mirror(X);

            // Setup values in X, Y and copy them to Xref and Yref
            const scalar_type range = 10 * flare::ArithTraits<scalar_type>::one();
            flare::Random_XorShift64_Pool<execution_space> rand_pool(13718);
            flare::fill_random(X, rand_pool, range);
            flare::fill_random(Y, rand_pool, range);

            flare::deep_copy(Xref, Y);
            flare::deep_copy(Yref, X);

            flare::blas::swap(X, Y);
            flare::fence();

            typename vector_type::HostMirror Xtest = flare::create_mirror_tensor(X);
            typename vector_type::HostMirror Ytest = flare::create_mirror_tensor(Y);
            flare::deep_copy(Xtest, X);
            flare::deep_copy(Ytest, Y);

            const mag_type tol = 10 * flare::ArithTraits<scalar_type>::eps();
            for (int idx = 0; idx < vector_length; ++idx) {
                Test::EXPECT_NEAR_KK_REL(Xtest(idx), Xref(idx), tol);
                Test::EXPECT_NEAR_KK_REL(Ytest(idx), Yref(idx), tol);
            }
        }

    }  // namespace Impl
}  // namespace Test

template<class scalar_type, class execution_space>
int test_swap() {
    using Vector = flare::Tensor<scalar_type *, execution_space>;

    Test::Impl::test_swap<Vector>(0);
    Test::Impl::test_swap<Vector>(10);
    Test::Impl::test_swap<Vector>(256);
    Test::Impl::test_swap<Vector>(1024);

    return 0;
}

#if defined(FLARE_TEST_FLOAT)
TEST_CASE_FIXTURE(TestCategory, "swap_float") {
    flare::Profiling::pushRegion("flare::blas::Test::swap_float");
    test_swap<float, TestDevice>();
    flare::Profiling::popRegion();
}

#endif

#if defined(FLARE_TEST_DOUBLE)
TEST_CASE_FIXTURE(TestCategory, "swap_double") {
    flare::Profiling::pushRegion("flare::blas::Test::swap_double");
    test_swap<double, TestDevice>();
    flare::Profiling::popRegion();
}

#endif

#if defined(FLARE_TEST_COMPLEX_FLOAT)
TEST_CASE_FIXTURE(TestCategory, "swap_complex_float") {
    flare::Profiling::pushRegion("flare::blas::Test::swap_complex_float");
    test_swap<flare::complex<float>, TestDevice>();
    flare::Profiling::popRegion();
}

#endif

#if defined(FLARE_TEST_COMPLEX_DOUBLE)
TEST_CASE_FIXTURE(TestCategory, "swap_complex_double") {
    flare::Profiling::pushRegion("flare::blas::Test::swap_complex_double");
    test_swap<flare::complex<double>, TestDevice>();
    flare::Profiling::popRegion();
}

#endif
#endif //FLARE_BLAS_SWAP_TEST_H
