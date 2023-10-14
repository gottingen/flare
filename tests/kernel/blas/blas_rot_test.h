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

#ifndef FLARE_BLAS_ROT_TEST_H
#define FLARE_BLAS_ROT_TEST_H

#include <flare/kernel/blas/rot.h>
#include <kernel/common/test_utility.h>

template<class Scalar, class ExecutionSpace>
int test_rot() {
    using mag_type = typename flare::ArithTraits<Scalar>::mag_type;
    using vector_type = flare::Tensor<Scalar *, ExecutionSpace>;
    using scalar_type = flare::Tensor<mag_type, ExecutionSpace>;
    using vector_ref_type = flare::Tensor<Scalar *, flare::HostSpace>;

    vector_type X("X", 4), Y("Y", 4);
    vector_ref_type Xref("Xref", 4), Yref("Yref", 4);
    scalar_type c("c"), s("s");

    // Initialize inputs
    typename vector_type::HostMirror X_h = flare::create_mirror_tensor(X);
    typename vector_type::HostMirror Y_h = flare::create_mirror_tensor(Y);
    X_h(0) = 0.6;
    X_h(1) = 0.1;
    X_h(2) = -0.5;
    X_h(3) = 0.8;
    Y_h(0) = 0.5;
    Y_h(1) = -0.9;
    Y_h(2) = 0.3;
    Y_h(3) = 0.7;
    flare::deep_copy(X, X_h);
    flare::deep_copy(Y, Y_h);

    flare::deep_copy(c, 0.8);
    flare::deep_copy(s, 0.6);

    // Compute the rotated vectors
    flare::blas::rot(X, Y, c, s);
    flare::fence();

    // Bring solution back to host
    flare::deep_copy(X_h, X);
    flare::deep_copy(Y_h, Y);

    // Check outputs against reference values
    Xref(0) = 0.78;
    Xref(1) = -0.46;
    Xref(2) = -0.22;
    Xref(3) = 1.06;
    Yref(0) = 0.04;
    Yref(1) = -0.78;
    Yref(2) = 0.54;
    Yref(3) = 0.08;

    Scalar const tol = 10 * flare::ArithTraits<Scalar>::eps();
    for (int idx = 0; idx < 4; ++idx) {
        Test::EXPECT_NEAR_KK_REL(X_h(idx), Xref(idx), tol);
        Test::EXPECT_NEAR_KK_REL(Y_h(idx), Yref(idx), tol);
    }

    return 1;
}

#if defined(FLARE_TEST_FLOAT)
TEST_CASE_FIXTURE(TestCategory, "rot_float") {
    flare::Profiling::pushRegion("flare::blas::Test::rot");
    test_rot<float, TestDevice>();
    flare::Profiling::popRegion();
}

#endif

#if defined(FLARE_TEST_DOUBLE)
TEST_CASE_FIXTURE(TestCategory, "rot_double") {
    flare::Profiling::pushRegion("flare::blas::Test::rot");
    test_rot<double, TestDevice>();
    flare::Profiling::popRegion();
}

#endif

#if defined(FLARE_TEST_COMPLEX_FLOAT)
TEST_CASE_FIXTURE(TestCategory, "rot_complex_float") {
    flare::Profiling::pushRegion("flare::blas::Test::rot");
    test_rot<flare::complex<float>, TestDevice>();
    flare::Profiling::popRegion();
}

#endif

#if defined(FLARE_TEST_COMPLEX_DOUBLE)
TEST_CASE_FIXTURE(TestCategory, "rot_complex_double") {
    flare::Profiling::pushRegion("flare::blas::Test::rot");
    test_rot<flare::complex<double>, TestDevice>();
    flare::Profiling::popRegion();
}

#endif

#endif //FLARE_BLAS_ROT_TEST_H
