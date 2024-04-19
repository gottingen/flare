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

#pragma once

#include <flare.h>
#include <fly/defines.h>
#include <fly/dim4.hpp>
#include <fly/traits.hpp>

#include <complex>
#include <iostream>
#include <string>
#include <vector>

using fly::cdouble;
using fly::cfloat;
using std::abs;
using std::cout;
using std::endl;
using std::string;
using std::vector;

template<typename T>
void solveTester(const int m, const int n, const int k, double eps,
                 int targetDevice = -1) {
    if (targetDevice >= 0) fly::setDevice(targetDevice);

    fly::deviceGC();

    SUPPORTED_TYPE_CHECK(T);
    LAPACK_ENABLED_CHECK();

#if 1
    fly::array A  = cpu_randu<T>(fly::dim4(m, n));
    fly::array X0 = cpu_randu<T>(fly::dim4(n, k));
#else
    fly::array A  = fly::randu(m, n, (fly::dtype)fly::dtype_traits<T>::fly_type);
    fly::array X0 = fly::randu(n, k, (fly::dtype)fly::dtype_traits<T>::fly_type);
#endif
    fly::array B0 = fly::matmul(A, X0);

    //! [ex_solve]
    fly::array X1 = fly::solve(A, B0);
    //! [ex_solve]

    //! [ex_solve_recon]
    fly::array B1 = fly::matmul(A, X1);
    //! [ex_solve_recon]

    ASSERT_ARRAYS_NEAR(B0, B1, eps);
}

template<typename T>
void solveLUTester(const int n, const int k, double eps,
                   int targetDevice = -1) {
    if (targetDevice >= 0) fly::setDevice(targetDevice);

    fly::deviceGC();

    SUPPORTED_TYPE_CHECK(T);
    LAPACK_ENABLED_CHECK();

#if 1
    fly::array A  = cpu_randu<T>(fly::dim4(n, n));
    fly::array X0 = cpu_randu<T>(fly::dim4(n, k));
#else
    fly::array A  = fly::randu(n, n, (fly::dtype)fly::dtype_traits<T>::fly_type);
    fly::array X0 = fly::randu(n, k, (fly::dtype)fly::dtype_traits<T>::fly_type);
#endif
    fly::array B0 = fly::matmul(A, X0);

    //! [ex_solve_lu]
    fly::array A_lu, pivot;
    fly::lu(A_lu, pivot, A);
    fly::array X1 = fly::solveLU(A_lu, pivot, B0);
    //! [ex_solve_lu]

    fly::array B1 = fly::matmul(A, X1);

    ASSERT_ARRAYS_NEAR(B0, B1, eps);
}

template<typename T>
void solveTriangleTester(const int n, const int k, bool is_upper, double eps,
                         int targetDevice = -1) {
    if (targetDevice >= 0) fly::setDevice(targetDevice);

    fly::deviceGC();

    SUPPORTED_TYPE_CHECK(T);
    LAPACK_ENABLED_CHECK();

#if 1
    fly::array A  = cpu_randu<T>(fly::dim4(n, n));
    fly::array X0 = cpu_randu<T>(fly::dim4(n, k));
#else
    fly::array A  = fly::randu(n, n, (fly::dtype)fly::dtype_traits<T>::fly_type);
    fly::array X0 = fly::randu(n, k, (fly::dtype)fly::dtype_traits<T>::fly_type);
#endif

    fly::array L, U, pivot;
    fly::lu(L, U, pivot, A);

    fly::array AT = is_upper ? U : L;
    fly::array B0 = fly::matmul(AT, X0);
    fly::array X1;

    if (is_upper) {
        //! [ex_solve_upper]
        fly::array X = fly::solve(AT, B0, FLY_MAT_UPPER);
        //! [ex_solve_upper]

        X1 = X;
    } else {
        //! [ex_solve_lower]
        fly::array X = fly::solve(AT, B0, FLY_MAT_LOWER);
        //! [ex_solve_lower]

        X1 = X;
    }

    fly::array B1 = fly::matmul(AT, X1);

    ASSERT_ARRAYS_NEAR(B0, B1, eps);
}
