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

#include <flare.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <fly/defines.h>
#include <fly/dim4.hpp>
#include <fly/traits.hpp>
#include <complex>
#include <iostream>
#include <string>
#include <vector>

using fly::array;
using fly::cdouble;
using fly::cfloat;
using fly::dim4;
using fly::exception;
using fly::identity;
using fly::matmul;
using fly::max;
using std::abs;
using std::cout;
using std::endl;
using std::string;
using std::vector;

///////////////////////////////// CPP ////////////////////////////////////
TEST(QRFactorized, CPP) {
    LAPACK_ENABLED_CHECK();

    int resultIdx = 0;

    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;
    readTests<float, float, float>(string(TEST_DIR "/lapack/qrfactorized.test"),
                                   numDims, in, tests);

    dim4 idims = numDims[0];
    array input(idims, &(in[0].front()));

    array q, r, tau;
    qr(q, r, tau, input);

    dim4 qdims = q.dims();
    dim4 rdims = r.dims();

    // Get result
    float* qData = new float[qdims.elements()];
    q.host((void*)qData);
    float* rData = new float[rdims.elements()];
    r.host((void*)rData);

    // Compare result
    for (int y = 0; y < (int)qdims[1]; ++y) {
        for (int x = 0; x < (int)qdims[0]; ++x) {
            int elIter = y * qdims[0] + x;
            ASSERT_NEAR(tests[resultIdx][elIter], qData[elIter], 0.001)
                << "at: " << elIter << endl;
        }
    }

    resultIdx = 1;

    for (int y = 0; y < (int)rdims[1]; ++y) {
        for (int x = 0; x < (int)rdims[0]; ++x) {
            // Test only upper half
            if (x <= y) {
                int elIter = y * rdims[0] + x;
                ASSERT_NEAR(tests[resultIdx][elIter], rData[elIter], 0.001)
                    << "at: " << elIter << endl;
            }
        }
    }

    // Delete
    delete[] qData;
    delete[] rData;
}

template<typename T>
void qrTester(const int m, const int n, double eps) {
    try {
        SUPPORTED_TYPE_CHECK(T);
        LAPACK_ENABLED_CHECK();

#if 1
        array in = cpu_randu<T>(dim4(m, n));
#else
        array in = randu(m, n, (dtype)dtype_traits<T>::fly_type);
#endif

        //! [ex_qr_unpacked]
        array q, r, tau;
        qr(q, r, tau, in);
        //! [ex_qr_unpacked]

        array qq = matmul(q, q.H());
        array ii = identity(qq.dims(), qq.type());

        ASSERT_NEAR(0, max<double>(abs(real(qq - ii))), eps);
        ASSERT_NEAR(0, max<double>(abs(imag(qq - ii))), eps);

        //! [ex_qr_recon]
        array re = matmul(q, r);
        //! [ex_qr_recon]

        ASSERT_NEAR(0, max<double>(abs(real(re - in))), eps);
        ASSERT_NEAR(0, max<double>(abs(imag(re - in))), eps);

        //! [ex_qr_packed]
        array out = in.copy();
        array tau2;
        qrInPlace(tau2, out);
        //! [ex_qr_packed]

        array r2 = upper(out);

        ASSERT_NEAR(0, max<double>(abs(real(tau - tau2))), eps);
        ASSERT_NEAR(0, max<double>(abs(imag(tau - tau2))), eps);

        ASSERT_NEAR(0, max<double>(abs(real(r2 - r))), eps);
        ASSERT_NEAR(0, max<double>(abs(imag(r2 - r))), eps);

    } catch (exception& ex) {
        cout << ex.what() << endl;
        throw;
    }
}

template<typename T>
double eps();

template<>
double eps<float>() {
    return 1e-3;
}

template<>
double eps<double>() {
    return 1e-5;
}

template<>
double eps<cfloat>() {
    return 1e-3;
}

template<>
double eps<cdouble>() {
    return 1e-5;
}
template<typename T>
class QR : public ::testing::Test {};

typedef ::testing::Types<float, cfloat, double, cdouble> TestTypes;
TYPED_TEST_SUITE(QR, TestTypes);

TYPED_TEST(QR, RectangularLarge0) {
    qrTester<TypeParam>(1000, 500, eps<TypeParam>());
}

TYPED_TEST(QR, RectangularMultipleOfTwoLarge0) {
    qrTester<TypeParam>(1024, 512, eps<TypeParam>());
}

TYPED_TEST(QR, RectangularLarge1) {
    qrTester<TypeParam>(500, 1000, eps<TypeParam>());
}

TYPED_TEST(QR, RectangularMultipleOfTwoLarge1) {
    qrTester<TypeParam>(512, 1024, eps<TypeParam>());
}

TEST(QR, InPlaceNullOutput) {
    LAPACK_ENABLED_CHECK();
    dim4 dims(3, 3);
    fly_array in = 0;
    ASSERT_SUCCESS(fly_randu(&in, dims.ndims(), dims.get(), f32));

    ASSERT_EQ(FLY_ERR_ARG, fly_qr_inplace(NULL, in));
    ASSERT_SUCCESS(fly_release_array(in));
}
