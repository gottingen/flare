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

using fly::cdouble;
using fly::cfloat;
using fly::dim4;
using fly::dtype_traits;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class Grad : public ::testing::Test {
   public:
    virtual void SetUp() {
        subMat0.push_back(fly_make_seq(0, 4, 1));
        subMat0.push_back(fly_make_seq(2, 6, 1));
        subMat0.push_back(fly_make_seq(0, 2, 1));
    }
    vector<fly_seq> subMat0;
};

// create a list of types to be tested
typedef ::testing::Types<float, double, cfloat, cdouble> TestTypes;

// register the type list
TYPED_TEST_SUITE(Grad, TestTypes);

template<typename T>
void gradTest(string pTestFile, const unsigned resultIdx0,
              const unsigned resultIdx1, bool isSubRef = false,
              const vector<fly_seq>* seqv = NULL) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;
    vector<vector<T>> in;
    vector<vector<T>> tests;
    readTests<T, T, float>(pTestFile, numDims, in, tests);

    dim4 idims = numDims[0];

    fly_array inArray   = 0;
    fly_array tempArray = 0;
    fly_array g0Array   = 0;
    fly_array g1Array   = 0;

    if (isSubRef) {
        ASSERT_SUCCESS(fly_create_array(&tempArray, &(in[0].front()),
                                       idims.ndims(), idims.get(),
                                       (fly_dtype)dtype_traits<T>::fly_type));

        ASSERT_SUCCESS(
            fly_index(&inArray, tempArray, seqv->size(), &seqv->front()));
    } else {
        ASSERT_SUCCESS(fly_create_array(&inArray, &(in[0].front()),
                                       idims.ndims(), idims.get(),
                                       (fly_dtype)dtype_traits<T>::fly_type));
    }

    ASSERT_SUCCESS(fly_gradient(&g0Array, &g1Array, inArray));

    size_t nElems = tests[resultIdx0].size();
    // Get result
    T* grad0Data = new T[tests[resultIdx0].size()];
    ASSERT_SUCCESS(fly_get_data_ptr((void*)grad0Data, g0Array));

    // Compare result
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(tests[resultIdx0][elIter], grad0Data[elIter])
            << "at: " << elIter << endl;
    }

    // Get result
    T* grad1Data = new T[tests[resultIdx1].size()];
    ASSERT_SUCCESS(fly_get_data_ptr((void*)grad1Data, g1Array));

    // Compare result
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(tests[resultIdx1][elIter], grad1Data[elIter])
            << "at: " << elIter << endl;
    }

    // Delete
    delete[] grad0Data;
    delete[] grad1Data;

    if (inArray != 0) fly_release_array(inArray);
    if (g0Array != 0) fly_release_array(g0Array);
    if (g1Array != 0) fly_release_array(g1Array);
    if (tempArray != 0) fly_release_array(tempArray);
}

#define GRAD_INIT(desc, file, resultIdx0, resultIdx1)                \
    TYPED_TEST(Grad, desc) {                                         \
        gradTest<TypeParam>(string(TEST_DIR "/grad/" #file ".test"), \
                            resultIdx0, resultIdx1);                 \
    }

GRAD_INIT(Grad0, grad, 0, 1);
GRAD_INIT(Grad1, grad2D, 0, 1);
GRAD_INIT(Grad2, grad3D, 0, 1);

/////////////////////////////////////// CPP
//////////////////////////////////////////////
//

using fly::array;

TEST(Grad, CPP) {
    const unsigned resultIdx0 = 0;
    const unsigned resultIdx1 = 1;

    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;
    readTests<float, float, float>(string(TEST_DIR "/grad/grad3D.test"),
                                   numDims, in, tests);

    dim4 idims = numDims[0];

    array input(idims, &(in[0].front()));
    array g0, g1;
    grad(g0, g1, input);

    size_t nElems = tests[resultIdx0].size();
    // Get result
    float* grad0Data = new float[tests[resultIdx0].size()];
    g0.host((void*)grad0Data);

    // Compare result
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(tests[resultIdx0][elIter], grad0Data[elIter])
            << "at: " << elIter << endl;
    }

    // Get result
    float* grad1Data = new float[tests[resultIdx1].size()];
    g1.host((void*)grad1Data);

    // Compare result
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(tests[resultIdx1][elIter], grad1Data[elIter])
            << "at: " << elIter << endl;
    }

    // Delete
    delete[] grad0Data;
    delete[] grad1Data;
}

TEST(Grad, MaxDim) {
    using fly::constant;
    using fly::sum;

    const size_t largeDim = 65535 * 8 + 1;

    array input = constant(1, 2, largeDim);
    array g0, g1;
    grad(g0, g1, input);

    ASSERT_EQ(0.f, sum<float>(g0));
    ASSERT_EQ(0.f, sum<float>(g1));
}
