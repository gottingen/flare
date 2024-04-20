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
#include <fly/dim4.hpp>
#include <fly/traits.hpp>
#include <iostream>
#include <string>
#include <vector>

using fly::array;
using fly::dim4;
using fly::dtype_traits;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class TransformCoordinates : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

typedef ::testing::Types<float, double> TestTypes;

TYPED_TEST_SUITE(TransformCoordinates, TestTypes);

template<typename T>
void transformCoordinatesTest(string pTestFile) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> inDims;
    vector<vector<T>> in;
    vector<vector<float>> gold;

    readTests<T, float, float>(pTestFile, inDims, in, gold);

    fly_array tfArray  = 0;
    fly_array outArray = 0;
    ASSERT_SUCCESS(fly_create_array(&tfArray, &(in[0].front()),
                                   inDims[0].ndims(), inDims[0].get(),
                                   (fly_dtype)dtype_traits<T>::fly_type));

    int nTests = in.size();

    for (int test = 1; test < nTests; test++) {
        dim_t d0 = (dim_t)in[test][0];
        dim_t d1 = (dim_t)in[test][1];

        ASSERT_SUCCESS(fly_transform_coordinates(&outArray, tfArray, d0, d1));

        // Get result
        dim_t outEl = 0;
        ASSERT_SUCCESS(fly_get_elements(&outEl, outArray));
        vector<T> outData(outEl);
        ASSERT_SUCCESS(fly_get_data_ptr((void*)&outData.front(), outArray));

        ASSERT_SUCCESS(fly_release_array(outArray));
        const float thr = 1.f;

        for (dim_t elIter = 0; elIter < outEl; elIter++) {
            ASSERT_LE(fabs(outData[elIter] - gold[test - 1][elIter]), thr)
                << "at: " << elIter << endl;
        }
    }

    if (tfArray != 0) fly_release_array(tfArray);
}

TYPED_TEST(TransformCoordinates, RotateMatrix) {
    transformCoordinatesTest<TypeParam>(
        string(TEST_DIR "/transformCoordinates/rotate_matrix.test"));
}

TYPED_TEST(TransformCoordinates, 3DMatrix) {
    transformCoordinatesTest<TypeParam>(
        string(TEST_DIR "/transformCoordinates/3d_matrix.test"));
}

///////////////////////////////////// CPP ////////////////////////////////
//
TEST(TransformCoordinates, CPP) {
    vector<dim4> inDims;
    vector<vector<float>> in;
    vector<vector<float>> gold;

    readTests<float, float, float>(
        TEST_DIR "/transformCoordinates/3d_matrix.test", inDims, in, gold);

    array tf = array(inDims[0][0], inDims[0][1], &(in[0].front()));

    float d0 = in[1][0];
    float d1 = in[1][1];

    array out    = transformCoordinates(tf, d0, d1);
    dim4 outDims = out.dims();

    vector<float> h_out(outDims[0] * outDims[1]);
    out.host(&h_out.front());

    const size_t n  = gold[0].size();
    const float thr = 1.f;

    for (size_t elIter = 0; elIter < n; elIter++) {
        ASSERT_LE(fabs(h_out[elIter] - gold[0][elIter]), thr)
            << "at: " << elIter << endl;
    }
}
