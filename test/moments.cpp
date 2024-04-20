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
#include <fly/array.h>
#include <fly/dim4.hpp>
#include <fly/traits.hpp>
#include <iostream>
#include <string>
#include <vector>

using fly::array;
using fly::cdouble;
using fly::cfloat;
using fly::dim4;
using fly::identity;
using fly::loadImage;
using fly::max;
using fly::min;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class Image : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<float, double, int> TestTypes;

// register the type list
TYPED_TEST_SUITE(Image, TestTypes);

template<typename T>
void momentsTest(string pTestFile) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;

    vector<vector<T>> in;
    vector<vector<float>> tests;
    readTests<T, float, float>(pTestFile, numDims, in, tests);

    array imgArray(numDims.front(), &in.front()[0]);

    array momentsArray = moments(imgArray, FLY_MOMENT_M00);
    vector<float> mData(momentsArray.elements());
    momentsArray.host(&mData[0]);
    for (int i = 0; i < momentsArray.elements(); ++i) {
        ASSERT_NEAR(tests[0][i], mData[i], 4e-3 * tests[0][i])
            << "at: " << i << endl;
    }

    momentsArray = moments(imgArray, FLY_MOMENT_M01);
    momentsArray.host(&mData[0]);
    for (int i = 0; i < momentsArray.elements(); ++i) {
        ASSERT_NEAR(tests[1][i], mData[i], 8e-3 * tests[1][i])
            << "at: " << i << endl;
    }

    momentsArray = moments(imgArray, FLY_MOMENT_M10);
    momentsArray.host(&mData[0]);
    for (int i = 0; i < momentsArray.elements(); ++i) {
        ASSERT_NEAR(tests[2][i], mData[i], 3e-3 * tests[2][i])
            << "at: " << i << endl;
    }

    momentsArray = moments(imgArray, FLY_MOMENT_M11);
    momentsArray.host(&mData[0]);
    for (int i = 0; i < momentsArray.elements(); ++i) {
        ASSERT_NEAR(tests[3][i], mData[i], 7e-3 * tests[3][i])
            << "at: " << i << endl;
    }

    momentsArray = moments(imgArray, FLY_MOMENT_FIRST_ORDER);
    mData.resize(momentsArray.elements());
    momentsArray.host(&mData[0]);
    for (int i = 0; i < momentsArray.elements() / 4; i += 4) {
        ASSERT_NEAR(tests[0][i], mData[i], 1e-3 * tests[0][i])
            << "at: " << i << endl;
        ASSERT_NEAR(tests[1][i], mData[i + 1], 1e-3 * tests[1][i])
            << "at: " << i + 1 << endl;
        ASSERT_NEAR(tests[2][i], mData[i + 2], 1e-3 * tests[2][i])
            << "at: " << i + 2 << endl;
        ASSERT_NEAR(tests[3][i], mData[i + 3], 1e-3 * tests[3][i])
            << "at: " << i + 3 << endl;
    }
}

void momentsOnImageTest(string pTestFile, string pImageFile, bool isColor) {
    IMAGEIO_ENABLED_CHECK();
    vector<dim4> numDims;

    vector<vector<float>> in;
    vector<vector<float>> tests;
    readTests<float, float, float>(pTestFile, numDims, in, tests);

    array imgArray = loadImage(pImageFile.c_str(), isColor);

    double maxVal = max<double>(imgArray);
    double minVal = min<double>(imgArray);
    imgArray -= minVal;
    imgArray /= maxVal - minVal;

    array momentsArray = moments(imgArray, FLY_MOMENT_M00);

    vector<float> mData(momentsArray.elements());
    momentsArray.host(&mData[0]);
    for (int i = 0; i < momentsArray.elements(); ++i) {
        ASSERT_NEAR(tests[0][i], mData[i], 1e-2 * tests[0][i])
            << "at: " << i << endl;
    }

    momentsArray = moments(imgArray, FLY_MOMENT_M01);
    momentsArray.host(&mData[0]);
    for (int i = 0; i < momentsArray.elements(); ++i) {
        ASSERT_NEAR(tests[1][i], mData[i], 1e-2 * tests[1][i])
            << "at: " << i << endl;
    }

    momentsArray = moments(imgArray, FLY_MOMENT_M10);
    momentsArray.host(&mData[0]);
    for (int i = 0; i < momentsArray.elements(); ++i) {
        ASSERT_NEAR(tests[2][i], mData[i], 1e-2 * tests[2][i])
            << "at: " << i << endl;
    }

    momentsArray = moments(imgArray, FLY_MOMENT_M11);
    momentsArray.host(&mData[0]);
    for (int i = 0; i < momentsArray.elements(); ++i) {
        ASSERT_NEAR(tests[3][i], mData[i], 1e-2 * tests[3][i])
            << "at: " << i << endl;
    }

    momentsArray = moments(imgArray, FLY_MOMENT_FIRST_ORDER);
    mData.resize(momentsArray.elements());
    momentsArray.host(&mData[0]);
    for (int i = 0; i < momentsArray.elements() / 4; i += 4) {
        ASSERT_NEAR(tests[0][i], mData[i], 1e-2 * tests[0][i])
            << "at: " << i << endl;
        ASSERT_NEAR(tests[1][i], mData[i + 1], 1e-2 * tests[1][i])
            << "at: " << i + 1 << endl;
        ASSERT_NEAR(tests[2][i], mData[i + 2], 1e-2 * tests[2][i])
            << "at: " << i + 2 << endl;
        ASSERT_NEAR(tests[3][i], mData[i + 3], 1e-2 * tests[3][i])
            << "at: " << i + 3 << endl;
    }
}

TEST(IMAGE, MomentsImage) {
    momentsOnImageTest(string(TEST_DIR "/moments/gray_seq_16_moments.test"),
                       string(TEST_DIR "/imageio/gray_seq_16.png"), false);
}

TEST(Image, MomentsImageBatch) {
    momentsTest<float>(
        string(TEST_DIR "/moments/simple_mat_batch_moments.test"));
}

TEST(Image, MomentsBatch2D) {
    momentsOnImageTest(string(TEST_DIR "/moments/color_seq_16_moments.test"),
                       string(TEST_DIR "/imageio/color_seq_16.png"), true);
}

TYPED_TEST(Image, MomentsSynthTypes) {
    momentsTest<TypeParam>(string(TEST_DIR "/moments/simple_mat_moments.test"));
}

TEST(Image, Moment_Issue1957) {
    array A = identity(3, 3, b8);

    double m00;
    moments(&m00, A, FLY_MOMENT_M00);
    ASSERT_EQ(m00, 3);
}
