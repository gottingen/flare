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
#include <fly/vision.h>
#include <string>
#include <vector>

using fly::array;
using fly::convolve2;
using fly::dim4;
using fly::dog;
using fly::dtype_traits;
using fly::exception;
using fly::gaussianKernel;
using fly::randu;
using fly::sum;

template<typename T>
class DOG : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<float, double, int, uint, char, uchar, short, ushort>
    TestTypes;

// register the type list
TYPED_TEST_SUITE(DOG, TestTypes);

TYPED_TEST(DOG, Basic) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    dim4 iDims(512, 512, 1, 1);
    array in = constant(1, iDims, (fly_dtype)dtype_traits<float>::fly_type);
    /* calculate DOG using Flare functions */
    array k1    = gaussianKernel(3, 3);
    array k2    = gaussianKernel(2, 2);
    array smth1 = convolve2(in, k1);
    array smth2 = convolve2(in, k2);
    array diff  = smth1 - smth2;
    /* calcuate DOG using new function */
    array out = dog(in, 3, 2);
    /* compare both the values */
    float accumErr = sum<float>(out - diff);
    EXPECT_EQ(true, accumErr < 1.0e-2);
}

TYPED_TEST(DOG, Batch) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    dim4 iDims(512, 512, 3, 1);
    array in = constant(1, iDims, (fly_dtype)dtype_traits<float>::fly_type);
    /* calculate DOG using Flare functions */
    array k1    = gaussianKernel(3, 3);
    array k2    = gaussianKernel(2, 2);
    array smth1 = convolve2(in, k1);
    array smth2 = convolve2(in, k2);
    array diff  = smth1 - smth2;
    /* calcuate DOG using new function */
    array out = dog(in, 3, 2);
    /* compare both the values */
    float accumErr = sum<float>(out - diff);
    EXPECT_EQ(true, accumErr < 1.0e-2);
}

TYPED_TEST(DOG, InvalidArray) {
    array in = randu(512);
    EXPECT_THROW(dog(in, 3, 2), exception);
}
