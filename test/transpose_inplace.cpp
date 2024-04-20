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
#include <string>
#include <vector>

using fly::array;
using fly::cdouble;
using fly::cfloat;
using fly::dim4;
using fly::dtype_traits;
using std::endl;
using std::vector;

template<typename T>
class Transpose : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<float, cfloat, double, cdouble, int, uint, char, uchar,
                         short, ushort>
    TestTypes;

// register the type list
TYPED_TEST_SUITE(Transpose, TestTypes);

template<typename T>
void transposeip_test(dim4 dims) {
    SUPPORTED_TYPE_CHECK(T);

    fly_array inArray  = 0;
    fly_array outArray = 0;

    ASSERT_SUCCESS(fly_randu(&inArray, dims.ndims(), dims.get(),
                            (fly_dtype)dtype_traits<T>::fly_type));

    ASSERT_SUCCESS(fly_transpose(&outArray, inArray, false));
    ASSERT_SUCCESS(fly_transpose_inplace(inArray, false));

    ASSERT_ARRAYS_EQ(inArray, outArray);

    // cleanup
    ASSERT_SUCCESS(fly_release_array(inArray));
    ASSERT_SUCCESS(fly_release_array(outArray));
}

#define INIT_TEST(Side, D3, D4)                                \
    TYPED_TEST(Transpose, TranposeIP_##Side) {                 \
        transposeip_test<TypeParam>(dim4(Side, Side, D3, D4)); \
    }

INIT_TEST(10, 1, 1);
INIT_TEST(64, 1, 1);
INIT_TEST(300, 1, 1);
INIT_TEST(1000, 1, 1);
INIT_TEST(100, 2, 1);
INIT_TEST(25, 2, 2);

////////////////////////////////////// CPP //////////////////////////////////
//
void transposeInPlaceCPPTest() {
    dim4 dims(64, 64, 1, 1);

    array input  = randu(dims);
    array output = transpose(input);
    transposeInPlace(input);

    ASSERT_ARRAYS_EQ(input, output);
}
