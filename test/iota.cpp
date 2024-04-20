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
class Iota : public ::testing::Test {
   public:
    virtual void SetUp() {
        subMat0.push_back(fly_make_seq(0, 4, 1));
        subMat0.push_back(fly_make_seq(2, 6, 1));
        subMat0.push_back(fly_make_seq(0, 2, 1));
    }
    vector<fly_seq> subMat0;
};

// create a list of types to be tested
typedef ::testing::Types<float, double, int, unsigned int, intl, uintl,
                         unsigned char, short, ushort, half_float::half>
    TestTypes;

// register the type list
TYPED_TEST_SUITE(Iota, TestTypes);

template<typename T>
void iotaTest(const dim4 idims, const dim4 tdims) {
    SUPPORTED_TYPE_CHECK(T);

    fly_array outArray = 0;

    ASSERT_SUCCESS(fly_iota(&outArray, idims.ndims(), idims.get(), tdims.ndims(),
                           tdims.get(), (fly_dtype)dtype_traits<T>::fly_type));

    fly_array temp0 = 0, temp1 = 0, temp2 = 0;
    dim4 tempdims(idims.elements());
    dim4 fulldims;
    for (unsigned i = 0; i < 4; i++) { fulldims[i] = idims[i] * tdims[i]; }
    ASSERT_SUCCESS(fly_range(&temp2, tempdims.ndims(), tempdims.get(), 0,
                            (fly_dtype)dtype_traits<T>::fly_type));
    ASSERT_SUCCESS(fly_moddims(&temp1, temp2, idims.ndims(), idims.get()));
    ASSERT_SUCCESS(
        fly_tile(&temp0, temp1, tdims[0], tdims[1], tdims[2], tdims[3]));

    ASSERT_ARRAYS_EQ(temp0, outArray);

    if (outArray != 0) fly_release_array(outArray);
    if (temp0 != 0) fly_release_array(temp0);
    if (temp1 != 0) fly_release_array(temp1);
    if (temp2 != 0) fly_release_array(temp2);
}

#define IOTA_INIT(desc, x, y, z, w, a, b, c, d)                  \
    TYPED_TEST(Iota, desc) {                                     \
        iotaTest<TypeParam>(dim4(x, y, z, w), dim4(a, b, c, d)); \
    }

IOTA_INIT(Iota1D0, 100, 1, 1, 1, 2, 3, 1, 1);

IOTA_INIT(Iota2D0, 10, 20, 1, 1, 3, 1, 2, 1);
IOTA_INIT(Iota2D1, 100, 5, 1, 1, 1, 2, 4, 2);

IOTA_INIT(Iota3D0, 20, 6, 3, 1, 1, 1, 1, 1);
IOTA_INIT(Iota3D1, 10, 12, 5, 1, 2, 3, 4, 5);
IOTA_INIT(Iota3D2, 25, 30, 2, 1, 1, 2, 2, 1);

IOTA_INIT(Iota4D0, 20, 6, 3, 2, 2, 3, 1, 2);
IOTA_INIT(Iota4D1, 10, 12, 5, 2, 1, 2, 2, 2);
IOTA_INIT(Iota4D2, 25, 30, 2, 2, 3, 2, 1, 1);
IOTA_INIT(Iota4D3, 25, 30, 2, 2, 4, 2, 4, 2);

IOTA_INIT(IotaMaxDimY, 1, 65535 * 32 + 1, 1, 1, 1, 1, 1, 1);
IOTA_INIT(IotaMaxDimZ, 1, 1, 65535 * 32 + 1, 1, 1, 1, 1, 1);
IOTA_INIT(IotaMaxDimW, 1, 1, 1, 65535 * 32 + 1, 1, 1, 1, 1);

///////////////////////////////// CPP ////////////////////////////////////
//

using fly::array;
using fly::iota;

TEST(Iota, CPP) {
    dim4 idims(23, 15, 1, 1);
    dim4 tdims(2, 2, 1, 1);
    dim4 fulldims;
    for (unsigned i = 0; i < 4; i++) { fulldims[i] = idims[i] * tdims[i]; }

    array output = iota(idims, tdims);
    array tileArray =
        tile(moddims(range(dim4(idims.elements()), 0), idims), tdims);

    ASSERT_ARRAYS_EQ(tileArray, output);
}
