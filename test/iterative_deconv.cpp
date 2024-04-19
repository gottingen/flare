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
#include <fly/data.h>
#include <fly/dim4.hpp>
#include <fly/traits.hpp>
#include <string>
#include <vector>

using std::abs;
using std::string;
using std::vector;
using namespace fly;

template<typename T>
class IterativeDeconvolution : public ::testing::Test {};

// create a list of types to be tested
typedef ::testing::Types<float, uchar, short, ushort> TestTypes;

// register the type list
TYPED_TEST_SUITE(IterativeDeconvolution, TestTypes);

template<typename T, bool isColor>
void iterDeconvImageTest(string pTestFile, const unsigned iters, const float rf,
                         const fly::iterativeDeconvAlgo algo) {
    typedef
        typename cond_type<is_same_type<T, double>::value, double, float>::type
            OutType;

    SUPPORTED_TYPE_CHECK(T);
    IMAGEIO_ENABLED_CHECK();

    using fly::dim4;

    vector<dim4> inDims;
    vector<string> inFiles;
    vector<dim_t> outSizes;
    vector<string> outFiles;

    readImageTests(pTestFile, inDims, inFiles, outSizes, outFiles);

    size_t testCount = inDims.size();

    for (size_t testId = 0; testId < testCount; ++testId) {
        inFiles[testId].insert(0, string(TEST_DIR "/iterative_deconv/"));
        outFiles[testId].insert(0, string(TEST_DIR "/iterative_deconv/"));

        fly_array _inArray   = 0;
        fly_array inArray    = 0;
        fly_array kerArray   = 0;
        fly_array _outArray  = 0;
        fly_array cstArray   = 0;
        fly_array minArray   = 0;
        fly_array numArray   = 0;
        fly_array denArray   = 0;
        fly_array divArray   = 0;
        fly_array outArray   = 0;
        fly_array goldArray  = 0;
        fly_array _goldArray = 0;
        dim_t nElems        = 0;

        ASSERT_SUCCESS(fly_gaussian_kernel(&kerArray, 13, 13, 2.25, 2.25));

        fly_dtype otype = (fly_dtype)fly::dtype_traits<OutType>::fly_type;

        ASSERT_SUCCESS(
            fly_load_image(&_inArray, inFiles[testId].c_str(), isColor));
        ASSERT_SUCCESS(conv_image<T>(&inArray, _inArray));

        ASSERT_SUCCESS(
            fly_load_image(&_goldArray, outFiles[testId].c_str(), isColor));
        ASSERT_SUCCESS(conv_image<OutType>(&goldArray, _goldArray));
        ASSERT_SUCCESS(fly_get_elements(&nElems, goldArray));

        unsigned ndims;
        dim_t dims[4];
        ASSERT_SUCCESS(fly_get_numdims(&ndims, goldArray));
        ASSERT_SUCCESS(
            fly_get_dims(dims, dims + 1, dims + 2, dims + 3, goldArray));

        ASSERT_SUCCESS(fly_iterative_deconv(&_outArray, inArray, kerArray, iters,
                                           rf, algo));

        double maxima, minima, imag;
        ASSERT_SUCCESS(fly_min_all(&minima, &imag, _outArray));
        ASSERT_SUCCESS(fly_max_all(&maxima, &imag, _outArray));
        ASSERT_SUCCESS(fly_constant(&cstArray, 255.0, ndims, dims, otype));
        ASSERT_SUCCESS(
            fly_constant(&denArray, (maxima - minima), ndims, dims, otype));
        ASSERT_SUCCESS(fly_constant(&minArray, minima, ndims, dims, otype));
        ASSERT_SUCCESS(fly_sub(&numArray, _outArray, minArray, false));
        ASSERT_SUCCESS(fly_div(&divArray, numArray, denArray, false));
        ASSERT_SUCCESS(fly_mul(&outArray, divArray, cstArray, false));

        ASSERT_IMAGES_NEAR(goldArray, outArray, 0.03);

        ASSERT_SUCCESS(fly_release_array(_inArray));
        ASSERT_SUCCESS(fly_release_array(inArray));
        ASSERT_SUCCESS(fly_release_array(kerArray));
        ASSERT_SUCCESS(fly_release_array(cstArray));
        ASSERT_SUCCESS(fly_release_array(minArray));
        ASSERT_SUCCESS(fly_release_array(denArray));
        ASSERT_SUCCESS(fly_release_array(numArray));
        ASSERT_SUCCESS(fly_release_array(divArray));
        ASSERT_SUCCESS(fly_release_array(_outArray));
        ASSERT_SUCCESS(fly_release_array(outArray));
        ASSERT_SUCCESS(fly_release_array(_goldArray));
        ASSERT_SUCCESS(fly_release_array(goldArray));
    }
}

TYPED_TEST(IterativeDeconvolution, LandweberOnGrayscale) {
    // Test file name format: <colorspace>_<iterations>_<number/1000:relaxation
    // factor>_<algo>.test
    iterDeconvImageTest<TypeParam, false>(
        string(TEST_DIR "/iterative_deconv/gray_100_50_landweber.test"), 100,
        0.05, FLY_ITERATIVE_DECONV_LANDWEBER);
}

TYPED_TEST(IterativeDeconvolution, RichardsonLucyOnGrayscale) {
    // Test file name format: <colorspace>_<iterations>_<number/1000:relaxation
    // factor>_<algo>.test For RichardsonLucy algorithm, relaxation factor is
    // not used.
    iterDeconvImageTest<TypeParam, false>(
        string(TEST_DIR "/iterative_deconv/gray_100_50_lucy.test"), 100, 0.05,
        FLY_ITERATIVE_DECONV_RICHARDSONLUCY);
}
