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

using fly::array;
using fly::exception;
using fly::fluxFunction;
using fly::max;
using fly::min;
using fly::randu;
using std::abs;
using std::string;
using std::vector;

template<typename T>
class AnisotropicDiffusion : public ::testing::Test {};

typedef ::testing::Types<float, double, int, uint, uchar, short, ushort>
    TestTypes;

TYPED_TEST_SUITE(AnisotropicDiffusion, TestTypes);

template<typename T>
array normalize(const array &p_in) {
    T mx = max<T>(p_in);
    T mn = min<T>(p_in);
    return (p_in - mn) / (mx - mn);
}

template<typename T, bool isColor>
void imageTest(string pTestFile, const float dt, const float K,
               const uint iters, fluxFunction fluxKind,
               bool isCurvatureDiffusion = false) {
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
        if (isCurvatureDiffusion) {
            inFiles[testId].insert(0, string(TEST_DIR "/curvature_diffusion/"));
            outFiles[testId].insert(0,
                                    string(TEST_DIR "/curvature_diffusion/"));
        } else {
            inFiles[testId].insert(0, string(TEST_DIR "/gradient_diffusion/"));
            outFiles[testId].insert(0, string(TEST_DIR "/gradient_diffusion/"));
        }

        fly_array _inArray   = 0;
        fly_array inArray    = 0;
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

        ASSERT_SUCCESS(
            fly_load_image(&_inArray, inFiles[testId].c_str(), isColor));
        ASSERT_SUCCESS(conv_image<T>(&inArray, _inArray));

        ASSERT_SUCCESS(
            fly_load_image(&_goldArray, outFiles[testId].c_str(), isColor));
        // fly_load_image always returns float array, so convert to output type
        ASSERT_SUCCESS(conv_image<OutType>(&goldArray, _goldArray));
        ASSERT_SUCCESS(fly_get_elements(&nElems, goldArray));

        if (isCurvatureDiffusion) {
            ASSERT_SUCCESS(fly_anisotropic_diffusion(&_outArray, inArray, dt, K,
                                                    iters, fluxKind,
                                                    FLY_DIFFUSION_MCDE));
        } else {
            ASSERT_SUCCESS(fly_anisotropic_diffusion(&_outArray, inArray, dt, K,
                                                    iters, fluxKind,
                                                    FLY_DIFFUSION_GRAD));
        }

        double maxima, minima, imag;
        ASSERT_SUCCESS(fly_min_all(&minima, &imag, _outArray));
        ASSERT_SUCCESS(fly_max_all(&maxima, &imag, _outArray));

        unsigned ndims;
        dim_t dims[4];
        ASSERT_SUCCESS(fly_get_numdims(&ndims, _outArray));
        ASSERT_SUCCESS(
            fly_get_dims(dims, dims + 1, dims + 2, dims + 3, _outArray));

        fly_dtype otype = (fly_dtype)fly::dtype_traits<OutType>::fly_type;
        ASSERT_SUCCESS(fly_constant(&cstArray, 255.0, ndims, dims, otype));
        ASSERT_SUCCESS(
            fly_constant(&denArray, (maxima - minima), ndims, dims, otype));
        ASSERT_SUCCESS(fly_constant(&minArray, minima, ndims, dims, otype));
        ASSERT_SUCCESS(fly_sub(&numArray, _outArray, minArray, false));
        ASSERT_SUCCESS(fly_div(&divArray, numArray, denArray, false));
        ASSERT_SUCCESS(fly_mul(&outArray, divArray, cstArray, false));

        ASSERT_IMAGES_NEAR(goldArray, outArray, 0.025);

        ASSERT_SUCCESS(fly_release_array(_inArray));
        ASSERT_SUCCESS(fly_release_array(_outArray));
        ASSERT_SUCCESS(fly_release_array(inArray));
        ASSERT_SUCCESS(fly_release_array(cstArray));
        ASSERT_SUCCESS(fly_release_array(minArray));
        ASSERT_SUCCESS(fly_release_array(denArray));
        ASSERT_SUCCESS(fly_release_array(numArray));
        ASSERT_SUCCESS(fly_release_array(divArray));
        ASSERT_SUCCESS(fly_release_array(outArray));
        ASSERT_SUCCESS(fly_release_array(_goldArray));
        ASSERT_SUCCESS(fly_release_array(goldArray));
    }
}

TYPED_TEST(AnisotropicDiffusion, GradientGrayscale) {
    // Numeric values separated by underscore are arguments to fn being tested.
    // Divide first value by 1000 to get time step `dt`
    // Divide second value by 100 to get time step `K`
    // Divide third value stays as it is since it is iteration count
    // Fourth value is a 4-character string indicating the flux kind
    imageTest<TypeParam, false>(
        string(TEST_DIR "/gradient_diffusion/gray_00125_100_2_exp.test"),
        0.125f, 1.0, 2, FLY_FLUX_EXPONENTIAL);
}

TYPED_TEST(AnisotropicDiffusion, GradientColorImage) {
    imageTest<TypeParam, true>(
        string(TEST_DIR "/gradient_diffusion/color_00125_100_2_exp.test"),
        0.125f, 1.0, 2, FLY_FLUX_EXPONENTIAL);
}

TEST(AnisotropicDiffusion, GradientInvalidInputArray) {
    try {
        array out = anisotropicDiffusion(randu(100), 0.125f, 0.2f, 10,
                                         FLY_FLUX_QUADRATIC);
    } catch (exception &exp) { ASSERT_EQ(FLY_ERR_SIZE, exp.err()); }
}

TYPED_TEST(AnisotropicDiffusion, CurvatureGrayscale) {
    // Numeric values separated by underscore are arguments to fn being tested.
    // Divide first value by 1000 to get time step `dt`
    // Divide second value by 100 to get time step `K`
    // Divide third value stays as it is since it is iteration count
    // Fourth value is a 4-character string indicating the flux kind
    imageTest<TypeParam, false>(
        string(TEST_DIR "/curvature_diffusion/gray_00125_100_2_mcde.test"),
        0.125f, 1.0, 2, FLY_FLUX_EXPONENTIAL, true);
}

TYPED_TEST(AnisotropicDiffusion, CurvatureColorImage) {
    imageTest<TypeParam, true>(
        string(TEST_DIR "/curvature_diffusion/color_00125_100_2_mcde.test"),
        0.125f, 1.0, 2, FLY_FLUX_EXPONENTIAL, true);
}

TEST(AnisotropicDiffusion, CurvatureInvalidInputArray) {
    try {
        array out = anisotropicDiffusion(randu(100), 0.125f, 0.2f, 10);
    } catch (exception &exp) { ASSERT_EQ(FLY_ERR_SIZE, exp.err()); }
}
