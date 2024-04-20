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

using fly::dim4;
using fly::dtype_traits;
using std::abs;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class Morph : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<float, double, int, uint, char, uchar, short, ushort>
    TestTypes;

// register the type list
TYPED_TEST_SUITE(Morph, TestTypes);

template<typename inType, bool isDilation, bool isVolume>
void morphTest(string pTestFile) {
    SUPPORTED_TYPE_CHECK(inType);

    vector<dim4> numDims;
    vector<vector<inType>> in;
    vector<vector<inType>> tests;

    readTests<inType, inType, int>(pTestFile, numDims, in, tests);

    dim4 dims          = numDims[0];
    dim4 maskDims      = numDims[1];
    fly_array outArray  = 0;
    fly_array inArray   = 0;
    fly_array maskArray = 0;

    ASSERT_SUCCESS(fly_create_array(&inArray, &(in[0].front()), dims.ndims(),
                                   dims.get(),
                                   (fly_dtype)dtype_traits<inType>::fly_type));
    ASSERT_SUCCESS(fly_create_array(&maskArray, &(in[1].front()),
                                   maskDims.ndims(), maskDims.get(),
                                   (fly_dtype)dtype_traits<inType>::fly_type));

    if (isDilation) {
        if (isVolume)
            ASSERT_SUCCESS(fly_dilate3(&outArray, inArray, maskArray));
        else
            ASSERT_SUCCESS(fly_dilate(&outArray, inArray, maskArray));
    } else {
        if (isVolume)
            ASSERT_SUCCESS(fly_erode3(&outArray, inArray, maskArray));
        else
            ASSERT_SUCCESS(fly_erode(&outArray, inArray, maskArray));
    }

    for (size_t testIter = 0; testIter < tests.size(); ++testIter) {
        vector<inType> currGoldBar = tests[testIter];
        ASSERT_VEC_ARRAY_EQ(currGoldBar, dims, outArray);
    }

    // cleanup
    ASSERT_SUCCESS(fly_release_array(inArray));
    ASSERT_SUCCESS(fly_release_array(maskArray));
    ASSERT_SUCCESS(fly_release_array(outArray));
}

TYPED_TEST(Morph, Dilate3x3) {
    morphTest<TypeParam, true, false>(string(TEST_DIR "/morph/dilate3x3.test"));
}

TYPED_TEST(Morph, Erode3x3) {
    morphTest<TypeParam, false, false>(string(TEST_DIR "/morph/erode3x3.test"));
}

TYPED_TEST(Morph, Dilate4x4) {
    morphTest<TypeParam, true, false>(string(TEST_DIR "/morph/dilate4x4.test"));
}

TYPED_TEST(Morph, Dilate12x12) {
    morphTest<TypeParam, true, false>(
        string(TEST_DIR "/morph/dilate12x12.test"));
}

TYPED_TEST(Morph, Erode4x4) {
    morphTest<TypeParam, false, false>(string(TEST_DIR "/morph/erode4x4.test"));
}

TYPED_TEST(Morph, Dilate3x3_Batch) {
    morphTest<TypeParam, true, false>(
        string(TEST_DIR "/morph/dilate3x3_batch.test"));
}

TYPED_TEST(Morph, Erode3x3_Batch) {
    morphTest<TypeParam, false, false>(
        string(TEST_DIR "/morph/erode3x3_batch.test"));
}

TYPED_TEST(Morph, Dilate3x3x3) {
    morphTest<TypeParam, true, true>(
        string(TEST_DIR "/morph/dilate3x3x3.test"));
}

TYPED_TEST(Morph, Erode3x3x3) {
    morphTest<TypeParam, false, true>(
        string(TEST_DIR "/morph/erode3x3x3.test"));
}

TYPED_TEST(Morph, Dilate4x4x4) {
    morphTest<TypeParam, true, true>(
        string(TEST_DIR "/morph/dilate4x4x4.test"));
}

TYPED_TEST(Morph, Erode4x4x4) {
    morphTest<TypeParam, false, true>(
        string(TEST_DIR "/morph/erode4x4x4.test"));
}

template<typename T, bool isDilation, bool isColor>
void morphImageTest(string pTestFile, dim_t seLen) {
    SUPPORTED_TYPE_CHECK(T);
    IMAGEIO_ENABLED_CHECK();

    vector<dim4> inDims;
    vector<string> inFiles;
    vector<dim_t> outSizes;
    vector<string> outFiles;

    readImageTests(pTestFile, inDims, inFiles, outSizes, outFiles);

    size_t testCount = inDims.size();

    for (size_t testId = 0; testId < testCount; ++testId) {
        fly_array _inArray   = 0;
        fly_array inArray    = 0;
        fly_array maskArray  = 0;
        fly_array outArray   = 0;
        fly_array _goldArray = 0;
        fly_array goldArray  = 0;
        dim_t nElems        = 0;

        inFiles[testId].insert(0, string(TEST_DIR "/morph/"));
        outFiles[testId].insert(0, string(TEST_DIR "/morph/"));

        fly_dtype targetType = static_cast<fly_dtype>(dtype_traits<T>::fly_type);

        dim4 mdims(seLen, seLen, 1, 1);
        ASSERT_SUCCESS(fly_constant(&maskArray, 1.0, mdims.ndims(), mdims.get(),
                                   targetType));

        ASSERT_SUCCESS(
            fly_load_image(&_inArray, inFiles[testId].c_str(), isColor));
        ASSERT_SUCCESS(fly_cast(&inArray, _inArray, targetType));

        ASSERT_SUCCESS(
            fly_load_image(&_goldArray, outFiles[testId].c_str(), isColor));
        ASSERT_SUCCESS(fly_cast(&goldArray, _goldArray, targetType));

        ASSERT_SUCCESS(fly_get_elements(&nElems, goldArray));

        fly_err error_code = FLY_SUCCESS;
        if (isDilation) {
            error_code = fly_dilate(&outArray, inArray, maskArray);
        } else {
            error_code = fly_erode(&outArray, inArray, maskArray);
        }

#if defined(FLY_CPU)
        ASSERT_SUCCESS(error_code);
        ASSERT_IMAGES_NEAR(goldArray, outArray, 0.018f);
#else
        ASSERT_EQ(error_code,
                  (targetType != b8 && seLen > 19 ? FLY_ERR_NOT_SUPPORTED
                                                  : FLY_SUCCESS));
        if (!(targetType != b8 && seLen > 19)) {
            ASSERT_IMAGES_NEAR(goldArray, outArray, 0.018f);
        }
#endif

        ASSERT_SUCCESS(fly_release_array(_inArray));
        ASSERT_SUCCESS(fly_release_array(inArray));
        ASSERT_SUCCESS(fly_release_array(maskArray));
        ASSERT_SUCCESS(fly_release_array(outArray));
        ASSERT_SUCCESS(fly_release_array(_goldArray));
        ASSERT_SUCCESS(fly_release_array(goldArray));
    }
}

TEST(Morph, GrayscaleDilation3x3StructuringElement) {
    morphImageTest<float, true, false>(string(TEST_DIR "/morph/gray.test"), 3);
}

TEST(Morph, ColorImageErosion3x3StructuringElement) {
    morphImageTest<float, false, true>(string(TEST_DIR "/morph/color.test"), 3);
}

TEST(Morph, BinaryImageDilationBy33x33Kernel) {
    morphImageTest<char, true, false>(
        string(TEST_DIR "/morph/zag_dilation.test"), 33);
}

TEST(Morph, BinaryImageErosionBy33x33Kernel) {
    morphImageTest<char, false, false>(
        string(TEST_DIR "/morph/zag_erosion.test"), 33);
}

TEST(Morph, DilationBy33x33Kernel) {
    morphImageTest<float, true, true>(
        string(TEST_DIR "/morph/baboon_dilation.test"), 33);
}

TEST(Morph, ErosionBy33x33Kernel) {
    morphImageTest<float, false, true>(
        string(TEST_DIR "/morph/baboon_erosion.test"), 33);
}

template<typename T, bool isDilation>
void morphInputTest(void) {
    SUPPORTED_TYPE_CHECK(T);

    fly_array inArray   = 0;
    fly_array maskArray = 0;
    fly_array outArray  = 0;

    vector<T> in(100, 1);
    vector<T> mask(9, 1);

    // Check for 1D inputs
    dim4 dims = dim4(100, 1, 1, 1);
    dim4 mdims(3, 3, 1, 1);

    ASSERT_SUCCESS(fly_create_array(&maskArray, &mask.front(), mdims.ndims(),
                                   mdims.get(),
                                   (fly_dtype)dtype_traits<T>::fly_type));

    ASSERT_SUCCESS(fly_create_array(&inArray, &in.front(), dims.ndims(),
                                   dims.get(),
                                   (fly_dtype)dtype_traits<T>::fly_type));

    if (isDilation)
        ASSERT_EQ(FLY_ERR_SIZE, fly_dilate(&outArray, inArray, maskArray));
    else
        ASSERT_EQ(FLY_ERR_SIZE, fly_erode(&outArray, inArray, maskArray));

    ASSERT_SUCCESS(fly_release_array(inArray));

    ASSERT_SUCCESS(fly_release_array(maskArray));
}

TYPED_TEST(Morph, DilateInvalidInput) { morphInputTest<TypeParam, true>(); }

TYPED_TEST(Morph, ErodeInvalidInput) { morphInputTest<TypeParam, false>(); }

template<typename T, bool isDilation>
void morphMaskTest(void) {
    SUPPORTED_TYPE_CHECK(T);

    fly_array inArray   = 0;
    fly_array maskArray = 0;
    fly_array outArray  = 0;

    vector<T> in(100, 1);
    vector<T> mask(16, 1);

    // Check for 4D mask
    dim4 dims(10, 10, 1, 1);
    dim4 mdims(2, 2, 2, 2);

    ASSERT_SUCCESS(fly_create_array(&inArray, &in.front(), dims.ndims(),
                                   dims.get(),
                                   (fly_dtype)dtype_traits<T>::fly_type));

    ASSERT_SUCCESS(fly_create_array(&maskArray, &mask.front(), mdims.ndims(),
                                   mdims.get(),
                                   (fly_dtype)dtype_traits<T>::fly_type));

    if (isDilation)
        ASSERT_EQ(FLY_ERR_SIZE, fly_dilate(&outArray, inArray, maskArray));
    else
        ASSERT_EQ(FLY_ERR_SIZE, fly_erode(&outArray, inArray, maskArray));

    ASSERT_SUCCESS(fly_release_array(maskArray));

    // Check for 1D mask
    mdims = dim4(16, 1, 1, 1);

    ASSERT_SUCCESS(fly_create_array(&maskArray, &mask.front(), mdims.ndims(),
                                   mdims.get(),
                                   (fly_dtype)dtype_traits<T>::fly_type));

    if (isDilation)
        ASSERT_EQ(FLY_ERR_SIZE, fly_dilate(&outArray, inArray, maskArray));
    else
        ASSERT_EQ(FLY_ERR_SIZE, fly_erode(&outArray, inArray, maskArray));

    ASSERT_SUCCESS(fly_release_array(maskArray));

    ASSERT_SUCCESS(fly_release_array(inArray));
}

TYPED_TEST(Morph, DilateInvalidMask) { morphMaskTest<TypeParam, true>(); }

TYPED_TEST(Morph, ErodeInvalidMask) { morphMaskTest<TypeParam, false>(); }

template<typename T, bool isDilation>
void morph3DMaskTest(void) {
    SUPPORTED_TYPE_CHECK(T);

    fly_array inArray   = 0;
    fly_array maskArray = 0;
    fly_array outArray  = 0;

    vector<T> in(1000, 1);
    vector<T> mask(81, 1);

    // Check for 2D mask
    dim4 dims(10, 10, 10, 1);
    dim4 mdims(9, 9, 1, 1);

    ASSERT_SUCCESS(fly_create_array(&inArray, &in.front(), dims.ndims(),
                                   dims.get(),
                                   (fly_dtype)dtype_traits<T>::fly_type));

    ASSERT_SUCCESS(fly_create_array(&maskArray, &mask.front(), mdims.ndims(),
                                   mdims.get(),
                                   (fly_dtype)dtype_traits<T>::fly_type));

    if (isDilation)
        ASSERT_EQ(FLY_ERR_SIZE, fly_dilate3(&outArray, inArray, maskArray));
    else
        ASSERT_EQ(FLY_ERR_SIZE, fly_erode3(&outArray, inArray, maskArray));

    ASSERT_SUCCESS(fly_release_array(maskArray));

    // Check for 4D mask
    mdims = dim4(3, 3, 3, 3);

    ASSERT_SUCCESS(fly_create_array(&maskArray, &mask.front(), mdims.ndims(),
                                   mdims.get(),
                                   (fly_dtype)dtype_traits<T>::fly_type));

    if (isDilation)
        ASSERT_EQ(FLY_ERR_SIZE, fly_dilate3(&outArray, inArray, maskArray));
    else
        ASSERT_EQ(FLY_ERR_SIZE, fly_erode3(&outArray, inArray, maskArray));

    ASSERT_SUCCESS(fly_release_array(maskArray));

    ASSERT_SUCCESS(fly_release_array(inArray));
}

TYPED_TEST(Morph, DilateVolumeInvalidMask) {
    morph3DMaskTest<TypeParam, true>();
}

TYPED_TEST(Morph, ErodeVolumeInvalidMask) {
    morph3DMaskTest<TypeParam, false>();
}

////////////////////////////////////// CPP //////////////////////////////////
//

using fly::array;
using fly::constant;
using fly::erode;
using fly::iota;
using fly::loadImage;
using fly::max;
using fly::randu;
using fly::seq;
using fly::span;

template<typename T, bool isDilation, bool isColor>
void cppMorphImageTest(string pTestFile) {
    SUPPORTED_TYPE_CHECK(T);
    IMAGEIO_ENABLED_CHECK();

    vector<dim4> inDims;
    vector<string> inFiles;
    vector<dim_t> outSizes;
    vector<string> outFiles;

    readImageTests(pTestFile, inDims, inFiles, outSizes, outFiles);

    size_t testCount = inDims.size();

    for (size_t testId = 0; testId < testCount; ++testId) {
        inFiles[testId].insert(0, string(TEST_DIR "/morph/"));
        outFiles[testId].insert(0, string(TEST_DIR "/morph/"));

        array mask   = constant(1.0, 3, 3);
        array img    = loadImage(inFiles[testId].c_str(), isColor);
        array gold   = loadImage(outFiles[testId].c_str(), isColor);
        dim_t nElems = gold.elements();
        array output;

        if (isDilation)
            output = dilate(img, mask);
        else
            output = erode(img, mask);

        vector<T> outData(nElems);
        output.host((void*)outData.data());

        vector<T> goldData(nElems);
        gold.host((void*)goldData.data());

        ASSERT_EQ(true, compareArraysRMSD(nElems, goldData.data(),
                                          outData.data(), 0.018f));
    }
}

TEST(Morph, Grayscale_CPP) {
    cppMorphImageTest<float, true, false>(string(TEST_DIR "/morph/gray.test"));
}

TEST(Morph, ColorImage_CPP) {
    cppMorphImageTest<float, false, true>(string(TEST_DIR "/morph/color.test"));
}

TEST(Morph, GFOR) {
    dim4 dims  = dim4(10, 10, 3);
    array A    = iota(dims);
    array B    = constant(0, dims);
    array mask = randu(3, 3) > 0.3;

    gfor(seq ii, 3) { B(span, span, ii) = erode(A(span, span, ii), mask); }

    for (int ii = 0; ii < 3; ii++) {
        array c_ii = erode(A(span, span, ii), mask);
        array b_ii = B(span, span, ii);
        ASSERT_EQ(max<double>(abs(c_ii - b_ii)) < 1E-5, true);
    }
}

TEST(Morph, EdgeIssue1564) {
    int inputData[10 * 10] = {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
                              0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    int goldData[10 * 10]  = {0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                              0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1,
                              1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1,
                              1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
    array input(10, 10, inputData);
    int maskData[3 * 3] = {1, 1, 1, 1, 0, 1, 1, 1, 1};
    array mask(3, 3, maskData);
    array dilated = dilate(input.as(b8), mask.as(b8));

    size_t nElems = dilated.elements();
    vector<char> outData(nElems);
    dilated.host((void*)outData.data());

    for (size_t i = 0; i < nElems; ++i) {
        ASSERT_EQ((int)outData[i], goldData[i]);
    }
}

TEST(Morph, UnsupportedKernel2D) {
    const unsigned ndims = 2;
    const dim_t dims[2]  = {10, 10};
    const dim_t kdims[2] = {32, 32};

    fly_array in, mask, out;

    ASSERT_SUCCESS(fly_constant(&mask, 1.0, ndims, kdims, f32));
    ASSERT_SUCCESS(fly_randu(&in, ndims, dims, f32));

#if defined(FLY_CPU)
    ASSERT_SUCCESS(fly_dilate(&out, in, mask));
    ASSERT_SUCCESS(fly_release_array(out));
#else
    ASSERT_EQ(FLY_ERR_NOT_SUPPORTED, fly_dilate(&out, in, mask));
#endif
    ASSERT_SUCCESS(fly_release_array(in));
    ASSERT_SUCCESS(fly_release_array(mask));
}
