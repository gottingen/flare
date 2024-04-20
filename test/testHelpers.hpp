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
#pragma once
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wparentheses"
#endif
#include <fly/half.hpp>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#include <fly/array.h>
#include <fly/defines.h>
#include <fly/dim4.hpp>
#include <fly/traits.hpp>

#include <gtest/gtest.h>

#include <cfloat>
#include <string>
#include <vector>

#if defined(USE_MTX)
#include <mmio.h>
#include <cstdlib>
#endif

/// GTest deprecated the INSTANTIATED_TEST_CASE_P macro in favor of the
/// INSTANTIATE_TEST_SUITE_P macro which has the same syntax but the older
/// versions of gtest do not support this new macro adds the
/// INSTANTIATE_TEST_SUITE_P macro and maps it to the old macro
#ifndef INSTANTIATE_TEST_SUITE_P
#define INSTANTIATE_TEST_SUITE_P INSTANTIATE_TEST_CASE_P
#endif
#ifndef TYPED_TEST_SUITE
#define TYPED_TEST_SUITE TYPED_TEST_CASE
#endif

bool operator==(const fly_half &lhs, const fly_half &rhs);

std::ostream &operator<<(std::ostream &os, const fly_half &val);

#define UNUSED(expr) \
    do { (void)(expr); } while (0)

namespace aft {
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
typedef intl intl;
typedef uintl uintl;
#ifdef __GNUC__
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
}  // namespace aft

using aft::intl;
using aft::uintl;

std::ostream &operator<<(std::ostream &os, fly::Backend bk);

std::ostream &operator<<(std::ostream &os, fly_err e);

std::ostream &operator<<(std::ostream &os, fly::dtype type);

namespace fly {
template<>
struct dtype_traits<half_float::half> {
    enum { fly_type = f16, ctype = f16 };
    typedef half_float::half base_type;
    static const char *getName() { return "half"; }
};

}  // namespace fly

typedef unsigned char uchar;
typedef unsigned int uint;
typedef unsigned short ushort;

std::string getBackendName();
std::string getTestName();

std::string readNextNonEmptyLine(std::ifstream &file);

namespace half_float {
std::ostream &operator<<(std::ostream &os, half_float::half val);
}  // namespace half_float

template<typename To, typename Ti>
To convert(Ti in) {
    return static_cast<To>(in);
}

#ifndef EXTERN_TEMPLATE
extern template float convert(fly::half in);
extern template fly_half convert(int in);
#endif

template<typename inType, typename outType, typename FileElementType>
void readTests(const std::string &FileName, std::vector<fly::dim4> &inputDims,
               std::vector<std::vector<inType>> &testInputs,
               std::vector<std::vector<outType>> &testOutputs);

template<typename inType, typename outType>
void readTestsFromFile(const std::string &FileName,
                       std::vector<fly::dim4> &inputDims,
                       std::vector<std::vector<inType>> &testInputs,
                       std::vector<std::vector<outType>> &testOutputs);

void readImageTests(const std::string &pFileName,
                    std::vector<fly::dim4> &pInputDims,
                    std::vector<std::string> &pTestInputs,
                    std::vector<dim_t> &pTestOutSizes,
                    std::vector<std::string> &pTestOutputs);

template<typename outType>
void readImageTests(const std::string &pFileName,
                    std::vector<fly::dim4> &pInputDims,
                    std::vector<std::string> &pTestInputs,
                    std::vector<std::vector<outType>> &pTestOutputs);

template<typename descType>
void readImageFeaturesDescriptors(
    const std::string &pFileName, std::vector<fly::dim4> &pInputDims,
    std::vector<std::string> &pTestInputs,
    std::vector<std::vector<float>> &pTestFeats,
    std::vector<std::vector<descType>> &pTestDescs);

/**
 * Below is not a pair wise comparition method, rather
 * it computes the accumulated error of the computed
 * output and gold output.
 *
 * The cut off is decided based on root mean square
 * deviation from cpu result
 *
 * For images, the maximum possible error will happen if all
 * the observed values are zeros and all the predicted values
 * are 255's. In such case, the value of NRMSD will be 1.0
 * Similarly, we can deduce that 0.0 will be the minimum
 * value of NRMSD. Hence, the range of RMSD is [0,255] for image inputs.
 */
template<typename T>
bool compareArraysRMSD(dim_t data_size, T *gold, T *data, double tolerance);

template<typename T>
double computeArraysRMSD(dim_t data_size, T *gold, T *data);

template<typename T, typename Other>
struct is_same_type {
    static const bool value = false;
};

template<typename T>
struct is_same_type<T, T> {
    static const bool value = true;
};

template<bool, typename T, typename O>
struct cond_type;

template<typename T, typename Other>
struct cond_type<true, T, Other> {
    typedef T type;
};

template<typename T, typename Other>
struct cond_type<false, T, Other> {
    typedef Other type;
};

template<bool B, class T = void>
struct enable_if {};

template<class T>
struct enable_if<true, T> {
    typedef T type;
};

template<typename T>
inline double real(T val) {
    return (double)val;
}
template<>
inline double real<fly::cdouble>(fly::cdouble val) {
    return real(val);
}
template<>
inline double real<fly::cfloat>(fly::cfloat val) {
    return real(val);
}

template<typename T>
inline double imag(T val) {
    return (double)val;
}
template<>
inline double imag<fly::cdouble>(fly::cdouble val) {
    return imag(val);
}
template<>
inline double imag<fly::cfloat>(fly::cfloat val) {
    return imag(val);
}

template<class T>
struct IsComplex {
    static const bool value = is_same_type<fly::cfloat, T>::value ||
                              is_same_type<fly::cdouble, T>::value;
};

template<class T>
struct IsFloatingPoint {
    static const bool value = is_same_type<half_float::half, T>::value ||
                              is_same_type<float, T>::value ||
                              is_same_type<double, T>::value ||
                              is_same_type<long double, T>::value;
};

bool noDoubleTests(fly::dtype ty);

bool noHalfTests(fly::dtype ty);

#define SUPPORTED_TYPE_CHECK(type)                                \
    if (noDoubleTests((fly_dtype)fly::dtype_traits<type>::fly_type)) \
        GTEST_SKIP() << "Device doesn't support Doubles";         \
    if (noHalfTests((fly_dtype)fly::dtype_traits<type>::fly_type))   \
    GTEST_SKIP() << "Device doesn't support Half"

#define LAPACK_ENABLED_CHECK() \
    if (!fly::isLAPACKAvailable()) GTEST_SKIP() << "LAPACK Not Configured."

#define IMAGEIO_ENABLED_CHECK() \
    if (!fly::isImageIOAvailable()) GTEST_SKIP() << "Image IO Not Configured"

#ifdef FLY_WITH_FAST_MATH
#define SKIP_IF_FAST_MATH_ENABLED() \
    GTEST_SKIP() << "Flare compiled with FLY_WITH_FAST_MATH"
#else
#define SKIP_IF_FAST_MATH_ENABLED()
#endif

template<typename TO, typename FROM>
TO convert_to(FROM in) {
    return TO(in);
}

// TODO: perform conversion on device for CUDA
template<typename T>
fly_err conv_image(fly_array *out, fly_array in);

template<typename T>
fly::array cpu_randu(const fly::dim4 dims);

void cleanSlate();

//********** flare custom test asserts ***********

// Overloading unary + op is needed to make unsigned char values printable
//  as numbers
fly_half abs(fly_half in);

fly_half operator-(fly_half lhs, fly_half rhs);

const fly::cfloat &operator+(const fly::cfloat &val);

const fly::cdouble &operator+(const fly::cdouble &val);

const fly_half &operator+(const fly_half &val);

// Calculate a multi-dimensional coordinates' linearized index
dim_t ravelIdx(fly::dim4 coords, fly::dim4 strides);

// Calculate a linearized index's multi-dimensonal coordinates in an fly::array,
//  given its dimension sizes and strides
fly::dim4 unravelIdx(dim_t idx, fly::dim4 dims, fly::dim4 strides);

fly::dim4 unravelIdx(dim_t idx, fly::array arr);

fly::dim4 calcStrides(const fly::dim4 &parentDim);

std::string minimalDim4(fly::dim4 coords, fly::dim4 dims);

template<typename T>
std::string printContext(const std::vector<T> &hGold, std::string goldName,
                         const std::vector<T> &hOut, std::string outName,
                         fly::dim4 arrDims, fly::dim4 arrStrides, dim_t idx);

struct FloatTag {};
struct IntegerTag {};

template<typename T>
::testing::AssertionResult elemWiseEq(std::string aName, std::string bName,
                                      const std::vector<T> &a, fly::dim4 aDims,
                                      const std::vector<T> &b, fly::dim4 bDims,
                                      float maxAbsDiff, IntegerTag);

template<typename T>
::testing::AssertionResult elemWiseEq(std::string aName, std::string bName,
                                      const std::vector<T> &a, fly::dim4 aDims,
                                      const std::vector<T> &b, fly::dim4 bDims,
                                      float maxAbsDiff, FloatTag);

template<typename T>
::testing::AssertionResult elemWiseEq(std::string aName, std::string bName,
                                      const fly::array &a, const fly::array &b,
                                      float maxAbsDiff);

::testing::AssertionResult assertArrayEq(std::string aName, std::string bName,
                                         const fly::array &a, const fly::array &b,
                                         float maxAbsDiff = 0.f);

// Called by ASSERT_VEC_ARRAY_EQ
template<typename T>
::testing::AssertionResult assertArrayEq(std::string aName,
                                         std::string aDimsName,
                                         std::string bName,
                                         const std::vector<T> &hA,
                                         fly::dim4 aDims, const fly::array &b,
                                         float maxAbsDiff = 0.0f);

// To support C API
::testing::AssertionResult assertArrayEq(std::string aName, std::string bName,
                                         const fly_array a, const fly_array b);

// To support C API
template<typename T>
::testing::AssertionResult assertArrayEq(std::string hA_name,
                                         std::string aDimsName,
                                         std::string bName,
                                         const std::vector<T> &hA,
                                         fly::dim4 aDims, const fly_array b);

// Called by ASSERT_ARRAYS_NEAR
::testing::AssertionResult assertArrayNear(std::string aName, std::string bName,
                                           std::string maxAbsDiffName,
                                           const fly::array &a,
                                           const fly::array &b,
                                           float maxAbsDiff);

::testing::AssertionResult assertImageNear(std::string aName, std::string bName,
                                           std::string maxAbsDiffName,
                                           const fly_array &a, const fly_array &b,
                                           float maxAbsDiff);

::testing::AssertionResult assertImageNear(std::string aName, std::string bName,
                                           std::string maxAbsDiffName,
                                           const fly::array &a,
                                           const fly::array &b,
                                           float maxAbsDiff);

// Called by ASSERT_VEC_ARRAY_NEAR
template<typename T>
::testing::AssertionResult assertArrayNear(
    std::string hA_name, std::string aDimsName, std::string bName,
    std::string maxAbsDiffName, const std::vector<T> &hA, fly::dim4 aDims,
    const fly::array &b, float maxAbsDiff);

// To support C API
::testing::AssertionResult assertArrayNear(std::string aName, std::string bName,
                                           std::string maxAbsDiffName,
                                           const fly_array a, const fly_array b,
                                           float maxAbsDiff);

// To support C API
template<typename T>
::testing::AssertionResult assertArrayNear(
    std::string hA_name, std::string aDimsName, std::string bName,
    std::string maxAbsDiffName, const std::vector<T> &hA, fly::dim4 aDims,
    const fly_array b, float maxAbsDiff);

::testing::AssertionResult assertRefEq(std::string hA_name,
                                       std::string expected_name,
                                       const fly::array &a, int expected);

/// Checks if the C-API flare function returns successfully
///
/// \param[in] CALL This is the flare C function
#define ASSERT_SUCCESS(CALL) ASSERT_EQ(FLY_SUCCESS, CALL)

/// Compares two fly::array or fly_arrays for their types, dims, and values
/// (strict equality).
///
/// \param[in] EXPECTED The expected array of the assertion
/// \param[in] ACTUAL The actual resulting array from the calculation
#define ASSERT_ARRAYS_EQ(EXPECTED, ACTUAL) \
    ASSERT_PRED_FORMAT2(assertArrayEq, EXPECTED, ACTUAL)

/// Same as ASSERT_ARRAYS_EQ, but for cases when a "special" output array is
/// given to the function.
/// The special array can be null, a full-sized array, a subarray, or reordered
/// Can only be used for testing C-API functions currently
///
/// \param[in] EXPECTED The expected array of the assertion
/// \param[in] ACTUAL The actual resulting array from the calculation
#define ASSERT_SPECIAL_ARRAYS_EQ(EXPECTED, ACTUAL, META) \
    ASSERT_PRED_FORMAT3(assertArrayEq, EXPECTED, ACTUAL, META)

/// Compares a std::vector with an fly::/fly_array for their types, dims, and
/// values (strict equality).
///
/// \param[in] EXPECTED_VEC The vector that represents the expected array
/// \param[in] EXPECTED_ARR_DIMS The dimensions of the expected array
/// \param[in] ACTUAL_ARR The actual resulting array from the calculation
#define ASSERT_VEC_ARRAY_EQ(EXPECTED_VEC, EXPECTED_ARR_DIMS, ACTUAL_ARR) \
    ASSERT_PRED_FORMAT3(assertArrayEq, EXPECTED_VEC, EXPECTED_ARR_DIMS,  \
                        ACTUAL_ARR)

/// Compares two fly::array or fly_arrays for their types, dims, and values
/// (strict equality).
///
/// \param[in] EXPECTED The expected array of the assertion
/// \param[in] ACTUAL The actual resulting array from the calculation
#define EXPECT_ARRAYS_EQ(EXPECTED, ACTUAL) \
    EXPECT_PRED_FORMAT2(assertArrayEq, EXPECTED, ACTUAL)

/// Same as EXPECT_ARRAYS_EQ, but for cases when a "special" output array is
/// given to the function.
/// The special array can be null, a full-sized array, a subarray, or reordered
/// Can only be used for testing C-API functions currently
///
/// \param[in] EXPECTED The expected array of the assertion
/// \param[in] ACTUAL The actual resulting array from the calculation
#define EXPECT_SPECIAL_ARRAYS_EQ(EXPECTED, ACTUAL, META) \
    EXPECT_PRED_FORMAT3(assertArrayEq, EXPECTED, ACTUAL, META)

/// Compares a std::vector with an fly::/fly_array for their types, dims, and
/// values (strict equality).
///
/// \param[in] EXPECTED_VEC The vector that represents the expected array
/// \param[in] EXPECTED_ARR_DIMS The dimensions of the expected array
/// \param[in] ACTUAL_ARR The actual resulting array from the calculation
#define EXPECT_VEC_ARRAY_EQ(EXPECTED_VEC, EXPECTED_ARR_DIMS, ACTUAL_ARR) \
    EXPECT_PRED_FORMAT3(assertArrayEq, EXPECTED_VEC, EXPECTED_ARR_DIMS,  \
                        ACTUAL_ARR)

/// Compares two fly::array or fly_arrays for their type, dims, and values (with a
/// given tolerance).
///
/// \param[in] EXPECTED Expected value of the assertion
/// \param[in] ACTUAL Actual value of the calculation
/// \param[in] MAX_ABSDIFF Expected maximum absolute difference between
///            elements of EXPECTED and ACTUAL
///
/// \NOTE: This macro will deallocate the fly_arrays after the call
#define ASSERT_ARRAYS_NEAR(EXPECTED, ACTUAL, MAX_ABSDIFF) \
    ASSERT_PRED_FORMAT3(assertArrayNear, EXPECTED, ACTUAL, MAX_ABSDIFF)

/// Compares two fly::array or fly_arrays for their type, dims, and values (with a
/// given tolerance).
///
/// \param[in] EXPECTED Expected value of the assertion
/// \param[in] ACTUAL Actual value of the calculation
/// \param[in] MAX_ABSDIFF Expected maximum absolute difference between
///            elements of EXPECTED and ACTUAL
///
/// \NOTE: This macro will deallocate the fly_arrays after the call
#define ASSERT_IMAGES_NEAR(EXPECTED, ACTUAL, MAX_ABSDIFF) \
    ASSERT_PRED_FORMAT3(assertImageNear, EXPECTED, ACTUAL, MAX_ABSDIFF)

/// Compares a std::vector with an fly::array for their dims and values (with a
/// given tolerance).
///
/// \param[in] EXPECTED_VEC The vector that represents the expected array
/// \param[in] EXPECTED_ARR_DIMS The dimensions of the expected array
/// \param[in] ACTUAL_ARR The actual array from the calculation
/// \param[in] MAX_ABSDIFF Expected maximum absolute difference between
///            elements of EXPECTED and ACTUAL
#define ASSERT_VEC_ARRAY_NEAR(EXPECTED_VEC, EXPECTED_ARR_DIMS, ACTUAL_ARR, \
                              MAX_ABSDIFF)                                 \
    ASSERT_PRED_FORMAT4(assertArrayNear, EXPECTED_VEC, EXPECTED_ARR_DIMS,  \
                        ACTUAL_ARR, MAX_ABSDIFF)

/// Compares two fly::array or fly_arrays for their type, dims, and values (with a
/// given tolerance).
///
/// \param[in] EXPECTED Expected value of the assertion
/// \param[in] ACTUAL Actual value of the calculation
/// \param[in] MAX_ABSDIFF Expected maximum absolute difference between
///            elements of EXPECTED and ACTUAL
///
/// \NOTE: This macro will deallocate the fly_arrays after the call
#define EXPECT_ARRAYS_NEAR(EXPECTED, ACTUAL, MAX_ABSDIFF) \
    EXPECT_PRED_FORMAT3(assertArrayNear, EXPECTED, ACTUAL, MAX_ABSDIFF)

/// Compares two fly::array or fly_arrays for their type, dims, and values (with a
/// given tolerance).
///
/// \param[in] EXPECTED Expected value of the assertion
/// \param[in] ACTUAL Actual value of the calculation
/// \param[in] MAX_ABSDIFF Expected maximum absolute difference between
///            elements of EXPECTED and ACTUAL
///
/// \NOTE: This macro will deallocate the fly_arrays after the call
#define EXPECT_IMAGES_NEAR(EXPECTED, ACTUAL, MAX_ABSDIFF) \
    EXPECT_PRED_FORMAT3(assertImageNear, EXPECTED, ACTUAL, MAX_ABSDIFF)

/// Compares a std::vector with an fly::array for their dims and values (with a
/// given tolerance).
///
/// \param[in] EXPECTED_VEC The vector that represents the expected array
/// \param[in] EXPECTED_ARR_DIMS The dimensions of the expected array
/// \param[in] ACTUAL_ARR The actual array from the calculation
/// \param[in] MAX_ABSDIFF Expected maximum absolute difference between
///            elements of EXPECTED and ACTUAL
#define EXPECT_VEC_ARRAY_NEAR(EXPECTED_VEC, EXPECTED_ARR_DIMS, ACTUAL_ARR, \
                              MAX_ABSDIFF)                                 \
    EXPECT_PRED_FORMAT4(assertArrayNear, EXPECTED_VEC, EXPECTED_ARR_DIMS,  \
                        ACTUAL_ARR, MAX_ABSDIFF)

#define ASSERT_REF(arr, expected) \
    ASSERT_PRED_FORMAT2(assertRefEq, arr, expected)

#if defined(USE_MTX)
::testing::AssertionResult mtxReadSparseMatrix(fly::array &out,
                                               const char *fileName);
#endif  // USE_MTX

enum TestOutputArrayType {
    // Test fly_* function when given a null array as its output
    NULL_ARRAY,

    // Test fly_* function when given an output array that is the same size as
    // the expected output
    FULL_ARRAY,

    // Test fly_* function when given an output array that is a sub-array of a
    // larger array (the sub-array size is still the same size as the expected
    // output). Only the sub-array must be modified by the fly_* function
    SUB_ARRAY,

    // Test fly_* function when given an output array that was previously
    // reordered (but after the reorder, has still the same shape as the
    // expected
    // output). This specifically uses the reorder behavior when dim0 is kept,
    // and thus no data movement is done - only the dims and strides are
    // modified
    REORDERED_ARRAY
};

class TestOutputArrayInfo {
    fly_array out_arr;
    fly_array out_arr_cpy;
    fly_array out_subarr;
    dim_t out_subarr_ndims;
    fly_seq out_subarr_idxs[4];
    TestOutputArrayType out_arr_type;

   public:
    TestOutputArrayInfo();

    TestOutputArrayInfo(TestOutputArrayType arr_type);

    ~TestOutputArrayInfo();

    void init(const unsigned ndims, const dim_t *const dims, const fly_dtype ty);

    void init(const unsigned ndims, const dim_t *const dims, const fly_dtype ty,
              const fly_seq *const subarr_idxs);

    void init(double val, const unsigned ndims, const dim_t *const dims,
              const fly_dtype ty);

    void init(double val, const unsigned ndims, const dim_t *const dims,
              const fly_dtype ty, const fly_seq *const subarr_idxs);

    fly_array getOutput();

    void setOutput(fly_array array);

    fly_array getFullOutput();
    fly_array getFullOutputCopy();
    fly_seq *getSubArrayIdxs();
    dim_t getSubArrayNumDims();
    TestOutputArrayType getOutputArrayType();
};

// Generates a random array. testWriteToOutputArray expects that it will receive
// the same fly_array that this generates after the fly_* function is called
void genRegularArray(TestOutputArrayInfo *metadata, const unsigned ndims,
                     const dim_t *const dims, const fly_dtype ty);

void genRegularArray(TestOutputArrayInfo *metadata, double val,
                     const unsigned ndims, const dim_t *const dims,
                     const fly_dtype ty);

// Generates a large, random array, and extracts a subarray for the fly_*
// function to use. testWriteToOutputArray expects that the large array that it
// receives is equal to the same large array with the gold array injected on the
// same subarray location
void genSubArray(TestOutputArrayInfo *metadata, const unsigned ndims,
                 const dim_t *const dims, const fly_dtype ty);

void genSubArray(TestOutputArrayInfo *metadata, double val,
                 const unsigned ndims, const dim_t *const dims,
                 const fly_dtype ty);

// Generates a reordered array. testWriteToOutputArray expects that this array
// will still have the correct output values from the fly_* function, even though
// the array was initially reordered.
void genReorderedArray(TestOutputArrayInfo *metadata, const unsigned ndims,
                       const dim_t *const dims, const fly_dtype ty);

void genReorderedArray(TestOutputArrayInfo *metadata, double val,
                       const unsigned ndims, const dim_t *const dims,
                       const fly_dtype ty);
// Partner function of testWriteToOutputArray. This generates the "special"
// array that testWriteToOutputArray will use to check if the fly_* function
// correctly uses an existing array as its output
void genTestOutputArray(fly_array *out_ptr, const unsigned ndims,
                        const dim_t *const dims, const fly_dtype ty,
                        TestOutputArrayInfo *metadata);

void genTestOutputArray(fly_array *out_ptr, double val, const unsigned ndims,
                        const dim_t *const dims, const fly_dtype ty,
                        TestOutputArrayInfo *metadata);

// Partner function of genTestOutputArray. This uses the same "special"
// array that genTestOutputArray generates, and checks whether the
// fly_* function wrote to that array correctly
::testing::AssertionResult testWriteToOutputArray(
    std::string gold_name, std::string result_name, const fly_array gold,
    const fly_array out, TestOutputArrayInfo *metadata);

// Called by ASSERT_SPECIAL_ARRAYS_EQ
::testing::AssertionResult assertArrayEq(std::string aName, std::string bName,
                                         std::string metadataName,
                                         const fly_array a, const fly_array b,
                                         TestOutputArrayInfo *metadata);

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
