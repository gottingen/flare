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

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using fly::array;
using fly::cdouble;
using fly::cfloat;
using fly::constant;
using fly::dim4;
using fly::dtype_traits;
using fly::fft;
using fly::fft2;
using fly::fft2InPlace;
using fly::fft3;
using fly::fft3InPlace;
using fly::fftInPlace;
using fly::ifft;
using fly::ifft2;
using fly::ifft2InPlace;
using fly::ifft3;
using fly::ifft3InPlace;
using fly::ifftInPlace;
using fly::moddims;
using fly::randu;
using fly::seq;
using fly::span;
using std::abs;
using std::endl;
using std::string;
using std::stringstream;
using std::vector;

TEST(fft, Invalid_Type) {
    vector<char> in(100, 1);

    fly_array inArray  = 0;
    fly_array outArray = 0;

    dim4 dims(5 * 5 * 2 * 2);
    ASSERT_SUCCESS(fly_create_array(&inArray, &(in.front()), dims.ndims(),
                                   dims.get(),
                                   (fly_dtype)dtype_traits<char>::fly_type));

    ASSERT_EQ(FLY_ERR_TYPE, fly_fft(&outArray, inArray, 1.0, 0));
    ASSERT_SUCCESS(fly_release_array(inArray));
}

TEST(fft2, Invalid_Array) {
    vector<float> in(100, 1);

    fly_array inArray  = 0;
    fly_array outArray = 0;

    dim4 dims(5 * 5 * 2 * 2);
    ASSERT_SUCCESS(fly_create_array(&inArray, &(in.front()), dims.ndims(),
                                   dims.get(),
                                   (fly_dtype)dtype_traits<float>::fly_type));

    ASSERT_EQ(FLY_ERR_SIZE, fly_fft2(&outArray, inArray, 1.0, 0, 0));
    ASSERT_SUCCESS(fly_release_array(inArray));
}

TEST(fft3, Invalid_Array) {
    vector<float> in(100, 1);

    fly_array inArray  = 0;
    fly_array outArray = 0;

    dim4 dims(10, 10, 1, 1);
    ASSERT_SUCCESS(fly_create_array(&inArray, &(in.front()), dims.ndims(),
                                   dims.get(),
                                   (fly_dtype)dtype_traits<float>::fly_type));

    ASSERT_EQ(FLY_ERR_SIZE, fly_fft3(&outArray, inArray, 1.0, 0, 0, 0));
    ASSERT_SUCCESS(fly_release_array(inArray));
}

TEST(ifft2, Invalid_Array) {
    vector<float> in(100, 1);

    fly_array inArray  = 0;
    fly_array outArray = 0;

    dim4 dims(100, 1, 1, 1);
    ASSERT_SUCCESS(fly_create_array(&inArray, &(in.front()), dims.ndims(),
                                   dims.get(),
                                   (fly_dtype)dtype_traits<float>::fly_type));

    ASSERT_EQ(FLY_ERR_SIZE, fly_ifft2(&outArray, inArray, 0.01, 0, 0));
    ASSERT_SUCCESS(fly_release_array(inArray));
}

TEST(ifft3, Invalid_Array) {
    vector<float> in(100, 1);

    fly_array inArray  = 0;
    fly_array outArray = 0;

    dim4 dims(10, 10, 1, 1);
    ASSERT_SUCCESS(fly_create_array(&inArray, &(in.front()), dims.ndims(),
                                   dims.get(),
                                   (fly_dtype)dtype_traits<float>::fly_type));

    ASSERT_EQ(FLY_ERR_SIZE, fly_ifft3(&outArray, inArray, 0.01, 0, 0, 0));
    ASSERT_SUCCESS(fly_release_array(inArray));
}

template<typename inType, typename outType, bool isInverse>
void fftTest(string pTestFile, dim_t pad0 = 0, dim_t pad1 = 0, dim_t pad2 = 0) {
    SUPPORTED_TYPE_CHECK(inType);
    SUPPORTED_TYPE_CHECK(outType);

    vector<dim4> numDims;
    vector<vector<inType>> in;
    vector<vector<outType>> tests;

    readTestsFromFile<inType, outType>(pTestFile, numDims, in, tests);

    dim4 dims         = numDims[0];
    fly_array outArray = 0;
    fly_array inArray  = 0;

    ASSERT_SUCCESS(fly_create_array(&inArray, &(in[0].front()), dims.ndims(),
                                   dims.get(),
                                   (fly_dtype)dtype_traits<inType>::fly_type));

    if (isInverse) {
        switch (dims.ndims()) {
            case 1:
                ASSERT_SUCCESS(fly_ifft(&outArray, inArray, 1.0, pad0));
                break;
            case 2:
                ASSERT_SUCCESS(fly_ifft2(&outArray, inArray, 1.0, pad0, pad1));
                break;
            case 3:
                ASSERT_SUCCESS(
                    fly_ifft3(&outArray, inArray, 1.0, pad0, pad1, pad2));
                break;
            default:
                throw std::runtime_error(
                    "This error shouldn't happen, pls check");
        }
    } else {
        switch (dims.ndims()) {
            case 1:
                ASSERT_SUCCESS(fly_fft(&outArray, inArray, 1.0, pad0));
                break;
            case 2:
                ASSERT_SUCCESS(fly_fft2(&outArray, inArray, 1.0, pad0, pad1));
                break;
            case 3:
                ASSERT_SUCCESS(
                    fly_fft3(&outArray, inArray, 1.0, pad0, pad1, pad2));
                break;
            default:
                throw std::runtime_error(
                    "This error shouldn't happen, pls check");
        }
    }

    size_t out_size  = tests[0].size();
    outType *outData = new outType[out_size];
    ASSERT_SUCCESS(fly_get_data_ptr((void *)outData, outArray));

    vector<outType> goldBar(tests[0].begin(), tests[0].end());

    size_t test_size = 0;
    switch (dims.ndims()) {
        case 1: test_size = dims[0] / 2 + 1; break;
        case 2: test_size = dims[1] * (dims[0] / 2 + 1); break;
        case 3: test_size = dims[2] * dims[1] * (dims[0] / 2 + 1); break;
        default: test_size = dims[0] / 2 + 1; break;
    }
    outType output_scale = (outType)(isInverse ? test_size : 1);
    for (size_t elIter = 0; elIter < test_size; ++elIter) {
        bool isUnderTolerance = abs(goldBar[elIter] - outData[elIter]) < 0.001;
        ASSERT_EQ(true, isUnderTolerance)
            << "Expected value=" << goldBar[elIter]
            << "\t Actual Value=" << (output_scale * outData[elIter])
            << " at: " << elIter << endl;
    }

    // cleanup
    delete[] outData;
    ASSERT_SUCCESS(fly_release_array(inArray));
    ASSERT_SUCCESS(fly_release_array(outArray));
}

#define INSTANTIATE_TEST(func, name, is_inverse, in_t, out_t, ...) \
    TEST(func, name) { fftTest<in_t, out_t, is_inverse>(__VA_ARGS__); }

// Real to complex transforms
INSTANTIATE_TEST(fft, R2C_Float, false, float, cfloat,
                 string(TEST_DIR "/signal/fft_r2c.test"));
INSTANTIATE_TEST(fft, R2C_Double, false, double, cdouble,
                 string(TEST_DIR "/signal/fft_r2c.test"));
INSTANTIATE_TEST(fft2, R2C_Float, false, float, cfloat,
                 string(TEST_DIR "/signal/fft2_r2c.test"));
INSTANTIATE_TEST(fft2, R2C_Double, false, double, cdouble,
                 string(TEST_DIR "/signal/fft2_r2c.test"));
INSTANTIATE_TEST(fft3, R2C_Float, false, float, cfloat,
                 string(TEST_DIR "/signal/fft3_r2c.test"));
INSTANTIATE_TEST(fft3, R2C_Double, false, double, cdouble,
                 string(TEST_DIR "/signal/fft3_r2c.test"));

// complex to complex transforms
INSTANTIATE_TEST(fft, C2C_Float, false, cfloat, cfloat,
                 string(TEST_DIR "/signal/fft_c2c.test"));
INSTANTIATE_TEST(fft, C2C_Double, false, cdouble, cdouble,
                 string(TEST_DIR "/signal/fft_c2c.test"));
INSTANTIATE_TEST(fft2, C2C_Float, false, cfloat, cfloat,
                 string(TEST_DIR "/signal/fft2_c2c.test"));
INSTANTIATE_TEST(fft2, C2C_Double, false, cdouble, cdouble,
                 string(TEST_DIR "/signal/fft2_c2c.test"));
INSTANTIATE_TEST(fft3, C2C_Float, false, cfloat, cfloat,
                 string(TEST_DIR "/signal/fft3_c2c.test"));
INSTANTIATE_TEST(fft3, C2C_Double, false, cdouble, cdouble,
                 string(TEST_DIR "/signal/fft3_c2c.test"));

// Factors 7, 11, 13
INSTANTIATE_TEST(fft, R2C_Float_7_11_13, false, float, cfloat,
                 string(TEST_DIR "/signal/fft_r2c_7_11_13.test"));
INSTANTIATE_TEST(fft, R2C_Double_7_11_13, false, double, cdouble,
                 string(TEST_DIR "/signal/fft_r2c_7_11_13.test"));
INSTANTIATE_TEST(fft2, R2C_Float_7_11_13, false, float, cfloat,
                 string(TEST_DIR "/signal/fft2_r2c_7_11_13.test"));
INSTANTIATE_TEST(fft2, R2C_Double_7_11_13, false, double, cdouble,
                 string(TEST_DIR "/signal/fft2_r2c_7_11_13.test"));
INSTANTIATE_TEST(fft3, R2C_Float_7_11_13, false, float, cfloat,
                 string(TEST_DIR "/signal/fft3_r2c_7_11_13.test"));
INSTANTIATE_TEST(fft3, R2C_Double_7_11_13, false, double, cdouble,
                 string(TEST_DIR "/signal/fft3_r2c_7_11_13.test"));

INSTANTIATE_TEST(fft, C2C_Float_7_11_13, false, cfloat, cfloat,
                 string(TEST_DIR "/signal/fft_c2c_7_11_13.test"));
INSTANTIATE_TEST(fft, C2C_Double_7_11_13, false, cdouble, cdouble,
                 string(TEST_DIR "/signal/fft_c2c_7_11_13.test"));
INSTANTIATE_TEST(fft2, C2C_Float_7_11_13, false, cfloat, cfloat,
                 string(TEST_DIR "/signal/fft2_c2c_7_11_13.test"));
INSTANTIATE_TEST(fft2, C2C_Double_7_11_13, false, cdouble, cdouble,
                 string(TEST_DIR "/signal/fft2_c2c_7_11_13.test"));
INSTANTIATE_TEST(fft3, C2C_Float_7_11_13, false, cfloat, cfloat,
                 string(TEST_DIR "/signal/fft3_c2c_7_11_13.test"));
INSTANTIATE_TEST(fft3, C2C_Double_7_11_13, false, cdouble, cdouble,
                 string(TEST_DIR "/signal/fft3_c2c_7_11_13.test"));

// transforms on padded and truncated arrays
INSTANTIATE_TEST(fft2, R2C_Float_Trunc, false, float, cfloat,
                 string(TEST_DIR "/signal/fft2_r2c_trunc.test"), 16, 16);
INSTANTIATE_TEST(fft2, R2C_Double_Trunc, false, double, cdouble,
                 string(TEST_DIR "/signal/fft2_r2c_trunc.test"), 16, 16);

INSTANTIATE_TEST(fft2, C2C_Float_Pad, false, cfloat, cfloat,
                 string(TEST_DIR "/signal/fft2_c2c_pad.test"), 16, 16);
INSTANTIATE_TEST(fft2, C2C_Double_Pad, false, cdouble, cdouble,
                 string(TEST_DIR "/signal/fft2_c2c_pad.test"), 16, 16);

// inverse transforms
// complex to complex transforms
INSTANTIATE_TEST(ifft, C2C_Float, true, cfloat, cfloat,
                 string(TEST_DIR "/signal/ifft_c2c.test"));
INSTANTIATE_TEST(ifft, C2C_Double, true, cdouble, cdouble,
                 string(TEST_DIR "/signal/ifft_c2c.test"));
INSTANTIATE_TEST(ifft2, C2C_Float, true, cfloat, cfloat,
                 string(TEST_DIR "/signal/ifft2_c2c.test"));
INSTANTIATE_TEST(ifft2, C2C_Double, true, cdouble, cdouble,
                 string(TEST_DIR "/signal/ifft2_c2c.test"));
INSTANTIATE_TEST(ifft3, C2C_Float, true, cfloat, cfloat,
                 string(TEST_DIR "/signal/ifft3_c2c.test"));
INSTANTIATE_TEST(ifft3, C2C_Double, true, cdouble, cdouble,
                 string(TEST_DIR "/signal/ifft3_c2c.test"));

template<typename inType, typename outType, int rank, bool isInverse>
void fftBatchTest(string pTestFile, dim_t pad0 = 0, dim_t pad1 = 0,
                  dim_t pad2 = 0) {
    SUPPORTED_TYPE_CHECK(inType);
    SUPPORTED_TYPE_CHECK(outType);

    vector<dim4> numDims;
    vector<vector<inType>> in;
    vector<vector<outType>> tests;

    readTestsFromFile<inType, outType>(pTestFile, numDims, in, tests);

    dim4 dims         = numDims[0];
    fly_array outArray = 0;
    fly_array inArray  = 0;

    ASSERT_SUCCESS(fly_create_array(&inArray, &(in[0].front()), dims.ndims(),
                                   dims.get(),
                                   (fly_dtype)dtype_traits<inType>::fly_type));

    if (isInverse) {
        switch (rank) {
            case 1:
                ASSERT_SUCCESS(fly_ifft(&outArray, inArray, 1.0, pad0));
                break;
            case 2:
                ASSERT_SUCCESS(fly_ifft2(&outArray, inArray, 1.0, pad0, pad1));
                break;
            case 3:
                ASSERT_SUCCESS(
                    fly_ifft3(&outArray, inArray, 1.0, pad0, pad1, pad2));
                break;
            default:
                throw std::runtime_error(
                    "This error shouldn't happen, pls check");
        }
    } else {
        switch (rank) {
            case 1:
                ASSERT_SUCCESS(fly_fft(&outArray, inArray, 1.0, pad0));
                break;
            case 2:
                ASSERT_SUCCESS(fly_fft2(&outArray, inArray, 1.0, pad0, pad1));
                break;
            case 3:
                ASSERT_SUCCESS(
                    fly_fft3(&outArray, inArray, 1.0, pad0, pad1, pad2));
                break;
            default:
                throw std::runtime_error(
                    "This error shouldn't happen, pls check");
        }
    }

    size_t out_size  = tests[0].size();
    outType *outData = new outType[out_size];
    ASSERT_SUCCESS(fly_get_data_ptr((void *)outData, outArray));

    vector<outType> goldBar(tests[0].begin(), tests[0].end());

    size_t test_size   = 0;
    size_t batch_count = dims[rank];
    switch (rank) {
        case 1: test_size = dims[0] / 2 + 1; break;
        case 2: test_size = dims[1] * (dims[0] / 2 + 1); break;
        case 3: test_size = dims[2] * dims[1] * (dims[0] / 2 + 1); break;
        default: test_size = dims[0] / 2 + 1; break;
    }

    size_t batch_stride = 1;
    for (int i = 0; i < rank; ++i) batch_stride *= dims[i];

    outType output_scale = (outType)(isInverse ? test_size : 1);
    for (size_t batchId = 0; batchId < batch_count; ++batchId) {
        size_t off = batchId * batch_stride;
        for (size_t elIter = 0; elIter < test_size; ++elIter) {
            bool isUnderTolerance =
                abs(goldBar[elIter + off] - outData[elIter + off]) < 0.001;
            ASSERT_EQ(true, isUnderTolerance)
                << "Batch id = " << batchId
                << "; Expected value=" << goldBar[elIter + off]
                << "\t Actual Value=" << (output_scale * outData[elIter + off])
                << " at: " << elIter << endl;
        }
    }

    // cleanup
    delete[] outData;
    ASSERT_SUCCESS(fly_release_array(inArray));
    ASSERT_SUCCESS(fly_release_array(outArray));
}

#define INSTANTIATE_BATCH_TEST(func, name, rank, is_inverse, in_t, out_t, ...) \
    TEST(func, name##_Batch) {                                                 \
        fftBatchTest<in_t, out_t, rank, is_inverse>(__VA_ARGS__);              \
    }

// real to complex transforms
INSTANTIATE_BATCH_TEST(fft, R2C_Float, 1, false, float, cfloat,
                       string(TEST_DIR "/signal/fft_r2c_batch.test"));
INSTANTIATE_BATCH_TEST(fft2, R2C_Float, 2, false, float, cfloat,
                       string(TEST_DIR "/signal/fft2_r2c_batch.test"));
INSTANTIATE_BATCH_TEST(fft3, R2C_Float, 3, false, float, cfloat,
                       string(TEST_DIR "/signal/fft3_r2c_batch.test"));

// complex to complex transforms
INSTANTIATE_BATCH_TEST(fft, C2C_Float, 1, false, cfloat, cfloat,
                       string(TEST_DIR "/signal/fft_c2c_batch.test"));
INSTANTIATE_BATCH_TEST(fft2, C2C_Float, 2, false, cfloat, cfloat,
                       string(TEST_DIR "/signal/fft2_c2c_batch.test"));
INSTANTIATE_BATCH_TEST(fft3, C2C_Float, 3, false, cfloat, cfloat,
                       string(TEST_DIR "/signal/fft3_c2c_batch.test"));

// inverse transforms
// complex to complex transforms
INSTANTIATE_BATCH_TEST(ifft, C2C_Float, 1, true, cfloat, cfloat,
                       string(TEST_DIR "/signal/ifft_c2c_batch.test"));
INSTANTIATE_BATCH_TEST(ifft2, C2C_Float, 2, true, cfloat, cfloat,
                       string(TEST_DIR "/signal/ifft2_c2c_batch.test"));
INSTANTIATE_BATCH_TEST(ifft3, C2C_Float, 3, true, cfloat, cfloat,
                       string(TEST_DIR "/signal/ifft3_c2c_batch.test"));

// transforms on padded and truncated arrays
INSTANTIATE_BATCH_TEST(fft2, R2C_Float_Trunc, 2, false, float, cfloat,
                       string(TEST_DIR "/signal/fft2_r2c_trunc_batch.test"), 16,
                       16);
INSTANTIATE_BATCH_TEST(fft2, R2C_Double_Trunc, 2, false, double, cdouble,
                       string(TEST_DIR "/signal/fft2_r2c_trunc_batch.test"), 16,
                       16);
INSTANTIATE_BATCH_TEST(fft2, C2C_Float_Pad, 2, false, cfloat, cfloat,
                       string(TEST_DIR "/signal/fft2_c2c_pad_batch.test"), 16,
                       16);
INSTANTIATE_BATCH_TEST(fft2, C2C_Double_Pad, 2, false, cdouble, cdouble,
                       string(TEST_DIR "/signal/fft2_c2c_pad_batch.test"), 16,
                       16);

/////////////////////////////////////// CPP ////////////////////////////////////
//
template<typename inType, typename outType, bool isInverse>
void cppFFTTest(string pTestFile) {
    SUPPORTED_TYPE_CHECK(inType);
    SUPPORTED_TYPE_CHECK(outType);

    vector<dim4> numDims;
    vector<vector<inType>> in;
    vector<vector<outType>> tests;

    readTestsFromFile<inType, outType>(pTestFile, numDims, in, tests);

    dim4 dims = numDims[0];
    array signal(dims, &(in[0].front()));
    array output;

    if (isInverse) {
        output = ifft3Norm(signal, 1.0);
    } else {
        output = fft3Norm(signal, 1.0);
    }

    size_t out_size = tests[0].size();
    cfloat *outData = new cfloat[out_size];
    output.host((void *)outData);

    vector<cfloat> goldBar(tests[0].begin(), tests[0].end());

    size_t test_size = 0;
    switch (dims.ndims()) {
        case 1: test_size = dims[0] / 2 + 1; break;
        case 2: test_size = dims[1] * (dims[0] / 2 + 1); break;
        case 3: test_size = dims[2] * dims[1] * (dims[0] / 2 + 1); break;
        default: test_size = dims[0] / 2 + 1; break;
    }
    outType output_scale = (outType)(isInverse ? test_size : 1);
    for (size_t elIter = 0; elIter < test_size; ++elIter) {
        bool isUnderTolerance = abs(goldBar[elIter] - outData[elIter]) < 0.001;
        ASSERT_EQ(true, isUnderTolerance)
            << "Expected value=" << goldBar[elIter]
            << "\t Actual Value=" << (output_scale * outData[elIter])
            << " at: " << elIter << endl;
    }
    // cleanup
    delete[] outData;
}

template<typename inType, typename outType, bool isInverse>
void cppDFTTest(string pTestFile) {
    SUPPORTED_TYPE_CHECK(inType);
    SUPPORTED_TYPE_CHECK(outType);

    vector<dim4> numDims;
    vector<vector<inType>> in;
    vector<vector<outType>> tests;

    readTestsFromFile<inType, outType>(pTestFile, numDims, in, tests);

    dim4 dims = numDims[0];
    array signal(dims, &(in[0].front()));
    array output;

    if (isInverse) {
        output = idft(signal);
    } else {
        output = dft(signal);
    }

    size_t out_size = tests[0].size();
    cfloat *outData = new cfloat[out_size];
    output.host((void *)outData);

    vector<cfloat> goldBar(tests[0].begin(), tests[0].end());

    size_t test_size = 0;
    switch (dims.ndims()) {
        case 1: test_size = dims[0] / 2 + 1; break;
        case 2: test_size = dims[1] * (dims[0] / 2 + 1); break;
        case 3: test_size = dims[2] * dims[1] * (dims[0] / 2 + 1); break;
        default: test_size = dims[0] / 2 + 1; break;
    }
    outType output_scale = (outType)(isInverse ? test_size : 1);
    for (size_t elIter = 0; elIter < test_size; ++elIter) {
        bool isUnderTolerance = abs(goldBar[elIter] - outData[elIter]) < 0.001;
        ASSERT_EQ(true, isUnderTolerance)
            << "Expected value=" << goldBar[elIter]
            << "\t Actual Value=" << (output_scale * outData[elIter])
            << " at: " << elIter << endl;
    }
    // cleanup
    delete[] outData;
}

TEST(fft3, CPP) {
    cppFFTTest<cfloat, cfloat, false>(string(TEST_DIR "/signal/fft3_c2c.test"));
}

TEST(ifft3, CPP) {
    cppFFTTest<cfloat, cfloat, true>(string(TEST_DIR "/signal/ifft3_c2c.test"));
}

TEST(fft3, RandomData) {
    array a = randu(31, 31, 31);
    array b = fft3(a, 64, 64, 64);
    array c = ifft3(b);

    dim4 aDims = a.dims();
    dim4 cDims = c.dims();
    dim4 aStrides(1, aDims[0], aDims[0] * aDims[1],
                  aDims[0] * aDims[1] * aDims[2]);
    dim4 cStrides(1, cDims[0], cDims[0] * cDims[1],
                  cDims[0] * cDims[1] * cDims[2]);

    float *gold = new float[a.elements()];
    float *out  = new float[2 * c.elements()];

    a.host((void *)gold);
    c.host((void *)out);

    for (int k = 0; k < (int)aDims[2]; ++k) {
        int gkOff = k * aStrides[2];
        int okOff = k * cStrides[2];
        for (int j = 0; j < (int)aDims[1]; ++j) {
            int gjOff = j * aStrides[1];
            int ojOff = j * cStrides[1];
            for (int i = 0; i < (int)aDims[0]; ++i) {
                int giOff = i * aStrides[0];
                int oiOff = i * cStrides[0];

                int gi = gkOff + gjOff + giOff;
                int oi = okOff + ojOff + oiOff;

                bool isUnderTolerance =
                    std::abs(gold[gi] - out[2 * oi]) < 0.001;
                ASSERT_EQ(true, isUnderTolerance)
                    << "Expected value=" << gold[gi]
                    << "\t Actual Value=" << out[2 * oi] << " at: " << gi
                    << endl;
            }
        }
    }

    delete[] gold;
    delete[] out;
}

TEST(dft, CPP) {
    cppDFTTest<cfloat, cfloat, false>(string(TEST_DIR "/signal/fft_c2c.test"));
}

TEST(idft, CPP) {
    cppDFTTest<cfloat, cfloat, true>(string(TEST_DIR "/signal/ifft_c2c.test"));
}

TEST(dft2, CPP) {
    cppDFTTest<cfloat, cfloat, false>(string(TEST_DIR "/signal/fft2_c2c.test"));
}

TEST(idft2, CPP) {
    cppDFTTest<cfloat, cfloat, true>(string(TEST_DIR "/signal/ifft2_c2c.test"));
}

TEST(dft3, CPP) {
    cppDFTTest<cfloat, cfloat, false>(string(TEST_DIR "/signal/fft3_c2c.test"));
}

TEST(idft3, CPP) {
    cppDFTTest<cfloat, cfloat, true>(string(TEST_DIR "/signal/ifft3_c2c.test"));
}

TEST(fft, CPP_4D) {
    array a = randu(1024, 1024);
    array b = fft(a);

    array A = moddims(a, 1024, 32, 16, 2);
    array B = fft(A);

    cfloat *h_b = b.host<cfloat>();
    cfloat *h_B = B.host<cfloat>();

    for (int i = 0; i < (int)a.elements(); i++) {
        ASSERT_EQ(h_b[i], h_B[i]) << "at: " << i << endl;
    }

    freeHost(h_b);
    freeHost(h_B);
}

TEST(ifft, CPP_4D) {
    array a = randu(1024, 1024, c32);
    array b = ifft(a);

    array A = moddims(a, 1024, 32, 16, 2);
    array B = ifft(A);

    cfloat *h_b = b.host<cfloat>();
    cfloat *h_B = B.host<cfloat>();

    for (int i = 0; i < (int)a.elements(); i++) {
        ASSERT_EQ(h_b[i], h_B[i]) << "at: " << i << endl;
    }

    freeHost(h_b);
    freeHost(h_B);
}

TEST(fft, GFOR) {
    array a = randu(1024, 1024);
    array b = constant(0, 1024, 1024, c32);
    array c = fft(a);

    gfor(seq ii, a.dims(1)) { b(span, ii) = fft(a(span, ii)); }

    cfloat *h_b = b.host<cfloat>();
    cfloat *h_c = c.host<cfloat>();

    for (int i = 0; i < (int)a.elements(); i++) {
        ASSERT_EQ(h_b[i], h_c[i]) << "at: " << i << endl;
    }

    freeHost(h_b);
    freeHost(h_c);
}

TEST(fft2, GFOR) {
    array a = randu(1024, 1024, 4);
    array b = constant(0, 1024, 1024, 4, c32);
    array c = fft2(a);

    gfor(seq ii, a.dims(2)) { b(span, span, ii) = fft2(a(span, span, ii)); }

    cfloat *h_b = b.host<cfloat>();
    cfloat *h_c = c.host<cfloat>();

    for (int i = 0; i < (int)a.elements(); i++) {
        ASSERT_EQ(h_b[i], h_c[i]) << "at: " << i << endl;
    }

    freeHost(h_b);
    freeHost(h_c);
}

TEST(fft3, GFOR) {
    array a = randu(32, 32, 32, 4);
    array b = constant(0, 32, 32, 32, 4, c32);
    array c = fft3(a);

    gfor(seq ii, a.dims(3)) {
        b(span, span, span, ii) = fft3(a(span, span, span, ii));
    }

    cfloat *h_b = b.host<cfloat>();
    cfloat *h_c = c.host<cfloat>();

    for (int i = 0; i < (int)a.elements(); i++) {
        ASSERT_EQ(h_b[i], h_c[i]) << "at: " << i << endl;
    }

    freeHost(h_b);
    freeHost(h_c);
}

void fft2InPlaceFunc() {
    array a = randu(1024, 1024, c32);
    array b = fft2(a);
    fft2InPlace(a);

    ASSERT_ARRAYS_EQ(a, b);
}

using fly::getDevice;
using fly::getDeviceCount;
using fly::setDevice;

#define DEVICE_ITERATE(func)                             \
    do {                                                 \
        const char *ENV = getenv("FLY_MULTI_GPU_TESTS");  \
        if (ENV && ENV[0] == '0') {                      \
            func;                                        \
        } else {                                         \
            int oldDevice = getDevice();                 \
            for (int i = 0; i < getDeviceCount(); i++) { \
                setDevice(i);                            \
                func;                                    \
            }                                            \
            setDevice(oldDevice);                        \
        }                                                \
    } while (0);

TEST(FFT2, MultiGPUInPlaceSquare_CPP) { DEVICE_ITERATE((fft2InPlaceFunc())); }

struct fft_params {
    dim4 input_dims_;
    bool is_odd_;
    double norm_factor_;
    fft_params(dim4 dim, bool is_odd, double norm_factor)
        : input_dims_(dim), is_odd_(is_odd), norm_factor_(norm_factor) {}
};

class FFTBase : public ::testing::TestWithParam<fft_params> {};

class FFTC2R2D : public FFTBase {};
class FFT2D : public FFTBase {};
class FFTC2R3D : public FFTBase {};
class FFT3D : public FFTBase {};
class FFTC2R : public FFTBase {};
class FFTND : public FFTBase {};

string to_test_params(const ::testing::TestParamInfo<FFTBase::ParamType> info) {
    stringstream ss;
    ss << "d0_" << info.param.input_dims_[0] << "_d1_"
       << info.param.input_dims_[1] << "_d2_" << info.param.input_dims_[2]
       << "_d3_" << info.param.input_dims_[3] << "_"
       << ((info.param.is_odd_) ? string("odd") : string("even")) << "_norm_"
       << info.param.norm_factor_;
    string out = ss.str();
    return out.replace(out.find("."), 1, "_");
}

// INSTANTIATE_TEST_SUITE_P(
//     Inputs2D, FFTC2R2D,
//     ::testing::Values(fft_params(dim4(513, 512), false, 0.5),
//                       fft_params(dim4(1025, 1024), false, 0.5),
//                       fft_params(dim4(2049, 2048), false, 0.5)),
//     to_test_params);

INSTANTIATE_TEST_SUITE_P(
    Inputs2D, FFT2D,
    ::testing::Values(fft_params(dim4(512, 512), false, 0.5),
                      fft_params(dim4(1024, 1024), false, 0.5),
                      fft_params(dim4(2048, 2048), false, 0.5)),
    to_test_params);

INSTANTIATE_TEST_SUITE_P(
    Inputs3D, FFTC2R3D,
    ::testing::Values(fft_params(dim4(512, 512, 3), false, 0.5),
                      fft_params(dim4(1024, 1024, 3), false, 0.5),
                      fft_params(dim4(2048, 2048, 3), false, 0.5)),
    to_test_params);

INSTANTIATE_TEST_SUITE_P(
    Inputs3D, FFT3D,
    ::testing::Values(fft_params(dim4(1024, 1024, 3), true, 0.5),
                      fft_params(dim4(1024, 1024, 3), false, 0.5)),
    to_test_params);

INSTANTIATE_TEST_SUITE_P(
    InputsND, FFTND,
    ::testing::Values(fft_params(dim4(512), false, 0.5),
                      fft_params(dim4(1024), false, 0.5),
                      fft_params(dim4(1024, 1024), false, 0.5),
                      fft_params(dim4(1024, 1024, 3), false, 0.5)),
    to_test_params);

INSTANTIATE_TEST_SUITE_P(
    InputsND, FFTC2R,
    ::testing::Values(fft_params(dim4(513), false, 0.5),
                      fft_params(dim4(1025), false, 0.5),
                      fft_params(dim4(1025, 1024), false, 0.5),
                      fft_params(dim4(1025, 1024, 3), false, 0.5)),
    to_test_params);

// Does not work well with CUDA 10.1
// TEST_P(FFTC2R2D, Complex32ToRealInputsPreserved) {
//     fft_params params = GetParam();
//     fly::array a       = fly::randu(params.input_dims_, c32);
//     fly::array a_copy  = a.copy();
//     fly::array out     = fly::fftC2R<2>(a, params.is_odd_,
//     params.norm_factor_);
//
//     ASSERT_ARRAYS_EQ(a_copy, a);
// }
//
// TEST_P(FFTC2R2D, Complex64ToRealInputsPreserved) {
//     fft_params params = GetParam();
//     fly::array a       = fly::randu(params.input_dims_, c64);
//     fly::array a_copy  = a.copy();
//     fly::array out     = fly::fftC2R<2>(a, params.is_odd_,
//     params.norm_factor_);
//
//     ASSERT_ARRAYS_EQ(a_copy, a);
// }

TEST_P(FFT2D, Real32ToComplexInputsPreserved) {
    fft_params params = GetParam();
    fly::array a       = fly::randu(params.input_dims_, f32);
    fly::array a_copy  = a.copy();
    fly::array out     = fly::fftR2C<2>(a, a.dims(), params.norm_factor_);

    ASSERT_ARRAYS_EQ(a_copy, a);
}

TEST_P(FFT2D, Real64ToComplexInputsPreserved) {
    SUPPORTED_TYPE_CHECK(double);
    fft_params params = GetParam();
    fly::array a       = fly::randu(params.input_dims_, f64);
    fly::array a_copy  = a.copy();
    fly::array out     = fly::fftR2C<2>(a, a.dims(), params.norm_factor_);

    ASSERT_ARRAYS_EQ(a_copy, a);
}

TEST_P(FFTC2R, Complex32ToRInputsPreserved) {
    fft_params params = GetParam();
    fly::array a       = fly::randu(params.input_dims_, c32);
    fly::array a_copy  = a.copy();
    fly::array out     = fly::fftC2R<1>(a, params.is_odd_, params.norm_factor_);

    ASSERT_ARRAYS_EQ(a_copy, a);
}

TEST_P(FFTC2R, Complex64ToRInputsPreserved) {
    SUPPORTED_TYPE_CHECK(double);
    fft_params params = GetParam();
    fly::array a       = fly::randu(params.input_dims_, c64);
    fly::array a_copy  = a.copy();
    fly::array out     = fly::fftC2R<1>(a, params.is_odd_, params.norm_factor_);

    ASSERT_ARRAYS_EQ(a_copy, a);
}

TEST_P(FFTND, Real32ToComplexInputsPreserved) {
    fft_params params = GetParam();
    fly::array a       = fly::randu(params.input_dims_, f32);
    fly::array a_copy  = a.copy();
    fly::array out     = fly::fftR2C<1>(a, a.dims(), params.norm_factor_);

    ASSERT_ARRAYS_EQ(a_copy, a);
}

TEST_P(FFTND, Real64ToComplexInputsPreserved) {
    SUPPORTED_TYPE_CHECK(double);
    fft_params params = GetParam();
    fly::array a       = fly::randu(params.input_dims_, f64);
    fly::array a_copy  = a.copy();
    fly::array out     = fly::fftR2C<1>(a, a.dims(), params.norm_factor_);

    ASSERT_ARRAYS_EQ(a_copy, a);
}

TEST_P(FFTND, InPlaceFFTMatchesOutOfPlace) {
    fft_params params = GetParam();
    array a           = randu(params.input_dims_, c32);
    array b           = fft(a);
    fftInPlace(a);

    ASSERT_ARRAYS_EQ(a, b);
}

TEST_P(FFTND, InPlaceIFFTMatchesOutOfPlace) {
    fft_params params = GetParam();
    array a           = randu(params.input_dims_, c32);
    array b           = ifft(a);
    ifftInPlace(a);

    ASSERT_ARRAYS_EQ(a, b);
}

TEST_P(FFT2D, InPlaceFFT2MatchesOutOfPlace) {
    fft_params params = GetParam();
    array a           = randu(params.input_dims_, c32);
    array b           = fft2(a);
    fft2InPlace(a);

    ASSERT_ARRAYS_EQ(a, b);
}

TEST_P(FFT2D, InPlaceIFFT2MatchesOutOfPlace) {
    fft_params params = GetParam();
    array a           = randu(params.input_dims_, c32);
    array b           = ifft2(a);
    ifft2InPlace(a);

    ASSERT_ARRAYS_EQ(a, b);
}

TEST_P(FFT3D, InPlaceFFT3MatchesOutOfPlace) {
    fft_params params = GetParam();
    array a           = randu(params.input_dims_, c32);
    array b           = fft3(a);
    fft3InPlace(a);

    ASSERT_ARRAYS_EQ(a, b);
}

TEST_P(FFTC2R3D, InPlaceIFFT3MatchesOutOfPlace) {
    fft_params params = GetParam();
    array a           = randu(params.input_dims_, c32);
    array b           = ifft3(a);
    ifft3InPlace(a);

    ASSERT_ARRAYS_EQ(a, b);
}
