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
#include <string>
#include <vector>

using fly::array;
using fly::dim4;
using fly::exception;
using fly::hsv2rgb;
using std::endl;
using std::string;
using std::vector;

TEST(hsv_rgb, InvalidArray) {
    vector<float> in(100, 1);

    dim4 dims(100);
    array input(dims, &(in.front()));

    try {
        array output = hsv2rgb(input);
        ASSERT_EQ(true, false);
    } catch (const exception & /* ex */) {
        ASSERT_EQ(true, true);
        return;
    }
}

TEST(hsv2rgb, CPP) {
    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;

    readTestsFromFile<float, float>(string(TEST_DIR "/hsv_rgb/hsv2rgb.test"),
                                    numDims, in, tests);

    dim4 dims = numDims[0];
    array input(dims, &(in[0].front()));
    array output = hsv2rgb(input);

    vector<float> currGoldBar = tests[0];
    ASSERT_VEC_ARRAY_NEAR(currGoldBar, dims, output, 1.0e-3);
}

TEST(rgb2hsv, CPP) {
    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;

    readTestsFromFile<float, float>(string(TEST_DIR "/hsv_rgb/rgb2hsv.test"),
                                    numDims, in, tests);

    dim4 dims = numDims[0];
    array input(dims, &(in[0].front()));
    array output = rgb2hsv(input);

    vector<float> currGoldBar = tests[0];
    ASSERT_VEC_ARRAY_NEAR(currGoldBar, dims, output, 1.0e-3);
}

TEST(rgb2hsv, MaxDim) {
    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;

    readTestsFromFile<float, float>(string(TEST_DIR "/hsv_rgb/rgb2hsv.test"),
                                    numDims, in, tests);

    dim4 dims = numDims[0];
    array input(dims, &(in[0].front()));

    const size_t largeDim = 65535 * 16 + 1;
    unsigned int ntile    = (largeDim + dims[1] - 1) / dims[1];
    input                 = tile(input, 1, ntile);
    array output          = rgb2hsv(input);
    dim4 outDims          = output.dims();

    float *outData = new float[outDims.elements()];
    output.host((void *)outData);

    vector<float> currGoldBar = tests[0];
    for (int z = 0; z < outDims[2]; ++z) {
        for (int y = 0; y < outDims[1]; ++y) {
            for (int x = 0; x < outDims[0]; ++x) {
                int outIter =
                    (z * outDims[1] * outDims[0]) + (y * outDims[0]) + x;
                int goldIter =
                    (z * dims[1] * dims[0]) + ((y % dims[1]) * dims[0]) + x;
                ASSERT_NEAR(currGoldBar[goldIter], outData[outIter], 1.0e-3)
                    << "at: " << outIter << endl;
            }
        }
    }

    // cleanup
    delete[] outData;
}

TEST(hsv2rgb, MaxDim) {
    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;

    readTestsFromFile<float, float>(string(TEST_DIR "/hsv_rgb/hsv2rgb.test"),
                                    numDims, in, tests);

    dim4 dims = numDims[0];
    array input(dims, &(in[0].front()));

    const size_t largeDim = 65535 * 16 + 1;
    unsigned int ntile    = (largeDim + dims[1] - 1) / dims[1];
    input                 = tile(input, 1, ntile);
    array output          = hsv2rgb(input);
    dim4 outDims          = output.dims();

    float *outData = new float[outDims.elements()];
    output.host((void *)outData);

    vector<float> currGoldBar = tests[0];
    for (int z = 0; z < outDims[2]; ++z) {
        for (int y = 0; y < outDims[1]; ++y) {
            for (int x = 0; x < outDims[0]; ++x) {
                int outIter =
                    (z * outDims[1] * outDims[0]) + (y * outDims[0]) + x;
                int goldIter =
                    (z * dims[1] * dims[0]) + ((y % dims[1]) * dims[0]) + x;
                ASSERT_NEAR(currGoldBar[goldIter], outData[outIter], 1.0e-3)
                    << "at: " << outIter << endl;
            }
        }
    }

    // cleanup
    delete[] outData;
}
