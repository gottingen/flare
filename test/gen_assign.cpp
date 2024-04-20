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
#include <fly/data.h>
#include <fly/defines.h>
#include <fly/dim4.hpp>
#include <fly/traits.hpp>

#include <testHelpers.hpp>
#include <algorithm>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

using fly::array;
using fly::dim4;
using fly::dtype_traits;
using fly::exception;
using fly::freeHost;
using fly::randu;
using fly::seq;
using fly::span;
using fly::where;
using std::endl;
using std::ostream_iterator;
using std::string;
using std::vector;

void testGeneralAssignOneArray(string pTestFile, const dim_t ndims,
                               fly_index_t *indexs, int arrayDim) {
    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;

    readTestsFromFile<float, float>(pTestFile, numDims, in, tests);

    dim4 dims0        = numDims[0];
    dim4 dims1        = numDims[1];
    dim4 dims2        = numDims[2];
    fly_array outArray = 0;
    fly_array rhsArray = 0;
    fly_array lhsArray = 0;
    fly_array idxArray = 0;

    ASSERT_SUCCESS(fly_create_array(&lhsArray, &(in[0].front()), dims0.ndims(),
                                   dims0.get(),
                                   (fly_dtype)dtype_traits<float>::fly_type));

    ASSERT_SUCCESS(fly_create_array(&rhsArray, &(in[1].front()), dims1.ndims(),
                                   dims1.get(),
                                   (fly_dtype)dtype_traits<float>::fly_type));

    ASSERT_SUCCESS(fly_create_array(&idxArray, &(in[2].front()), dims2.ndims(),
                                   dims2.get(),
                                   (fly_dtype)dtype_traits<float>::fly_type));
    indexs[arrayDim].idx.arr = idxArray;

    ASSERT_SUCCESS(fly_assign_gen(&outArray, lhsArray, ndims, indexs, rhsArray));

    vector<float> currGoldBar = tests[0];
    size_t nElems             = currGoldBar.size();
    vector<float> outData(nElems);

    ASSERT_SUCCESS(fly_get_data_ptr((void *)outData.data(), outArray));

    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])
            << "at: " << elIter << endl;
    }

    ASSERT_SUCCESS(fly_release_array(rhsArray));
    ASSERT_SUCCESS(fly_release_array(lhsArray));
    ASSERT_SUCCESS(fly_release_array(idxArray));
    ASSERT_SUCCESS(fly_release_array(outArray));
}

TEST(GeneralAssign, ASSS) {
    fly_index_t indexs[2];
    indexs[1].idx.seq = fly_make_seq(0, 9, 1);
    indexs[0].isSeq   = false;
    indexs[1].isSeq   = true;

    testGeneralAssignOneArray(string(TEST_DIR "/gen_assign/as0_9s0_ns0_n.test"),
                              2, indexs, 0);
}

TEST(GeneralAssign, SASS) {
    fly_index_t indexs[2];
    indexs[0].idx.seq = fly_make_seq(10, 14, 1);
    indexs[0].isSeq   = true;
    indexs[1].isSeq   = false;

    testGeneralAssignOneArray(
        string(TEST_DIR "/gen_assign/s10_14as0_ns0_n.test"), 2, indexs, 1);
}

TEST(GeneralAssign, SSSS) {
    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;

    readTestsFromFile<float, float>(
        string(TEST_DIR "/gen_assign/s10_14s0_9s0_ns0_n.test"), numDims, in,
        tests);

    dim4 dims0        = numDims[0];
    dim4 dims1        = numDims[1];
    fly_array outArray = 0;
    fly_array rhsArray = 0;
    fly_array lhsArray = 0;

    fly_index_t indexs[2];
    indexs[0].idx.seq = fly_make_seq(10, 14, 1);
    indexs[1].idx.seq = fly_make_seq(0, 9, 1);
    indexs[0].isSeq   = true;
    indexs[1].isSeq   = true;

    ASSERT_SUCCESS(fly_create_array(&lhsArray, &(in[0].front()), dims0.ndims(),
                                   dims0.get(),
                                   (fly_dtype)dtype_traits<float>::fly_type));

    ASSERT_SUCCESS(fly_create_array(&rhsArray, &(in[1].front()), dims1.ndims(),
                                   dims1.get(),
                                   (fly_dtype)dtype_traits<float>::fly_type));

    ASSERT_SUCCESS(fly_assign_gen(&outArray, lhsArray, 2, indexs, rhsArray));

    vector<float> currGoldBar = tests[0];
    size_t nElems             = currGoldBar.size();
    vector<float> outData(nElems);

    ASSERT_SUCCESS(fly_get_data_ptr((void *)outData.data(), outArray));

    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])
            << "at: " << elIter << endl;
    }

    ASSERT_SUCCESS(fly_release_array(rhsArray));
    ASSERT_SUCCESS(fly_release_array(lhsArray));
    ASSERT_SUCCESS(fly_release_array(outArray));
}

TEST(GeneralAssign, AAAA) {
    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;

    readTestsFromFile<float, float>(string(TEST_DIR "/gen_assign/aaaa.test"),
                                    numDims, in, tests);

    dim4 dims0         = numDims[0];
    dim4 dims1         = numDims[1];
    dim4 dims2         = numDims[2];
    dim4 dims3         = numDims[3];
    dim4 dims4         = numDims[4];
    dim4 dims5         = numDims[5];
    fly_array outArray  = 0;
    fly_array rhsArray  = 0;
    fly_array lhsArray  = 0;
    fly_array idxArray0 = 0;
    fly_array idxArray1 = 0;
    fly_array idxArray2 = 0;
    fly_array idxArray3 = 0;

    fly_index_t indexs[4];
    indexs[0].isSeq = false;
    indexs[1].isSeq = false;
    indexs[2].isSeq = false;
    indexs[3].isSeq = false;

    ASSERT_SUCCESS(fly_create_array(&lhsArray, &(in[0].front()), dims0.ndims(),
                                   dims0.get(),
                                   (fly_dtype)dtype_traits<float>::fly_type));

    ASSERT_SUCCESS(fly_create_array(&rhsArray, &(in[1].front()), dims1.ndims(),
                                   dims1.get(),
                                   (fly_dtype)dtype_traits<float>::fly_type));

    ASSERT_SUCCESS(fly_create_array(&idxArray0, &(in[2].front()), dims2.ndims(),
                                   dims2.get(),
                                   (fly_dtype)dtype_traits<float>::fly_type));
    indexs[0].idx.arr = idxArray0;

    ASSERT_SUCCESS(fly_create_array(&idxArray1, &(in[3].front()), dims3.ndims(),
                                   dims3.get(),
                                   (fly_dtype)dtype_traits<float>::fly_type));
    indexs[1].idx.arr = idxArray1;

    ASSERT_SUCCESS(fly_create_array(&idxArray2, &(in[4].front()), dims4.ndims(),
                                   dims4.get(),
                                   (fly_dtype)dtype_traits<float>::fly_type));
    indexs[2].idx.arr = idxArray2;

    ASSERT_SUCCESS(fly_create_array(&idxArray3, &(in[5].front()), dims5.ndims(),
                                   dims5.get(),
                                   (fly_dtype)dtype_traits<float>::fly_type));
    indexs[3].idx.arr = idxArray3;

    ASSERT_SUCCESS(fly_assign_gen(&outArray, lhsArray, 4, indexs, rhsArray));

    vector<float> currGoldBar = tests[0];
    size_t nElems             = currGoldBar.size();
    vector<float> outData(nElems);

    ASSERT_SUCCESS(fly_get_data_ptr((void *)outData.data(), outArray));

    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])
            << "at: " << elIter << endl;
    }

    ASSERT_SUCCESS(fly_release_array(rhsArray));
    ASSERT_SUCCESS(fly_release_array(lhsArray));
    ASSERT_SUCCESS(fly_release_array(outArray));
    ASSERT_SUCCESS(fly_release_array(idxArray0));
    ASSERT_SUCCESS(fly_release_array(idxArray1));
    ASSERT_SUCCESS(fly_release_array(idxArray2));
    ASSERT_SUCCESS(fly_release_array(idxArray3));
}

TEST(ArrayAssign, CPP_ASSIGN_INDEX) {
    using fly::array;

    const int num = 20000;

    array a    = randu(num);
    float *hAO = a.host<float>();

    array a_copy  = a;
    array idx     = where(a < 0.5);
    const int len = idx.elements();
    array b       = randu(len);
    a(idx)        = b;

    float *hA  = a.host<float>();
    float *hB  = b.host<float>();
    float *hAC = a_copy.host<float>();
    uint *hIdx = idx.host<uint>();

    for (int i = 0; i < num; i++) {
        int j = 0;
        while (j < len) {
            // If index found, value should match B
            if ((int)hIdx[j] == i) {
                ASSERT_EQ(hA[i], hB[j]);
                break;
            }
            j++;
        }

        // If index not found, value should match original
        if (j >= len) { ASSERT_EQ(hA[i], hAO[i]); }
    }

    // hAC should not be modified, i.e. same as original
    for (int i = 0; i < num; i++) { ASSERT_EQ(hAO[i], hAC[i]); }

    freeHost(hA);
    freeHost(hB);
    freeHost(hAC);
    freeHost(hAO);
    freeHost(hIdx);
}

TEST(ArrayAssign, CPP_ASSIGN_INDEX_LOGICAL) {
    try {
        using fly::array;

        const int num = 20000;

        array a    = randu(num);
        float *hAO = a.host<float>();

        array a_copy  = a;
        array idx     = where(a < 0.5);
        const int len = idx.elements();
        array b       = randu(len);
        a(a < 0.5)    = b;

        float *hA  = a.host<float>();
        float *hB  = b.host<float>();
        float *hAC = a_copy.host<float>();
        uint *hIdx = idx.host<uint>();

        for (int i = 0; i < num; i++) {
            int j = 0;
            while (j < len) {
                // If index found, value should match B
                if ((int)hIdx[j] == i) {
                    ASSERT_EQ(hA[i], hB[j]);
                    break;
                }
                j++;
            }

            // If index not found, value should match original
            if (j >= len) { ASSERT_EQ(hA[i], hAO[i]); }
        }

        // hAC should not be modified, i.e. same as original
        for (int i = 0; i < num; i++) { ASSERT_EQ(hAO[i], hAC[i]); }

        freeHost(hA);
        freeHost(hB);
        freeHost(hAC);
        freeHost(hAO);
        freeHost(hIdx);
    } catch (exception &ex) { FAIL() << ex.what() << endl; }
}

TEST(GeneralAssign, CPP_ASNN) {
    const int nx = 1000;
    const int ny = 1000;
    const int st = 200;
    const int en = 805;

    array a   = randu(nx, ny);
    array idx = where(randu(ny) > 0.5);

    const int nyb = (en - st) + 1;
    const int nxb = idx.elements();

    array b = randu(nxb, nyb);

    a(idx, seq(st, en)) = b;

    float *hA  = a.host<float>();
    uint *hIdx = idx.host<uint>();
    float *hB  = b.host<float>();

    for (int j = 0; j < nyb; j++) {
        float *hAt = hA + (st + j) * nx;
        float *hBt = hB + j * nxb;
        for (int i = 0; i < nxb; i++) {
            ASSERT_EQ(hAt[hIdx[i]], hBt[i]) << "at " << i << " " << j << endl;
        }
    }

    freeHost(hA);
    freeHost(hB);
    freeHost(hIdx);
}

TEST(GeneralAssign, CPP_SANN) {
    const int nx = 1000;
    const int ny = 1000;
    const int st = 200;
    const int en = 805;

    array a   = randu(nx, ny);
    array idx = where(randu(ny) > 0.5);

    const int nxb = (en - st) + 1;
    const int nyb = idx.elements();

    array b = randu(nxb, nyb);

    a(seq(st, en), idx) = b;

    float *hA  = a.host<float>();
    uint *hIdx = idx.host<uint>();
    float *hB  = b.host<float>();

    for (int j = 0; j < nyb; j++) {
        float *hAt = hA + hIdx[j] * nx;
        float *hBt = hB + j * nxb;

        for (int i = 0; i < nxb; i++) {
            ASSERT_EQ(hAt[i + st], hBt[i]) << "at " << i << " " << j << endl;
        }
    }

    freeHost(hA);
    freeHost(hB);
    freeHost(hIdx);
}

TEST(GeneralAssign, CPP_SSAN) {
    const int nx = 100;
    const int ny = 100;
    const int nz = 100;
    const int st = 20;
    const int en = 85;

    array a   = randu(nx, ny, nz);
    array idx = where(randu(nz) > 0.5);

    const int nxb = (en - st) + 1;
    const int nyb = ny;
    const int nzb = idx.elements();
    array b       = randu(nxb, nyb, nzb);

    a(seq(st, en), span, idx) = b;

    float *hA  = a.host<float>();
    uint *hIdx = idx.host<uint>();
    float *hB  = b.host<float>();

    for (int k = 0; k < nzb; k++) {
        float *hAt = hA + hIdx[k] * nx * ny;
        float *hBt = hB + k * nxb * nyb;

        for (int j = 0; j < nyb; j++) {
            for (int i = 0; i < nxb; i++) {
                ASSERT_EQ(hAt[j * nx + i + st], hBt[j * nxb + i])
                    << "at " << i << " " << j << " " << k << endl;
            }
        }
    }

    freeHost(hA);
    freeHost(hB);
    freeHost(hIdx);
}

TEST(GeneralAssign, CPP_AANN) {
    const int nx = 1000;
    const int ny = 1000;

    array a    = randu(nx, ny);
    array idx0 = where(randu(nx) > 0.5);
    array idx1 = where(randu(ny) > 0.5);

    const int nxb = idx0.elements();
    const int nyb = idx1.elements();
    array b       = randu(nxb, nyb);

    a(idx0, idx1) = b;

    float *hA   = a.host<float>();
    uint *hIdx0 = idx0.host<uint>();
    uint *hIdx1 = idx1.host<uint>();
    float *hB   = b.host<float>();

    for (int j = 0; j < nyb; j++) {
        float *hAt = hA + hIdx1[j] * nx;
        float *hBt = hB + j * nxb;
        for (int i = 0; i < nxb; i++) {
            ASSERT_EQ(hAt[hIdx0[i]], hBt[i]) << "at " << i << " " << j << endl;
        }
    }

    freeHost(hA);
    freeHost(hB);
    freeHost(hIdx0);
    freeHost(hIdx1);
}
