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

using fly::cdouble;
using fly::cfloat;
using fly::dtype_traits;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class HammingMatcher8 : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

template<typename T>
class HammingMatcher32 : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

// create lists of types to be tested
typedef ::testing::Types<uchar, ushort> TestTypes8;
typedef ::testing::Types<uint, uintl> TestTypes32;

// register the type list
TYPED_TEST_SUITE(HammingMatcher8, TestTypes8);
TYPED_TEST_SUITE(HammingMatcher32, TestTypes32);

template<typename T>
void hammingMatcherTest(string pTestFile, int feat_dim) {
    using fly::dim4;

    vector<dim4> numDims;
    vector<vector<uint>> in32;
    vector<vector<uint>> tests;

    readTests<uint, uint, int>(pTestFile, numDims, in32, tests);

    vector<vector<T>> in(in32.size());
    for (size_t i = 0; i < in32[0].size(); i++) in[0].push_back((T)in32[0][i]);
    for (size_t i = 0; i < in32[1].size(); i++) in[1].push_back((T)in32[1][i]);

    dim4 qDims     = numDims[0];
    dim4 tDims     = numDims[1];
    fly_array query = 0;
    fly_array train = 0;
    fly_array idx   = 0;
    fly_array dist  = 0;

    ASSERT_SUCCESS(fly_create_array(&query, &(in[0].front()), qDims.ndims(),
                                   qDims.get(),
                                   (fly_dtype)dtype_traits<T>::fly_type));
    ASSERT_SUCCESS(fly_create_array(&train, &(in[1].front()), tDims.ndims(),
                                   tDims.get(),
                                   (fly_dtype)dtype_traits<T>::fly_type));

    ASSERT_SUCCESS(fly_hamming_matcher(&idx, &dist, query, train, feat_dim, 1));

    vector<uint> goldIdx  = tests[0];
    vector<uint> goldDist = tests[1];
    size_t nElems         = goldIdx.size();
    uint *outIdx          = new uint[nElems];
    uint *outDist         = new uint[nElems];

    ASSERT_SUCCESS(fly_get_data_ptr((void *)outIdx, idx));
    ASSERT_SUCCESS(fly_get_data_ptr((void *)outDist, dist));

    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(goldDist[elIter], outDist[elIter])
            << "at: " << elIter << endl;
    }

    delete[] outIdx;
    delete[] outDist;
    ASSERT_SUCCESS(fly_release_array(query));
    ASSERT_SUCCESS(fly_release_array(train));
    ASSERT_SUCCESS(fly_release_array(idx));
    ASSERT_SUCCESS(fly_release_array(dist));
}

TYPED_TEST(HammingMatcher8, Hamming_500_5000_Dim0) {
    hammingMatcherTest<TypeParam>(
        string(TEST_DIR "/hamming/hamming_500_5000_dim0_u8.test"), 0);
}

TYPED_TEST(HammingMatcher8, Hamming_500_5000_Dim1) {
    hammingMatcherTest<TypeParam>(
        string(TEST_DIR "/hamming/hamming_500_5000_dim1_u8.test"), 1);
}

TYPED_TEST(HammingMatcher32, Hamming_500_5000_Dim0) {
    hammingMatcherTest<TypeParam>(
        string(TEST_DIR "/hamming/hamming_500_5000_dim0_u32.test"), 0);
}

TYPED_TEST(HammingMatcher32, Hamming_500_5000_Dim1) {
    hammingMatcherTest<TypeParam>(
        string(TEST_DIR "/hamming/hamming_500_5000_dim1_u32.test"), 1);
}

///////////////////////////////////// CPP ////////////////////////////////
//
TEST(HammingMatcher, CPP) {
    using fly::array;
    using fly::dim4;

    vector<dim4> numDims;
    vector<vector<uint>> in;
    vector<vector<uint>> tests;

    readTests<uint, uint, int>(
        TEST_DIR "/hamming/hamming_500_5000_dim0_u32.test", numDims, in, tests);

    dim4 qDims = numDims[0];
    dim4 tDims = numDims[1];

    array query(qDims, &(in[0].front()));
    array train(tDims, &(in[1].front()));

    array idx, dist;
    hammingMatcher(idx, dist, query, train, 0, 1);

    vector<uint> goldIdx  = tests[0];
    vector<uint> goldDist = tests[1];
    size_t nElems         = goldIdx.size();
    uint *outIdx          = new uint[nElems];
    uint *outDist         = new uint[nElems];

    idx.host(outIdx);
    dist.host(outDist);

    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(goldDist[elIter], outDist[elIter])
            << "at: " << elIter << endl;
    }

    delete[] outIdx;
    delete[] outDist;
}

TEST(HammingMatcher64bit, CPP) {
    using fly::array;
    using fly::dim4;

    vector<dim4> numDims;
    vector<vector<unsigned long long>> in;
    vector<vector<unsigned long long>> tests;

    readTests<unsigned long long, unsigned long long, int>(
        TEST_DIR "/hamming/hamming_500_5000_dim0_u32.test", numDims, in, tests);

    dim4 qDims = numDims[0];
    dim4 tDims = numDims[1];

    array query(qDims, &(in[0].front()));
    array train(tDims, &(in[1].front()));

    array idx, dist;
    hammingMatcher(idx, dist, query, train, 0, 1);

    vector<unsigned long long> goldIdx  = tests[0];
    vector<unsigned long long> goldDist = tests[1];
    size_t nElems                       = goldIdx.size();
    uint *outIdx                        = new uint[nElems];
    uint *outDist                       = new uint[nElems];

    idx.host(outIdx);
    dist.host(outDist);

    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(goldDist[elIter], outDist[elIter])
            << "at: " << elIter << endl;
    }

    delete[] outIdx;
    delete[] outDist;
}
