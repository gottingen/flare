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
#include <fly/algorithm.h>
#include <fly/dim4.hpp>
#include <fly/traits.hpp>
#include <iostream>
#include <string>
#include <vector>

using fly::cdouble;
using fly::cfloat;
using fly::dim4;
using fly::dtype_traits;
using std::cout;
using std::endl;
using std::string;
using std::vector;

template<typename T>
void uniqueTest(string pTestFile) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;

    vector<vector<int>> data;
    vector<vector<int>> tests;
    readTests<int, int, int>(pTestFile, numDims, data, tests);

    // Compare result
    for (int d = 0; d < (int)tests.size(); ++d) {
        dim4 dims = numDims[d];
        vector<T> in(data[d].begin(), data[d].end());

        fly_array inArray  = 0;
        fly_array outArray = 0;

        // Get input array
        ASSERT_SUCCESS(fly_create_array(&inArray, &in.front(), dims.ndims(),
                                       dims.get(),
                                       (fly_dtype)dtype_traits<T>::fly_type));

        vector<T> currGoldBar(tests[d].begin(), tests[d].end());

        // Run sum
        ASSERT_SUCCESS(
            fly_set_unique(&outArray, inArray, d == 0 ? false : true));

        // Get result
        vector<T> outData(currGoldBar.size());
        ASSERT_SUCCESS(fly_get_data_ptr((void *)&outData.front(), outArray));

        size_t nElems = currGoldBar.size();
        for (size_t elIter = 0; elIter < nElems; ++elIter) {
            ASSERT_EQ(currGoldBar[elIter], outData[elIter])
                << "at: " << elIter << " for test: " << d << endl;
        }

        if (inArray != 0) fly_release_array(inArray);
        if (outArray != 0) fly_release_array(outArray);
    }
}

#define UNIQUE_TESTS(T) \
    TEST(Set, Test_Unique_##T) { uniqueTest<T>(TEST_DIR "/set/unique.test"); }

UNIQUE_TESTS(float)
UNIQUE_TESTS(double)
UNIQUE_TESTS(int)
UNIQUE_TESTS(uint)
UNIQUE_TESTS(uchar)
UNIQUE_TESTS(short)
UNIQUE_TESTS(ushort)
UNIQUE_TESTS(intl)
UNIQUE_TESTS(uintl)

typedef fly_err (*setFunc)(fly_array *, const fly_array, const fly_array,
                          const bool);

template<typename T, setFunc fly_set_func>
void setTest(string pTestFile) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;

    vector<vector<int>> data;
    vector<vector<int>> tests;
    readTests<int, int, int>(pTestFile, numDims, data, tests);

    // Compare result
    for (int d = 0; d < (int)tests.size(); d += 2) {
        dim4 dims0 = numDims[d + 0];
        vector<T> in0(data[d + 0].begin(), data[d + 0].end());

        dim4 dims1 = numDims[d + 1];
        vector<T> in1(data[d + 1].begin(), data[d + 1].end());

        fly_array inArray0 = 0;
        fly_array inArray1 = 0;
        fly_array outArray = 0;

        ASSERT_SUCCESS(fly_create_array(&inArray0, &in0.front(), dims0.ndims(),
                                       dims0.get(),
                                       (fly_dtype)dtype_traits<T>::fly_type));

        ASSERT_SUCCESS(fly_create_array(&inArray1, &in1.front(), dims1.ndims(),
                                       dims1.get(),
                                       (fly_dtype)dtype_traits<T>::fly_type));
        vector<T> currGoldBar(tests[d].begin(), tests[d].end());

        // Run sum
        ASSERT_SUCCESS(
            fly_set_func(&outArray, inArray0, inArray1, d == 0 ? false : true));

        // Get result
        vector<T> outData(currGoldBar.size());
        ASSERT_SUCCESS(fly_get_data_ptr((void *)&outData.front(), outArray));

        size_t nElems = currGoldBar.size();
        for (size_t elIter = 0; elIter < nElems; ++elIter) {
            ASSERT_EQ(currGoldBar[elIter], outData[elIter])
                << "at: " << elIter << " for test: " << d << endl;
        }

        if (inArray0 != 0) fly_release_array(inArray0);
        if (inArray1 != 0) fly_release_array(inArray1);
        if (outArray != 0) fly_release_array(outArray);
    }
}

#define SET_TESTS(T)                                                  \
    TEST(Set, Test_Union_##T) {                                       \
        setTest<T, fly_set_union>(TEST_DIR "/set/union.test");         \
    }                                                                 \
    TEST(Set, Test_Intersect_##T) {                                   \
        setTest<T, fly_set_intersect>(TEST_DIR "/set/intersect.test"); \
    }

SET_TESTS(float)
SET_TESTS(double)
SET_TESTS(int)
SET_TESTS(uint)
SET_TESTS(uchar)
SET_TESTS(short)
SET_TESTS(ushort)
SET_TESTS(intl)
SET_TESTS(uintl)

// Documentation examples for setUnique
TEST(Set, SNIPPET_setUniqueSorted) {
    //! [ex_set_unique_sorted]

    // input data
    int h_set[6] = {1, 2, 2, 3, 3, 3};
    fly::array set(6, h_set);

    // is_sorted flag specifies if input is sorted,
    // allows algorithm to skip internal sorting step
    const bool is_sorted = true;
    fly::array unique     = setUnique(set, is_sorted);
    // unique == { 1, 2, 3 };

    //! [ex_set_unique_sorted]

    vector<int> unique_gold = {1, 2, 3};
    dim4 gold_dim(3, 1, 1, 1);
    ASSERT_VEC_ARRAY_EQ(unique_gold, gold_dim, unique);
}

TEST(Set, SNIPPET_setUniqueSortedDesc) {
    //! [ex_set_unique_desc]

    // input data
    int h_set[6] = {3, 3, 3, 2, 2, 1};
    fly::array set(6, h_set);

    // is_sorted flag specifies if input is sorted,
    // allows algorithm to skip internal sorting step
    // input can be sorted in ascending or descending order
    const bool is_sorted = true;
    fly::array unique     = setUnique(set, is_sorted);
    // unique == { 3, 2, 1 };

    //! [ex_set_unique_desc]

    vector<int> unique_gold = {3, 2, 1};
    dim4 gold_dim(3, 1, 1, 1);
    ASSERT_VEC_ARRAY_EQ(unique_gold, gold_dim, unique);
}

TEST(Set, SNIPPET_setUniqueSimple) {
    //! [ex_set_unique_simple]

    // input data
    int h_set[6] = {3, 2, 3, 3, 2, 1};
    fly::array set(6, h_set);

    fly::array unique = setUnique(set);
    // unique == { 1, 2, 3 };

    //! [ex_set_unique_simple]

    vector<int> unique_gold = {1, 2, 3};
    dim4 gold_dim(3, 1, 1, 1);
    ASSERT_VEC_ARRAY_EQ(unique_gold, gold_dim, unique);
}

// Documentation examples for setUnion
TEST(Set, SNIPPET_setUnion) {
    //! [ex_set_union]

    // input data
    int h_setA[4] = {1, 2, 3, 4};
    int h_setB[4] = {2, 3, 4, 5};
    fly::array setA(4, h_setA);
    fly::array setB(4, h_setB);

    const bool is_unique = true;
    // is_unique flag specifies if inputs are unique,
    // allows algorithm to skip internal calls to setUnique
    // inputs must be unique and sorted in increasing order
    fly::array setAB = setUnion(setA, setB, is_unique);
    // setAB == { 1, 2, 3, 4, 5 };

    //! [ex_set_union]

    vector<int> union_gold = {1, 2, 3, 4, 5};
    dim4 gold_dim(5, 1, 1, 1);
    ASSERT_VEC_ARRAY_EQ(union_gold, gold_dim, setAB);
}

TEST(Set, SNIPPET_setUnionSimple) {
    //! [ex_set_union_simple]

    // input data
    int h_setA[4] = {1, 2, 3, 3};
    int h_setB[4] = {3, 4, 5, 5};
    fly::array setA(4, h_setA);
    fly::array setB(4, h_setB);

    fly::array setAB = setUnion(setA, setB);
    // setAB == { 1, 2, 3, 4, 5 };

    //! [ex_set_union_simple]

    vector<int> union_gold = {1, 2, 3, 4, 5};
    dim4 gold_dim(5, 1, 1, 1);
    ASSERT_VEC_ARRAY_EQ(union_gold, gold_dim, setAB);
}

// Documentation examples for setIntersect()
TEST(Set, SNIPPET_setIntersect) {
    //! [ex_set_intersect]

    // input data
    int h_setA[4] = {1, 2, 3, 4};
    int h_setB[4] = {2, 3, 4, 5};
    fly::array setA(4, h_setA);
    fly::array setB(4, h_setB);

    const bool is_unique = true;
    // is_unique flag specifies if inputs are unique,
    // allows algorithm to skip internal calls to setUnique
    // inputs must be unique and sorted in increasing order
    fly::array setA_B = setIntersect(setA, setB, is_unique);
    // setA_B == { 2, 3, 4 };

    //! [ex_set_intersect]

    vector<int> intersect_gold = {2, 3, 4};
    dim4 gold_dim(3, 1, 1, 1);
    ASSERT_VEC_ARRAY_EQ(intersect_gold, gold_dim, setA_B);
}

TEST(Set, SNIPPET_setIntersectSimple) {
    //! [ex_set_intersect_simple]

    // input data
    int h_setA[4] = {1, 2, 3, 3};
    int h_setB[4] = {3, 3, 4, 5};
    fly::array setA(4, h_setA);
    fly::array setB(4, h_setB);

    fly::array setA_B = setIntersect(setA, setB);
    // setA_B == { 3 };

    //! [ex_set_intersect_simple]

    vector<int> intersect_gold = {3};
    dim4 gold_dim(1, 1, 1, 1);
    ASSERT_VEC_ARRAY_EQ(intersect_gold, gold_dim, setA_B);
}
