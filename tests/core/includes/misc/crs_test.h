// Copyright 2023 The Elastic-AI Authors.
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

#include <vector>

#include <flare/core.h>
#include <doctest.h>

namespace Test {

    namespace {

        template<class ExecSpace>
        struct CountFillFunctor {
            FLARE_INLINE_FUNCTION
            std::int32_t operator()(std::int32_t row, float *fill) const {
                auto n = (row % 4) + 1;
                if (fill) {
                    for (std::int32_t j = 0; j < n; ++j) {
                        fill[j] = j + 1;
                    }
                }
                return n;
            }
        };

/* RunUpdateCrsTest
 *   4 test cases:
 *     1. use member object version which is constructed directly using the copy
 * constructor
 *     2. excplicity copy construct in local variable
 *     3. construct default and assign to input object
 *     4. construct object from views
 */
        template<class CrsType, class ExecSpace, class scalarType>
        struct RunUpdateCrsTest {
            struct TestOne {
            };
            struct TestTwo {
            };
            struct TestThree {
            };
            struct TestFour {
            };

            CrsType graph;

            RunUpdateCrsTest(CrsType g_in) : graph(g_in) {}

            void run_test(int nTest) {
                switch (nTest) {
                    case 1:
                        parallel_for(
                                "TestCrs1",
                                flare::RangePolicy<ExecSpace, TestOne>(0, graph.numRows()), *this);
                        break;
                    case 2:
                        parallel_for(
                                "TestCrs2",
                                flare::RangePolicy<ExecSpace, TestTwo>(0, graph.numRows()), *this);
                        break;
                    case 3:
                        parallel_for(
                                "TestCrs3",
                                flare::RangePolicy<ExecSpace, TestThree>(0, graph.numRows()),
                                *this);
                        break;
                    case 4:
                        parallel_for(
                                "TestCrs4",
                                flare::RangePolicy<ExecSpace, TestFour>(0, graph.numRows()),
                                *this);
                        break;
                    default:
                        break;
                }
            }

            FLARE_INLINE_FUNCTION
            void updateGraph(const CrsType &g_in, const scalarType row) const {
                auto row_map = g_in.row_map;
                auto entries = g_in.entries;
                auto j_start = row_map(row);
                auto j_end = row_map(row + 1) - j_start;
                for (scalarType j = 0; j < j_end; ++j) {
                    entries(j_start + j) = (j + 1) * (j + 1);
                }
            }

            // Test Crs class from class member
            FLARE_INLINE_FUNCTION
            void operator()(const TestOne &, const scalarType row) const {
                updateGraph(graph, row);
            }

            // Test Crs class from copy constructor (local_graph(graph)
            FLARE_INLINE_FUNCTION
            void operator()(const TestTwo &, const scalarType row) const {
                CrsType local_graph(graph);
                updateGraph(local_graph, row);
            }

            // Test Crs class from default constructor assigned to function parameter
            FLARE_INLINE_FUNCTION
            void operator()(const TestThree &, const scalarType row) const {
                CrsType local_graph;
                local_graph = graph;
                updateGraph(local_graph, row);
            }

            // Test Crs class from local graph constructed from row_map and entities
            // access on input parameter)
            FLARE_INLINE_FUNCTION
            void operator()(const TestFour &, const scalarType row) const {
                CrsType local_graph(graph.row_map, graph.entries);
                updateGraph(local_graph, row);
            }
        };

        template<class ExecSpace>
        void test_count_fill(std::int32_t nrows) {
            flare::Crs<float, ExecSpace, void, std::int32_t> graph;
            flare::count_and_fill_crs(graph, nrows, CountFillFunctor<ExecSpace>());
            REQUIRE_EQ(graph.numRows(), nrows);
            auto row_map = flare::create_mirror_view(graph.row_map);
            flare::deep_copy(row_map, graph.row_map);
            auto entries = flare::create_mirror_view(graph.entries);
            flare::deep_copy(entries, graph.entries);
            for (std::int32_t row = 0; row < nrows; ++row) {
                auto n = (row % 4) + 1;
                REQUIRE_EQ(row_map(row + 1) - row_map(row), n);
                for (std::int32_t j = 0; j < n; ++j) {
                    REQUIRE_EQ(entries(row_map(row) + j), j + 1);
                }
            }
        }

// Test Crs Constructor / assignment operation by
// using count and fill to create/populate initial graph,
// then use parallel_for with Crs directly to update content
// then verify results
        template<class ExecSpace>
        void test_constructor(std::int32_t nrows) {
            for (int nTest = 1; nTest < 5; nTest++) {
                using crs_type = flare::Crs<float, ExecSpace, void, std::int32_t>;
                crs_type graph;
                flare::count_and_fill_crs(graph, nrows, CountFillFunctor<ExecSpace>());
                REQUIRE_EQ(graph.numRows(), nrows);

                RunUpdateCrsTest<crs_type, ExecSpace, std::int32_t> crstest(graph);
                crstest.run_test(nTest);

                auto row_map = flare::create_mirror_view(graph.row_map);
                flare::deep_copy(row_map, graph.row_map);
                auto entries = flare::create_mirror_view(graph.entries);
                flare::deep_copy(entries, graph.entries);

                for (std::int32_t row = 0; row < nrows; ++row) {
                    auto n = (row % 4) + 1;
                    REQUIRE_EQ(row_map(row + 1) - row_map(row), n);
                    for (std::int32_t j = 0; j < n; ++j) {
                        REQUIRE_EQ(entries(row_map(row) + j), (j + 1) * (j + 1));
                    }
                }
            }
        }

    }  // anonymous namespace

    TEST_CASE("TEST_CATEGORY, crs_count_fill") {
        test_count_fill<TEST_EXECSPACE>(0);
        test_count_fill<TEST_EXECSPACE>(1);
        test_count_fill<TEST_EXECSPACE>(2);
        test_count_fill<TEST_EXECSPACE>(3);
        test_count_fill<TEST_EXECSPACE>(13);
        test_count_fill<TEST_EXECSPACE>(100);
        test_count_fill<TEST_EXECSPACE>(1000);
        test_count_fill<TEST_EXECSPACE>(10000);
    }

    TEST_CASE("TEST_CATEGORY, crs_copy_constructor") {
        test_constructor<TEST_EXECSPACE>(0);
        test_constructor<TEST_EXECSPACE>(1);
        test_constructor<TEST_EXECSPACE>(2);
        test_constructor<TEST_EXECSPACE>(3);
        test_constructor<TEST_EXECSPACE>(13);
        test_constructor<TEST_EXECSPACE>(100);
        test_constructor<TEST_EXECSPACE>(1000);
        test_constructor<TEST_EXECSPACE>(10000);
    }

}  // namespace Test
