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

#include <cstdio>
#include <doctest.h>
#include <flare/core.h>

namespace Test {

    namespace {

        using namespace flare;

        template<typename ExecSpace>
        struct TestMDRange_ReduceArray_2D {
            using DataType = int;
            using ViewType_2 = typename flare::View<DataType **, ExecSpace>;
            using HostViewType_2 = typename ViewType_2::HostMirror;

            ViewType_2 input_view;

            using scalar_type = double;
            using value_type = scalar_type[];
            const unsigned value_count;

            TestMDRange_ReduceArray_2D(const int N0, const int N1,
                                       const unsigned array_size)
                    : input_view("input_view", N0, N1), value_count(array_size) {}

            FLARE_INLINE_FUNCTION
            void operator()(const int i, const int j) const { input_view(i, j) = 1; }

            FLARE_INLINE_FUNCTION
            void operator()(const int i, const int j, value_type lsum) const {
                lsum[0] += input_view(i, j) * 2;  //+=6 each time if InitTag => N0*N1*6
                lsum[1] += input_view(i, j);      //+=3 each time if InitTag => N0*N1*3
            }

            // tagged operators
            struct InitTag {
            };

            FLARE_INLINE_FUNCTION
            void operator()(const InitTag &, const int i, const int j) const {
                input_view(i, j) = 3;
            }

            static void test_arrayreduce2(const int N0, const int N1) {
                {
                    using range_type_init =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<2>,
                                    flare::IndexType<int>, InitTag>;
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<2>,
                                    flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type_init range_init(point_type{{0, 0}}, point_type{{N0, N1}},
                                               tile_type{{3, 3}});
                    range_type range(point_type{{0, 0}}, point_type{{N0, N1}},
                                     tile_type{{3, 3}});

                    const unsigned array_size = 2;

                    TestMDRange_ReduceArray_2D functor(N0, N1, array_size);

                    parallel_for(range_init, functor);  // Init the view to 3's

                    double sums[array_size];
                    flare::fence("Fence before accessing result on the host");
                    parallel_reduce(range, functor, sums);

                    // Check output
                    // printf("Array Reduce result. N0 = %d  N1 = %d  N0*N1 = %d  sums[0] =
                    // %lf  sums[1] = %lf \n", N0, N1, N0*N1, sums[0], sums[1]);

                    REQUIRE_EQ(sums[0], 6 * N0 * N1);
                    REQUIRE_EQ(sums[1], 3 * N0 * N1);
                }
            }
        };

        template<typename ExecSpace>
        struct TestMDRange_ReduceArray_3D {
            using DataType = int;
            using ViewType_3 = typename flare::View<DataType ***, ExecSpace>;
            using HostViewType_3 = typename ViewType_3::HostMirror;

            ViewType_3 input_view;

            using scalar_type = double;
            using value_type = scalar_type[];
            const unsigned value_count;

            TestMDRange_ReduceArray_3D(const int N0, const int N1, const int N2,
                                       const unsigned array_size)
                    : input_view("input_view", N0, N1, N2), value_count(array_size) {}

            FLARE_INLINE_FUNCTION
            void init(scalar_type dst[]) const {
                for (unsigned i = 0; i < value_count; ++i) {
                    dst[i] = 0.0;
                }
            }

            FLARE_INLINE_FUNCTION
            void join(scalar_type dst[], const scalar_type src[]) const {
                for (unsigned i = 0; i < value_count; ++i) {
                    dst[i] += src[i];
                }
            }

            FLARE_INLINE_FUNCTION
            void operator()(const int i, const int j, const int k) const {
                input_view(i, j, k) = 1;
            }

            FLARE_INLINE_FUNCTION
            void operator()(const int i, const int j, const int k,
                            value_type lsum) const {
                lsum[0] +=
                        input_view(i, j, k) * 2;     //+=6 each time if InitTag => N0*N1*N2*6
                lsum[1] += input_view(i, j, k);  //+=3 each time if InitTag => N0*N1*N2*3
            }

            // tagged operators
            struct InitTag {
            };

            FLARE_INLINE_FUNCTION
            void operator()(const InitTag &, const int i, const int j,
                            const int k) const {
                input_view(i, j, k) = 3;
            }

            static void test_arrayreduce3(const int N0, const int N1, const int N2) {
                {
                    using range_type_init =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<3>,
                                    flare::IndexType<int>, InitTag>;
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<3>,
                                    flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type_init range_init(point_type{{0, 0, 0}},
                                               point_type{{N0, N1, N2}},
                                               tile_type{{3, 3, 3}});
                    range_type range(point_type{{0, 0, 0}}, point_type{{N0, N1, N2}},
                                     tile_type{{3, 3, 3}});

                    const unsigned array_size = 2;

                    TestMDRange_ReduceArray_3D functor(N0, N1, N2, array_size);

                    parallel_for(range_init, functor);  // Init the view to 3's

                    double sums[array_size];
                    parallel_reduce(range, functor, sums);

                    REQUIRE_EQ(sums[0], 6 * N0 * N1 * N2);
                    REQUIRE_EQ(sums[1], 3 * N0 * N1 * N2);
                }
            }
        };

        template<typename ExecSpace>
        struct TestMDRange_2D {
            using DataType = int;
            using ViewType = typename flare::View<DataType **, ExecSpace>;
            using HostViewType = typename ViewType::HostMirror;

            ViewType input_view;
            using value_type = double;

            TestMDRange_2D(const DataType N0, const DataType N1)
                    : input_view("input_view", N0, N1) {}

            FLARE_INLINE_FUNCTION
            void operator()(const int i, const int j) const { input_view(i, j) = 1; }

            FLARE_INLINE_FUNCTION
            void operator()(const int i, const int j, value_type &lsum) const {
                lsum += input_view(i, j) * 2;
            }

            // tagged operators
            struct InitTag {
            };

            FLARE_INLINE_FUNCTION
            void operator()(const InitTag &, const int i, const int j) const {
                input_view(i, j) = 3;
            }

            // reduction tagged operators
            FLARE_INLINE_FUNCTION
            void operator()(const InitTag &, const int i, const int j,
                            value_type &lsum) const {
                lsum += input_view(i, j) * 3;
            }

            static void test_reduce2(const int N0, const int N1) {
#if defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)
                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<2>,
                                    flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0}}, point_type{{N0, N1}},
                                     tile_type{{3, 3}});
                    double sum = 0.0;
                    parallel_reduce(
                            range,
                            FLARE_LAMBDA(const int /*i*/, const int /*j*/, double &lsum) {
                                lsum += 1.0;
                            },
                            sum);
                    REQUIRE_EQ(sum, N0 * N1);
                }
#endif

                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<2>,
                                    flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0}}, point_type{{N0, N1}},
                                     tile_type{{3, 3}});

                    TestMDRange_2D functor(N0, N1);

                    parallel_for(range, functor);
                    double sum = 0.0;
                    parallel_reduce(range, functor, sum);

                    REQUIRE_EQ(sum, 2 * N0 * N1);
                }

                // Test with reducers - scalar
                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<2>,
                                    flare::IndexType<int>>;
                    int s0 = 1;
                    int s1 = 1;
                    range_type range({{s0, s1}}, {{N0, N1}}, {{3, 3}});

                    TestMDRange_2D functor(N0, N1);

                    parallel_for(range, functor);

                    value_type sum = 0.0;
                    flare::Sum<value_type> reducer_scalar(sum);

                    parallel_reduce(range, functor, reducer_scalar);

                    REQUIRE_EQ(sum, 2 * (N0 - s0) * (N1 - s1));
                }
                // Test with reducers - scalar + label
                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<2>,
                                    flare::IndexType<int>>;
                    int s0 = 1;
                    int s1 = 1;
                    range_type range({{s0, s1}}, {{N0, N1}}, {{3, 3}});

                    TestMDRange_2D functor(N0, N1);

                    parallel_for("rank2-parfor-label", range, functor);

                    value_type sum = 0.0;
                    flare::Sum<value_type> reducer_scalar(sum);

                    parallel_reduce("rank2-reducer-label", range, functor, reducer_scalar);

                    REQUIRE_EQ(sum, 2 * (N0 - s0) * (N1 - s1));
                }
                // Test with reducers - scalar view
                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<2>,
                                    flare::IndexType<int>>;
                    range_type range({{0, 0}}, {{N0, N1}}, {{3, 3}});

                    TestMDRange_2D functor(N0, N1);

                    parallel_for(range, functor);

                    value_type sum = 0.0;
                    flare::View<value_type, flare::HostSpace> sum_view("sum_view");
                    sum_view() = sum;
                    flare::Sum<value_type> reducer_view(sum_view);

                    parallel_reduce(range, functor, reducer_view);
                    flare::fence();
                    sum = sum_view();

                    REQUIRE_EQ(sum, 2 * N0 * N1);
                }
                // Test Min reducer with lambda
#if defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)
                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<2>,
                                    flare::IndexType<int>>;
                    range_type range({{1, 1}}, {{N0, N1}}, {{3, 3}});

                    flare::View<double **, ExecSpace> v_in("v_in", N0, N1);

                    parallel_for(
                            "rank2-init-lambda", range, FLARE_LAMBDA(const int i, const int j) {
                                v_in(i, j) = (i + 1) * (j + 1);
                            });

                    double min;
                    flare::Min<double> reducer_scalar(min);

                    parallel_reduce(
                            "rank2-min-reducer", range,
                            FLARE_LAMBDA(const int i, const int j, double &min_val) {
                                min_val = flare::fmin(v_in(i, j), min_val);
                            },
                            reducer_scalar);

                    REQUIRE_EQ(min, 4.0);
                }
#endif
                // Tagged operator test
                {
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::Rank<2, Iterate::Default, Iterate::Default>,
                            flare::IndexType<int>, InitTag>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0}}, point_type{{N0, N1}},
                                     tile_type{{2, 4}});

                    TestMDRange_2D functor(N0, N1);

                    parallel_for(range, functor);

                    // check parallel_for results correct with InitTag
                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);
                    int counter = 0;
                    for (int i = 0; i < N0; ++i)
                        for (int j = 0; j < N1; ++j) {
                            if (h_view(i, j) != 3) {
                                ++counter;
                            }
                        }

                    if (counter != 0) {
                        printf(
                                "Defaults + InitTag op(): Errors in test_for3; mismatches = %d\n\n",
                                counter);
                    }
                    REQUIRE_EQ(counter, 0);

                    double sum = 0.0;
                    parallel_reduce(range, functor, sum);

                    REQUIRE_EQ(sum, 9 * N0 * N1);
                }

                {
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::Rank<2, Iterate::Default, Iterate::Default>,
                            flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0}}, point_type{{N0, N1}},
                                     tile_type{{2, 6}});

                    TestMDRange_2D functor(N0, N1);

                    parallel_for(range, functor);
                    double sum = 0.0;
                    parallel_reduce(range, functor, sum);

                    REQUIRE_EQ(sum, 2 * N0 * N1);
                }

                {
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::Rank<2, Iterate::Left, Iterate::Left>,
                            flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0}}, point_type{{N0, N1}},
                                     tile_type{{2, 6}});

                    TestMDRange_2D functor(N0, N1);

                    parallel_for(range, functor);
                    double sum = 0.0;
                    parallel_reduce(range, functor, sum);

                    REQUIRE_EQ(sum, 2 * N0 * N1);
                }

                {
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::Rank<2, Iterate::Left, Iterate::Right>,
                            flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0}}, point_type{{N0, N1}},
                                     tile_type{{2, 6}});

                    TestMDRange_2D functor(N0, N1);

                    parallel_for(range, functor);
                    double sum = 0.0;
                    parallel_reduce(range, functor, sum);

                    REQUIRE_EQ(sum, 2 * N0 * N1);
                }

                {
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::Rank<2, Iterate::Right, Iterate::Left>,
                            flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0}}, point_type{{N0, N1}},
                                     tile_type{{2, 6}});

                    TestMDRange_2D functor(N0, N1);

                    parallel_for(range, functor);
                    double sum = 0.0;
                    parallel_reduce(range, functor, sum);

                    REQUIRE_EQ(sum, 2 * N0 * N1);
                }

                {
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::Rank<2, Iterate::Right, Iterate::Right>,
                            flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0}}, point_type{{N0, N1}},
                                     tile_type{{2, 6}});

                    TestMDRange_2D functor(N0, N1);

                    parallel_for(range, functor);
                    double sum = 0.0;
                    parallel_reduce(range, functor, sum);

                    REQUIRE_EQ(sum, 2 * N0 * N1);
                }
            }  // end test_reduce2

            static void test_for2(const int N0, const int N1) {
#if defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)
                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<2>,
                                    flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    const int s0 = 1;
                    const int s1 = 1;

                    range_type range(point_type{{s0, s1}}, point_type{{N0, N1}},
                                     tile_type{{3, 3}});

                    TestMDRange_2D::ViewType v("v", N0, N1);

                    parallel_for(
                            range, FLARE_LAMBDA(const int i, const int j) { v(i, j) = 3; });

                    TestMDRange_2D::HostViewType h_view = flare::create_mirror_view(v);
                    flare::deep_copy(h_view, v);

                    int counter = 0;
                    for (int i = s0; i < N0; ++i)
                        for (int j = s1; j < N1; ++j) {
                            if (h_view(i, j) != 3) {
                                ++counter;
                            }
                        }

                    if (counter != 0) {
                        printf(
                                "Offset Start + Default Layouts + InitTag op(): Errors in "
                                "test_for2; mismatches = %d\n\n",
                                counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }
#endif

                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<2>,
                                    flare::IndexType<int>, InitTag>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    const int s0 = 1;
                    const int s1 = 1;
                    range_type range(point_type{{s0, s1}}, point_type{{N0, N1}},
                                     tile_type{{3, 3}});
                    TestMDRange_2D functor(N0, N1);

                    parallel_for(range, functor);

                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);

                    int counter = 0;
                    for (int i = s0; i < N0; ++i)
                        for (int j = s1; j < N1; ++j) {
                            if (h_view(i, j) != 3) {
                                ++counter;
                            }
                        }

                    if (counter != 0) {
                        printf(
                                "Offset Start + Default Layouts + InitTag op(): Errors in "
                                "test_for2; mismatches = %d\n\n",
                                counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }

                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<2>, InitTag>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0}}, point_type{{N0, N1}},
                                     tile_type{{3, 3}});
                    TestMDRange_2D functor(N0, N1);

                    parallel_for(range, functor);

                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);

                    int counter = 0;
                    for (int i = 0; i < N0; ++i)
                        for (int j = 0; j < N1; ++j) {
                            if (h_view(i, j) != 3) {
                                ++counter;
                            }
                        }

                    if (counter != 0) {
                        printf(
                                "Default Layouts + InitTag op(): Errors in test_for2; mismatches = "
                                "%d\n\n",
                                counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }

                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<2>, InitTag>;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0}}, point_type{{N0, N1}});
                    TestMDRange_2D functor(N0, N1);

                    parallel_for(range, functor);

                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);

                    int counter = 0;
                    for (int i = 0; i < N0; ++i)
                        for (int j = 0; j < N1; ++j) {
                            if (h_view(i, j) != 3) {
                                ++counter;
                            }
                        }

                    if (counter != 0) {
                        printf(
                                "Default Layouts + InitTag op() + Default Tile: Errors in "
                                "test_for2; mismatches = %d\n\n",
                                counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }

                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<2>,
                                    flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0}}, point_type{{N0, N1}},
                                     tile_type{{3, 3}});
                    TestMDRange_2D functor(N0, N1);

                    parallel_for(range, functor);

                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);

                    int counter = 0;
                    for (int i = 0; i < N0; ++i)
                        for (int j = 0; j < N1; ++j) {
                            if (h_view(i, j) != 1) {
                                ++counter;
                            }
                        }

                    if (counter != 0) {
                        printf("No info: Errors in test_for2; mismatches = %d\n\n", counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }

                {
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::Rank<2, Iterate::Default, Iterate::Default>,
                            flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0}}, point_type{{N0, N1}},
                                     tile_type{{4, 4}});
                    TestMDRange_2D functor(N0, N1);

                    parallel_for(range, functor);

                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);

                    int counter = 0;
                    for (int i = 0; i < N0; ++i)
                        for (int j = 0; j < N1; ++j) {
                            if (h_view(i, j) != 1) {
                                ++counter;
                            }
                        }

                    if (counter != 0) {
                        printf("D D: Errors in test_for2; mismatches = %d\n\n", counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }

                {
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::Rank<2, Iterate::Left, Iterate::Left>,
                            flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0}}, point_type{{N0, N1}},
                                     tile_type{{3, 3}});
                    TestMDRange_2D functor(N0, N1);

                    parallel_for(range, functor);

                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);

                    int counter = 0;
                    for (int i = 0; i < N0; ++i)
                        for (int j = 0; j < N1; ++j) {
                            if (h_view(i, j) != 1) {
                                ++counter;
                            }
                        }

                    if (counter != 0) {
                        printf("L L: Errors in test_for2; mismatches = %d\n\n", counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }

                {
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::Rank<2, Iterate::Left, Iterate::Right>,
                            flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0}}, point_type{{N0, N1}},
                                     tile_type{{7, 7}});
                    TestMDRange_2D functor(N0, N1);

                    parallel_for(range, functor);

                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);

                    int counter = 0;
                    for (int i = 0; i < N0; ++i)
                        for (int j = 0; j < N1; ++j) {
                            if (h_view(i, j) != 1) {
                                ++counter;
                            }
                        }

                    if (counter != 0) {
                        printf("L R: Errors in test_for2; mismatches = %d\n\n", counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }

                {
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::Rank<2, Iterate::Right, Iterate::Left>,
                            flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0}}, point_type{{N0, N1}},
                                     tile_type{{16, 16}});
                    TestMDRange_2D functor(N0, N1);

                    parallel_for(range, functor);

                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);

                    int counter = 0;
                    for (int i = 0; i < N0; ++i)
                        for (int j = 0; j < N1; ++j) {
                            if (h_view(i, j) != 1) {
                                ++counter;
                            }
                        }

                    if (counter != 0) {
                        printf("R L: Errors in test_for2; mismatches = %d\n\n", counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }

                {
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::Rank<2, Iterate::Right, Iterate::Right>,
                            flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0}}, point_type{{N0, N1}},
                                     tile_type{{5, 16}});
                    TestMDRange_2D functor(N0, N1);

                    parallel_for(range, functor);

                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);

                    int counter = 0;
                    for (int i = 0; i < N0; ++i)
                        for (int j = 0; j < N1; ++j) {
                            if (h_view(i, j) != 1) {
                                ++counter;
                            }
                        }

                    if (counter != 0) {
                        printf("R R: Errors in test_for2; mismatches = %d\n\n", counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }

            }  // end test_for2
        };   // MDRange_2D

        template<typename ExecSpace>
        struct TestMDRange_3D {
            using DataType = int;
            using ViewType = typename flare::View<DataType ***, ExecSpace>;
            using HostViewType = typename ViewType::HostMirror;

            ViewType input_view;
            using value_type = double;

            TestMDRange_3D(const DataType N0, const DataType N1, const DataType N2)
                    : input_view("input_view", N0, N1, N2) {}

            FLARE_INLINE_FUNCTION
            void operator()(const int i, const int j, const int k) const {
                input_view(i, j, k) = 1;
            }

            FLARE_INLINE_FUNCTION
            void operator()(const int i, const int j, const int k, double &lsum) const {
                lsum += input_view(i, j, k) * 2;
            }

            // tagged operators
            struct InitTag {
            };

            FLARE_INLINE_FUNCTION
            void operator()(const InitTag &, const int i, const int j,
                            const int k) const {
                input_view(i, j, k) = 3;
            }

            // reduction tagged operators
            FLARE_INLINE_FUNCTION
            void operator()(const InitTag &, const int i, const int j, const int k,
                            value_type &lsum) const {
                lsum += input_view(i, j, k) * 3;
            }

            static void test_reduce3(const int N0, const int N1, const int N2) {
#if defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)
                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<3>,
                                    flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0}}, point_type{{N0, N1, N2}},
                                     tile_type{{3, 3, 3}});
                    double sum = 0.0;
                    parallel_reduce(
                            range,
                            FLARE_LAMBDA(const int /*i*/, const int /*j*/, const int /*k*/,
                                         double &lsum) { lsum += 1.0; },
                            sum);
                    REQUIRE_EQ(sum, N0 * N1 * N2);
                }
#endif

                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<3>,
                                    flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    int s0 = 1;
                    int s1 = 1;
                    int s2 = 1;
                    range_type range(point_type{{s0, s1, s2}}, point_type{{N0, N1, N2}},
                                     tile_type{{3, 3, 3}});

                    TestMDRange_3D functor(N0, N1, N2);

                    parallel_for(range, functor);
                    double sum = 0.0;
                    parallel_reduce(range, functor, sum);

                    REQUIRE_EQ(sum, 2 * (N0 - s0) * (N1 - s1) * (N2 - s2));
                }

                // Test with reducers - scalar
                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<3>,
                                    flare::IndexType<int>>;
                    range_type range({{0, 0, 0}}, {{N0, N1, N2}}, {{3, 3, 3}});

                    TestMDRange_3D functor(N0, N1, N2);

                    parallel_for(range, functor);

                    value_type sum = 0.0;
                    flare::Sum<value_type> reducer_scalar(sum);

                    parallel_reduce(range, functor, reducer_scalar);

                    REQUIRE_EQ(sum, 2 * N0 * N1 * N2);
                }
                // Test with reducers - scalar + label
                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<3>,
                                    flare::IndexType<int>>;
                    range_type range({{0, 0, 0}}, {{N0, N1, N2}}, {{3, 3, 3}});

                    TestMDRange_3D functor(N0, N1, N2);

                    parallel_for("rank3-parfor-label", range, functor);

                    value_type sum = 0.0;
                    flare::Sum<value_type> reducer_scalar(sum);

                    parallel_reduce("rank3-reducer-label", range, functor, reducer_scalar);

                    REQUIRE_EQ(sum, 2 * N0 * N1 * N2);
                }
                // Test with reducers - scalar view
                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<3>,
                                    flare::IndexType<int>>;
                    range_type range({{0, 0, 0}}, {{N0, N1, N2}}, {{3, 3, 3}});

                    TestMDRange_3D functor(N0, N1, N2);

                    parallel_for(range, functor);

                    value_type sum = 0.0;
                    flare::View<value_type, flare::HostSpace> sum_view("sum_view");
                    sum_view() = sum;
                    flare::Sum<value_type> reducer_view(sum_view);

                    parallel_reduce(range, functor, reducer_view);
                    flare::fence();
                    sum = sum_view();

                    REQUIRE_EQ(sum, 2 * N0 * N1 * N2);
                }
                // Test Min reducer with lambda
#if defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)
                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<3>,
                                    flare::IndexType<int>>;

                    range_type range({{1, 1, 1}}, {{N0, N1, N2}}, {{3, 3, 3}});

                    flare::View<double ***, ExecSpace> v_in("v_in", N0, N1, N2);

                    parallel_for(
                            "rank3-init-lambda", range,
                            FLARE_LAMBDA(const int i, const int j, const int k) {
                                v_in(i, j, k) = (i + 1) * (j + 1) * (k + 1);
                            });

                    double min;

                    parallel_reduce(
                            "rank3-min-reducer", range,
                            FLARE_LAMBDA(const int i, const int j, const int k,
                                         double &min_val) {
                                min_val = (v_in(i, j, k) < min_val) ? v_in(i, j, k) : min_val;
                            },
                            flare::Min<double>(min));

                    if ((N0 - 1) * (N1 - 1) * (N2 - 1) > 0)
                        REQUIRE_EQ(min, 8.0);
                    else {
                        double min_identity = flare::reduction_identity<double>::min();
                        REQUIRE_EQ(min, min_identity);
                    }
                }
#endif

                // Tagged operator test
                {
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::Rank<3, Iterate::Default, Iterate::Default>,
                            flare::IndexType<int>, InitTag>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0}}, point_type{{N0, N1, N2}},
                                     tile_type{{2, 4, 6}});

                    TestMDRange_3D functor(N0, N1, N2);

                    parallel_for(range, functor);

                    // check parallel_for results correct with InitTag
                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);
                    int counter = 0;
                    for (int i = 0; i < N0; ++i)
                        for (int j = 0; j < N1; ++j)
                            for (int k = 0; k < N2; ++k) {
                                if (h_view(i, j, k) != 3) {
                                    ++counter;
                                }
                            }

                    if (counter != 0) {
                        printf(
                                "Defaults + InitTag op(): Errors in test_for3; mismatches = %d\n\n",
                                counter);
                    }
                    REQUIRE_EQ(counter, 0);

                    double sum = 0.0;
                    parallel_reduce(range, functor, sum);

                    REQUIRE_EQ(sum, 9 * N0 * N1 * N2);
                }

                {
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::Rank<3, Iterate::Default, Iterate::Default>,
                            flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0}}, point_type{{N0, N1, N2}},
                                     tile_type{{2, 4, 6}});

                    TestMDRange_3D functor(N0, N1, N2);

                    parallel_for(range, functor);
                    double sum = 0.0;
                    parallel_reduce(range, functor, sum);

                    REQUIRE_EQ(sum, 2 * N0 * N1 * N2);
                }

                {
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::Rank<3, Iterate::Left, Iterate::Left>,
                            flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0}}, point_type{{N0, N1, N2}},
                                     tile_type{{2, 4, 6}});

                    TestMDRange_3D functor(N0, N1, N2);

                    parallel_for(range, functor);
                    double sum = 0.0;
                    parallel_reduce(range, functor, sum);

                    REQUIRE_EQ(sum, 2 * N0 * N1 * N2);
                }

                {
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::Rank<3, Iterate::Left, Iterate::Right>,
                            flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0}}, point_type{{N0, N1, N2}},
                                     tile_type{{2, 4, 6}});

                    TestMDRange_3D functor(N0, N1, N2);

                    parallel_for(range, functor);
                    double sum = 0.0;
                    parallel_reduce(range, functor, sum);

                    REQUIRE_EQ(sum, 2 * N0 * N1 * N2);
                }

                {
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::Rank<3, Iterate::Right, Iterate::Left>,
                            flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0}}, point_type{{N0, N1, N2}},
                                     tile_type{{2, 4, 6}});

                    TestMDRange_3D functor(N0, N1, N2);

                    parallel_for(range, functor);
                    double sum = 0.0;
                    parallel_reduce(range, functor, sum);

                    REQUIRE_EQ(sum, 2 * N0 * N1 * N2);
                }

                {
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::Rank<3, Iterate::Right, Iterate::Right>,
                            flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0}}, point_type{{N0, N1, N2}},
                                     tile_type{{2, 4, 6}});

                    TestMDRange_3D functor(N0, N1, N2);

                    parallel_for(range, functor);
                    double sum = 0.0;
                    parallel_reduce(range, functor, sum);

                    REQUIRE_EQ(sum, 2 * N0 * N1 * N2);
                }
            }  // end test_reduce3

            static void test_for3(const int N0, const int N1, const int N2) {
#if defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)
                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<3>,
                                    flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    const int s0 = 1;
                    const int s1 = 1;
                    const int s2 = 1;

                    range_type range(point_type{{s0, s1, s2}}, point_type{{N0, N1, N2}},
                                     tile_type{{3, 3, 3}});

                    TestMDRange_3D::ViewType v("v", N0, N1, N2);

                    parallel_for(
                            range, FLARE_LAMBDA(const int i, const int j, const int k) {
                                v(i, j, k) = 3;
                            });

                    TestMDRange_3D::HostViewType h_view = flare::create_mirror_view(v);
                    flare::deep_copy(h_view, v);

                    int counter = 0;
                    for (int i = s0; i < N0; ++i)
                        for (int j = s1; j < N1; ++j)
                            for (int k = s2; k < N2; ++k) {
                                if (h_view(i, j, k) != 3) {
                                    ++counter;
                                }
                            }

                    if (counter != 0) {
                        printf(
                                "Offset Start + Default Layouts + InitTag op(): Errors in "
                                "test_for3; mismatches = %d\n\n",
                                counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }
#endif

                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<3>>;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0}}, point_type{{N0, N1, N2}});
                    TestMDRange_3D functor(N0, N1, N2);

                    parallel_for(range, functor);

                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);

                    int counter = 0;
                    for (int i = 0; i < N0; ++i)
                        for (int j = 0; j < N1; ++j)
                            for (int k = 0; k < N2; ++k) {
                                if (h_view(i, j, k) != 1) {
                                    ++counter;
                                }
                            }

                    if (counter != 0) {
                        printf("Defaults + No Tile: Errors in test_for3; mismatches = %d\n\n",
                               counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }

                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<3>,
                                    flare::IndexType<int>, InitTag>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    int s0 = 1;
                    int s1 = 1;
                    int s2 = 1;
                    range_type range(point_type{{s0, s1, s2}}, point_type{{N0, N1, N2}},
                                     tile_type{{3, 3, 3}});
                    TestMDRange_3D functor(N0, N1, N2);

                    parallel_for(range, functor);

                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);

                    int counter = 0;
                    for (int i = s0; i < N0; ++i)
                        for (int j = s1; j < N1; ++j)
                            for (int k = s2; k < N2; ++k) {
                                if (h_view(i, j, k) != 3) {
                                    ++counter;
                                }
                            }

                    if (counter != 0) {
                        printf(
                                "Offset Start + Defaults + InitTag op(): Errors in test_for3; "
                                "mismatches = %d\n\n",
                                counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }

                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<3>,
                                    flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0}}, point_type{{N0, N1, N2}},
                                     tile_type{{3, 3, 3}});

                    TestMDRange_3D functor(N0, N1, N2);

                    parallel_for(range, functor);

                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);

                    int counter = 0;
                    for (int i = 0; i < N0; ++i)
                        for (int j = 0; j < N1; ++j)
                            for (int k = 0; k < N2; ++k) {
                                if (h_view(i, j, k) != 1) {
                                    ++counter;
                                }
                            }

                    if (counter != 0) {
                        printf(" Errors in test_for3; mismatches = %d\n\n", counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }

                {
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::Rank<3, Iterate::Default, Iterate::Default>,
                            flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0}}, point_type{{N0, N1, N2}},
                                     tile_type{{3, 3, 3}});
                    TestMDRange_3D functor(N0, N1, N2);

                    parallel_for(range, functor);

                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);

                    int counter = 0;
                    for (int i = 0; i < N0; ++i)
                        for (int j = 0; j < N1; ++j)
                            for (int k = 0; k < N2; ++k) {
                                if (h_view(i, j, k) != 1) {
                                    ++counter;
                                }
                            }

                    if (counter != 0) {
                        printf(" Errors in test_for3; mismatches = %d\n\n", counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }

                {
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::Rank<3, Iterate::Left, Iterate::Left>,
                            flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0}}, point_type{{N0, N1, N2}},
                                     tile_type{{2, 4, 2}});
                    TestMDRange_3D functor(N0, N1, N2);

                    parallel_for(range, functor);

                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);

                    int counter = 0;
                    for (int i = 0; i < N0; ++i)
                        for (int j = 0; j < N1; ++j)
                            for (int k = 0; k < N2; ++k) {
                                if (h_view(i, j, k) != 1) {
                                    ++counter;
                                }
                            }

                    if (counter != 0) {
                        printf(" Errors in test_for3; mismatches = %d\n\n", counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }

                {
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::Rank<3, Iterate::Left, Iterate::Right>,
                            flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0}}, point_type{{N0, N1, N2}},
                                     tile_type{{3, 5, 7}});
                    TestMDRange_3D functor(N0, N1, N2);

                    parallel_for(range, functor);

                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);

                    int counter = 0;
                    for (int i = 0; i < N0; ++i)
                        for (int j = 0; j < N1; ++j)
                            for (int k = 0; k < N2; ++k) {
                                if (h_view(i, j, k) != 1) {
                                    ++counter;
                                }
                            }

                    if (counter != 0) {
                        printf(" Errors in test_for3; mismatches = %d\n\n", counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }

                {
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::Rank<3, Iterate::Right, Iterate::Left>,
                            flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0}}, point_type{{N0, N1, N2}},
                                     tile_type{{8, 8, 8}});
                    TestMDRange_3D functor(N0, N1, N2);

                    parallel_for(range, functor);

                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);

                    int counter = 0;
                    for (int i = 0; i < N0; ++i)
                        for (int j = 0; j < N1; ++j)
                            for (int k = 0; k < N2; ++k) {
                                if (h_view(i, j, k) != 1) {
                                    ++counter;
                                }
                            }

                    if (counter != 0) {
                        printf(" Errors in test_for3; mismatches = %d\n\n", counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }

                {
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::Rank<3, Iterate::Right, Iterate::Right>,
                            flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0}}, point_type{{N0, N1, N2}},
                                     tile_type{{2, 4, 2}});
                    TestMDRange_3D functor(N0, N1, N2);

                    parallel_for(range, functor);

                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);

                    int counter = 0;
                    for (int i = 0; i < N0; ++i)
                        for (int j = 0; j < N1; ++j)
                            for (int k = 0; k < N2; ++k) {
                                if (h_view(i, j, k) != 1) {
                                    ++counter;
                                }
                            }

                    if (counter != 0) {
                        printf(" Errors in test_for3; mismatches = %d\n\n", counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }
            }  // end test_for3
        };

        template<typename ExecSpace>
        struct TestMDRange_4D {
            using DataType = int;
            using ViewType = typename flare::View<DataType ****, ExecSpace>;
            using HostViewType = typename ViewType::HostMirror;

            ViewType input_view;
            using value_type = double;

            TestMDRange_4D(const DataType N0, const DataType N1, const DataType N2,
                           const DataType N3)
                    : input_view("input_view", N0, N1, N2, N3) {}

            FLARE_INLINE_FUNCTION
            void operator()(const int i, const int j, const int k, const int l) const {
                input_view(i, j, k, l) = 1;
            }

            FLARE_INLINE_FUNCTION
            void operator()(const int i, const int j, const int k, const int l,
                            double &lsum) const {
                lsum += input_view(i, j, k, l) * 2;
            }

            // tagged operators
            struct InitTag {
            };

            FLARE_INLINE_FUNCTION
            void operator()(const InitTag &, const int i, const int j, const int k,
                            const int l) const {
                input_view(i, j, k, l) = 3;
            }

            // reduction tagged operators
            FLARE_INLINE_FUNCTION
            void operator()(const InitTag &, const int i, const int j, const int k,
                            const int l, value_type &lsum) const {
                lsum += input_view(i, j, k, l) * 3;
            }

            static void test_reduce4(const int N0, const int N1, const int N2,
                                     const int N3) {
#if defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)
                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<4>,
                                    flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0, 0}}, point_type{{N0, N1, N2, N3}},
                                     tile_type{{3, 3, 3, 3}});
                    double sum = 0.0;
                    parallel_reduce(
                            range,
                            FLARE_LAMBDA(const int /*i*/, const int /*j*/, const int /*k*/,
                                         const int /*l*/, double &lsum) { lsum += 1.0; },
                            sum);
                    REQUIRE_EQ(sum, N0 * N1 * N2 * N3);
                }
#endif

                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<4>,
                                    flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    int s0 = 1;
                    int s1 = 1;
                    int s2 = 1;
                    int s3 = 1;
                    range_type range(point_type{{s0, s1, s2, s3}},
                                     point_type{{N0, N1, N2, N3}}, tile_type{{3, 3, 3, 3}});

                    TestMDRange_4D functor(N0, N1, N2, N3);

                    parallel_for(range, functor);
                    double sum = 0.0;
                    parallel_reduce(range, functor, sum);

                    REQUIRE_EQ(sum, 2 * (N0 - s0) * (N1 - s1) * (N2 - s2) * (N3 - s3));
                }

                // Test with reducers - scalar
                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<4>,
                                    flare::IndexType<int>>;
                    range_type range({{0, 0, 0, 0}}, {{N0, N1, N2, N3}}, {{3, 3, 3, 3}});

                    TestMDRange_4D functor(N0, N1, N2, N3);

                    parallel_for(range, functor);

                    value_type sum = 0.0;
                    flare::Sum<value_type> reducer_scalar(sum);

                    parallel_reduce(range, functor, reducer_scalar);

                    REQUIRE_EQ(sum, 2 * N0 * N1 * N2 * N3);
                }

                // Test with reducers - scalar + label
                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<4>,
                                    flare::IndexType<int>>;
                    range_type range({{0, 0, 0, 0}}, {{N0, N1, N2, N3}}, {{3, 3, 3, 3}});

                    TestMDRange_4D functor(N0, N1, N2, N3);

                    parallel_for("rank4-parfor-label", range, functor);

                    value_type sum = 0.0;
                    flare::Sum<value_type> reducer_scalar(sum);

                    parallel_reduce("rank4-reducer-label", range, functor, reducer_scalar);

                    REQUIRE_EQ(sum, 2 * N0 * N1 * N2 * N3);
                }

                // Test with reducers - scalar view
                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<4>,
                                    flare::IndexType<int>>;
                    range_type range({{0, 0, 0, 0}}, {{N0, N1, N2, N3}}, {{3, 3, 3, 3}});

                    TestMDRange_4D functor(N0, N1, N2, N3);

                    parallel_for(range, functor);

                    value_type sum = 0.0;
                    flare::View<value_type, flare::HostSpace> sum_view("sum_view");
                    sum_view() = sum;
                    flare::Sum<value_type> reducer_view(sum_view);

                    parallel_reduce(range, functor, reducer_view);
                    flare::fence();
                    sum = sum_view();

                    REQUIRE_EQ(sum, 2 * N0 * N1 * N2 * N3);
                }

                // Test Min reducer with lambda
#if defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)
                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<4>,
                                    flare::IndexType<int>>;

                    range_type range({{1, 1, 1, 1}}, {{N0, N1, N2, N3}}, {{3, 3, 3, 3}});

                    flare::View<double ****, ExecSpace> v_in("v_in", N0, N1, N2, N3);

                    parallel_for(
                            "rank4-init-lambda", range,
                            FLARE_LAMBDA(const int i, const int j, const int k, const int l) {
                                v_in(i, j, k, l) = (i + 1) * (j + 1) * (k + 1) * (l + 1);
                            });

                    double min;

                    parallel_reduce(
                            "rank4-min-reducer", range,
                            FLARE_LAMBDA(const int i, const int j, const int k, const int l,
                                         double &min_val) {
                                min_val = (v_in(i, j, k, l) < min_val) ? v_in(i, j, k, l) : min_val;
                            },
                            flare::Min<double>(min));

                    REQUIRE_EQ(min, 16.0);
                }
#endif

                // Tagged operator test
                {
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::Rank<4, Iterate::Default, Iterate::Default>,
                            flare::IndexType<int>, InitTag>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0, 0}}, point_type{{N0, N1, N2, N3}},
                                     tile_type{{2, 4, 6, 2}});

                    TestMDRange_4D functor(N0, N1, N2, N3);

                    parallel_for(range, functor);

                    // check parallel_for results correct with InitTag
                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);
                    int counter = 0;
                    for (int i = 0; i < N0; ++i)
                        for (int j = 0; j < N1; ++j)
                            for (int k = 0; k < N2; ++k)
                                for (int l = 0; l < N3; ++l) {
                                    if (h_view(i, j, k, l) != 3) {
                                        ++counter;
                                    }
                                }

                    if (counter != 0) {
                        printf(
                                "Defaults + InitTag op(): Errors in test_reduce4 parallel_for "
                                "init; mismatches = %d\n\n",
                                counter);
                    }
                    REQUIRE_EQ(counter, 0);

                    double sum = 0.0;
                    parallel_reduce(range, functor, sum);

                    REQUIRE_EQ(sum, 9 * N0 * N1 * N2 * N3);
                }

                {
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::Rank<4, Iterate::Default, Iterate::Default>,
                            flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0, 0}}, point_type{{N0, N1, N2, N3}},
                                     tile_type{{2, 4, 6, 2}});

                    TestMDRange_4D functor(N0, N1, N2, N3);

                    parallel_for(range, functor);
                    double sum = 0.0;
                    parallel_reduce(range, functor, sum);

                    REQUIRE_EQ(sum, 2 * N0 * N1 * N2 * N3);
                }

                {
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::Rank<4, Iterate::Left, Iterate::Left>,
                            flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0, 0}}, point_type{{N0, N1, N2, N3}},
                                     tile_type{{2, 4, 6, 2}});

                    TestMDRange_4D functor(N0, N1, N2, N3);

                    parallel_for(range, functor);
                    double sum = 0.0;
                    parallel_reduce(range, functor, sum);

                    REQUIRE_EQ(sum, 2 * N0 * N1 * N2 * N3);
                }

                {
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::Rank<4, Iterate::Left, Iterate::Right>,
                            flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0, 0}}, point_type{{N0, N1, N2, N3}},
                                     tile_type{{2, 4, 6, 2}});

                    TestMDRange_4D functor(N0, N1, N2, N3);

                    parallel_for(range, functor);
                    double sum = 0.0;
                    parallel_reduce(range, functor, sum);

                    REQUIRE_EQ(sum, 2 * N0 * N1 * N2 * N3);
                }

                {
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::Rank<4, Iterate::Right, Iterate::Left>,
                            flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0, 0}}, point_type{{N0, N1, N2, N3}},
                                     tile_type{{2, 4, 6, 2}});

                    TestMDRange_4D functor(N0, N1, N2, N3);

                    parallel_for(range, functor);
                    double sum = 0.0;
                    parallel_reduce(range, functor, sum);

                    REQUIRE_EQ(sum, 2 * N0 * N1 * N2 * N3);
                }

                {
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::Rank<4, Iterate::Right, Iterate::Right>,
                            flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0, 0}}, point_type{{N0, N1, N2, N3}},
                                     tile_type{{2, 4, 6, 2}});

                    TestMDRange_4D functor(N0, N1, N2, N3);

                    parallel_for(range, functor);
                    double sum = 0.0;
                    parallel_reduce(range, functor, sum);

                    REQUIRE_EQ(sum, 2 * N0 * N1 * N2 * N3);
                }
            }  // end test_reduce

            static void test_for4(const int N0, const int N1, const int N2,
                                  const int N3) {
#if defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)
                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<4>,
                                    flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    const int s0 = 1;
                    const int s1 = 1;
                    const int s2 = 1;
                    const int s3 = 1;

                    range_type range(point_type{{s0, s1, s2, s3}},
                                     point_type{{N0, N1, N2, N3}}, tile_type{{3, 3, 3, 3}});

                    TestMDRange_4D::ViewType v("v", N0, N1, N2, N3);

                    parallel_for(
                            range, FLARE_LAMBDA(const int i, const int j, const int k,
                                                const int l) { v(i, j, k, l) = 3; });

                    TestMDRange_4D::HostViewType h_view = flare::create_mirror_view(v);
                    flare::deep_copy(h_view, v);

                    int counter = 0;
                    for (int i = s0; i < N0; ++i)
                        for (int j = s1; j < N1; ++j)
                            for (int k = s2; k < N2; ++k)
                                for (int l = s3; l < N3; ++l) {
                                    if (h_view(i, j, k, l) != 3) {
                                        ++counter;
                                    }
                                }

                    if (counter != 0) {
                        printf(
                                "Offset Start + Default Layouts + InitTag op(): Errors in "
                                "test_for4; mismatches = %d\n\n",
                                counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }
#endif

                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<4>>;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0, 0}}, point_type{{N0, N1, N2, N3}});
                    TestMDRange_4D functor(N0, N1, N2, N3);

                    parallel_for(range, functor);

                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);

                    int counter = 0;
                    for (int i = 0; i < N0; ++i)
                        for (int j = 0; j < N1; ++j)
                            for (int k = 0; k < N2; ++k)
                                for (int l = 0; l < N3; ++l) {
                                    if (h_view(i, j, k, l) != 1) {
                                        ++counter;
                                    }
                                }

                    if (counter != 0) {
                        printf("Defaults + No Tile: Errors in test_for4; mismatches = %d\n\n",
                               counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }

                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<4>,
                                    flare::IndexType<int>, InitTag>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    int s0 = 1;
                    int s1 = 1;
                    int s2 = 1;
                    int s3 = 1;
                    range_type range(point_type{{s0, s1, s2, s3}},
                                     point_type{{N0, N1, N2, N3}}, tile_type{{3, 11, 3, 3}});
                    TestMDRange_4D functor(N0, N1, N2, N3);

                    parallel_for(range, functor);

                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);

                    int counter = 0;
                    for (int i = s0; i < N0; ++i)
                        for (int j = s1; j < N1; ++j)
                            for (int k = s2; k < N2; ++k)
                                for (int l = s3; l < N3; ++l) {
                                    if (h_view(i, j, k, l) != 3) {
                                        ++counter;
                                    }
                                }

                    if (counter != 0) {
                        printf(
                                "Offset Start + Defaults +m_tile > m_upper dim2 InitTag op(): "
                                "Errors in test_for4; mismatches = %d\n\n",
                                counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }

                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<4>,
                                    flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0, 0}}, point_type{{N0, N1, N2, N3}},
                                     tile_type{{4, 4, 4, 4}});

                    TestMDRange_4D functor(N0, N1, N2, N3);

                    parallel_for(range, functor);

                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);

                    int counter = 0;
                    for (int i = 0; i < N0; ++i)
                        for (int j = 0; j < N1; ++j)
                            for (int k = 0; k < N2; ++k)
                                for (int l = 0; l < N3; ++l) {
                                    if (h_view(i, j, k, l) != 1) {
                                        ++counter;
                                    }
                                }

                    if (counter != 0) {
                        printf(" Errors in test_for4; mismatches = %d\n\n", counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }

                {
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::Rank<4, Iterate::Default, Iterate::Default>,
                            flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0, 0}}, point_type{{N0, N1, N2, N3}},
                                     tile_type{{4, 4, 4, 4}});

                    TestMDRange_4D functor(N0, N1, N2, N3);

                    parallel_for(range, functor);

                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);

                    int counter = 0;
                    for (int i = 0; i < N0; ++i)
                        for (int j = 0; j < N1; ++j)
                            for (int k = 0; k < N2; ++k)
                                for (int l = 0; l < N3; ++l) {
                                    if (h_view(i, j, k, l) != 1) {
                                        ++counter;
                                    }
                                }

                    if (counter != 0) {
                        printf(" Errors in test_for4; mismatches = %d\n\n", counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }

                {
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::Rank<4, Iterate::Left, Iterate::Left>,
                            flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0, 0}}, point_type{{N0, N1, N2, N3}},
                                     tile_type{{4, 4, 4, 4}});

                    TestMDRange_4D functor(N0, N1, N2, N3);

                    parallel_for(range, functor);

                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);

                    int counter = 0;
                    for (int i = 0; i < N0; ++i)
                        for (int j = 0; j < N1; ++j)
                            for (int k = 0; k < N2; ++k)
                                for (int l = 0; l < N3; ++l) {
                                    if (h_view(i, j, k, l) != 1) {
                                        ++counter;
                                    }
                                }

                    if (counter != 0) {
                        printf(" Errors in test_for4; mismatches = %d\n\n", counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }

                {
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::Rank<4, Iterate::Left, Iterate::Right>,
                            flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0, 0}}, point_type{{N0, N1, N2, N3}},
                                     tile_type{{4, 4, 4, 4}});

                    TestMDRange_4D functor(N0, N1, N2, N3);

                    parallel_for(range, functor);

                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);

                    int counter = 0;
                    for (int i = 0; i < N0; ++i)
                        for (int j = 0; j < N1; ++j)
                            for (int k = 0; k < N2; ++k)
                                for (int l = 0; l < N3; ++l) {
                                    if (h_view(i, j, k, l) != 1) {
                                        ++counter;
                                    }
                                }

                    if (counter != 0) {
                        printf(" Errors in test_for4; mismatches = %d\n\n", counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }

                {
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::Rank<4, Iterate::Right, Iterate::Left>,
                            flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0, 0}}, point_type{{N0, N1, N2, N3}},
                                     tile_type{{4, 4, 4, 4}});

                    TestMDRange_4D functor(N0, N1, N2, N3);

                    parallel_for(range, functor);

                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);

                    int counter = 0;
                    for (int i = 0; i < N0; ++i)
                        for (int j = 0; j < N1; ++j)
                            for (int k = 0; k < N2; ++k)
                                for (int l = 0; l < N3; ++l) {
                                    if (h_view(i, j, k, l) != 1) {
                                        ++counter;
                                    }
                                }

                    if (counter != 0) {
                        printf(" Errors in test_for4; mismatches = %d\n\n", counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }

                {
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::Rank<4, Iterate::Right, Iterate::Right>,
                            flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0, 0}}, point_type{{N0, N1, N2, N3}},
                                     tile_type{{4, 4, 4, 4}});

                    TestMDRange_4D functor(N0, N1, N2, N3);

                    parallel_for(range, functor);

                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);

                    int counter = 0;
                    for (int i = 0; i < N0; ++i)
                        for (int j = 0; j < N1; ++j)
                            for (int k = 0; k < N2; ++k)
                                for (int l = 0; l < N3; ++l) {
                                    if (h_view(i, j, k, l) != 1) {
                                        ++counter;
                                    }
                                }

                    if (counter != 0) {
                        printf(" Errors in test_for4; mismatches = %d\n\n", counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }
            }  // end test_for4
        };

        template<typename ExecSpace>
        struct TestMDRange_5D {
            using DataType = int;
            using ViewType = typename flare::View<DataType *****, ExecSpace>;
            using HostViewType = typename ViewType::HostMirror;

            ViewType input_view;
            using value_type = double;

            TestMDRange_5D(const DataType N0, const DataType N1, const DataType N2,
                           const DataType N3, const DataType N4)
                    : input_view("input_view", N0, N1, N2, N3, N4) {}

            FLARE_INLINE_FUNCTION
            void operator()(const int i, const int j, const int k, const int l,
                            const int m) const {
                input_view(i, j, k, l, m) = 1;
            }

            FLARE_INLINE_FUNCTION
            void operator()(const int i, const int j, const int k, const int l,
                            const int m, value_type &lsum) const {
                lsum += input_view(i, j, k, l, m) * 2;
            }

            // tagged operators
            struct InitTag {
            };

            FLARE_INLINE_FUNCTION
            void operator()(const InitTag &, const int i, const int j, const int k,
                            const int l, const int m) const {
                input_view(i, j, k, l, m) = 3;
            }

            // reduction tagged operators
            FLARE_INLINE_FUNCTION
            void operator()(const InitTag &, const int i, const int j, const int k,
                            const int l, const int m, value_type &lsum) const {
                lsum += input_view(i, j, k, l, m) * 3;
            }

            static void test_reduce5(const int N0, const int N1, const int N2,
                                     const int N3, const int N4) {
#if defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)
                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<5>,
                                    flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0, 0, 0}},
                                     point_type{{N0, N1, N2, N3, N4}},
                                     tile_type{{3, 3, 3, 3, 1}});
                    double sum = 0.0;
                    parallel_reduce(
                            range,
                            FLARE_LAMBDA(const int /*i*/, const int /*j*/, const int /*k*/,
                                         const int /*l*/, const int /*m*/,
                                         double &lsum) { lsum += 1.0; },
                            sum);
                    REQUIRE_EQ(sum, N0 * N1 * N2 * N3 * N4);
                }
#endif

                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<5>,
                                    flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    int s0 = 1;
                    int s1 = 1;
                    int s2 = 1;
                    int s3 = 1;
                    int s4 = 1;
                    range_type range(point_type{{s0, s1, s2, s3, s4}},
                                     point_type{{N0, N1, N2, N3, N4}},
                                     tile_type{{3, 3, 3, 3, 3}});

                    TestMDRange_5D functor(N0, N1, N2, N3, N4);

                    parallel_for(range, functor);
                    double sum = 0.0;
                    parallel_reduce(range, functor, sum);

                    REQUIRE_EQ(sum,
                               2 * (N0 - s0) * (N1 - s1) * (N2 - s2) * (N3 - s3) * (N4 - s4));
                }

                // Test with reducers - scalar
                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<5>,
                                    flare::IndexType<int>>;
                    range_type range({{0, 0, 0, 0, 0}}, {{N0, N1, N2, N3, N4}},
                                     {{3, 3, 3, 3, 3}});

                    TestMDRange_5D functor(N0, N1, N2, N3, N4);

                    parallel_for(range, functor);

                    value_type sum = 0.0;
                    flare::Sum<value_type> reducer_scalar(sum);

                    parallel_reduce(range, functor, reducer_scalar);

                    REQUIRE_EQ(sum, 2 * N0 * N1 * N2 * N3 * N4);
                }

                // Test with reducers - scalar + label
                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<5>,
                                    flare::IndexType<int>>;
                    range_type range({{0, 0, 0, 0, 0}}, {{N0, N1, N2, N3, N4}},
                                     {{3, 3, 3, 3, 3}});

                    TestMDRange_5D functor(N0, N1, N2, N3, N4);

                    parallel_for("rank5-parfor-label", range, functor);

                    value_type sum = 0.0;
                    flare::Sum<value_type> reducer_scalar(sum);

                    parallel_reduce("rank5-reducer-label", range, functor, reducer_scalar);

                    REQUIRE_EQ(sum, 2 * N0 * N1 * N2 * N3 * N4);
                }

                // Test with reducers - scalar view
                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<5>,
                                    flare::IndexType<int>>;
                    range_type range({{0, 0, 0, 0, 0}}, {{N0, N1, N2, N3, N4}},
                                     {{3, 3, 3, 3, 3}});

                    TestMDRange_5D functor(N0, N1, N2, N3, N4);

                    parallel_for(range, functor);

                    value_type sum = 0.0;
                    flare::View<value_type, flare::HostSpace> sum_view("sum_view");
                    sum_view() = sum;
                    flare::Sum<value_type> reducer_view(sum_view);

                    parallel_reduce(range, functor, reducer_view);
                    flare::fence();
                    sum = sum_view();

                    REQUIRE_EQ(sum, 2 * N0 * N1 * N2 * N3 * N4);
                }

                // Test Min reducer with lambda
#if defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)
                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<5>,
                                    flare::IndexType<int>>;

                    range_type range({{1, 1, 1, 1, 1}}, {{N0, N1, N2, N3, N4}},
                                     {{3, 3, 3, 2, 2}});

                    flare::View<double *****, ExecSpace> v_in("v_in", N0, N1, N2, N3, N4);

                    parallel_for(
                            "rank5-init-lambda", range,
                            FLARE_LAMBDA(const int i, const int j, const int k, const int l,
                                         const int m) {
                                v_in(i, j, k, l, m) =
                                        (i + 1) * (j + 1) * (k + 1) * (l + 1) * (m + 1);
                            });

                    double min;

                    parallel_reduce(
                            "rank5-min-reducer", range,
                            FLARE_LAMBDA(const int i, const int j, const int k, const int l,
                                         const int m, double &min_val) {
                                min_val =
                                        (v_in(i, j, k, l, m) < min_val) ? v_in(i, j, k, l, m) : min_val;
                            },
                            flare::Min<double>(min));

                    REQUIRE_EQ(min, 32.0);
                }
#endif

                // Tagged operator test
                {
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::Rank<5, Iterate::Default, Iterate::Default>,
                            flare::IndexType<int>, InitTag>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0, 0, 0}},
                                     point_type{{N0, N1, N2, N3, N4}},
                                     tile_type{{2, 4, 6, 2, 2}});

                    TestMDRange_5D functor(N0, N1, N2, N3, N4);

                    parallel_for(range, functor);

                    // check parallel_for results correct with InitTag
                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);
                    int counter = 0;
                    for (int i = 0; i < N0; ++i)
                        for (int j = 0; j < N1; ++j)
                            for (int k = 0; k < N2; ++k)
                                for (int l = 0; l < N3; ++l)
                                    for (int m = 0; m < N4; ++m) {
                                        if (h_view(i, j, k, l, m) != 3) {
                                            ++counter;
                                        }
                                    }

                    if (counter != 0) {
                        printf(
                                "Defaults + InitTag op(): Errors in test_reduce5 parallel_for "
                                "init; mismatches = %d\n\n",
                                counter);
                    }
                    REQUIRE_EQ(counter, 0);

                    double sum = 0.0;
                    parallel_reduce(range, functor, sum);

                    REQUIRE_EQ(sum, 9 * N0 * N1 * N2 * N3 * N4);
                }
            }

            static void test_for5(const int N0, const int N1, const int N2, const int N3,
                                  const int N4) {
#if defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)
                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<5>,
                                    flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    const int s0 = 1;
                    const int s1 = 1;
                    const int s2 = 1;
                    const int s3 = 1;
                    const int s4 = 1;

                    range_type range(point_type{{s0, s1, s2, s3, s4}},
                                     point_type{{N0, N1, N2, N3, N4}},
                                     tile_type{{3, 3, 3, 3, 1}});

                    TestMDRange_5D::ViewType v("v", N0, N1, N2, N3, N4);

                    parallel_for(
                            range,
                            FLARE_LAMBDA(const int i, const int j, const int k, const int l,
                                         const int m) { v(i, j, k, l, m) = 3; });

                    TestMDRange_5D::HostViewType h_view = flare::create_mirror_view(v);
                    flare::deep_copy(h_view, v);

                    int counter = 0;
                    for (int i = s0; i < N0; ++i)
                        for (int j = s1; j < N1; ++j)
                            for (int k = s2; k < N2; ++k)
                                for (int l = s3; l < N3; ++l)
                                    for (int m = s4; m < N4; ++m) {
                                        if (h_view(i, j, k, l, m) != 3) {
                                            ++counter;
                                        }
                                    }

                    if (counter != 0) {
                        printf(
                                "Offset Start + Default Layouts + InitTag op(): Errors in "
                                "test_for5; mismatches = %d\n\n",
                                counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }
#endif

                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<5>>;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0, 0, 0}},
                                     point_type{{N0, N1, N2, N3, N4}});
                    TestMDRange_5D functor(N0, N1, N2, N3, N4);

                    parallel_for(range, functor);

                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);

                    int counter = 0;
                    for (int i = 0; i < N0; ++i)
                        for (int j = 0; j < N1; ++j)
                            for (int k = 0; k < N2; ++k)
                                for (int l = 0; l < N3; ++l)
                                    for (int m = 0; m < N4; ++m) {
                                        if (h_view(i, j, k, l, m) != 1) {
                                            ++counter;
                                        }
                                    }

                    if (counter != 0) {
                        printf("Defaults + No Tile: Errors in test_for5; mismatches = %d\n\n",
                               counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }

                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<5>,
                                    flare::IndexType<int>, InitTag>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    int s0 = 1;
                    int s1 = 1;
                    int s2 = 1;
                    int s3 = 1;
                    int s4 = 1;

                    range_type range(point_type{{s0, s1, s2, s3, s4}},
                                     point_type{{N0, N1, N2, N3, N4}},
                                     tile_type{{3, 3, 3, 3, 5}});

                    TestMDRange_5D functor(N0, N1, N2, N3, N4);

                    parallel_for(range, functor);

                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);

                    int counter = 0;
                    for (int i = s0; i < N0; ++i)
                        for (int j = s1; j < N1; ++j)
                            for (int k = s2; k < N2; ++k)
                                for (int l = s3; l < N3; ++l)
                                    for (int m = s4; m < N4; ++m) {
                                        if (h_view(i, j, k, l, m) != 3) {
                                            ++counter;
                                        }
                                    }

                    if (counter != 0) {
                        printf(
                                "Offset Start + Defaults + InitTag op(): Errors in test_for5; "
                                "mismatches = %d\n\n",
                                counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }

                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<5>,
                                    flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0, 0, 0}},
                                     point_type{{N0, N1, N2, N3, N4}},
                                     tile_type{{4, 4, 4, 2, 2}});

                    TestMDRange_5D functor(N0, N1, N2, N3, N4);

                    parallel_for(range, functor);

                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);

                    int counter = 0;
                    for (int i = 0; i < N0; ++i)
                        for (int j = 0; j < N1; ++j)
                            for (int k = 0; k < N2; ++k)
                                for (int l = 0; l < N3; ++l)
                                    for (int m = 0; m < N4; ++m) {
                                        if (h_view(i, j, k, l, m) != 1) {
                                            ++counter;
                                        }
                                    }

                    if (counter != 0) {
                        printf(" Errors in test_for5; mismatches = %d\n\n", counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }

                {
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::Rank<5, Iterate::Default, Iterate::Default>,
                            flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0, 0, 0}},
                                     point_type{{N0, N1, N2, N3, N4}},
                                     tile_type{{4, 4, 4, 2, 2}});

                    TestMDRange_5D functor(N0, N1, N2, N3, N4);

                    parallel_for(range, functor);

                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);

                    int counter = 0;
                    for (int i = 0; i < N0; ++i)
                        for (int j = 0; j < N1; ++j)
                            for (int k = 0; k < N2; ++k)
                                for (int l = 0; l < N3; ++l)
                                    for (int m = 0; m < N4; ++m) {
                                        if (h_view(i, j, k, l, m) != 1) {
                                            ++counter;
                                        }
                                    }

                    if (counter != 0) {
                        printf(" Errors in test_for5; mismatches = %d\n\n", counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }

                {
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::Rank<5, Iterate::Left, Iterate::Left>,
                            flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0, 0, 0}},
                                     point_type{{N0, N1, N2, N3, N4}},
                                     tile_type{{4, 4, 4, 2, 2}});

                    TestMDRange_5D functor(N0, N1, N2, N3, N4);

                    parallel_for(range, functor);

                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);

                    int counter = 0;
                    for (int i = 0; i < N0; ++i)
                        for (int j = 0; j < N1; ++j)
                            for (int k = 0; k < N2; ++k)
                                for (int l = 0; l < N3; ++l)
                                    for (int m = 0; m < N4; ++m) {
                                        if (h_view(i, j, k, l, m) != 1) {
                                            ++counter;
                                        }
                                    }

                    if (counter != 0) {
                        printf(" Errors in test_for5; mismatches = %d\n\n", counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }

                {
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::Rank<5, Iterate::Left, Iterate::Right>,
                            flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0, 0, 0}},
                                     point_type{{N0, N1, N2, N3, N4}},
                                     tile_type{{4, 4, 4, 2, 2}});

                    TestMDRange_5D functor(N0, N1, N2, N3, N4);

                    parallel_for(range, functor);

                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);

                    int counter = 0;
                    for (int i = 0; i < N0; ++i)
                        for (int j = 0; j < N1; ++j)
                            for (int k = 0; k < N2; ++k)
                                for (int l = 0; l < N3; ++l)
                                    for (int m = 0; m < N4; ++m) {
                                        if (h_view(i, j, k, l, m) != 1) {
                                            ++counter;
                                        }
                                    }

                    if (counter != 0) {
                        printf(" Errors in test_for5; mismatches = %d\n\n", counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }

                {
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::Rank<5, Iterate::Right, Iterate::Left>,
                            flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0, 0, 0}},
                                     point_type{{N0, N1, N2, N3, N4}},
                                     tile_type{{4, 4, 4, 2, 2}});

                    TestMDRange_5D functor(N0, N1, N2, N3, N4);

                    parallel_for(range, functor);

                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);

                    int counter = 0;
                    for (int i = 0; i < N0; ++i)
                        for (int j = 0; j < N1; ++j)
                            for (int k = 0; k < N2; ++k)
                                for (int l = 0; l < N3; ++l)
                                    for (int m = 0; m < N4; ++m) {
                                        if (h_view(i, j, k, l, m) != 1) {
                                            ++counter;
                                        }
                                    }

                    if (counter != 0) {
                        printf(" Errors in test_for5; mismatches = %d\n\n", counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }

                {
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::Rank<5, Iterate::Right, Iterate::Right>,
                            flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0, 0, 0}},
                                     point_type{{N0, N1, N2, N3, N4}},
                                     tile_type{{4, 4, 4, 2, 2}});

                    TestMDRange_5D functor(N0, N1, N2, N3, N4);

                    parallel_for(range, functor);

                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);

                    int counter = 0;
                    for (int i = 0; i < N0; ++i)
                        for (int j = 0; j < N1; ++j)
                            for (int k = 0; k < N2; ++k)
                                for (int l = 0; l < N3; ++l)
                                    for (int m = 0; m < N4; ++m) {
                                        if (h_view(i, j, k, l, m) != 1) {
                                            ++counter;
                                        }
                                    }

                    if (counter != 0) {
                        printf(" Errors in test_for5; mismatches = %d\n\n", counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }
            }
        };

        template<typename ExecSpace>
        struct TestMDRange_6D {
            using DataType = int;
            using ViewType = typename flare::View<DataType ******, ExecSpace>;
            using HostViewType = typename ViewType::HostMirror;

            ViewType input_view;
            using value_type = double;

            TestMDRange_6D(const DataType N0, const DataType N1, const DataType N2,
                           const DataType N3, const DataType N4, const DataType N5)
                    : input_view("input_view", N0, N1, N2, N3, N4, N5) {}

            FLARE_INLINE_FUNCTION
            void operator()(const int i, const int j, const int k, const int l,
                            const int m, const int n) const {
                input_view(i, j, k, l, m, n) = 1;
            }

            FLARE_INLINE_FUNCTION
            void operator()(const int i, const int j, const int k, const int l,
                            const int m, const int n, value_type &lsum) const {
                lsum += input_view(i, j, k, l, m, n) * 2;
            }

            // tagged operators
            struct InitTag {
            };

            FLARE_INLINE_FUNCTION
            void operator()(const InitTag &, const int i, const int j, const int k,
                            const int l, const int m, const int n) const {
                input_view(i, j, k, l, m, n) = 3;
            }

            // reduction tagged operators
            FLARE_INLINE_FUNCTION
            void operator()(const InitTag &, const int i, const int j, const int k,
                            const int l, const int m, const int n,
                            value_type &lsum) const {
                lsum += input_view(i, j, k, l, m, n) * 3;
            }

            static void test_reduce6(const int N0, const int N1, const int N2,
                                     const int N3, const int N4, const int N5) {
#if defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)
                {
#if defined(FLARE_COMPILER_INTEL)
                                                                                                                                            // Launchbounds causes hang with intel compilers
      using range_type =
          typename flare::MDRangePolicy<ExecSpace, flare::Rank<6>,
                                         flare::IndexType<int>>;
#else
                    // Launchbounds to ensure the tile fits into a CUDA block under register
                    // constraints
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::LaunchBounds<128, 1>, flare::Rank<6>,
                            flare::IndexType<int>>;
#endif
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0, 0, 0, 0}},
                                     point_type{{N0, N1, N2, N3, N4, N5}},
                                     tile_type{{3, 3, 3, 3, 1, 1}});
                    double sum = 0.0;
                    parallel_reduce(
                            range,
                            FLARE_LAMBDA(const int /*i*/, const int /*j*/, const int /*k*/,
                                         const int /*l*/, const int /*m*/, const int /*n*/,
                                         double &lsum) { lsum += 1.0; },
                            sum);
                    REQUIRE_EQ(sum, N0 * N1 * N2 * N3 * N4 * N5);
                }
#endif

                {
#if defined(FLARE_COMPILER_INTEL)
                                                                                                                                            // Launchbounds causes hang with intel compilers
      using range_type =
          typename flare::MDRangePolicy<ExecSpace, flare::Rank<6>,
                                         flare::IndexType<int>>;
#else
                    // Launchbounds to ensure the tile fits into a CUDA block under register
                    // constraints
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::LaunchBounds<512, 1>, flare::Rank<6>,
                            flare::IndexType<int>>;
#endif
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    int s0 = 1;
                    int s1 = 1;
                    int s2 = 1;
                    int s3 = 1;
                    int s4 = 1;
                    int s5 = 1;

                    range_type range(point_type{{s0, s1, s2, s3, s4, s5}},
                                     point_type{{N0, N1, N2, N3, N4, N5}},
                                     tile_type{{3, 3, 3, 3, 3, 2}});

                    TestMDRange_6D functor(N0, N1, N2, N3, N4, N5);

                    parallel_for(range, functor);
                    double sum = 0.0;
                    parallel_reduce(range, functor, sum);

                    REQUIRE_EQ(sum, 2 * (N0 - s0) * (N1 - s1) * (N2 - s2) * (N3 - s3) *
                                    (N4 - s4) * (N5 - s5));
                }

                // Test with reducers - scalar
                {
#if defined(FLARE_COMPILER_INTEL)
                                                                                                                                            // Launchbounds causes hang with intel compilers
      using range_type =
          typename flare::MDRangePolicy<ExecSpace, flare::Rank<6>,
                                         flare::IndexType<int>>;
#else
                    // Launchbounds to ensure the tile fits into a CUDA block under register
                    // constraints
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::LaunchBounds<512, 1>, flare::Rank<6>,
                            flare::IndexType<int>>;
#endif
                    range_type range({{0, 0, 0, 0, 0, 0}}, {{N0, N1, N2, N3, N4, N5}},
                                     {{3, 3, 3, 3, 3, 2}});

                    TestMDRange_6D functor(N0, N1, N2, N3, N4, N5);

                    parallel_for(range, functor);

                    value_type sum = 0.0;
                    flare::Sum<value_type> reducer_scalar(sum);

                    parallel_reduce(range, functor, reducer_scalar);

                    REQUIRE_EQ(sum, 2 * N0 * N1 * N2 * N3 * N4 * N5);
                }

                // Test with reducers - scalar + label
                {
#if defined(FLARE_COMPILER_INTEL)
                                                                                                                                            // Launchbounds causes hang with intel compilers
      using range_type =
          typename flare::MDRangePolicy<ExecSpace, flare::Rank<6>,
                                         flare::IndexType<int>>;
#else
                    // Launchbounds to ensure the tile fits into a CUDA block under register
                    // constraints
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::LaunchBounds<512, 1>, flare::Rank<6>,
                            flare::IndexType<int>>;
#endif

                    range_type range({{0, 0, 0, 0, 0, 0}}, {{N0, N1, N2, N3, N4, N5}},
                                     {{3, 3, 3, 3, 3, 2}});

                    TestMDRange_6D functor(N0, N1, N2, N3, N4, N5);

                    parallel_for("rank6-parfor-label", range, functor);

                    value_type sum = 0.0;
                    flare::Sum<value_type> reducer_scalar(sum);

                    parallel_reduce("rank6-reducer-label", range, functor, reducer_scalar);

                    REQUIRE_EQ(sum, 2 * N0 * N1 * N2 * N3 * N4 * N5);
                }

                // Test with reducers - scalar view
                {
#if defined(FLARE_COMPILER_INTEL)
                                                                                                                                            // Launchbounds causes hang with intel compilers
      using range_type =
          typename flare::MDRangePolicy<ExecSpace, flare::Rank<6>,
                                         flare::IndexType<int>>;
#else
                    // Launchbounds to ensure the tile fits into a CUDA block under register
                    // constraints
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<6>,
                                    flare::IndexType<int>,
                                    flare::LaunchBounds<512, 1>>;
#endif

                    range_type range({{0, 0, 0, 0, 0, 0}}, {{N0, N1, N2, N3, N4, N5}},
                                     {{3, 3, 3, 3, 3, 2}});

                    TestMDRange_6D functor(N0, N1, N2, N3, N4, N5);

                    parallel_for(range, functor);

                    value_type sum = 0.0;
                    flare::View<value_type, flare::HostSpace> sum_view("sum_view");
                    sum_view() = sum;
                    flare::Sum<value_type> reducer_view(sum_view);

                    parallel_reduce(range, functor, reducer_view);
                    flare::fence();
                    sum = sum_view();

                    REQUIRE_EQ(sum, 2 * N0 * N1 * N2 * N3 * N4 * N5);
                }

                // Test Min reducer with lambda
#if defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)
                {
#if defined(FLARE_COMPILER_INTEL)
                                                                                                                                            // Launchbounds causes hang with intel compilers
      using range_type =
          typename flare::MDRangePolicy<ExecSpace, flare::Rank<6>,
                                         flare::IndexType<int>>;
#else
                    // Launchbounds to ensure the tile fits into a CUDA block under register
                    // constraints
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::LaunchBounds<128, 1>, flare::Rank<6>,
                            flare::IndexType<int>>;
#endif
                    range_type range({{1, 1, 1, 1, 1, 1}}, {{N0, N1, N2, N3, N4, N5}},
                                     {{3, 3, 3, 2, 2, 1}});

                    flare::View<double ******, ExecSpace> v_in("v_in", N0, N1, N2, N3, N4,
                                                               N5);

                    parallel_for(
                            "rank6-init-lambda", range,
                            FLARE_LAMBDA(const int i, const int j, const int k, const int l,
                                         const int m, const int n) {
                                v_in(i, j, k, l, m, n) =
                                        (i + 1) * (j + 1) * (k + 1) * (l + 1) * (m + 1) * (n + 1);
                            });

                    double min;

                    parallel_reduce(
                            "rank6-min-reducer", range,
                            FLARE_LAMBDA(const int i, const int j, const int k, const int l,
                                         const int m, const int n, double &min_val) {
                                min_val = (v_in(i, j, k, l, m, n) < min_val)
                                          ? v_in(i, j, k, l, m, n)
                                          : min_val;
                            },
                            flare::Min<double>(min));

                    REQUIRE_EQ(min, 64.0);
                }
#endif

                // Tagged operator test
                {
#if defined(FLARE_COMPILER_INTEL)
                                                                                                                                            // Launchbounds causes hang with intel compilers
      using range_type = typename flare::MDRangePolicy<
          ExecSpace, flare::Rank<6, Iterate::Default, Iterate::Default>,
          flare::IndexType<int>, InitTag>;
#else
                    // Launchbounds to ensure the tile fits into a CUDA block under register
                    // constraints
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::LaunchBounds<512, 1>,
                            flare::Rank<6, Iterate::Default, Iterate::Default>,
                            flare::IndexType<int>, InitTag>;
#endif
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0, 0, 0, 0}},
                                     point_type{{N0, N1, N2, N3, N4, N5}},
                                     tile_type{{2, 4, 6, 2, 2, 2}});

                    TestMDRange_6D functor(N0, N1, N2, N3, N4, N5);

                    parallel_for(range, functor);

                    // check parallel_for results correct with InitTag
                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);
                    int counter = 0;
                    for (int i = 0; i < N0; ++i)
                        for (int j = 0; j < N1; ++j)
                            for (int k = 0; k < N2; ++k)
                                for (int l = 0; l < N3; ++l)
                                    for (int m = 0; m < N4; ++m)
                                        for (int n = 0; n < N5; ++n) {
                                            if (h_view(i, j, k, l, m, n) != 3) {
                                                ++counter;
                                            }
                                        }

                    if (counter != 0) {
                        printf(
                                "Defaults + InitTag op(): Errors in test_reduce6 parallel_for "
                                "init; mismatches = %d\n\n",
                                counter);
                    }
                    REQUIRE_EQ(counter, 0);

                    double sum = 0.0;
                    parallel_reduce(range, functor, sum);

                    REQUIRE_EQ(sum, 9 * N0 * N1 * N2 * N3 * N4 * N5);
                }
            }

            static void test_for6(const int N0, const int N1, const int N2, const int N3,
                                  const int N4, const int N5) {
#if defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)
                {
#if defined(FLARE_COMPILER_INTEL)
                                                                                                                                            // Launchbounds causes hang with intel compilers
      using range_type =
          typename flare::MDRangePolicy<ExecSpace, flare::Rank<6>,
                                         flare::IndexType<int>>;
#else
                    // Launchbounds to ensure the tile fits into a CUDA block under register
                    // constraints
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::LaunchBounds<128, 1>, flare::Rank<6>,
                            flare::IndexType<int>>;
#endif
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    const int s0 = 1;
                    const int s1 = 1;
                    const int s2 = 1;
                    const int s3 = 1;
                    const int s4 = 1;
                    const int s5 = 1;

                    range_type range(point_type{{s0, s1, s2, s3, s4, s5}},
                                     point_type{{N0, N1, N2, N3, N4, N5}},
                                     tile_type{{3, 3, 3, 3, 1, 1}});

                    TestMDRange_6D::ViewType v("v", N0, N1, N2, N3, N4, N5);

                    parallel_for(
                            range,
                            FLARE_LAMBDA(const int i, const int j, const int k, const int l,
                                         const int m, const int n) { v(i, j, k, l, m, n) = 3; });

                    TestMDRange_6D::HostViewType h_view = flare::create_mirror_view(v);
                    flare::deep_copy(h_view, v);

                    int counter = 0;
                    for (int i = s0; i < N0; ++i)
                        for (int j = s1; j < N1; ++j)
                            for (int k = s2; k < N2; ++k)
                                for (int l = s3; l < N3; ++l)
                                    for (int m = s4; m < N4; ++m)
                                        for (int n = s5; n < N5; ++n) {
                                            if (h_view(i, j, k, l, m, n) != 3) {
                                                ++counter;
                                            }
                                        }

                    if (counter != 0) {
                        printf(
                                "Offset Start + Default Layouts + InitTag op(): Errors in "
                                "test_for6; mismatches = %d\n\n",
                                counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }
#endif

                {
#if defined(FLARE_COMPILER_INTEL)
                                                                                                                                            // Launchbounds causes hang with intel compilers
      using range_type =
          typename flare::MDRangePolicy<ExecSpace, flare::Rank<6>>;
#else
                    // Launchbounds to ensure the tile fits into a CUDA block under register
                    // constraints
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::LaunchBounds<512, 1>, flare::Rank<6>>;
#endif
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0, 0, 0, 0}},
                                     point_type{{N0, N1, N2, N3, N4, N5}});
                    TestMDRange_6D functor(N0, N1, N2, N3, N4, N5);

                    parallel_for(range, functor);

                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);

                    int counter = 0;
                    for (int i = 0; i < N0; ++i)
                        for (int j = 0; j < N1; ++j)
                            for (int k = 0; k < N2; ++k)
                                for (int l = 0; l < N3; ++l)
                                    for (int m = 0; m < N4; ++m)
                                        for (int n = 0; n < N5; ++n) {
                                            if (h_view(i, j, k, l, m, n) != 1) {
                                                ++counter;
                                            }
                                        }

                    if (counter != 0) {
                        printf("Defaults + No Tile: Errors in test_for6; mismatches = %d\n\n",
                               counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }

                {
#if defined(FLARE_COMPILER_INTEL)
                                                                                                                                            // Launchbounds causes hang with intel compilers
      using range_type =
          typename flare::MDRangePolicy<ExecSpace, flare::Rank<6>,
                                         flare::IndexType<int>, InitTag>;
#else
                    // Launchbounds to ensure the tile fits into a CUDA block under register
                    // constraints
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::LaunchBounds<512, 1>, flare::Rank<6>,
                            flare::IndexType<int>, InitTag>;
#endif
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    int s0 = 1;
                    int s1 = 1;
                    int s2 = 1;
                    int s3 = 1;
                    int s4 = 1;
                    int s5 = 1;
                    // tile dims 3,3,3,3,3,3 more than cuda can handle with debugging
                    range_type range(point_type{{s0, s1, s2, s3, s4, s5}},
                                     point_type{{N0, N1, N2, N3, N4, N5}},
                                     tile_type{{3, 3, 3, 3, 2, 3}});

                    TestMDRange_6D functor(N0, N1, N2, N3, N4, N5);

                    parallel_for(range, functor);

                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);

                    int counter = 0;
                    for (int i = s0; i < N0; ++i)
                        for (int j = s1; j < N1; ++j)
                            for (int k = s2; k < N2; ++k)
                                for (int l = s3; l < N3; ++l)
                                    for (int m = s4; m < N4; ++m)
                                        for (int n = s5; n < N5; ++n) {
                                            if (h_view(i, j, k, l, m, n) != 3) {
                                                ++counter;
                                            }
                                        }

                    if (counter != 0) {
                        printf(
                                "Offset Start + Defaults + InitTag op(): Errors in test_for6; "
                                "mismatches = %d\n\n",
                                counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }

                {
#if defined(FLARE_COMPILER_INTEL)
                                                                                                                                            // Launchbounds causes hang with intel compilers
      using range_type =
          typename flare::MDRangePolicy<ExecSpace, flare::Rank<6>,
                                         flare::IndexType<int>>;
#else
                    // Launchbounds to ensure the tile fits into a CUDA block under register
                    // constraints
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::LaunchBounds<512, 1>, flare::Rank<6>,
                            flare::IndexType<int>>;
#endif
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0, 0, 0, 0}},
                                     point_type{{N0, N1, N2, N3, N4, N5}},
                                     tile_type{{4, 4, 4, 2, 2, 2}});

                    TestMDRange_6D functor(N0, N1, N2, N3, N4, N5);

                    parallel_for(range, functor);

                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);

                    int counter = 0;
                    for (int i = 0; i < N0; ++i)
                        for (int j = 0; j < N1; ++j)
                            for (int k = 0; k < N2; ++k)
                                for (int l = 0; l < N3; ++l)
                                    for (int m = 0; m < N4; ++m)
                                        for (int n = 0; n < N5; ++n) {
                                            if (h_view(i, j, k, l, m, n) != 1) {
                                                ++counter;
                                            }
                                        }

                    if (counter != 0) {
                        printf(" Errors in test_for6; mismatches = %d\n\n", counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }

                {
#if defined(FLARE_COMPILER_INTEL)
                                                                                                                                            // Launchbounds causes hang with intel compilers
      using range_type = typename flare::MDRangePolicy<
          ExecSpace, flare::Rank<6, Iterate::Default, Iterate::Default>,
          flare::IndexType<int>>;
#else
                    // Launchbounds to ensure the tile fits into a CUDA block under register
                    // constraints
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::LaunchBounds<512, 1>,
                            flare::Rank<6, Iterate::Default, Iterate::Default>,
                            flare::IndexType<int>>;
#endif
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0, 0, 0, 0}},
                                     point_type{{N0, N1, N2, N3, N4, N5}},
                                     tile_type{{4, 4, 4, 2, 2, 2}});

                    TestMDRange_6D functor(N0, N1, N2, N3, N4, N5);

                    parallel_for(range, functor);

                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);

                    int counter = 0;
                    for (int i = 0; i < N0; ++i)
                        for (int j = 0; j < N1; ++j)
                            for (int k = 0; k < N2; ++k)
                                for (int l = 0; l < N3; ++l)
                                    for (int m = 0; m < N4; ++m)
                                        for (int n = 0; n < N5; ++n) {
                                            if (h_view(i, j, k, l, m, n) != 1) {
                                                ++counter;
                                            }
                                        }

                    if (counter != 0) {
                        printf(" Errors in test_for6; mismatches = %d\n\n", counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }

                {
#if defined(FLARE_COMPILER_INTEL)
                                                                                                                                            // Launchbounds causes hang with intel compilers
      using range_type = typename flare::MDRangePolicy<
          ExecSpace, flare::Rank<6, Iterate::Left, Iterate::Left>,
          flare::IndexType<int>>;
#else
                    // Launchbounds to ensure the tile fits into a CUDA block under register
                    // constraints
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::LaunchBounds<512, 1>,
                            flare::Rank<6, Iterate::Left, Iterate::Left>,
                            flare::IndexType<int>>;
#endif
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0, 0, 0, 0}},
                                     point_type{{N0, N1, N2, N3, N4, N5}},
                                     tile_type{{4, 4, 4, 2, 2, 2}});

                    TestMDRange_6D functor(N0, N1, N2, N3, N4, N5);

                    parallel_for(range, functor);

                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);

                    int counter = 0;
                    for (int i = 0; i < N0; ++i)
                        for (int j = 0; j < N1; ++j)
                            for (int k = 0; k < N2; ++k)
                                for (int l = 0; l < N3; ++l)
                                    for (int m = 0; m < N4; ++m)
                                        for (int n = 0; n < N5; ++n) {
                                            if (h_view(i, j, k, l, m, n) != 1) {
                                                ++counter;
                                            }
                                        }

                    if (counter != 0) {
                        printf(" Errors in test_for6; mismatches = %d\n\n", counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }

                {
#if defined(FLARE_COMPILER_INTEL)
                                                                                                                                            // Launchbounds causes hang with intel compilers
      using range_type = typename flare::MDRangePolicy<
          ExecSpace, flare::Rank<6, Iterate::Left, Iterate::Right>,
          flare::IndexType<int>>;
#else
                    // Launchbounds to ensure the tile fits into a CUDA block under register
                    // constraints
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::LaunchBounds<512, 1>,
                            flare::Rank<6, Iterate::Left, Iterate::Right>,
                            flare::IndexType<int>>;
#endif
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0, 0, 0, 0}},
                                     point_type{{N0, N1, N2, N3, N4, N5}},
                                     tile_type{{4, 4, 4, 2, 2, 2}});
                    TestMDRange_6D functor(N0, N1, N2, N3, N4, N5);

                    parallel_for(range, functor);

                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);

                    int counter = 0;
                    for (int i = 0; i < N0; ++i)
                        for (int j = 0; j < N1; ++j)
                            for (int k = 0; k < N2; ++k)
                                for (int l = 0; l < N3; ++l)
                                    for (int m = 0; m < N4; ++m)
                                        for (int n = 0; n < N5; ++n) {
                                            if (h_view(i, j, k, l, m, n) != 1) {
                                                ++counter;
                                            }
                                        }

                    if (counter != 0) {
                        printf(" Errors in test_for6; mismatches = %d\n\n", counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }

                {
#if defined(FLARE_COMPILER_INTEL)
                                                                                                                                            // Launchbounds causes hang with intel compilers
      using range_type = typename flare::MDRangePolicy<
          ExecSpace, flare::Rank<6, Iterate::Right, Iterate::Left>,
          flare::IndexType<int>>;
#else
                    // Launchbounds to ensure the tile fits into a CUDA block under register
                    // constraints
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::LaunchBounds<512, 1>,
                            flare::Rank<6, Iterate::Right, Iterate::Left>,
                            flare::IndexType<int>>;
#endif
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0, 0, 0, 0}},
                                     point_type{{N0, N1, N2, N3, N4, N5}},
                                     tile_type{{4, 4, 4, 2, 2, 2}});

                    TestMDRange_6D functor(N0, N1, N2, N3, N4, N5);

                    parallel_for(range, functor);

                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);

                    int counter = 0;
                    for (int i = 0; i < N0; ++i)
                        for (int j = 0; j < N1; ++j)
                            for (int k = 0; k < N2; ++k)
                                for (int l = 0; l < N3; ++l)
                                    for (int m = 0; m < N4; ++m)
                                        for (int n = 0; n < N5; ++n) {
                                            if (h_view(i, j, k, l, m, n) != 1) {
                                                ++counter;
                                            }
                                        }

                    if (counter != 0) {
                        printf(" Errors in test_for6; mismatches = %d\n\n", counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }

                {
#if defined(FLARE_COMPILER_INTEL)
                                                                                                                                            // Launchbounds causes hang with intel compilers
      using range_type = typename flare::MDRangePolicy<
          ExecSpace, flare::Rank<6, Iterate::Right, Iterate::Right>,
          flare::IndexType<int>>;
#else
                    // Launchbounds to ensure the tile fits into a CUDA block under register
                    // constraints
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::LaunchBounds<512, 1>,
                            flare::Rank<6, Iterate::Right, Iterate::Right>,
                            flare::IndexType<int>>;
#endif
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    range_type range(point_type{{0, 0, 0, 0, 0, 0}},
                                     point_type{{N0, N1, N2, N3, N4, N5}},
                                     tile_type{{4, 4, 4, 2, 2, 2}});

                    TestMDRange_6D functor(N0, N1, N2, N3, N4, N5);

                    parallel_for(range, functor);

                    HostViewType h_view = flare::create_mirror_view(functor.input_view);
                    flare::deep_copy(h_view, functor.input_view);

                    int counter = 0;
                    for (int i = 0; i < N0; ++i)
                        for (int j = 0; j < N1; ++j)
                            for (int k = 0; k < N2; ++k)
                                for (int l = 0; l < N3; ++l)
                                    for (int m = 0; m < N4; ++m)
                                        for (int n = 0; n < N5; ++n) {
                                            if (h_view(i, j, k, l, m, n) != 1) {
                                                ++counter;
                                            }
                                        }

                    if (counter != 0) {
                        printf(" Errors in test_for6; mismatches = %d\n\n", counter);
                    }

                    REQUIRE_EQ(counter, 0);
                }
            }
        };

        template<typename ExecSpace>
        struct TestMDRange_2D_NegIdx {
            using value_type = double;

            using DataType = int;
            using ViewType = typename flare::View<DataType **, ExecSpace>;
            using HostViewType = typename ViewType::HostMirror;

            ViewType input_view;
            DataType lower_offset[2];

            TestMDRange_2D_NegIdx(const DataType L0, const DataType L1, const DataType N0,
                                  const DataType N1)
                    : input_view("input_view", N0 - L0, N1 - L1) {
                lower_offset[0] = L0;
                lower_offset[1] = L1;
            }

            // When using negative indices, must offset View appropriately as views cannot
            // take a negative index
            FLARE_INLINE_FUNCTION
            void operator()(const int i, const int j) const {
                input_view(i - lower_offset[0], j - lower_offset[1]) = 1;
            }

            FLARE_INLINE_FUNCTION
            void operator()(const int i, const int j, value_type &lsum) const {
                lsum += input_view(i - lower_offset[0], j - lower_offset[1]) * 2;
            }

            static void test_2D_negidx(const int N0, const int N1) {
                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<2>,
                                    flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    const point_type lower{{-1, -1}};
                    const point_type upper{{N0, N1}};
                    const tile_type tile{{8, 8}};

                    range_type range(point_type{{lower[0], lower[1]}},
                                     point_type{{upper[0], upper[1]}},
                                     tile_type{{tile[0], tile[1]}});

                    TestMDRange_2D_NegIdx functor(lower[0], lower[1], upper[0], upper[1]);

                    parallel_for(range, functor);
                    double sum = 0.0;
                    parallel_reduce(range, functor, sum);

                    REQUIRE_EQ(sum, 2 * (upper[0] - lower[0]) * (upper[1] - lower[1]));
                }
            }
        };

        template<typename ExecSpace>
        struct TestMDRange_3D_NegIdx {
            using value_type = double;

            using DataType = int;
            using ViewType = typename flare::View<DataType ***, ExecSpace>;
            using HostViewType = typename ViewType::HostMirror;

            ViewType input_view;
            DataType lower_offset[3];

            TestMDRange_3D_NegIdx(const DataType L0, const DataType L1, const DataType L2,
                                  const DataType N0, const DataType N1, const DataType N2)
                    : input_view("input_view", N0 - L0, N1 - L1, N2 - L2) {
                lower_offset[0] = L0;
                lower_offset[1] = L1;
                lower_offset[2] = L2;
            }

            // When using negative indices, must offset View appropriately as views cannot
            // take a negative index
            FLARE_INLINE_FUNCTION
            void operator()(const int i, const int j, const int k) const {
                input_view(i - lower_offset[0], j - lower_offset[1], k - lower_offset[2]) =
                        1;
            }

            FLARE_INLINE_FUNCTION
            void operator()(const int i, const int j, const int k,
                            value_type &lsum) const {
                lsum += input_view(i - lower_offset[0], j - lower_offset[1],
                                   k - lower_offset[2]) *
                        2;
            }

            static void test_3D_negidx(const int N0, const int N1, const int N2) {
                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<3>,
                                    flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    const point_type lower{{-1, -1, -1}};
                    const point_type upper{{N0, N1, N2}};
                    const tile_type tile{{8, 8, 2}};

                    range_type range(point_type{{lower[0], lower[1], lower[2]}},
                                     point_type{{upper[0], upper[1], upper[2]}},
                                     tile_type{{tile[0], tile[1], tile[2]}});

                    TestMDRange_3D_NegIdx functor(lower[0], lower[1], lower[2], upper[0],
                                                  upper[1], upper[2]);

                    parallel_for(range, functor);
                    double sum = 0.0;
                    parallel_reduce(range, functor, sum);

                    REQUIRE_EQ(sum, 2 * (upper[0] - lower[0]) * (upper[1] - lower[1]) *
                                    (upper[2] - lower[2]));
                }
            }
        };

        template<typename ExecSpace>
        struct TestMDRange_4D_NegIdx {
            using value_type = double;

            using DataType = int;
            using ViewType = typename flare::View<DataType ****, ExecSpace>;
            using HostViewType = typename ViewType::HostMirror;

            ViewType input_view;
            DataType lower_offset[4];

            TestMDRange_4D_NegIdx(const DataType L0, const DataType L1, const DataType L2,
                                  const DataType L3, const DataType N0, const DataType N1,
                                  const DataType N2, const DataType N3)
                    : input_view("input_view", N0 - L0, N1 - L1, N2 - L2, N3 - L3) {
                lower_offset[0] = L0;
                lower_offset[1] = L1;
                lower_offset[2] = L2;
                lower_offset[3] = L3;
            }

            // When using negative indices, must offset View appropriately as views cannot
            // take a negative index
            FLARE_INLINE_FUNCTION
            void operator()(const int i, const int j, const int k, const int l) const {
                input_view(i - lower_offset[0], j - lower_offset[1], k - lower_offset[2],
                           l - lower_offset[3]) = 1;
            }

            FLARE_INLINE_FUNCTION
            void operator()(const int i, const int j, const int k, const int l,
                            value_type &lsum) const {
                lsum += input_view(i - lower_offset[0], j - lower_offset[1],
                                   k - lower_offset[2], l - lower_offset[3]) *
                        2;
            }

            static void test_4D_negidx(const int N0, const int N1, const int N2,
                                       const int N3) {
                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<4>,
                                    flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    const point_type lower{{-1, -1, -1, -1}};
                    const point_type upper{{N0, N1, N2, N3}};
                    const tile_type tile{{8, 8, 2, 2}};

                    range_type range(point_type{{lower[0], lower[1], lower[2], lower[3]}},
                                     point_type{{upper[0], upper[1], upper[2], upper[3]}},
                                     tile_type{{tile[0], tile[1], tile[2], tile[3]}});

                    TestMDRange_4D_NegIdx functor(lower[0], lower[1], lower[2], lower[3],
                                                  upper[0], upper[1], upper[2], upper[3]);

                    parallel_for(range, functor);
                    double sum = 0.0;
                    parallel_reduce(range, functor, sum);

                    REQUIRE_EQ(sum, 2 * (upper[0] - lower[0]) * (upper[1] - lower[1]) *
                                    (upper[2] - lower[2]) * (upper[3] - lower[3]));
                }
            }
        };

        template<typename ExecSpace>
        struct TestMDRange_5D_NegIdx {
            using value_type = double;

            using DataType = int;
            using ViewType = typename flare::View<DataType *****, ExecSpace>;
            using HostViewType = typename ViewType::HostMirror;

            ViewType input_view;
            DataType lower_offset[5];

            TestMDRange_5D_NegIdx(const DataType L0, const DataType L1, const DataType L2,
                                  const DataType L3, const DataType L4, const DataType N0,
                                  const DataType N1, const DataType N2, const DataType N3,
                                  const DataType N4)
                    : input_view("input_view", N0 - L0, N1 - L1, N2 - L2, N3 - L3, N4 - L4) {
                lower_offset[0] = L0;
                lower_offset[1] = L1;
                lower_offset[2] = L2;
                lower_offset[3] = L3;
                lower_offset[4] = L4;
            }

            // When using negative indices, must offset View appropriately as views cannot
            // take a negative index
            FLARE_INLINE_FUNCTION
            void operator()(const int i, const int j, const int k, const int l,
                            const int m) const {
                input_view(i - lower_offset[0], j - lower_offset[1], k - lower_offset[2],
                           l - lower_offset[3], m - lower_offset[4]) = 1;
            }

            FLARE_INLINE_FUNCTION
            void operator()(const int i, const int j, const int k, const int l,
                            const int m, value_type &lsum) const {
                lsum += input_view(i - lower_offset[0], j - lower_offset[1],
                                   k - lower_offset[2], l - lower_offset[3],
                                   m - lower_offset[4]) *
                        2;
            }

            static void test_5D_negidx(const int N0, const int N1, const int N2,
                                       const int N3, const int N4) {
                {
                    using range_type =
                            typename flare::MDRangePolicy<ExecSpace, flare::Rank<5>,
                                    flare::IndexType<int>>;
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    const point_type lower{{-1, -1, -1, -1, -1}};
                    const point_type upper{{N0, N1, N2, N3, N4}};
                    const tile_type tile{{8, 4, 2, 2, 2}};

                    range_type range(
                            point_type{{lower[0], lower[1], lower[2], lower[3], lower[4]}},
                            point_type{{upper[0], upper[1], upper[2], upper[3], upper[4]}},
                            tile_type{{tile[0], tile[1], tile[2], tile[3], tile[4]}});

                    TestMDRange_5D_NegIdx functor(lower[0], lower[1], lower[2], lower[3],
                                                  lower[4], upper[0], upper[1], upper[2],
                                                  upper[3], upper[4]);

                    parallel_for(range, functor);
                    double sum = 0.0;
                    parallel_reduce(range, functor, sum);

                    REQUIRE_EQ(sum, 2 * (upper[0] - lower[0]) * (upper[1] - lower[1]) *
                                    (upper[2] - lower[2]) * (upper[3] - lower[3]) *
                                    (upper[4] - lower[4]));
                }
            }
        };

        template<typename ExecSpace>
        struct TestMDRange_6D_NegIdx {
            using value_type = double;

            using DataType = int;
            using ViewType = typename flare::View<DataType ******, ExecSpace>;
            using HostViewType = typename ViewType::HostMirror;

            ViewType input_view;
            DataType lower_offset[6];

            TestMDRange_6D_NegIdx(const DataType L0, const DataType L1, const DataType L2,
                                  const DataType L3, const DataType L4, const DataType L5,
                                  const DataType N0, const DataType N1, const DataType N2,
                                  const DataType N3, const DataType N4, const DataType N5)
                    : input_view("input_view", N0 - L0, N1 - L1, N2 - L2, N3 - L3, N4 - L4,
                                 N5 - L5) {
                lower_offset[0] = L0;
                lower_offset[1] = L1;
                lower_offset[2] = L2;
                lower_offset[3] = L3;
                lower_offset[4] = L4;
                lower_offset[5] = L5;
            }

            // When using negative indices, must offset View appropriately as views cannot
            // take a negative index
            FLARE_INLINE_FUNCTION
            void operator()(const int i, const int j, const int k, const int l,
                            const int m, const int n) const {
                input_view(i - lower_offset[0], j - lower_offset[1], k - lower_offset[2],
                           l - lower_offset[3], m - lower_offset[4], n - lower_offset[5]) =
                        1;
            }

            FLARE_INLINE_FUNCTION
            void operator()(const int i, const int j, const int k, const int l,
                            const int m, const int n, value_type &lsum) const {
                lsum += input_view(i - lower_offset[0], j - lower_offset[1],
                                   k - lower_offset[2], l - lower_offset[3],
                                   m - lower_offset[4], n - lower_offset[5]) *
                        2;
            }

            static void test_6D_negidx(const int N0, const int N1, const int N2,
                                       const int N3, const int N4, const int N5) {
                {
#if defined(FLARE_COMPILER_INTEL)
                                                                                                                                            // Launchbounds causes hang with intel compilers
      using range_type =
          typename flare::MDRangePolicy<ExecSpace, flare::Rank<6>,
                                         flare::IndexType<int>>;
#else
                    // Launchbounds to ensure the tile fits into a CUDA block under register
                    // constraints
                    using range_type = typename flare::MDRangePolicy<
                            ExecSpace, flare::LaunchBounds<256, 1>, flare::Rank<6>,
                            flare::IndexType<int>>;
#endif
                    using tile_type = typename range_type::tile_type;
                    using point_type = typename range_type::point_type;

                    const point_type lower{{-1, -1, -1, -1, -1, -1}};
                    const point_type upper{{N0, N1, N2, N3, N4, N5}};
                    const tile_type tile{{8, 4, 2, 2, 2, 1}};

                    range_type range(
                            point_type{
                                    {lower[0], lower[1], lower[2], lower[3], lower[4], lower[5]}},
                            point_type{
                                    {upper[0], upper[1], upper[2], upper[3], upper[4], upper[5]}},
                            tile_type{{tile[0], tile[1], tile[2], tile[3], tile[4], tile[5]}});

                    TestMDRange_6D_NegIdx functor(lower[0], lower[1], lower[2], lower[3],
                                                  lower[4], lower[5], upper[0], upper[1],
                                                  upper[2], upper[3], upper[4], upper[5]);

                    parallel_for(range, functor);
                    double sum = 0.0;
                    parallel_reduce(range, functor, sum);

                    REQUIRE_EQ(sum, 2 * (upper[0] - lower[0]) * (upper[1] - lower[1]) *
                                    (upper[2] - lower[2]) * (upper[3] - lower[3]) *
                                    (upper[4] - lower[4]) * (upper[5] - lower[5]));
                }
            }
        };

        template<typename ExecSpace>
        struct TestMDRange_ReduceScalar {
            struct Scalar {
                double v[4];

                FLARE_INLINE_FUNCTION
                Scalar() {
                    for (int i = 0; i < 4; i++) v[i] = 0;
                }

                FLARE_INLINE_FUNCTION
                Scalar(const Scalar &src) {
                    for (int i = 0; i < 4; i++) v[i] = src.v[i];
                }

                FLARE_INLINE_FUNCTION
                void operator=(const Scalar &src) {
                    for (int i = 0; i < 4; i++) v[i] = src.v[i];
                }

                FLARE_INLINE_FUNCTION
                void operator+=(const Scalar &src) {
                    for (int i = 0; i < 4; i++) v[i] += src.v[i];
                }
            };

            static void test_scalar_reduce(const int N0, const int N1) {
#if defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)
                Scalar sum;
                using range_type =
                        typename flare::MDRangePolicy<ExecSpace, flare::Rank<2>,
                                flare::IndexType<int>>;
                using tile_type = typename range_type::tile_type;
                using point_type = typename range_type::point_type;

                range_type range(point_type{{0, 0}}, point_type{{N0, N1}},
                                 tile_type{{3, 3}});

                parallel_reduce(
                        range,
                        FLARE_LAMBDA(int, int, Scalar &lsum) {
                            for (int i = 0; i < 4; i++) lsum.v[i]++;
                        },
                        sum);
                for (int i = 0; i < 4; i++) REQUIRE_EQ(sum.v[i], N0 * N1);
#else
                                                                                                                                        std::ignore = N0;
    std::ignore = N1;
#endif
            }
        };

    }  // namespace

}  // namespace Test
