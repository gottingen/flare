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

#ifndef CONTAINERS_OFFSET_VIEW_TEST_H_
#define CONTAINERS_OFFSET_VIEW_TEST_H_

#include <doctest.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <flare/timer.h>
#include <flare/offset_view.h>

using std::cout;
using std::endl;

namespace Test {

    template<typename Scalar, typename Device>
    void test_offsetview_construction() {
        using offset_view_type = flare::experimental::OffsetView<Scalar **, Device>;
        using view_type = flare::View<Scalar **, Device>;

        std::pair<int64_t, int64_t> range0 = {-1, 3};
        std::pair<int64_t, int64_t> range1 = {-2, 2};

        {
            offset_view_type o1;
            REQUIRE_FALSE(o1.is_allocated());

            o1 = offset_view_type("o1", {-1, 3}, {-2, 2});
            offset_view_type o2(o1);
            offset_view_type o3("o3", range0, range1);

            REQUIRE(o1.is_allocated());
            REQUIRE(o2.is_allocated());
            REQUIRE(o3.is_allocated());
        }

        offset_view_type ov("firstOV", range0, range1);

        REQUIRE_EQ("firstOV", ov.label());
        REQUIRE_EQ(2, ov.Rank);

        REQUIRE_EQ(ov.begin(0), -1);
        REQUIRE_EQ(ov.end(0), 4);

        REQUIRE_EQ(ov.begin(1), -2);
        REQUIRE_EQ(ov.end(1), 3);

        REQUIRE_EQ(ov.extent(0), 5u);
        REQUIRE_EQ(ov.extent(1), 5u);

        {
            flare::experimental::OffsetView<Scalar *, Device> offsetV1("OneDOffsetView",
                                                                       range0);

            flare::RangePolicy<Device, int> rangePolicy1(offsetV1.begin(0),
                                                         offsetV1.end(0));
            flare::parallel_for(
                    rangePolicy1, FLARE_LAMBDA(const int i) { offsetV1(i) = 1; });
            flare::fence();

            int OVResult = 0;
            flare::parallel_reduce(
                    rangePolicy1,
                    FLARE_LAMBDA(const int i, int &updateMe) { updateMe += offsetV1(i); },
                    OVResult);

            flare::fence();
            REQUIRE_EQ(OVResult, offsetV1.end(0) - offsetV1.begin(0));
        }
        {  // test deep copy of scalar const value into mirro
            const int constVal = 6;
            typename offset_view_type::HostMirror hostOffsetView =
                    flare::create_mirror_view(ov);

            flare::deep_copy(hostOffsetView, constVal);

            for (int i = hostOffsetView.begin(0); i < hostOffsetView.end(0); ++i) {
                for (int j = hostOffsetView.begin(1); j < hostOffsetView.end(1); ++j) {
                    REQUIRE_EQ(hostOffsetView(i, j), constVal);
                }
            }
        }

        const int ovmin0 = ov.begin(0);
        const int ovend0 = ov.end(0);
        const int ovmin1 = ov.begin(1);
        const int ovend1 = ov.end(1);

        using range_type =
                flare::MDRangePolicy<Device, flare::Rank<2>, flare::IndexType<int> >;
        using point_type = typename range_type::point_type;

        range_type rangePolicy2D(point_type{{ovmin0, ovmin1}},
                                 point_type{{ovend0, ovend1}});

        const int constValue = 9;
        flare::parallel_for(
                rangePolicy2D,
                FLARE_LAMBDA(const int i, const int j) { ov(i, j) = constValue; });

        // test offsetview to offsetviewmirror deep copy
        typename offset_view_type::HostMirror hostOffsetView =
                flare::create_mirror_view(ov);

        flare::deep_copy(hostOffsetView, ov);

        for (int i = hostOffsetView.begin(0); i < hostOffsetView.end(0); ++i) {
            for (int j = hostOffsetView.begin(1); j < hostOffsetView.end(1); ++j) {
                REQUIRE_EQ(hostOffsetView(i, j), constValue);
            }
        }

        int OVResult = 0;
        flare::parallel_reduce(
                rangePolicy2D,
                FLARE_LAMBDA(const int i, const int j, int &updateMe) {
                    updateMe += ov(i, j);
                },
                OVResult);

        int answer = 0;
        for (int i = ov.begin(0); i < ov.end(0); ++i) {
            for (int j = ov.begin(1); j < ov.end(1); ++j) {
                answer += constValue;
            }
        }

        REQUIRE_EQ(OVResult, answer);

        {
            offset_view_type ovCopy(ov);
            REQUIRE_EQ(ovCopy == ov, true);
        }

        {
            offset_view_type ovAssigned = ov;
            REQUIRE_EQ(ovAssigned == ov, true);
        }

        {  // construct OffsetView from a View plus begins array
            const int extent0 = 100;
            const int extent1 = 200;
            const int extent2 = 300;
            flare::View<Scalar ***, Device> view3D("view3D", extent0, extent1, extent2);

            flare::deep_copy(view3D, 1);

            using range3_type = flare::MDRangePolicy<Device, flare::Rank<3>,
                    flare::IndexType<int64_t> >;
            using point3_type = typename range3_type::point_type;

            typename point3_type::value_type begins0 = -10, begins1 = -20,
                    begins2 = -30;
            flare::Array<int64_t, 3> begins = {{begins0, begins1, begins2}};
            flare::experimental::OffsetView<Scalar ***, Device> offsetView3D(view3D,
                                                                             begins);

            range3_type rangePolicy3DZero(point3_type{{0, 0, 0}},
                                          point3_type{{extent0, extent1, extent2}});

            int view3DSum = 0;
            flare::parallel_reduce(
                    rangePolicy3DZero,
                    FLARE_LAMBDA(const int i, const int j, int k, int &updateMe) {
                        updateMe += view3D(i, j, k);
                    },
                    view3DSum);

            range3_type rangePolicy3D(
                    point3_type{{begins0, begins1, begins2}},
                    point3_type{{begins0 + extent0, begins1 + extent1, begins2 + extent2}});
            int offsetView3DSum = 0;

            flare::parallel_reduce(
                    rangePolicy3D,
                    FLARE_LAMBDA(const int i, const int j, int k, int &updateMe) {
                        updateMe += offsetView3D(i, j, k);
                    },
                    offsetView3DSum);

            REQUIRE_EQ(view3DSum, offsetView3DSum);
        }
        view_type viewFromOV = ov.view();

        REQUIRE_EQ(viewFromOV == ov, true);

        {
            offset_view_type ovFromV(viewFromOV, {-1, -2});

            REQUIRE_EQ(ovFromV == viewFromOV, true);
        }
        {
            offset_view_type ovFromV = viewFromOV;
            REQUIRE_EQ(ovFromV == viewFromOV, true);
        }

        {  // test offsetview to view deep copy
            view_type aView("aView", ov.extent(0), ov.extent(1));
            flare::deep_copy(aView, ov);

            int sum = 0;
            flare::parallel_reduce(
                    rangePolicy2D,
                    FLARE_LAMBDA(const int i, const int j, int &updateMe) {
                        updateMe += ov(i, j) - aView(i - ov.begin(0), j - ov.begin(1));
                    },
                    sum);

            REQUIRE_EQ(sum, 0);
        }

        {  // test view to  offsetview deep copy
            view_type aView("aView", ov.extent(0), ov.extent(1));

            flare::deep_copy(aView, 99);
            flare::deep_copy(ov, aView);

            int sum = 0;
            flare::parallel_reduce(
                    rangePolicy2D,
                    FLARE_LAMBDA(const int i, const int j, int &updateMe) {
                        updateMe += ov(i, j) - aView(i - ov.begin(0), j - ov.begin(1));
                    },
                    sum);

            REQUIRE_EQ(sum, 0);
        }
    }

    template<typename Scalar, typename Device>
    void test_offsetview_unmanaged_construction() {
        // Preallocated memory (Only need a valid address for this test)
        Scalar s;

        {
            // Constructing an OffsetView directly around our preallocated memory
            flare::Array<int64_t, 1> begins1{{2}};
            flare::Array<int64_t, 1> ends1{{3}};
            flare::experimental::OffsetView<Scalar *, Device> ov1(&s, begins1, ends1);

            // Constructing an OffsetView around an unmanaged View of our preallocated
            // memory
            flare::View<Scalar *, Device> v1(&s, ends1[0] - begins1[0]);
            flare::experimental::OffsetView<Scalar *, Device> ovv1(v1, begins1);

            // They should match
            REQUIRE_EQ(ovv1, ov1);
        }

        {
            flare::Array<int64_t, 2> begins2{{-2, -7}};
            flare::Array<int64_t, 2> ends2{{5, -3}};
            flare::experimental::OffsetView<Scalar **, Device> ov2(&s, begins2, ends2);

            flare::View<Scalar **, Device> v2(&s, ends2[0] - begins2[0],
                                              ends2[1] - begins2[1]);
            flare::experimental::OffsetView<Scalar **, Device> ovv2(v2, begins2);

            REQUIRE_EQ(ovv2, ov2);
        }

        {
            flare::Array<int64_t, 3> begins3{{2, 3, 5}};
            flare::Array<int64_t, 3> ends3{{7, 11, 13}};
            flare::experimental::OffsetView<Scalar ***, Device> ovv3(&s, begins3,
                                                                     ends3);

            flare::View<Scalar ***, Device> v3(&s, ends3[0] - begins3[0],
                                               ends3[1] - begins3[1],
                                               ends3[2] - begins3[2]);
            flare::experimental::OffsetView<Scalar ***, Device> ov3(v3, begins3);

            REQUIRE_EQ(ovv3, ov3);
        }

        {
            // Test all four public constructor overloads (begins_type x
            // index_list_type)
            flare::Array<int64_t, 1> begins{{-3}};
            flare::Array<int64_t, 1> ends{{2}};

            flare::experimental::OffsetView<Scalar *, Device> bb(&s, begins, ends);
            flare::experimental::OffsetView<Scalar *, Device> bi(&s, begins, {2});
            flare::experimental::OffsetView<Scalar *, Device> ib(&s, {-3}, ends);
            flare::experimental::OffsetView<Scalar *, Device> ii(&s, {-3}, {2});

            REQUIRE_EQ(bb, bi);
            REQUIRE_EQ(bb, ib);
            REQUIRE_EQ(bb, ii);
        }

        {
            using offset_view_type = flare::experimental::OffsetView<Scalar *, Device>;

            // Range calculations must be positive
            REQUIRE_NOTHROW(offset_view_type(&s, {0}, {1}));
            REQUIRE_NOTHROW(offset_view_type(&s, {0}, {0}));
            REQUIRE_THROWS_AS(offset_view_type(&s, {0}, {-1}), std::runtime_error);
        }

        {
            using offset_view_type = flare::experimental::OffsetView<Scalar *, Device>;

            // Range calculations must not overflow
            REQUIRE_NOTHROW(offset_view_type(&s, {0}, {0x7fffffffffffffffl}));
            REQUIRE_THROWS_AS(offset_view_type(&s, {-1}, {0x7fffffffffffffffl}),
                              std::runtime_error);
            REQUIRE_THROWS_AS(
                    offset_view_type(&s, {-0x7fffffffffffffffl - 1}, {0x7fffffffffffffffl}),
                    std::runtime_error);
            REQUIRE_THROWS_AS(offset_view_type(&s, {-0x7fffffffffffffffl - 1}, {0}),
                              std::runtime_error);
        }

        {
            using offset_view_type = flare::experimental::OffsetView<Scalar **, Device>;

            // Should throw when the rank of begins and/or ends doesn't match that of
            // OffsetView
            REQUIRE_THROWS_AS(offset_view_type(&s, {0}, {1}), std::runtime_error);
            REQUIRE_THROWS_AS(offset_view_type(&s, {0}, {1, 1}), std::runtime_error);
            REQUIRE_THROWS_AS(offset_view_type(&s, {0}, {1, 1, 1}), std::runtime_error);
            REQUIRE_THROWS_AS(offset_view_type(&s, {0, 0}, {1}), std::runtime_error);
            REQUIRE_NOTHROW(offset_view_type(&s, {0, 0}, {1, 1}));
            REQUIRE_THROWS_AS(offset_view_type(&s, {0, 0}, {1, 1, 1}), std::runtime_error);
            REQUIRE_THROWS_AS(offset_view_type(&s, {0, 0, 0}, {1}), std::runtime_error);
            REQUIRE_THROWS_AS(offset_view_type(&s, {0, 0, 0}, {1, 1}), std::runtime_error);
            REQUIRE_THROWS_AS(offset_view_type(&s, {0, 0, 0}, {1, 1, 1}),
                              std::runtime_error);
        }
    }

    template<typename Scalar, typename Device>
    void test_offsetview_subview() {
        {  // test subview 1
            flare::experimental::OffsetView<Scalar *, Device> sliceMe("offsetToSlice",
                                                                      {-10, 20});
            {
                auto offsetSubviewa = flare::experimental::subview(sliceMe, 0);
                REQUIRE_EQ(offsetSubviewa.Rank, 0);
            }
        }
        {  // test subview 2
            flare::experimental::OffsetView<Scalar **, Device> sliceMe(
                    "offsetToSlice", {-10, 20}, {-20, 30});
            {
                auto offsetSubview =
                        flare::experimental::subview(sliceMe, flare::ALL(), -2);
                REQUIRE_EQ(offsetSubview.Rank, 1);
            }

            {
                auto offsetSubview =
                        flare::experimental::subview(sliceMe, 0, flare::ALL());
                REQUIRE_EQ(offsetSubview.Rank, 1);
            }
        }

        {  // test subview rank 3

            flare::experimental::OffsetView<Scalar ***, Device> sliceMe(
                    "offsetToSlice", {-10, 20}, {-20, 30}, {-30, 40});

            // slice 1
            {
                auto offsetSubview = flare::experimental::subview(sliceMe, flare::ALL(),
                                                                  flare::ALL(), 0);
                REQUIRE_EQ(offsetSubview.Rank, 2);
            }
            {
                auto offsetSubview = flare::experimental::subview(sliceMe, flare::ALL(),
                                                                  0, flare::ALL());
                REQUIRE_EQ(offsetSubview.Rank, 2);
            }

            {
                auto offsetSubview = flare::experimental::subview(
                        sliceMe, 0, flare::ALL(), flare::ALL());
                REQUIRE_EQ(offsetSubview.Rank, 2);
            }
            {
                auto offsetSubview = flare::experimental::subview(
                        sliceMe, 0, flare::ALL(), flare::make_pair(-30, -21));
                REQUIRE_EQ(offsetSubview.Rank, 2);

                REQUIRE_EQ(offsetSubview.begin(0), -20);
                REQUIRE_EQ(offsetSubview.end(0), 31);
                REQUIRE_EQ(offsetSubview.begin(1), 0);
                REQUIRE_EQ(offsetSubview.end(1), 9);

                using range_type = flare::MDRangePolicy<Device, flare::Rank<2>,
                        flare::IndexType<int> >;
                using point_type = typename range_type::point_type;

                const int b0 = offsetSubview.begin(0);
                const int b1 = offsetSubview.begin(1);

                const int e0 = offsetSubview.end(0);
                const int e1 = offsetSubview.end(1);

                range_type rangeP2D(point_type{{b0, b1}}, point_type{{e0, e1}});

                flare::parallel_for(
                        rangeP2D,
                        FLARE_LAMBDA(const int i, const int j) { offsetSubview(i, j) = 6; });

                int sum = 0;
                flare::parallel_reduce(
                        rangeP2D,
                        FLARE_LAMBDA(const int i, const int j, int &updateMe) {
                            updateMe += offsetSubview(i, j);
                        },
                        sum);

                REQUIRE_EQ(sum, 6 * (e0 - b0) * (e1 - b1));
            }

            // slice 2
            {
                auto offsetSubview =
                        flare::experimental::subview(sliceMe, flare::ALL(), 0, 0);
                REQUIRE_EQ(offsetSubview.Rank, 1);
            }
            {
                auto offsetSubview =
                        flare::experimental::subview(sliceMe, 0, 0, flare::ALL());
                REQUIRE_EQ(offsetSubview.Rank, 1);
            }

            {
                auto offsetSubview =
                        flare::experimental::subview(sliceMe, 0, flare::ALL(), 0);
                REQUIRE_EQ(offsetSubview.Rank, 1);
            }
        }

        {  // test subview rank 4

            flare::experimental::OffsetView<Scalar ****, Device> sliceMe(
                    "offsetToSlice", {-10, 20}, {-20, 30}, {-30, 40}, {-40, 50});

            // slice 1
            {
                auto offsetSubview = flare::experimental::subview(
                        sliceMe, flare::ALL(), flare::ALL(), flare::ALL(), 0);
                REQUIRE_EQ(offsetSubview.Rank, 3);
            }
            {
                auto offsetSubview = flare::experimental::subview(
                        sliceMe, flare::ALL(), flare::ALL(), 0, flare::ALL());
                REQUIRE_EQ(offsetSubview.Rank, 3);
            }
            {
                auto offsetSubview = flare::experimental::subview(
                        sliceMe, flare::ALL(), 0, flare::ALL(), flare::ALL());
                REQUIRE_EQ(offsetSubview.Rank, 3);
            }
            {
                auto offsetSubview = flare::experimental::subview(
                        sliceMe, 0, flare::ALL(), flare::ALL(), flare::ALL());
                REQUIRE_EQ(offsetSubview.Rank, 3);
            }

            // slice 2
            auto offsetSubview2a = flare::experimental::subview(sliceMe, flare::ALL(),
                                                                flare::ALL(), 0, 0);
            REQUIRE_EQ(offsetSubview2a.Rank, 2);
            {
                auto offsetSubview2b = flare::experimental::subview(
                        sliceMe, flare::ALL(), 0, flare::ALL(), 0);
                REQUIRE_EQ(offsetSubview2b.Rank, 2);
            }
            {
                auto offsetSubview2b = flare::experimental::subview(
                        sliceMe, flare::ALL(), 0, 0, flare::ALL());
                REQUIRE_EQ(offsetSubview2b.Rank, 2);
            }
            {
                auto offsetSubview2b = flare::experimental::subview(
                        sliceMe, 0, flare::ALL(), 0, flare::ALL());
                REQUIRE_EQ(offsetSubview2b.Rank, 2);
            }
            {
                auto offsetSubview2b = flare::experimental::subview(
                        sliceMe, 0, 0, flare::ALL(), flare::ALL());
                REQUIRE_EQ(offsetSubview2b.Rank, 2);
            }
            // slice 3
            {
                auto offsetSubview =
                        flare::experimental::subview(sliceMe, flare::ALL(), 0, 0, 0);
                REQUIRE_EQ(offsetSubview.Rank, 1);
            }
            {
                auto offsetSubview =
                        flare::experimental::subview(sliceMe, 0, flare::ALL(), 0, 0);
                REQUIRE_EQ(offsetSubview.Rank, 1);
            }
            {
                auto offsetSubview =
                        flare::experimental::subview(sliceMe, 0, 0, flare::ALL(), 0);
                REQUIRE_EQ(offsetSubview.Rank, 1);
            }
            {
                auto offsetSubview =
                        flare::experimental::subview(sliceMe, 0, 0, 0, flare::ALL());
                REQUIRE_EQ(offsetSubview.Rank, 1);
            }
        }
    }

    template<class InputIt, class T, class BinaryOperation>
    FLARE_INLINE_FUNCTION T std_accumulate(InputIt first, InputIt last, T init,
                                           BinaryOperation op) {
        for (; first != last; ++first) {
            init = op(std::move(init), *first);
        }
        return init;
    }

    FLARE_INLINE_FUNCTION int element(std::initializer_list<int>
    il) {
    return
    std_accumulate(il
    .

    begin(), il

    .

    end(),

    0,
    [](
    int l,
    int r
    ) {
    return l * 10 +
    r;
}
);
}

template<typename DEVICE>
void test_offsetview_offsets_rank1() {
    using data_type = int *;
    using view_type = flare::View<data_type, DEVICE>;
    using index_type = flare::IndexType<int>;
    using execution_policy = flare::RangePolicy<DEVICE, index_type>;
    using offset_view_type = flare::experimental::OffsetView<data_type, DEVICE>;

    view_type v("View1", 10);
    flare::parallel_for(
            "For1", execution_policy(0, v.extent_int(0)),
            FLARE_LAMBDA(const int i) { v(i) = element({i}); });

    int errors;
    flare::parallel_reduce(
            "Reduce1", execution_policy(-3, 4),
            FLARE_LAMBDA(const int ii, int &lerrors) {
                offset_view_type ov(v, {ii});
                lerrors += (ov(3) != element({3 - ii}));
            },
            errors);

    REQUIRE_EQ(0, errors);
}

template<typename DEVICE>
void test_offsetview_offsets_rank2() {
    using data_type = int **;
    using view_type = flare::View<data_type, DEVICE>;
    using index_type = flare::IndexType<int>;
    using execution_policy = flare::RangePolicy<DEVICE, index_type>;
    using offset_view_type = flare::experimental::OffsetView<data_type, DEVICE>;

    view_type v("View2", 10, 10);
    flare::parallel_for(
            "For2", execution_policy(0, v.extent_int(0)), FLARE_LAMBDA(const int i) {
                for (int j = 0; j != v.extent_int(1); ++j) {
                    v(i, j) = element({i, j});
                }
            });

    int errors;
    flare::parallel_reduce(
            "Reduce2", execution_policy(-3, 4),
            FLARE_LAMBDA(const int ii, int &lerrors) {
                for (int jj = -3; jj <= 3; ++jj) {
                    offset_view_type ov(v, {ii, jj});
                    lerrors += (ov(3, 3) != element({3 - ii, 3 - jj}));
                }
            },
            errors);

    REQUIRE_EQ(0, errors);
}

template<typename DEVICE>
void test_offsetview_offsets_rank3() {
    using data_type = int ***;
    using view_type = flare::View<data_type, DEVICE>;
    using index_type = flare::IndexType<int>;
    using execution_policy = flare::RangePolicy<DEVICE, index_type>;
    using offset_view_type = flare::experimental::OffsetView<data_type, DEVICE>;

    view_type v("View3", 10, 10, 10);
    flare::parallel_for(
            "For3", execution_policy(0, v.extent_int(0)), FLARE_LAMBDA(const int i) {
                for (int j = 0; j != v.extent_int(1); ++j) {
                    for (int k = 0; k != v.extent_int(2); ++k) {
                        v(i, j, k) = element({i, j, k});
                    }
                }
            });

    int errors;
    flare::parallel_reduce(
            "Reduce3", execution_policy(-3, 4),
            FLARE_LAMBDA(const int ii, int &lerrors) {
                for (int jj = -3; jj <= 3; ++jj) {
                    for (int kk = -3; kk <= 3; ++kk) {
                        offset_view_type ov(v, {ii, jj, kk});
                        lerrors += (ov(3, 3, 3) != element({3 - ii, 3 - jj, 3 - kk}));
                    }
                }
            },
            errors);

    REQUIRE_EQ(0, errors);
}

TEST_CASE("TEST_CATEGORY, offsetview_construction") {
    test_offsetview_construction<int, TEST_EXECSPACE>();
}

TEST_CASE("TEST_CATEGORY, offsetview_unmanaged_construction") {
    test_offsetview_unmanaged_construction<int, TEST_EXECSPACE>();
}

TEST_CASE("TEST_CATEGORY, offsetview_subview") {
    test_offsetview_subview<int, TEST_EXECSPACE>();
}

TEST_CASE("TEST_CATEGORY, offsetview_offsets_rank1") {
    test_offsetview_offsets_rank1<TEST_EXECSPACE>();
}

TEST_CASE("TEST_CATEGORY, offsetview_offsets_rank2") {
    test_offsetview_offsets_rank2<TEST_EXECSPACE>();
}

TEST_CASE("TEST_CATEGORY, offsetview_offsets_rank3") {
    test_offsetview_offsets_rank3<TEST_EXECSPACE>();
}

}  // namespace Test

#endif  // CONTAINERS_OFFSET_VIEW_TEST_H_
