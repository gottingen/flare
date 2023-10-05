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
#ifndef TEST_VIEW_SUB_VIEW_H_
#define TEST_VIEW_SUB_VIEW_H_

#include <doctest.h>

#include <flare/core.h>
#include <sstream>
#include <iostream>
#include <type_traits>

// TODO @refactoring move this to somewhere common

//------------------------------------------------------------------------------

template<class...>
struct _flare____________________static_test_failure_____;

template<class...>
struct static_predicate_message {
};

//------------------------------------------------------------------------------

template<class, template<class...> class, class...>
struct static_assert_predicate_true_impl;

template<template<class...> class predicate, class... message, class... args>
struct static_assert_predicate_true_impl<
        std::enable_if_t<predicate<args...>::type::value>, predicate,
        static_predicate_message<message...>, args...> {
    using type = int;
};

template<template<class...> class predicate, class... message, class... args>
struct static_assert_predicate_true_impl<
        std::enable_if_t<!predicate<args...>::type::value>, predicate,
        static_predicate_message<message...>, args...> {
    using type = typename _flare____________________static_test_failure_____<
            message...>::type;
};

template<template<class...> class predicate, class... args>
struct static_assert_predicate_true
        : static_assert_predicate_true_impl<void, predicate,
                static_predicate_message<>, args...> {
};

template<template<class...> class predicate, class... message, class... args>
struct static_assert_predicate_true<
        predicate, static_predicate_message<message...>, args...>
        : static_assert_predicate_true_impl<
                void, predicate, static_predicate_message<message...>, args...> {
};

//------------------------------------------------------------------------------

// error "messages"
struct _flare__________types_should_be_the_same_____expected_type__ {
};
struct _flare__________actual_type_was__ {
};
template<class Expected, class Actual>
struct static_expect_same {
    using type = typename static_assert_predicate_true<
            std::is_same,
            static_predicate_message<
                    _flare__________types_should_be_the_same_____expected_type__,
                    Expected, _flare__________actual_type_was__, Actual>,
            Expected, Actual>::type;
};

//------------------------------------------------------------------------------

namespace TestViewSubview {

    template<class Layout, class Space>
    struct getView {
        static flare::View<double **, Layout, Space> get(int n, int m) {
            return flare::View<double **, Layout, Space>("G", n, m);
        }
    };

    template<class Space>
    struct getView<flare::LayoutStride, Space> {
        static flare::View<double **, flare::LayoutStride, Space> get(int n, int m) {
            const int rank = 2;
            const int order[] = {0, 1};
            const unsigned dim[] = {unsigned(n), unsigned(m)};
            flare::LayoutStride stride =
                    flare::LayoutStride::order_dimensions(rank, order, dim);

            return flare::View<double **, flare::LayoutStride, Space>("G", stride);
        }
    };

    template<class ViewType, class Space>
    struct fill_1D {
        using execution_space = typename Space::execution_space;
        using size_type = typename ViewType::size_type;

        ViewType a;
        double val;

        fill_1D(ViewType a_, double val_) : a(a_), val(val_) {}

        FLARE_INLINE_FUNCTION
        void operator()(const int i) const { a(i) = val; }
    };

    template<class ViewType, class Space>
    struct fill_2D {
        using execution_space = typename Space::execution_space;
        using size_type = typename ViewType::size_type;

        ViewType a;
        double val;

        fill_2D(ViewType a_, double val_) : a(a_), val(val_) {}

        FLARE_INLINE_FUNCTION
        void operator()(const int i) const {
            for (int j = 0; j < static_cast<int>(a.extent(1)); j++) {
                a(i, j) = val;
            }
        }
    };

    template<class Layout, class Space>
    void test_auto_1d() {
        using mv_type = flare::View<double **, Layout, Space>;
        using execution_space = typename Space::execution_space;
        using size_type = typename mv_type::size_type;

        const double ZERO = 0.0;
        const double ONE = 1.0;
        const double TWO = 2.0;

        const size_type numRows = 10;
        const size_type numCols = 3;

        mv_type X = getView<Layout, Space>::get(numRows, numCols);
        typename mv_type::HostMirror X_h = flare::create_mirror_view(X);

        fill_2D<mv_type, Space> f1(X, ONE);
        using Property = flare::experimental::WorkItemProperty::None_t;
        flare::parallel_for(
                flare::RangePolicy<execution_space, Property>(0, X.extent(0)), f1);
        flare::fence();
        flare::deep_copy(X_h, X);
        for (size_type j = 0; j < numCols; ++j) {
            for (size_type i = 0; i < numRows; ++i) {
                REQUIRE_EQ(X_h(i, j), ONE);
            }
        }

        fill_2D<mv_type, Space> f2(X, 0.0);
        flare::parallel_for(
                flare::RangePolicy<execution_space, Property>(0, X.extent(0)), f2);
        flare::fence();
        flare::deep_copy(X_h, X);
        for (size_type j = 0; j < numCols; ++j) {
            for (size_type i = 0; i < numRows; ++i) {
                REQUIRE_EQ(X_h(i, j), ZERO);
            }
        }

        fill_2D<mv_type, Space> f3(X, TWO);
        flare::parallel_for(
                flare::RangePolicy<execution_space, Property>(0, X.extent(0)), f3);
        flare::fence();
        flare::deep_copy(X_h, X);
        for (size_type j = 0; j < numCols; ++j) {
            for (size_type i = 0; i < numRows; ++i) {
                REQUIRE_EQ(X_h(i, j), TWO);
            }
        }

        for (size_type j = 0; j < numCols; ++j) {
            auto X_j = flare::subview(X, flare::ALL, j);

            fill_1D<decltype(X_j), Space> f4(X_j, ZERO);
            flare::parallel_for(
                    flare::RangePolicy<execution_space, Property>(0, X_j.extent(0)), f4);
            flare::fence();
            flare::deep_copy(X_h, X);
            for (size_type i = 0; i < numRows; ++i) {
                REQUIRE_EQ(X_h(i, j), ZERO);
            }

            for (size_type jj = 0; jj < numCols; ++jj) {
                auto X_jj = flare::subview(X, flare::ALL, jj);
                fill_1D<decltype(X_jj), Space> f5(X_jj, ONE);
                flare::parallel_for(
                        flare::RangePolicy<execution_space, Property>(0, X_jj.extent(0)),
                        f5);
                flare::fence();
                flare::deep_copy(X_h, X);
                for (size_type i = 0; i < numRows; ++i) {
                    REQUIRE_EQ(X_h(i, jj), ONE);
                }
            }
        }
    }

    template<class LD, class LS, class Space>
    void test_1d_strided_assignment_impl(bool a, bool b, bool c, bool d, int n,
                                         int m) {
        flare::View<double **, LS, Space> l2d("l2d", n, m);

        int col = n > 2 ? 2 : 0;
        int row = m > 2 ? 2 : 0;

        if (flare::SpaceAccessibility<flare::HostSpace,
                typename Space::memory_space>::accessible) {
            if (a) {
                flare::View<double *, LD, Space> l1da =
                        flare::subview(l2d, flare::ALL, row);
                REQUIRE_EQ(&l1da(0), &l2d(0, row));
                if (n > 1) {
                    REQUIRE_EQ(&l1da(1), &l2d(1, row));
                }
            }

            if (b && n > 13) {
                flare::View<double *, LD, Space> l1db =
                        flare::subview(l2d, std::pair<unsigned, unsigned>(2, 13), row);
                REQUIRE_EQ(&l1db(0), &l2d(2, row));
                REQUIRE_EQ(&l1db(1), &l2d(3, row));
            }

            if (c) {
                flare::View<double *, LD, Space> l1dc =
                        flare::subview(l2d, col, flare::ALL);
                REQUIRE_EQ(&l1dc(0), &l2d(col, 0));
                if (m > 1) {
                    REQUIRE_EQ(&l1dc(1), &l2d(col, 1));
                }
            }

            if (d && m > 13) {
                flare::View<double *, LD, Space> l1dd =
                        flare::subview(l2d, col, std::pair<unsigned, unsigned>(2, 13));
                REQUIRE_EQ(&l1dd(0), &l2d(col, 2));
                REQUIRE_EQ(&l1dd(1), &l2d(col, 3));
            }
        }
    }

    template<class Space>
    void test_1d_strided_assignment() {
        test_1d_strided_assignment_impl<flare::LayoutStride, flare::LayoutLeft,
                Space>(true, true, true, true, 17, 3);
        test_1d_strided_assignment_impl<flare::LayoutStride, flare::LayoutRight,
                Space>(true, true, true, true, 17, 3);

        test_1d_strided_assignment_impl<flare::LayoutLeft, flare::LayoutLeft,
                Space>(true, true, false, false, 17, 3);
        test_1d_strided_assignment_impl<flare::LayoutRight, flare::LayoutLeft,
                Space>(true, true, false, false, 17, 3);
        test_1d_strided_assignment_impl<flare::LayoutLeft, flare::LayoutRight,
                Space>(false, false, true, true, 17, 3);
        test_1d_strided_assignment_impl<flare::LayoutRight, flare::LayoutRight,
                Space>(false, false, true, true, 17, 3);

        test_1d_strided_assignment_impl<flare::LayoutLeft, flare::LayoutLeft,
                Space>(true, true, false, false, 17, 1);
        test_1d_strided_assignment_impl<flare::LayoutLeft, flare::LayoutLeft,
                Space>(true, true, true, true, 1, 17);
        test_1d_strided_assignment_impl<flare::LayoutRight, flare::LayoutLeft,
                Space>(true, true, true, true, 1, 17);
        test_1d_strided_assignment_impl<flare::LayoutRight, flare::LayoutLeft,
                Space>(true, true, false, false, 17, 1);

        test_1d_strided_assignment_impl<flare::LayoutLeft, flare::LayoutRight,
                Space>(true, true, true, true, 17, 1);
        test_1d_strided_assignment_impl<flare::LayoutLeft, flare::LayoutRight,
                Space>(false, false, true, true, 1, 17);
        test_1d_strided_assignment_impl<flare::LayoutRight, flare::LayoutRight,
                Space>(false, false, true, true, 1, 17);
        test_1d_strided_assignment_impl<flare::LayoutRight, flare::LayoutRight,
                Space>(true, true, true, true, 17, 1);
    }

    template<class NewView, class OrigView, class... Args>
    void make_subview(bool use_constructor, NewView &v, OrigView org,
                      Args... args) {
        if (use_constructor) {
            v = NewView(org, args...);
        } else {
            v = flare::subview(org, args...);
        }
    }

    template<class Space>
    void test_left_0(bool constr) {
        using view_static_8_type =
                flare::View<int[2][3][4][5][2][3][4][5], flare::LayoutLeft, Space>;

        if (flare::SpaceAccessibility<flare::HostSpace,
                typename Space::memory_space>::accessible) {
            view_static_8_type x_static_8("x_static_left_8");

            REQUIRE(x_static_8.span_is_contiguous());

            flare::View<int, flare::LayoutLeft, Space> x0;
            make_subview(constr, x0, x_static_8, 0, 0, 0, 0, 0, 0, 0, 0);

            REQUIRE(x0.span_is_contiguous());
            REQUIRE_EQ(x0.span(), 1u);
            REQUIRE_EQ(&x0(), &x_static_8(0, 0, 0, 0, 0, 0, 0, 0));

            flare::View<int *, flare::LayoutLeft, Space> x1;
            make_subview(constr, x1, x_static_8, flare::pair<int, int>(0, 2), 1, 2, 3,
                         0, 1, 2, 3);

            REQUIRE(x1.span_is_contiguous());
            REQUIRE_EQ(x1.span(), 2u);
            REQUIRE_EQ(&x1(0), &x_static_8(0, 1, 2, 3, 0, 1, 2, 3));
            REQUIRE_EQ(&x1(1), &x_static_8(1, 1, 2, 3, 0, 1, 2, 3));

            flare::View<int *, flare::LayoutLeft, Space> x_deg1;
            make_subview(constr, x_deg1, x_static_8, flare::pair<int, int>(0, 0), 1, 2,
                         3, 0, 1, 2, 3);

            REQUIRE(x_deg1.span_is_contiguous());
            REQUIRE_EQ(x_deg1.span(), 0u);
            REQUIRE_EQ(x_deg1.data(), &x_static_8(0, 1, 2, 3, 0, 1, 2, 3));

            flare::View<int *, flare::LayoutLeft, Space> x_deg2;
            make_subview(constr, x_deg2, x_static_8, flare::pair<int, int>(2, 2), 2, 3,
                         4, 1, 2, 3, 4);

            REQUIRE(x_deg2.span_is_contiguous());
            REQUIRE_EQ(x_deg2.span(), 0u);
            REQUIRE_EQ(x_deg2.data(), x_static_8.data() + x_static_8.span());

            flare::View<int **, flare::LayoutLeft, Space> x2;
            make_subview(constr, x2, x_static_8, flare::pair<int, int>(0, 2), 1, 2, 3,
                         flare::pair<int, int>(0, 2), 1, 2, 3);

            REQUIRE(!x2.span_is_contiguous());
            REQUIRE_EQ(&x2(0, 0), &x_static_8(0, 1, 2, 3, 0, 1, 2, 3));
            REQUIRE_EQ(&x2(1, 0), &x_static_8(1, 1, 2, 3, 0, 1, 2, 3));
            REQUIRE_EQ(&x2(0, 1), &x_static_8(0, 1, 2, 3, 1, 1, 2, 3));
            REQUIRE_EQ(&x2(1, 1), &x_static_8(1, 1, 2, 3, 1, 1, 2, 3));

            // flare::View< int**, flare::LayoutLeft, Space > error_2 =
            flare::View<int **, flare::LayoutStride, Space> sx2;
            make_subview(constr, sx2, x_static_8, 1, flare::pair<int, int>(0, 2), 2, 3,
                         flare::pair<int, int>(0, 2), 1, 2, 3);

            REQUIRE(!sx2.span_is_contiguous());
            REQUIRE_EQ(&sx2(0, 0), &x_static_8(1, 0, 2, 3, 0, 1, 2, 3));
            REQUIRE_EQ(&sx2(1, 0), &x_static_8(1, 1, 2, 3, 0, 1, 2, 3));
            REQUIRE_EQ(&sx2(0, 1), &x_static_8(1, 0, 2, 3, 1, 1, 2, 3));
            REQUIRE_EQ(&sx2(1, 1), &x_static_8(1, 1, 2, 3, 1, 1, 2, 3));

            flare::View<int ****, flare::LayoutStride, Space> sx4;
            make_subview(constr, sx4, x_static_8, 0,
                         flare::pair<int, int>(0, 2) /* of [3] */
                    ,
                         1, flare::pair<int, int>(1, 3) /* of [5] */
                    ,
                         1, flare::pair<int, int>(0, 2) /* of [3] */
                    ,
                         2, flare::pair<int, int>(2, 4) /* of [5] */
            );

            REQUIRE(!sx4.span_is_contiguous());

            for (int i0 = 0; i0 < (int) sx4.extent(0); ++i0)
                for (int i1 = 0; i1 < (int) sx4.extent(1); ++i1)
                    for (int i2 = 0; i2 < (int) sx4.extent(2); ++i2)
                        for (int i3 = 0; i3 < (int) sx4.extent(3); ++i3) {
                            REQUIRE_EQ(&sx4(i0, i1, i2, i3),
                                       &x_static_8(0, 0 + i0, 1, 1 + i1, 1, 0 + i2, 2, 2 + i3));
                        }
        }
    }

    template<class Space>
    void test_left_0() {
        test_left_0<Space>(true);
        test_left_0<Space>(false);
    }

    template<class Space>
    void test_left_1(bool use_constr) {
        using view_type =
                flare::View<int ****[2][3][4][5], flare::LayoutLeft, Space>;

        if (flare::SpaceAccessibility<flare::HostSpace,
                typename Space::memory_space>::accessible) {
            view_type x8("x_left_8", 2, 3, 4, 5);

            REQUIRE(x8.span_is_contiguous());

            flare::View<int, flare::LayoutLeft, Space> x0;
            make_subview(use_constr, x0, x8, 0, 0, 0, 0, 0, 0, 0, 0);

            REQUIRE(x0.span_is_contiguous());
            REQUIRE_EQ(&x0(), &x8(0, 0, 0, 0, 0, 0, 0, 0));

            flare::View<int *, flare::LayoutLeft, Space> x1;
            make_subview(use_constr, x1, x8, flare::pair<int, int>(0, 2), 1, 2, 3, 0,
                         1, 2, 3);

            REQUIRE(x1.span_is_contiguous());
            REQUIRE_EQ(&x1(0), &x8(0, 1, 2, 3, 0, 1, 2, 3));
            REQUIRE_EQ(&x1(1), &x8(1, 1, 2, 3, 0, 1, 2, 3));

            flare::View<int *, flare::LayoutLeft, Space> x1_deg1;
            make_subview(use_constr, x1_deg1, x8, flare::pair<int, int>(0, 0), 1, 2, 3,
                         0, 1, 2, 3);

            REQUIRE(x1_deg1.span_is_contiguous());
            REQUIRE_EQ(0u, x1_deg1.span());
            REQUIRE_EQ(x1_deg1.data(), &x8(0, 1, 2, 3, 0, 1, 2, 3));

            flare::View<int *, flare::LayoutLeft, Space> x1_deg2;
            make_subview(use_constr, x1_deg2, x8, flare::pair<int, int>(2, 2), 2, 3, 4,
                         1, 2, 3, 4);

            REQUIRE_EQ(0u, x1_deg2.span());
            REQUIRE(x1_deg2.span_is_contiguous());
            REQUIRE_EQ(x1_deg2.data(), x8.data() + x8.span());

            flare::View<int **, flare::LayoutLeft, Space> x2;
            make_subview(use_constr, x2, x8, flare::pair<int, int>(0, 2), 1, 2, 3,
                         flare::pair<int, int>(0, 2), 1, 2, 3);

            REQUIRE(!x2.span_is_contiguous());
            REQUIRE_EQ(&x2(0, 0), &x8(0, 1, 2, 3, 0, 1, 2, 3));
            REQUIRE_EQ(&x2(1, 0), &x8(1, 1, 2, 3, 0, 1, 2, 3));
            REQUIRE_EQ(&x2(0, 1), &x8(0, 1, 2, 3, 1, 1, 2, 3));
            REQUIRE_EQ(&x2(1, 1), &x8(1, 1, 2, 3, 1, 1, 2, 3));

            flare::View<int **, flare::LayoutLeft, Space> x2_deg2;
            make_subview(use_constr, x2_deg2, x8, flare::pair<int, int>(2, 2), 2, 3, 4,
                         1, 2, flare::pair<int, int>(2, 3), 4);
            REQUIRE_EQ(0u, x2_deg2.span());

            // flare::View< int**, flare::LayoutLeft, Space > error_2 =
            flare::View<int **, flare::LayoutStride, Space> sx2;
            make_subview(use_constr, sx2, x8, 1, flare::pair<int, int>(0, 2), 2, 3,
                         flare::pair<int, int>(0, 2), 1, 2, 3);

            REQUIRE(!sx2.span_is_contiguous());
            REQUIRE_EQ(&sx2(0, 0), &x8(1, 0, 2, 3, 0, 1, 2, 3));
            REQUIRE_EQ(&sx2(1, 0), &x8(1, 1, 2, 3, 0, 1, 2, 3));
            REQUIRE_EQ(&sx2(0, 1), &x8(1, 0, 2, 3, 1, 1, 2, 3));
            REQUIRE_EQ(&sx2(1, 1), &x8(1, 1, 2, 3, 1, 1, 2, 3));

            flare::View<int **, flare::LayoutStride, Space> sx2_deg;
            make_subview(use_constr, sx2, x8, 1, flare::pair<int, int>(0, 0), 2, 3,
                         flare::pair<int, int>(0, 2), 1, 2, 3);
            REQUIRE_EQ(0u, sx2_deg.span());

            flare::View<int ****, flare::LayoutStride, Space> sx4;
            make_subview(use_constr, sx4, x8, 0,
                         flare::pair<int, int>(0, 2) /* of [3] */
                    ,
                         1, flare::pair<int, int>(1, 3) /* of [5] */
                    ,
                         1, flare::pair<int, int>(0, 2) /* of [3] */
                    ,
                         2, flare::pair<int, int>(2, 4) /* of [5] */
            );

            REQUIRE(!sx4.span_is_contiguous());

            for (int i0 = 0; i0 < (int) sx4.extent(0); ++i0)
                for (int i1 = 0; i1 < (int) sx4.extent(1); ++i1)
                    for (int i2 = 0; i2 < (int) sx4.extent(2); ++i2)
                        for (int i3 = 0; i3 < (int) sx4.extent(3); ++i3) {
                            REQUIRE_EQ(&sx4(i0, i1, i2, i3),
                                       &x8(0, 0 + i0, 1, 1 + i1, 1, 0 + i2, 2, 2 + i3));
                        }
        }
    }

    template<class Space>
    void test_left_1() {
        test_left_1<Space>(true);
        test_left_1<Space>(false);
    }

    template<class Space>
    void test_left_2() {
        using view_type = flare::View<int ****, flare::LayoutLeft, Space>;

        if (flare::SpaceAccessibility<flare::HostSpace,
                typename Space::memory_space>::accessible) {
            view_type x4("x4", 2, 3, 4, 5);

            REQUIRE(x4.span_is_contiguous());

            flare::View<int, flare::LayoutLeft, Space> x0 =
                    flare::subview(x4, 0, 0, 0, 0);

            REQUIRE(x0.span_is_contiguous());
            REQUIRE_EQ(&x0(), &x4(0, 0, 0, 0));

            flare::View<int *, flare::LayoutLeft, Space> x1 =
                    flare::subview(x4, flare::pair<int, int>(0, 2), 1, 2, 3);

            REQUIRE(x1.span_is_contiguous());
            REQUIRE_EQ(&x1(0), &x4(0, 1, 2, 3));
            REQUIRE_EQ(&x1(1), &x4(1, 1, 2, 3));

            flare::View<int **, flare::LayoutLeft, Space> x2 = flare::subview(
                    x4, flare::pair<int, int>(0, 2), 1, flare::pair<int, int>(1, 3), 2);

            REQUIRE(!x2.span_is_contiguous());
            REQUIRE_EQ(&x2(0, 0), &x4(0, 1, 1, 2));
            REQUIRE_EQ(&x2(1, 0), &x4(1, 1, 1, 2));
            REQUIRE_EQ(&x2(0, 1), &x4(0, 1, 2, 2));
            REQUIRE_EQ(&x2(1, 1), &x4(1, 1, 2, 2));

            // flare::View< int**, flare::LayoutLeft, Space > error_2 =
            flare::View<int **, flare::LayoutStride, Space> sx2 = flare::subview(
                    x4, 1, flare::pair<int, int>(0, 2), 2, flare::pair<int, int>(1, 4));

            REQUIRE(!sx2.span_is_contiguous());
            REQUIRE_EQ(&sx2(0, 0), &x4(1, 0, 2, 1));
            REQUIRE_EQ(&sx2(1, 0), &x4(1, 1, 2, 1));
            REQUIRE_EQ(&sx2(0, 1), &x4(1, 0, 2, 2));
            REQUIRE_EQ(&sx2(1, 1), &x4(1, 1, 2, 2));
            REQUIRE_EQ(&sx2(0, 2), &x4(1, 0, 2, 3));
            REQUIRE_EQ(&sx2(1, 2), &x4(1, 1, 2, 3));

            flare::View<int ****, flare::LayoutStride, Space> sx4 =
                    flare::subview(x4, flare::pair<int, int>(1, 2) /* of [2] */
                            ,
                                   flare::pair<int, int>(1, 3) /* of [3] */
                            ,
                                   flare::pair<int, int>(0, 4) /* of [4] */
                            ,
                                   flare::pair<int, int>(2, 4) /* of [5] */
                    );

            REQUIRE(!sx4.span_is_contiguous());

            for (int i0 = 0; i0 < (int) sx4.extent(0); ++i0)
                for (int i1 = 0; i1 < (int) sx4.extent(1); ++i1)
                    for (int i2 = 0; i2 < (int) sx4.extent(2); ++i2)
                        for (int i3 = 0; i3 < (int) sx4.extent(3); ++i3) {
                            REQUIRE_EQ(&sx4(i0, i1, i2, i3),
                                       &x4(1 + i0, 1 + i1, 0 + i2, 2 + i3));
                        }
        }
    }

    template<class Space>
    void test_left_3() {
        using view_type = flare::View<int **, flare::LayoutLeft, Space>;

        if (flare::SpaceAccessibility<flare::HostSpace,
                typename Space::memory_space>::accessible) {
            view_type xm("x4", 10, 5);

            REQUIRE(xm.span_is_contiguous());

            flare::View<int, flare::LayoutLeft, Space> x0 = flare::subview(xm, 5, 3);

            REQUIRE(x0.span_is_contiguous());
            REQUIRE_EQ(&x0(), &xm(5, 3));

            flare::View<int *, flare::LayoutLeft, Space> x1 =
                    flare::subview(xm, flare::ALL, 3);

            REQUIRE(x1.span_is_contiguous());
            for (int i = 0; i < int(xm.extent(0)); ++i) {
                REQUIRE_EQ(&x1(i), &xm(i, 3));
            }

            flare::View<int **, flare::LayoutLeft, Space> x2 =
                    flare::subview(xm, flare::pair<int, int>(1, 9), flare::ALL);

            REQUIRE(!x2.span_is_contiguous());
            for (int j = 0; j < int(x2.extent(1)); ++j)
                for (int i = 0; i < int(x2.extent(0)); ++i) {
                    REQUIRE_EQ(&x2(i, j), &xm(1 + i, j));
                }

            flare::View<int **, flare::LayoutLeft, Space> x2c =
                    flare::subview(xm, flare::ALL, std::pair<int, int>(2, 4));

            REQUIRE(x2c.span_is_contiguous());
            for (int j = 0; j < int(x2c.extent(1)); ++j)
                for (int i = 0; i < int(x2c.extent(0)); ++i) {
                    REQUIRE_EQ(&x2c(i, j), &xm(i, 2 + j));
                }

            flare::View<int **, flare::LayoutLeft, Space> x2_n1 =
                    flare::subview(xm, std::pair<int, int>(1, 1), flare::ALL);

            REQUIRE_EQ(x2_n1.extent(0), 0u);
            REQUIRE_EQ(x2_n1.extent(1), xm.extent(1));

            flare::View<int **, flare::LayoutLeft, Space> x2_n2 =
                    flare::subview(xm, flare::ALL, std::pair<int, int>(1, 1));

            REQUIRE_EQ(x2_n2.extent(0), xm.extent(0));
            REQUIRE_EQ(x2_n2.extent(1), 0u);
        }
    }

//----------------------------------------------------------------------------

    template<class Space>
    void test_right_0(bool use_constr) {
        using view_static_8_type =
                flare::View<int[2][3][4][5][2][3][4][5], flare::LayoutRight, Space>;

        if (flare::SpaceAccessibility<flare::HostSpace,
                typename Space::memory_space>::accessible) {
            view_static_8_type x_static_8("x_static_right_8");

            flare::View<int, flare::LayoutRight, Space> x0;
            make_subview(use_constr, x0, x_static_8, 0, 0, 0, 0, 0, 0, 0, 0);

            REQUIRE_EQ(&x0(), &x_static_8(0, 0, 0, 0, 0, 0, 0, 0));

            flare::View<int *, flare::LayoutRight, Space> x1;
            make_subview(use_constr, x1, x_static_8, 0, 1, 2, 3, 0, 1, 2,
                         flare::pair<int, int>(1, 3));

            REQUIRE_EQ(x1.extent(0), 2u);
            REQUIRE_EQ(&x1(0), &x_static_8(0, 1, 2, 3, 0, 1, 2, 1));
            REQUIRE_EQ(&x1(1), &x_static_8(0, 1, 2, 3, 0, 1, 2, 2));

            flare::View<int **, flare::LayoutRight, Space> x2;
            make_subview(use_constr, x2, x_static_8, 0, 1, 2,
                         flare::pair<int, int>(1, 3), 0, 1, 2,
                         flare::pair<int, int>(1, 3));

            REQUIRE_EQ(x2.extent(0), 2u);
            REQUIRE_EQ(x2.extent(1), 2u);
            REQUIRE_EQ(&x2(0, 0), &x_static_8(0, 1, 2, 1, 0, 1, 2, 1));
            REQUIRE_EQ(&x2(1, 0), &x_static_8(0, 1, 2, 2, 0, 1, 2, 1));
            REQUIRE_EQ(&x2(0, 1), &x_static_8(0, 1, 2, 1, 0, 1, 2, 2));
            REQUIRE_EQ(&x2(1, 1), &x_static_8(0, 1, 2, 2, 0, 1, 2, 2));

            // flare::View< int**, flare::LayoutRight, Space > error_2 =
            flare::View<int **, flare::LayoutStride, Space> sx2;
            make_subview(use_constr, sx2, x_static_8, 1, flare::pair<int, int>(0, 2),
                         2, 3, flare::pair<int, int>(0, 2), 1, 2, 3);

            REQUIRE_EQ(sx2.extent(0), 2u);
            REQUIRE_EQ(sx2.extent(1), 2u);
            REQUIRE_EQ(&sx2(0, 0), &x_static_8(1, 0, 2, 3, 0, 1, 2, 3));
            REQUIRE_EQ(&sx2(1, 0), &x_static_8(1, 1, 2, 3, 0, 1, 2, 3));
            REQUIRE_EQ(&sx2(0, 1), &x_static_8(1, 0, 2, 3, 1, 1, 2, 3));
            REQUIRE_EQ(&sx2(1, 1), &x_static_8(1, 1, 2, 3, 1, 1, 2, 3));

            flare::View<int ****, flare::LayoutStride, Space> sx4;
            make_subview(use_constr, sx4, x_static_8, 0,
                         flare::pair<int, int>(0, 2) /* of [3] */
                    ,
                         1, flare::pair<int, int>(1, 3) /* of [5] */
                    ,
                         1, flare::pair<int, int>(0, 2) /* of [3] */
                    ,
                         2, flare::pair<int, int>(2, 4) /* of [5] */
            );

            REQUIRE_EQ(sx4.extent(0), 2u);
            REQUIRE_EQ(sx4.extent(1), 2u);
            REQUIRE_EQ(sx4.extent(2), 2u);
            REQUIRE_EQ(sx4.extent(3), 2u);
            for (int i0 = 0; i0 < (int) sx4.extent(0); ++i0)
                for (int i1 = 0; i1 < (int) sx4.extent(1); ++i1)
                    for (int i2 = 0; i2 < (int) sx4.extent(2); ++i2)
                        for (int i3 = 0; i3 < (int) sx4.extent(3); ++i3) {
                            REQUIRE_EQ(&sx4(i0, i1, i2, i3),
                                       &x_static_8(0, 0 + i0, 1, 1 + i1, 1, 0 + i2, 2, 2 + i3));
                        }
        }
    }

    template<class Space>
    void test_right_0() {
        test_right_0<Space>(true);
        test_right_0<Space>(false);
    }

    template<class Space>
    void test_right_1(bool use_constr) {
        using view_type =
                flare::View<int ****[2][3][4][5], flare::LayoutRight, Space>;

        if (flare::SpaceAccessibility<flare::HostSpace,
                typename Space::memory_space>::accessible) {
            view_type x8("x_right_8", 2, 3, 4, 5);

            flare::View<int, flare::LayoutRight, Space> x0;
            make_subview(use_constr, x0, x8, 0, 0, 0, 0, 0, 0, 0, 0);

            REQUIRE_EQ(&x0(), &x8(0, 0, 0, 0, 0, 0, 0, 0));

            flare::View<int *, flare::LayoutRight, Space> x1;
            make_subview(use_constr, x1, x8, 0, 1, 2, 3, 0, 1, 2,
                         flare::pair<int, int>(1, 3));

            REQUIRE_EQ(&x1(0), &x8(0, 1, 2, 3, 0, 1, 2, 1));
            REQUIRE_EQ(&x1(1), &x8(0, 1, 2, 3, 0, 1, 2, 2));

            flare::View<int *, flare::LayoutRight, Space> x1_deg1;
            make_subview(use_constr, x1_deg1, x8, 0, 1, 2, 3, 0, 1, 2,
                         flare::pair<int, int>(3, 3));
            REQUIRE_EQ(0u, x1_deg1.span());

            flare::View<int **, flare::LayoutRight, Space> x2;
            make_subview(use_constr, x2, x8, 0, 1, 2, flare::pair<int, int>(1, 3), 0,
                         1, 2, flare::pair<int, int>(1, 3));

            REQUIRE_EQ(&x2(0, 0), &x8(0, 1, 2, 1, 0, 1, 2, 1));
            REQUIRE_EQ(&x2(1, 0), &x8(0, 1, 2, 2, 0, 1, 2, 1));
            REQUIRE_EQ(&x2(0, 1), &x8(0, 1, 2, 1, 0, 1, 2, 2));
            REQUIRE_EQ(&x2(1, 1), &x8(0, 1, 2, 2, 0, 1, 2, 2));

            flare::View<int **, flare::LayoutRight, Space> x2_deg2;
            make_subview(use_constr, x2_deg2, x8, 0, 1, 2, flare::pair<int, int>(1, 3),
                         0, 1, 2, flare::pair<int, int>(3, 3));
            REQUIRE_EQ(0u, x2_deg2.span());

            // flare::View< int**, flare::LayoutRight, Space > error_2 =
            flare::View<int **, flare::LayoutStride, Space> sx2;
            make_subview(use_constr, sx2, x8, 1, flare::pair<int, int>(0, 2), 2, 3,
                         flare::pair<int, int>(0, 2), 1, 2, 3);

            REQUIRE_EQ(&sx2(0, 0), &x8(1, 0, 2, 3, 0, 1, 2, 3));
            REQUIRE_EQ(&sx2(1, 0), &x8(1, 1, 2, 3, 0, 1, 2, 3));
            REQUIRE_EQ(&sx2(0, 1), &x8(1, 0, 2, 3, 1, 1, 2, 3));
            REQUIRE_EQ(&sx2(1, 1), &x8(1, 1, 2, 3, 1, 1, 2, 3));

            flare::View<int **, flare::LayoutStride, Space> sx2_deg;
            make_subview(use_constr, sx2_deg, x8, 1, flare::pair<int, int>(0, 2), 2, 3,
                         1, 1, 2, flare::pair<int, int>(3, 3));
            REQUIRE_EQ(0u, sx2_deg.span());

            flare::View<int ****, flare::LayoutStride, Space> sx4;
            make_subview(use_constr, sx4, x8, 0,
                         flare::pair<int, int>(0, 2) /* of [3] */
                    ,
                         1, flare::pair<int, int>(1, 3) /* of [5] */
                    ,
                         1, flare::pair<int, int>(0, 2) /* of [3] */
                    ,
                         2, flare::pair<int, int>(2, 4) /* of [5] */
            );

            for (int i0 = 0; i0 < (int) sx4.extent(0); ++i0)
                for (int i1 = 0; i1 < (int) sx4.extent(1); ++i1)
                    for (int i2 = 0; i2 < (int) sx4.extent(2); ++i2)
                        for (int i3 = 0; i3 < (int) sx4.extent(3); ++i3) {
                            REQUIRE_EQ(&sx4(i0, i1, i2, i3),
                                       &x8(0, 0 + i0, 1, 1 + i1, 1, 0 + i2, 2, 2 + i3));
                        }
        }
    }

    template<class Space>
    void test_right_1() {
        test_right_1<Space>(true);
        test_right_1<Space>(false);
    }

    template<class Space>
    void test_right_3() {
        using view_type = flare::View<int **, flare::LayoutRight, Space>;

        if (flare::SpaceAccessibility<flare::HostSpace,
                typename Space::memory_space>::accessible) {
            view_type xm("x4", 10, 5);

            REQUIRE(xm.span_is_contiguous());

            flare::View<int, flare::LayoutRight, Space> x0 =
                    flare::subview(xm, 5, 3);

            REQUIRE(x0.span_is_contiguous());
            REQUIRE_EQ(&x0(), &xm(5, 3));

            flare::View<int *, flare::LayoutRight, Space> x1 =
                    flare::subview(xm, 3, flare::ALL);

            REQUIRE(x1.span_is_contiguous());
            for (int i = 0; i < int(xm.extent(1)); ++i) {
                REQUIRE_EQ(&x1(i), &xm(3, i));
            }

            flare::View<int **, flare::LayoutRight, Space> x2c =
                    flare::subview(xm, flare::pair<int, int>(1, 9), flare::ALL);

            REQUIRE(x2c.span_is_contiguous());
            for (int j = 0; j < int(x2c.extent(1)); ++j)
                for (int i = 0; i < int(x2c.extent(0)); ++i) {
                    REQUIRE_EQ(&x2c(i, j), &xm(1 + i, j));
                }

            flare::View<int **, flare::LayoutRight, Space> x2 =
                    flare::subview(xm, flare::ALL, std::pair<int, int>(2, 4));

            REQUIRE(!x2.span_is_contiguous());
            for (int j = 0; j < int(x2.extent(1)); ++j)
                for (int i = 0; i < int(x2.extent(0)); ++i) {
                    REQUIRE_EQ(&x2(i, j), &xm(i, 2 + j));
                }

            flare::View<int **, flare::LayoutRight, Space> x2_n1 =
                    flare::subview(xm, std::pair<int, int>(1, 1), flare::ALL);

            REQUIRE_EQ(x2_n1.extent(0), 0u);
            REQUIRE_EQ(x2_n1.extent(1), xm.extent(1));

            flare::View<int **, flare::LayoutRight, Space> x2_n2 =
                    flare::subview(xm, flare::ALL, std::pair<int, int>(1, 1));

            REQUIRE_EQ(x2_n2.extent(0), xm.extent(0));
            REQUIRE_EQ(x2_n2.extent(1), 0u);
        }
    }

    namespace detail {

        constexpr int N0 = 113;
        constexpr int N1 = 11;
        constexpr int N2 = 17;
        constexpr int N3 = 5;
        constexpr int N4 = 7;

        template<class Layout, class Space>
        struct FillView_1D {
            using view_t = flare::View<int *, Layout, Space>;
            view_t a;
            using policy_t = flare::RangePolicy<typename Space::execution_space>;

            FillView_1D(view_t a_) : a(a_) {}

            void run() {
                flare::parallel_for("FillView_1D", policy_t(0, a.extent(0)), *this);
            }

            FLARE_INLINE_FUNCTION
            void operator()(int i) const { a(i) = i; }
        };

        template<class Layout, class Space>
        struct FillView_3D {
            using exec_t = typename Space::execution_space;
            using view_t = flare::View<int ***, Layout, Space>;
            using rank_t = flare::Rank<
                    view_t::rank,
                    std::is_same<Layout, flare::LayoutLeft>::value ? flare::Iterate::Left
                                                                   : flare::Iterate::Right,
                    std::is_same<Layout, flare::LayoutLeft>::value ? flare::Iterate::Left
                                                                   : flare::Iterate::Right>;
            using policy_t = flare::MDRangePolicy<exec_t, rank_t>;

            view_t a;

            FillView_3D(view_t a_) : a(a_) {}

            void run() {
                flare::parallel_for(
                        "FillView_3D",
                        policy_t({0, 0, 0}, {a.extent(0), a.extent(1), a.extent(2)}), *this);
            }

            FLARE_INLINE_FUNCTION
            void operator()(int i0, int i1, int i2) const {
                a(i0, i1, i2) = 1000000 * i0 + 1000 * i1 + i2;
            }
        };

        template<class Layout, class Space>
        struct FillView_4D {
            using exec_t = typename Space::execution_space;
            using view_t = flare::View<int ****, Layout, Space>;
            using rank_t = flare::Rank<
                    view_t::rank,
                    std::is_same<Layout, flare::LayoutLeft>::value ? flare::Iterate::Left
                                                                   : flare::Iterate::Right,
                    std::is_same<Layout, flare::LayoutLeft>::value ? flare::Iterate::Left
                                                                   : flare::Iterate::Right>;
            using policy_t = flare::MDRangePolicy<exec_t, rank_t>;

            view_t a;

            FillView_4D(view_t a_) : a(a_) {}

            void run() {
                flare::parallel_for("FillView_4D",
                                    policy_t({0, 0, 0, 0}, {a.extent(0), a.extent(1),
                                                            a.extent(2), a.extent(3)}),
                                    *this);
            }

            FLARE_INLINE_FUNCTION
            void operator()(int i0, int i1, int i2, int i3) const {
                a(i0, i1, i2, i3) = 1000000 * i0 + 10000 * i1 + 100 * i2 + i3;
            }
        };

        template<class Layout, class Space>
        struct FillView_5D {
            using exec_t = typename Space::execution_space;
            using view_t = flare::View<int *****, Layout, Space>;
            using rank_t = flare::Rank<
                    view_t::rank,
                    std::is_same<Layout, flare::LayoutLeft>::value ? flare::Iterate::Left
                                                                   : flare::Iterate::Right,
                    std::is_same<Layout, flare::LayoutLeft>::value ? flare::Iterate::Left
                                                                   : flare::Iterate::Right>;
            using policy_t = flare::MDRangePolicy<exec_t, rank_t>;

            view_t a;

            FillView_5D(view_t a_) : a(a_) {}

            void run() {
                flare::parallel_for(
                        "FillView_5D",
                        policy_t({0, 0, 0, 0, 0}, {a.extent(0), a.extent(1), a.extent(2),
                                                   a.extent(3), a.extent(4)}),
                        *this);
            }

            FLARE_INLINE_FUNCTION
            void operator()(int i0, int i1, int i2, int i3, int i4) const {
                a(i0, i1, i2, i3, i4) = 1000000 * i0 + 10000 * i1 + 100 * i2 + 10 * i3 + i4;
            }
        };

        template<class View, class SubView>
        struct CheckSubviewCorrectness_1D_1D {
            using policy_t = flare::RangePolicy<typename View::execution_space>;
            View a;
            SubView b;
            int offset;

            CheckSubviewCorrectness_1D_1D(View a_, SubView b_, int o)
                    : a(a_), b(b_), offset(o) {}

            void run() {
                int errors = 0;
                flare::parallel_reduce("CheckSubView_1D_1D", policy_t(0, b.size()), *this,
                                       errors);
                REQUIRE_EQ(errors, 0);
            }

            FLARE_INLINE_FUNCTION
            void operator()(const int &i, int &e) const {
                if (a(i + offset) != b(i)) {
                    e++;
                }
            }
        };

        template<class View, class SubView>
        struct CheckSubviewCorrectness_1D_2D {
            using policy_t = flare::RangePolicy<typename View::execution_space>;
            View a;
            SubView b;
            int i0;
            int offset;

            CheckSubviewCorrectness_1D_2D(View a_, SubView b_, int i0_, int o)
                    : a(a_), b(b_), i0(i0_), offset(o) {}

            void run() {
                int errors = 0;
                flare::parallel_reduce("CheckSubView_1D_2D", policy_t(0, b.size()), *this,
                                       errors);
                REQUIRE_EQ(errors, 0);
            }

            FLARE_INLINE_FUNCTION
            void operator()(const int &i1, int &e) const {
                if (a(i0, i1 + offset) != b(i1)) {
                    e++;
                }
            }
        };

        template<class View, class SubView>
        struct CheckSubviewCorrectness_2D_3D {
            using policy_t = flare::RangePolicy<typename View::execution_space>;
            using layout = typename View::array_layout;
            View a;
            SubView b;
            int i0;
            int offset_1;
            int offset_2;

            CheckSubviewCorrectness_2D_3D(View a_, SubView b_, int i0_, int o1, int o2)
                    : a(a_), b(b_), i0(i0_), offset_1(o1), offset_2(o2) {}

            void run() {
                int errors = 0;
                flare::parallel_reduce("CheckSubView_2D_3D", policy_t(0, b.size()), *this,
                                       errors);
                REQUIRE_EQ(errors, 0);
            }

            FLARE_INLINE_FUNCTION
            void operator()(const int &ii, int &e) const {
                const int i1 = std::is_same<layout, flare::LayoutLeft>::value
                               ? ii % b.extent(0)
                               : ii / b.extent(1);

                const int i2 = std::is_same<layout, flare::LayoutLeft>::value
                               ? ii / b.extent(0)
                               : ii % b.extent(1);

                if (a(i0, i1 + offset_1, i2 + offset_2) != b(i1, i2)) {
                    e++;
                }
            }
        };

        template<class View, class SubView>
        struct CheckSubviewCorrectness_3D_3D {
            using policy_t = flare::RangePolicy<typename View::execution_space>;
            using layout = typename View::array_layout;
            View a;
            SubView b;
            int offset_0;
            int offset_2;

            CheckSubviewCorrectness_3D_3D(View a_, SubView b_, int o0, int o2)
                    : a(a_), b(b_), offset_0(o0), offset_2(o2) {}

            void run() {
                int errors = 0;
                flare::parallel_reduce("CheckSubView_3D_3D", policy_t(0, b.size()), *this,
                                       errors);
                REQUIRE_EQ(errors, 0);
            }

            FLARE_INLINE_FUNCTION
            void operator()(const int &ii, int &e) const {
                const int i0 = std::is_same<layout, flare::LayoutLeft>::value
                               ? ii % b.extent(0)
                               : ii / (b.extent(1) * b.extent(2));

                const int i1 = std::is_same<layout, flare::LayoutLeft>::value
                               ? (ii / b.extent(0)) % b.extent(1)
                               : (ii / b.extent(2)) % b.extent(1);

                const int i2 = std::is_same<layout, flare::LayoutLeft>::value
                               ? ii / (b.extent(0) * b.extent(1))
                               : ii % b.extent(2);

                if (a(i0 + offset_0, i1, i2 + offset_2) != b(i0, i1, i2)) {
                    e++;
                }
            }
        };

        template<class View, class SubView>
        struct CheckSubviewCorrectness_3D_4D {
            using policy_t = flare::RangePolicy<typename View::execution_space>;
            using layout = typename View::array_layout;
            View a;
            SubView b;
            int index;
            int offset_0, offset_2;

            CheckSubviewCorrectness_3D_4D(View a_, SubView b_, int index_, int o0, int o2)
                    : a(a_), b(b_), index(index_), offset_0(o0), offset_2(o2) {}

            void run() {
                int errors = 0;
                flare::parallel_reduce("CheckSubView_3D_4D", policy_t(0, b.size()), *this,
                                       errors);
                REQUIRE_EQ(errors, 0);
            }

            FLARE_INLINE_FUNCTION
            void operator()(const int &ii, int &e) const {
                const int i = std::is_same<layout, flare::LayoutLeft>::value
                              ? ii % b.extent(0)
                              : ii / (b.extent(1) * b.extent(2));

                const int j = std::is_same<layout, flare::LayoutLeft>::value
                              ? (ii / b.extent(0)) % b.extent(1)
                              : (ii / b.extent(2)) % b.extent(1);

                const int k = std::is_same<layout, flare::LayoutLeft>::value
                              ? ii / (b.extent(0) * b.extent(1))
                              : ii % b.extent(2);

                int i0, i1, i2, i3;

                if (std::is_same<layout, flare::LayoutLeft>::value) {
                    i0 = i + offset_0;
                    i1 = j;
                    i2 = k + offset_2;
                    i3 = index;
                } else {
                    i0 = index;
                    i1 = i + offset_0;
                    i2 = j;
                    i3 = k + offset_2;
                }

                if (a(i0, i1, i2, i3) != b(i, j, k)) e++;
            }
        };

        template<class View, class SubView>
        struct CheckSubviewCorrectness_3D_5D {
            using policy_t = flare::RangePolicy<typename View::execution_space>;
            using layout = typename View::array_layout;
            View a;
            SubView b;
            int i0, i1;
            int offset_2, offset_3, offset_4;

            CheckSubviewCorrectness_3D_5D(View a_, SubView b_, int i0_, int i1_, int o2,
                                          int o3, int o4)
                    : a(a_),
                      b(b_),
                      i0(i0_),
                      i1(i1_),
                      offset_2(o2),
                      offset_3(o3),
                      offset_4(o4) {}

            void run() {
                int errors = 0;
                flare::parallel_reduce("CheckSubView_3D_5D", policy_t(0, b.size()), *this,
                                       errors);
                REQUIRE_EQ(errors, 0);
            }

            FLARE_INLINE_FUNCTION
            void operator()(const int &ii, int &e) const {
                const int i2 = std::is_same<layout, flare::LayoutLeft>::value
                               ? ii % b.extent(0)
                               : ii / (b.extent(1) * b.extent(2));

                const int i3 = std::is_same<layout, flare::LayoutLeft>::value
                               ? (ii / b.extent(0)) % b.extent(1)
                               : (ii / b.extent(2)) % b.extent(1);

                const int i4 = std::is_same<layout, flare::LayoutLeft>::value
                               ? ii / (b.extent(0) * b.extent(1))
                               : ii % b.extent(2);

                if (a(i0, i1, i2 + offset_2, i3 + offset_3, i4 + offset_4) !=
                    b(i2, i3, i4)) {
                    e++;
                }
            }
        };

        template<class SubView, class View>
        void test_Check1D(SubView a, View b, flare::pair<int, int> range) {
            CheckSubviewCorrectness_1D_1D<View, SubView> check(b, a, range.first);
            check.run();
        }

        template<class SubView, class View>
        void test_Check1D2D(SubView a, View b, int i0, std::pair<int, int> range) {
            CheckSubviewCorrectness_1D_2D<View, SubView> check(b, a, i0, range.first);
            check.run();
        }

        template<class SubView, class View>
        void test_Check2D3D(SubView a, View b, int i0, std::pair<int, int> range1,
                            std::pair<int, int> range2) {
            CheckSubviewCorrectness_2D_3D<View, SubView> check(b, a, i0, range1.first,
                                                               range2.first);
            check.run();
        }

        template<class SubView, class View>
        void test_Check3D5D(SubView a, View b, int i0, int i1,
                            flare::pair<int, int> range2,
                            flare::pair<int, int> range3,
                            flare::pair<int, int> range4) {
            CheckSubviewCorrectness_3D_5D<View, SubView> check(
                    b, a, i0, i1, range2.first, range3.first, range4.first);
            check.run();
        }

        template<class Space, class LayoutSub, class Layout, class LayoutOrg,
                class MemTraits>
        void test_1d_assign_impl() {
            {  // Breaks.
                flare::View<int *, LayoutOrg, Space> a_org("A", N0);
                flare::View<int *, LayoutOrg, Space, MemTraits> a(a_org);
                flare::fence();

                detail::FillView_1D<LayoutOrg, Space> fill(a_org);
                fill.run();

                flare::View<int[N0], Layout, Space, MemTraits> a1(a);
                flare::fence();
                test_Check1D(a1, a, std::pair<int, int>(0, N0));

                flare::View<int[N0], LayoutSub, Space, MemTraits> a2(a1);
                flare::fence();
                test_Check1D(a2, a, std::pair<int, int>(0, N0));
                a1 = a;
                test_Check1D(a1, a, std::pair<int, int>(0, N0));

                // Runtime Fail expected.
                // flare::View< int[N1] > afail1( a );

                // Compile Time Fail expected.
                // flare::View< int[N1] > afail2( a1 );
            }

            {  // Works.
                flare::View<int[N0], LayoutOrg, Space, MemTraits> a("A");
                flare::View<int *, Layout, Space, MemTraits> a1(a);
                flare::fence();
                test_Check1D(a1, a, std::pair<int, int>(0, N0));
                a1 = a;
                flare::fence();
                test_Check1D(a1, a, std::pair<int, int>(0, N0));
            }
        }

        template<class Space, class Type, class TypeSub, class LayoutSub, class Layout,
                class LayoutOrg, class MemTraits>
        void test_2d_subview_3d_impl_type() {
            flare::View<int ***, LayoutOrg, Space> a_org("A", N0, N1, N2);
            flare::View<Type, Layout, Space, MemTraits> a(a_org);

            detail::FillView_3D<LayoutOrg, Space> fill(a_org);
            fill.run();

            flare::View<TypeSub, LayoutSub, Space, MemTraits> a1;
            a1 = flare::subview(a, 3, flare::ALL, flare::ALL);
            flare::fence();
            test_Check2D3D(a1, a, 3, std::pair<int, int>(0, N1),
                           std::pair<int, int>(0, N2));

            flare::View<TypeSub, LayoutSub, Space, MemTraits> a2(a, 3, flare::ALL,
                                                                 flare::ALL);
            flare::fence();
            test_Check2D3D(a2, a, 3, std::pair<int, int>(0, N1),
                           std::pair<int, int>(0, N2));
        }

        template<class Space, class LayoutSub, class Layout, class LayoutOrg,
                class MemTraits>
        void test_2d_subview_3d_impl_layout() {
            test_2d_subview_3d_impl_type<Space, int[N0][N1][N2], int[N1][N2], LayoutSub,
                    Layout, LayoutOrg, MemTraits>();
            test_2d_subview_3d_impl_type<Space, int[N0][N1][N2], int *[N2], LayoutSub,
                    Layout, LayoutOrg, MemTraits>();
            test_2d_subview_3d_impl_type<Space, int[N0][N1][N2], int **, LayoutSub, Layout,
                    LayoutOrg, MemTraits>();

            test_2d_subview_3d_impl_type<Space, int *[N1][N2], int[N1][N2], LayoutSub,
                    Layout, LayoutOrg, MemTraits>();
            test_2d_subview_3d_impl_type<Space, int *[N1][N2], int *[N2], LayoutSub,
                    Layout, LayoutOrg, MemTraits>();
            test_2d_subview_3d_impl_type<Space, int *[N1][N2], int **, LayoutSub, Layout,
                    LayoutOrg, MemTraits>();

            test_2d_subview_3d_impl_type<Space, int **[N2], int[N1][N2], LayoutSub,
                    Layout, LayoutOrg, MemTraits>();
            test_2d_subview_3d_impl_type<Space, int **[N2], int *[N2], LayoutSub,
                    Layout, LayoutOrg, MemTraits>();
            test_2d_subview_3d_impl_type<Space, int **[N2], int **, LayoutSub, Layout,
                    LayoutOrg, MemTraits>();

            test_2d_subview_3d_impl_type<Space, int ***, int[N1][N2], LayoutSub, Layout,
                    LayoutOrg, MemTraits>();
            test_2d_subview_3d_impl_type<Space, int ***, int *[N2], LayoutSub, Layout,
                    LayoutOrg, MemTraits>();
            test_2d_subview_3d_impl_type<Space, int ***, int **, LayoutSub, Layout,
                    LayoutOrg, MemTraits>();

            test_2d_subview_3d_impl_type<Space, const int[N0][N1][N2], const int[N1][N2],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_2d_subview_3d_impl_type<Space, const int[N0][N1][N2], const int *[N2],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_2d_subview_3d_impl_type<Space, const int[N0][N1][N2], const int **,
                    LayoutSub, Layout, LayoutOrg, MemTraits>();

            test_2d_subview_3d_impl_type<Space, const int *[N1][N2], const int[N1][N2],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_2d_subview_3d_impl_type<Space, const int *[N1][N2], const int *[N2],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_2d_subview_3d_impl_type<Space, const int *[N1][N2], const int **,
                    LayoutSub, Layout, LayoutOrg, MemTraits>();

            test_2d_subview_3d_impl_type<Space, const int **[N2], const int[N1][N2],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_2d_subview_3d_impl_type<Space, const int **[N2], const int *[N2],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_2d_subview_3d_impl_type<Space, const int **[N2], const int **, LayoutSub,
                    Layout, LayoutOrg, MemTraits>();

            test_2d_subview_3d_impl_type<Space, const int ***, const int[N1][N2],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_2d_subview_3d_impl_type<Space, const int ***, const int *[N2], LayoutSub,
                    Layout, LayoutOrg, MemTraits>();
            test_2d_subview_3d_impl_type<Space, const int ***, const int **, LayoutSub,
                    Layout, LayoutOrg, MemTraits>();
        }

        template<class Space, class Type, class TypeSub, class LayoutSub, class Layout,
                class LayoutOrg, class MemTraits>
        void test_3d_subview_5d_impl_type() {
            flare::View<int *****, LayoutOrg, Space> a_org("A", N0, N1, N2, N3, N4);
            flare::View<Type, Layout, Space, MemTraits> a(a_org);

            detail::FillView_5D<LayoutOrg, Space> fill(a_org);
            fill.run();

            flare::View<TypeSub, LayoutSub, Space, MemTraits> a1;
            a1 = flare::subview(a, 3, 5, flare::ALL, flare::ALL, flare::ALL);
            flare::fence();
            test_Check3D5D(a1, a, 3, 5, std::pair<int, int>(0, N2),
                           std::pair<int, int>(0, N3), std::pair<int, int>(0, N4));

            flare::View<TypeSub, LayoutSub, Space, MemTraits> a2(
                    a, 3, 5, flare::ALL, flare::ALL, flare::ALL);
            flare::fence();
            test_Check3D5D(a2, a, 3, 5, std::pair<int, int>(0, N2),
                           std::pair<int, int>(0, N3), std::pair<int, int>(0, N4));
        }

        template<class Space, class LayoutSub, class Layout, class LayoutOrg,
                class MemTraits>
        void test_3d_subview_5d_impl_layout() {
            test_3d_subview_5d_impl_type<Space, int[N0][N1][N2][N3][N4], int[N2][N3][N4],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_3d_subview_5d_impl_type<Space, int[N0][N1][N2][N3][N4], int *[N3][N4],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_3d_subview_5d_impl_type<Space, int[N0][N1][N2][N3][N4], int **[N4],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_3d_subview_5d_impl_type<Space, int[N0][N1][N2][N3][N4], int ***,
                    LayoutSub, Layout, LayoutOrg, MemTraits>();

            test_3d_subview_5d_impl_type<Space, int *[N1][N2][N3][N4], int[N2][N3][N4],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_3d_subview_5d_impl_type<Space, int *[N1][N2][N3][N4], int *[N3][N4],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_3d_subview_5d_impl_type<Space, int *[N1][N2][N3][N4], int **[N4],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_3d_subview_5d_impl_type<Space, int *[N1][N2][N3][N4], int ***, LayoutSub,
                    Layout, LayoutOrg, MemTraits>();

            test_3d_subview_5d_impl_type<Space, int **[N2][N3][N4], int[N2][N3][N4],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_3d_subview_5d_impl_type<Space, int **[N2][N3][N4], int *[N3][N4],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_3d_subview_5d_impl_type<Space, int **[N2][N3][N4], int **[N4],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_3d_subview_5d_impl_type<Space, int **[N2][N3][N4], int ***, LayoutSub,
                    Layout, LayoutOrg, MemTraits>();

            test_3d_subview_5d_impl_type<Space, int ***[N3][N4], int[N2][N3][N4],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_3d_subview_5d_impl_type<Space, int ***[N3][N4], int *[N3][N4],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_3d_subview_5d_impl_type<Space, int ***[N3][N4], int **[N4], LayoutSub,
                    Layout, LayoutOrg, MemTraits>();
            test_3d_subview_5d_impl_type<Space, int ***[N3][N4], int ***, LayoutSub,
                    Layout, LayoutOrg, MemTraits>();

            test_3d_subview_5d_impl_type<Space, int ****[N4], int[N2][N3][N4], LayoutSub,
                    Layout, LayoutOrg, MemTraits>();
            test_3d_subview_5d_impl_type<Space, int ****[N4], int *[N3][N4], LayoutSub,
                    Layout, LayoutOrg, MemTraits>();
            test_3d_subview_5d_impl_type<Space, int ****[N4], int **[N4], LayoutSub,
                    Layout, LayoutOrg, MemTraits>();
            test_3d_subview_5d_impl_type<Space, int ****[N4], int ***, LayoutSub, Layout,
                    LayoutOrg, MemTraits>();

            test_3d_subview_5d_impl_type<Space, int *****, int[N2][N3][N4], LayoutSub,
                    Layout, LayoutOrg, MemTraits>();
            test_3d_subview_5d_impl_type<Space, int *****, int *[N3][N4], LayoutSub,
                    Layout, LayoutOrg, MemTraits>();
            test_3d_subview_5d_impl_type<Space, int *****, int **[N4], LayoutSub, Layout,
                    LayoutOrg, MemTraits>();
            test_3d_subview_5d_impl_type<Space, int *****, int ***, LayoutSub, Layout,
                    LayoutOrg, MemTraits>();

            test_3d_subview_5d_impl_type<Space, const int[N0][N1][N2][N3][N4],
                    const int[N2][N3][N4], LayoutSub, Layout,
                    LayoutOrg, MemTraits>();
            test_3d_subview_5d_impl_type<Space, const int[N0][N1][N2][N3][N4],
                    const int *[N3][N4], LayoutSub, Layout,
                    LayoutOrg, MemTraits>();
            test_3d_subview_5d_impl_type<Space, const int[N0][N1][N2][N3][N4],
                    const int **[N4], LayoutSub, Layout, LayoutOrg,
                    MemTraits>();
            test_3d_subview_5d_impl_type<Space, const int[N0][N1][N2][N3][N4],
                    const int ***, LayoutSub, Layout, LayoutOrg,
                    MemTraits>();

            test_3d_subview_5d_impl_type<Space, const int *[N1][N2][N3][N4],
                    const int[N2][N3][N4], LayoutSub, Layout,
                    LayoutOrg, MemTraits>();
            test_3d_subview_5d_impl_type<Space, const int *[N1][N2][N3][N4],
                    const int *[N3][N4], LayoutSub, Layout,
                    LayoutOrg, MemTraits>();
            test_3d_subview_5d_impl_type<Space, const int *[N1][N2][N3][N4],
                    const int **[N4], LayoutSub, Layout, LayoutOrg,
                    MemTraits>();
            test_3d_subview_5d_impl_type<Space, const int *[N1][N2][N3][N4],
                    const int ***, LayoutSub, Layout, LayoutOrg,
                    MemTraits>();

            test_3d_subview_5d_impl_type<Space, const int **[N2][N3][N4],
                    const int[N2][N3][N4], LayoutSub, Layout,
                    LayoutOrg, MemTraits>();
            test_3d_subview_5d_impl_type<Space, const int **[N2][N3][N4],
                    const int *[N3][N4], LayoutSub, Layout,
                    LayoutOrg, MemTraits>();
            test_3d_subview_5d_impl_type<Space, const int **[N2][N3][N4],
                    const int **[N4], LayoutSub, Layout, LayoutOrg,
                    MemTraits>();
            test_3d_subview_5d_impl_type<Space, const int **[N2][N3][N4], const int ***,
                    LayoutSub, Layout, LayoutOrg, MemTraits>();

            test_3d_subview_5d_impl_type<Space, const int ***[N3][N4],
                    const int[N2][N3][N4], LayoutSub, Layout,
                    LayoutOrg, MemTraits>();
            test_3d_subview_5d_impl_type<Space, const int ***[N3][N4],
                    const int *[N3][N4], LayoutSub, Layout,
                    LayoutOrg, MemTraits>();
            test_3d_subview_5d_impl_type<Space, const int ***[N3][N4], const int **[N4],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_3d_subview_5d_impl_type<Space, const int ***[N3][N4], const int ***,
                    LayoutSub, Layout, LayoutOrg, MemTraits>();

            test_3d_subview_5d_impl_type<Space, const int ****[N4],
                    const int[N2][N3][N4], LayoutSub, Layout,
                    LayoutOrg, MemTraits>();
            test_3d_subview_5d_impl_type<Space, const int ****[N4], const int *[N3][N4],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_3d_subview_5d_impl_type<Space, const int ****[N4], const int **[N4],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_3d_subview_5d_impl_type<Space, const int ****[N4], const int ***,
                    LayoutSub, Layout, LayoutOrg, MemTraits>();

            test_3d_subview_5d_impl_type<Space, const int *****, const int[N2][N3][N4],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_3d_subview_5d_impl_type<Space, const int *****, const int *[N3][N4],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_3d_subview_5d_impl_type<Space, const int *****, const int **[N4],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_3d_subview_5d_impl_type<Space, const int *****, const int ***, LayoutSub,
                    Layout, LayoutOrg, MemTraits>();
        }

        inline void test_subview_legal_args_right() {
            REQUIRE_EQ(
                    0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, flare::ALL_t,
                    flare::ALL_t, flare::pair<int, int>, int, int>::value));
            REQUIRE_EQ(
                    0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, flare::ALL_t,
                    flare::ALL_t, flare::ALL_t, int, int>::value));
            REQUIRE_EQ(
                    0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, flare::ALL_t,
                    flare::pair<int, int>, flare::pair<int, int>, int, int>::value));
            REQUIRE_EQ(
                    0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, flare::ALL_t,
                    flare::pair<int, int>, flare::ALL_t, int, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0,
                    flare::pair<int, int>, flare::ALL_t,
                    flare::pair<int, int>, int, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0,
                    flare::pair<int, int>, flare::ALL_t, flare::ALL_t, int,
                    int>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0,
                    flare::pair<int, int>, flare::pair<int, int>,
                    flare::pair<int, int>, int, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0,
                    flare::pair<int, int>, flare::pair<int, int>,
                    flare::ALL_t, int, int>::value));

            REQUIRE_EQ(
                    0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, flare::ALL_t,
                    int, flare::ALL_t, flare::pair<int, int>, int>::value));
            REQUIRE_EQ(
                    0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, flare::ALL_t,
                    int, flare::ALL_t, flare::ALL_t, int>::value));
            REQUIRE_EQ(
                    0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, flare::ALL_t,
                    int, flare::pair<int, int>, flare::pair<int, int>, int>::value));
            REQUIRE_EQ(
                    0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, flare::ALL_t,
                    int, flare::pair<int, int>, flare::ALL_t, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0,
                    flare::pair<int, int>, int, flare::ALL_t,
                    flare::pair<int, int>, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0,
                    flare::pair<int, int>, int, flare::ALL_t, flare::ALL_t,
                    int>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0,
                    flare::pair<int, int>, int, flare::pair<int, int>,
                    flare::pair<int, int>, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0,
                    flare::pair<int, int>, int, flare::ALL_t,
                    flare::pair<int, int>, int>::value));

            REQUIRE_EQ(
                    0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, flare::ALL_t,
                    flare::ALL_t, int, flare::pair<int, int>, int>::value));
            REQUIRE_EQ(
                    0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, flare::ALL_t,
                    flare::ALL_t, int, flare::ALL_t, int>::value));
            REQUIRE_EQ(
                    0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, flare::ALL_t,
                    flare::pair<int, int>, int, flare::pair<int, int>, int>::value));
            REQUIRE_EQ(
                    0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, flare::ALL_t,
                    flare::pair<int, int>, int, flare::ALL_t, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0,
                    flare::pair<int, int>, flare::ALL_t, int,
                    flare::pair<int, int>, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0,
                    flare::pair<int, int>, flare::ALL_t, int, flare::ALL_t,
                    int>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0,
                    flare::pair<int, int>, flare::pair<int, int>, int,
                    flare::pair<int, int>, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0,
                    flare::pair<int, int>, flare::ALL_t, int,
                    flare::pair<int, int>, int>::value));

            REQUIRE_EQ(
                    0,
                    (flare::detail::SubviewLegalArgsCompileTime<
                            flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int, flare::ALL_t,
                            flare::ALL_t, flare::pair<int, int>, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int,
                    flare::ALL_t, flare::ALL_t, flare::ALL_t, int>::value));
            REQUIRE_EQ(
                    0,
                    (flare::detail::SubviewLegalArgsCompileTime<
                            flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int, flare::ALL_t,
                            flare::pair<int, int>, flare::pair<int, int>, int>::value));
            REQUIRE_EQ(
                    0,
                    (flare::detail::SubviewLegalArgsCompileTime<
                            flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int, flare::ALL_t,
                            flare::pair<int, int>, flare::ALL_t, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int,
                    flare::pair<int, int>, flare::ALL_t,
                    flare::pair<int, int>, int>::value));
            REQUIRE_EQ(
                    0,
                    (flare::detail::SubviewLegalArgsCompileTime<
                            flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int,
                            flare::pair<int, int>, flare::ALL_t, flare::ALL_t, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int,
                    flare::pair<int, int>, flare::pair<int, int>,
                    flare::pair<int, int>, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int,
                    flare::pair<int, int>, flare::pair<int, int>,
                    flare::ALL_t, int>::value));

            REQUIRE_EQ(
                    0,
                    (flare::detail::SubviewLegalArgsCompileTime<
                            flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int, flare::ALL_t,
                            flare::ALL_t, int, flare::pair<int, int>>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int,
                    flare::ALL_t, flare::ALL_t, int, flare::ALL_t>::value));
            REQUIRE_EQ(
                    0,
                    (flare::detail::SubviewLegalArgsCompileTime<
                            flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int, flare::ALL_t,
                            flare::pair<int, int>, int, flare::pair<int, int>>::value));
            REQUIRE_EQ(
                    0,
                    (flare::detail::SubviewLegalArgsCompileTime<
                            flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int, flare::ALL_t,
                            flare::pair<int, int>, int, flare::ALL_t>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int,
                    flare::pair<int, int>, flare::ALL_t, int,
                    flare::pair<int, int>>::value));
            REQUIRE_EQ(
                    0,
                    (flare::detail::SubviewLegalArgsCompileTime<
                            flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int,
                            flare::pair<int, int>, flare::ALL_t, int, flare::ALL_t>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int,
                    flare::pair<int, int>, flare::pair<int, int>, int,
                    flare::pair<int, int>>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int,
                    flare::pair<int, int>, flare::pair<int, int>, int,
                    flare::ALL_t>::value));

            REQUIRE_EQ(0,
                       (flare::detail::SubviewLegalArgsCompileTime<
                               flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int, int,
                               flare::ALL_t, flare::ALL_t, flare::pair<int, int>>::value));
            REQUIRE_EQ(1, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int, int,
                    flare::ALL_t, flare::ALL_t, flare::ALL_t>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int, int,
                    flare::ALL_t, flare::pair<int, int>,
                    flare::pair<int, int>>::value));
            REQUIRE_EQ(0,
                       (flare::detail::SubviewLegalArgsCompileTime<
                               flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int, int,
                               flare::ALL_t, flare::pair<int, int>, flare::ALL_t>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int, int,
                    flare::pair<int, int>, flare::ALL_t,
                    flare::pair<int, int>>::value));
            REQUIRE_EQ(1,
                       (flare::detail::SubviewLegalArgsCompileTime<
                               flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int, int,
                               flare::pair<int, int>, flare::ALL_t, flare::ALL_t>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int, int,
                    flare::pair<int, int>, flare::pair<int, int>,
                    flare::pair<int, int>>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int, int,
                    flare::pair<int, int>, flare::pair<int, int>,
                    flare::ALL_t>::value));

            REQUIRE_EQ(1, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 3, 0,
                    flare::ALL_t, flare::ALL_t, flare::ALL_t>::value));
            REQUIRE_EQ(
                    0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 3, 0, flare::ALL_t,
                    flare::ALL_t, flare::pair<int, int>>::value));
            REQUIRE_EQ(1,
                       (flare::detail::SubviewLegalArgsCompileTime<
                               flare::LayoutRight, flare::LayoutRight, 3, 3, 0,
                               flare::pair<int, int>, flare::ALL_t, flare::ALL_t>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 3, 0,
                    flare::pair<int, int>, flare::ALL_t,
                    flare::pair<int, int>>::value));
            REQUIRE_EQ(
                    0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 3, 0, flare::ALL_t,
                    flare::pair<int, int>, flare::ALL_t>::value));
            REQUIRE_EQ(
                    0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 3, 0, flare::ALL_t,
                    flare::pair<int, int>, flare::pair<int, int>>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 3, 0,
                    flare::pair<int, int>, flare::pair<int, int>,
                    flare::ALL_t>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 3, 0,
                    flare::pair<int, int>, flare::pair<int, int>,
                    flare::pair<int, int>>::value));
        }

        inline void test_subview_legal_args_left() {
            REQUIRE_EQ(1,
                       (flare::detail::SubviewLegalArgsCompileTime<
                               flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, flare::ALL_t,
                               flare::ALL_t, flare::pair<int, int>, int, int>::value));
            REQUIRE_EQ(1,
                       (flare::detail::SubviewLegalArgsCompileTime<
                               flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, flare::ALL_t,
                               flare::ALL_t, flare::ALL_t, int, int>::value));
            REQUIRE_EQ(
                    0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, flare::ALL_t,
                    flare::pair<int, int>, flare::pair<int, int>, int, int>::value));
            REQUIRE_EQ(0,
                       (flare::detail::SubviewLegalArgsCompileTime<
                               flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, flare::ALL_t,
                               flare::pair<int, int>, flare::ALL_t, int, int>::value));
            REQUIRE_EQ(1, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0,
                    flare::pair<int, int>, flare::ALL_t,
                    flare::pair<int, int>, int, int>::value));
            REQUIRE_EQ(1, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0,
                    flare::pair<int, int>, flare::ALL_t, flare::ALL_t, int,
                    int>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0,
                    flare::pair<int, int>, flare::pair<int, int>,
                    flare::pair<int, int>, int, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0,
                    flare::pair<int, int>, flare::pair<int, int>,
                    flare::ALL_t, int, int>::value));

            REQUIRE_EQ(0,
                       (flare::detail::SubviewLegalArgsCompileTime<
                               flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, flare::ALL_t,
                               int, flare::ALL_t, flare::pair<int, int>, int>::value));
            REQUIRE_EQ(0,
                       (flare::detail::SubviewLegalArgsCompileTime<
                               flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, flare::ALL_t,
                               int, flare::ALL_t, flare::ALL_t, int>::value));
            REQUIRE_EQ(
                    0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, flare::ALL_t,
                    int, flare::pair<int, int>, flare::pair<int, int>, int>::value));
            REQUIRE_EQ(0,
                       (flare::detail::SubviewLegalArgsCompileTime<
                               flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, flare::ALL_t,
                               int, flare::pair<int, int>, flare::ALL_t, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0,
                    flare::pair<int, int>, int, flare::ALL_t,
                    flare::pair<int, int>, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0,
                    flare::pair<int, int>, int, flare::ALL_t, flare::ALL_t,
                    int>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0,
                    flare::pair<int, int>, int, flare::pair<int, int>,
                    flare::pair<int, int>, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0,
                    flare::pair<int, int>, int, flare::ALL_t,
                    flare::pair<int, int>, int>::value));

            REQUIRE_EQ(0,
                       (flare::detail::SubviewLegalArgsCompileTime<
                               flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, flare::ALL_t,
                               flare::ALL_t, int, flare::pair<int, int>, int>::value));
            REQUIRE_EQ(0,
                       (flare::detail::SubviewLegalArgsCompileTime<
                               flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, flare::ALL_t,
                               flare::ALL_t, int, flare::ALL_t, int>::value));
            REQUIRE_EQ(
                    0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, flare::ALL_t,
                    flare::pair<int, int>, int, flare::pair<int, int>, int>::value));
            REQUIRE_EQ(0,
                       (flare::detail::SubviewLegalArgsCompileTime<
                               flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, flare::ALL_t,
                               flare::pair<int, int>, int, flare::ALL_t, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0,
                    flare::pair<int, int>, flare::ALL_t, int,
                    flare::pair<int, int>, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0,
                    flare::pair<int, int>, flare::ALL_t, int, flare::ALL_t,
                    int>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0,
                    flare::pair<int, int>, flare::pair<int, int>, int,
                    flare::pair<int, int>, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0,
                    flare::pair<int, int>, flare::ALL_t, int,
                    flare::pair<int, int>, int>::value));

            REQUIRE_EQ(
                    0,
                    (flare::detail::SubviewLegalArgsCompileTime<
                            flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int, flare::ALL_t,
                            flare::ALL_t, flare::pair<int, int>, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int,
                    flare::ALL_t, flare::ALL_t, flare::ALL_t, int>::value));
            REQUIRE_EQ(
                    0,
                    (flare::detail::SubviewLegalArgsCompileTime<
                            flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int, flare::ALL_t,
                            flare::pair<int, int>, flare::pair<int, int>, int>::value));
            REQUIRE_EQ(
                    0,
                    (flare::detail::SubviewLegalArgsCompileTime<
                            flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int, flare::ALL_t,
                            flare::pair<int, int>, flare::ALL_t, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int,
                    flare::pair<int, int>, flare::ALL_t,
                    flare::pair<int, int>, int>::value));
            REQUIRE_EQ(
                    0,
                    (flare::detail::SubviewLegalArgsCompileTime<
                            flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int,
                            flare::pair<int, int>, flare::ALL_t, flare::ALL_t, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int,
                    flare::pair<int, int>, flare::pair<int, int>,
                    flare::pair<int, int>, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int,
                    flare::pair<int, int>, flare::pair<int, int>,
                    flare::ALL_t, int>::value));

            REQUIRE_EQ(
                    0,
                    (flare::detail::SubviewLegalArgsCompileTime<
                            flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int, flare::ALL_t,
                            flare::ALL_t, int, flare::pair<int, int>>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int,
                    flare::ALL_t, flare::ALL_t, int, flare::ALL_t>::value));
            REQUIRE_EQ(
                    0,
                    (flare::detail::SubviewLegalArgsCompileTime<
                            flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int, flare::ALL_t,
                            flare::pair<int, int>, int, flare::pair<int, int>>::value));
            REQUIRE_EQ(
                    0,
                    (flare::detail::SubviewLegalArgsCompileTime<
                            flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int, flare::ALL_t,
                            flare::pair<int, int>, int, flare::ALL_t>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int,
                    flare::pair<int, int>, flare::ALL_t, int,
                    flare::pair<int, int>>::value));
            REQUIRE_EQ(
                    0,
                    (flare::detail::SubviewLegalArgsCompileTime<
                            flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int,
                            flare::pair<int, int>, flare::ALL_t, int, flare::ALL_t>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int,
                    flare::pair<int, int>, flare::pair<int, int>, int,
                    flare::pair<int, int>>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int,
                    flare::pair<int, int>, flare::pair<int, int>, int,
                    flare::ALL_t>::value));

            REQUIRE_EQ(0,
                       (flare::detail::SubviewLegalArgsCompileTime<
                               flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int, int,
                               flare::ALL_t, flare::ALL_t, flare::pair<int, int>>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int, int,
                    flare::ALL_t, flare::ALL_t, flare::ALL_t>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int, int,
                    flare::ALL_t, flare::pair<int, int>,
                    flare::pair<int, int>>::value));
            REQUIRE_EQ(0,
                       (flare::detail::SubviewLegalArgsCompileTime<
                               flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int, int,
                               flare::ALL_t, flare::pair<int, int>, flare::ALL_t>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int, int,
                    flare::pair<int, int>, flare::ALL_t,
                    flare::pair<int, int>>::value));
            REQUIRE_EQ(0,
                       (flare::detail::SubviewLegalArgsCompileTime<
                               flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int, int,
                               flare::pair<int, int>, flare::ALL_t, flare::ALL_t>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int, int,
                    flare::pair<int, int>, flare::pair<int, int>,
                    flare::pair<int, int>>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int, int,
                    flare::pair<int, int>, flare::pair<int, int>,
                    flare::ALL_t>::value));

            REQUIRE_EQ(1,
                       (flare::detail::SubviewLegalArgsCompileTime<
                               flare::LayoutLeft, flare::LayoutLeft, 3, 3, 0, flare::ALL_t,
                               flare::ALL_t, flare::pair<int, int>>::value));
            REQUIRE_EQ(1, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 3, 0,
                    flare::ALL_t, flare::ALL_t, flare::ALL_t>::value));
            REQUIRE_EQ(1, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 3, 0,
                    flare::pair<int, int>, flare::ALL_t,
                    flare::pair<int, int>>::value));
            REQUIRE_EQ(1,
                       (flare::detail::SubviewLegalArgsCompileTime<
                               flare::LayoutLeft, flare::LayoutLeft, 3, 3, 0,
                               flare::pair<int, int>, flare::ALL_t, flare::ALL_t>::value));
            REQUIRE_EQ(0,
                       (flare::detail::SubviewLegalArgsCompileTime<
                               flare::LayoutLeft, flare::LayoutLeft, 3, 3, 0, flare::ALL_t,
                               flare::pair<int, int>, flare::ALL_t>::value));
            REQUIRE_EQ(0,
                       (flare::detail::SubviewLegalArgsCompileTime<
                               flare::LayoutLeft, flare::LayoutLeft, 3, 3, 0, flare::ALL_t,
                               flare::pair<int, int>, flare::pair<int, int>>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 3, 0,
                    flare::pair<int, int>, flare::pair<int, int>,
                    flare::ALL_t>::value));
            REQUIRE_EQ(0, (flare::detail::SubviewLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 3, 0,
                    flare::pair<int, int>, flare::pair<int, int>,
                    flare::pair<int, int>>::value));
        }

    }  // namespace detail

    template<class Space, class MemTraits = void>
    void test_1d_assign() {
        detail::test_1d_assign_impl<Space, flare::LayoutLeft, flare::LayoutLeft,
                flare::LayoutLeft, MemTraits>();
        // detail::test_1d_assign_impl< Space, flare::LayoutRight, flare::LayoutLeft,
        // flare::LayoutLeft >();
        detail::test_1d_assign_impl<Space, flare::LayoutStride, flare::LayoutLeft,
                flare::LayoutLeft, MemTraits>();
        // detail::test_1d_assign_impl< Space, flare::LayoutLeft, flare::LayoutRight,
        // flare::LayoutLeft >();
        detail::test_1d_assign_impl<Space, flare::LayoutRight, flare::LayoutRight,
                flare::LayoutRight, MemTraits>();
        detail::test_1d_assign_impl<Space, flare::LayoutStride, flare::LayoutRight,
                flare::LayoutRight, MemTraits>();
        // detail::test_1d_assign_impl< Space, flare::LayoutLeft, flare::LayoutStride,
        // flare::LayoutLeft >(); detail::test_1d_assign_impl< Space,
        // flare::LayoutRight, flare::LayoutStride, flare::LayoutLeft >();
        detail::test_1d_assign_impl<Space, flare::LayoutStride, flare::LayoutStride,
                flare::LayoutLeft, MemTraits>();
    }

    template<class Space, class MemTraits = void>
    void test_2d_subview_3d() {
        detail::test_2d_subview_3d_impl_layout<Space, flare::LayoutRight,
                flare::LayoutRight, flare::LayoutRight,
                MemTraits>();
        detail::test_2d_subview_3d_impl_layout<Space, flare::LayoutStride,
                flare::LayoutRight, flare::LayoutRight,
                MemTraits>();
        detail::test_2d_subview_3d_impl_layout<Space, flare::LayoutStride,
                flare::LayoutStride,
                flare::LayoutRight, MemTraits>();
        detail::test_2d_subview_3d_impl_layout<Space, flare::LayoutStride,
                flare::LayoutLeft, flare::LayoutLeft,
                MemTraits>();
        detail::test_2d_subview_3d_impl_layout<Space, flare::LayoutStride,
                flare::LayoutStride, flare::LayoutLeft,
                MemTraits>();
    }

    template<class Space, class MemTraits = void>
    void test_3d_subview_5d_right() {
        detail::test_3d_subview_5d_impl_layout<Space, flare::LayoutStride,
                flare::LayoutRight, flare::LayoutRight,
                MemTraits>();
        detail::test_3d_subview_5d_impl_layout<Space, flare::LayoutStride,
                flare::LayoutStride,
                flare::LayoutRight, MemTraits>();
    }

    template<class Space, class MemTraits = void>
    void test_3d_subview_5d_left() {
        detail::test_3d_subview_5d_impl_layout<Space, flare::LayoutStride,
                flare::LayoutLeft, flare::LayoutLeft,
                MemTraits>();
        detail::test_3d_subview_5d_impl_layout<Space, flare::LayoutStride,
                flare::LayoutStride, flare::LayoutLeft,
                MemTraits>();
    }

    template<class Space, class MemTraits = void>
    void test_layoutleft_to_layoutleft() {
        detail::test_subview_legal_args_left();

        using view3D_t = flare::View<int ***, flare::LayoutLeft, Space>;
        using view4D_t = flare::View<int ****, flare::LayoutLeft, Space>;
        {
            view3D_t a("A", 100, 4, 3);
            view3D_t b(a, flare::pair<int, int>(16, 32), flare::ALL, flare::ALL);

            detail::FillView_3D<flare::LayoutLeft, Space> fill(a);
            fill.run();

            detail::CheckSubviewCorrectness_3D_3D<view3D_t, view3D_t> check(a, b, 16, 0);
            check.run();
        }

        {
            view3D_t a("A", 100, 4, 5);
            view3D_t b(a, flare::pair<int, int>(16, 32), flare::ALL,
                       flare::pair<int, int>(1, 3));

            detail::FillView_3D<flare::LayoutLeft, Space> fill(a);
            fill.run();

            detail::CheckSubviewCorrectness_3D_3D<view3D_t, view3D_t> check(a, b, 16, 1);
            check.run();
        }

        {
            view4D_t a("A", 100, 4, 5, 3);
            view3D_t b(a, flare::pair<int, int>(16, 32), flare::ALL,
                       flare::pair<int, int>(1, 3), 1);

            detail::FillView_4D<flare::LayoutLeft, Space> fill(a);
            fill.run();

            detail::CheckSubviewCorrectness_3D_4D<view4D_t, view3D_t> check(a, b, 1, 16,
                                                                            1);
            check.run();
        }
    }

    template<class Space, class MemTraits = void>
    void test_layoutright_to_layoutright() {
        detail::test_subview_legal_args_right();

        using view3D_t = flare::View<int ***, flare::LayoutRight, Space>;
        using view4D_t = flare::View<int ****, flare::LayoutRight, Space>;
        {
            view3D_t a("A", 100, 4, 3);
            view3D_t b(a, flare::pair<int, int>(16, 32), flare::ALL, flare::ALL);

            detail::FillView_3D<flare::LayoutRight, Space> fill(a);
            fill.run();

            detail::CheckSubviewCorrectness_3D_3D<view3D_t, view3D_t> check(a, b, 16, 0);
            check.run();
        }
        {
            view4D_t a("A", 3, 4, 5, 100);
            view3D_t b(a, 1, flare::pair<int, int>(1, 3), flare::ALL, flare::ALL);

            detail::FillView_4D<flare::LayoutRight, Space> fill(a);
            fill.run();

            detail::CheckSubviewCorrectness_3D_4D<view4D_t, view3D_t> check(a, b, 1, 1,
                                                                            0);
            check.run();
        }
    }
//----------------------------------------------------------------------------

    template<class Space>
    struct TestUnmanagedSubviewReset {
        flare::View<int ****, Space> a;

        FLARE_INLINE_FUNCTION
        void operator()(int) const noexcept {
            auto sub_a = flare::subview(a, 0, flare::ALL, flare::ALL, flare::ALL);

            for (int i = 0; i < int(a.extent(0)); ++i) {
                sub_a.assign_data(&a(i, 0, 0, 0));
                if (&sub_a(1, 1, 1) != &a(i, 1, 1, 1)) {
                    flare::abort("TestUnmanagedSubviewReset");
                }
            }
        }

        TestUnmanagedSubviewReset() : a(flare::view_alloc(), 20, 10, 5, 2) {}
    };

    template<class Space>
    void test_unmanaged_subview_reset() {
        flare::parallel_for(
                flare::RangePolicy<typename Space::execution_space>(0, 1),
                TestUnmanagedSubviewReset<Space>());
    }

//----------------------------------------------------------------------------

    template<std::underlying_type_t<flare::MemoryTraitsFlags> MTF>
    struct TestSubviewMemoryTraitsConstruction {
        void operator()() const noexcept {
            using memory_traits_type = flare::MemoryTraits<MTF>;
            using view_type =
                    flare::View<double *, flare::HostSpace, memory_traits_type>;
            using size_type = typename view_type::size_type;

            // Create a managed View first and then apply the desired memory traits to
            // an unmanaged version of it since a managed View can't use the Unmanaged
            // trait.
            flare::View<double *, flare::HostSpace> v_original("v", 7);
            view_type v(v_original.data(), v_original.size());
            for (size_type i = 0; i != v.size(); ++i) v[i] = static_cast<double>(i);

            std::pair<int, int> range(3, 5);
            auto sv = flare::subview(v, range);

            // check that the subview memory traits are the same as the original view
            // (with the Aligned trait stripped).
            using view_memory_traits = typename decltype(v)::memory_traits;
            using subview_memory_traits = typename decltype(sv)::memory_traits;
            static_assert(view_memory_traits::impl_value ==
                          memory_traits_type::impl_value);
            if constexpr (memory_traits_type::is_aligned)
                static_assert(subview_memory_traits::impl_value + flare::Aligned ==
                              memory_traits_type::impl_value);
            else
                static_assert(subview_memory_traits::impl_value ==
                              memory_traits_type::impl_value);

            REQUIRE_EQ(2u, sv.size());
            REQUIRE_EQ(3., sv[0]);
            REQUIRE_EQ(4., sv[1]);
        }
    };

    inline void test_subview_memory_traits_construction() {
        // Test all combinations of MemoryTraits:
        // Unmanaged (1)
        // RandomAccess (2)
        // Atomic (4)
        // Restricted (8)
        // Aligned (16)
        TestSubviewMemoryTraitsConstruction<0>()();
        TestSubviewMemoryTraitsConstruction<1>()();
        TestSubviewMemoryTraitsConstruction<2>()();
        TestSubviewMemoryTraitsConstruction<3>()();
        TestSubviewMemoryTraitsConstruction<4>()();
        TestSubviewMemoryTraitsConstruction<5>()();
        TestSubviewMemoryTraitsConstruction<6>()();
        TestSubviewMemoryTraitsConstruction<7>()();
        TestSubviewMemoryTraitsConstruction<8>()();
        TestSubviewMemoryTraitsConstruction<9>()();
        TestSubviewMemoryTraitsConstruction<10>()();
        TestSubviewMemoryTraitsConstruction<11>()();
        TestSubviewMemoryTraitsConstruction<12>()();
        TestSubviewMemoryTraitsConstruction<13>()();
        TestSubviewMemoryTraitsConstruction<14>()();
        TestSubviewMemoryTraitsConstruction<15>()();
        TestSubviewMemoryTraitsConstruction<16>()();
        TestSubviewMemoryTraitsConstruction<17>()();
        TestSubviewMemoryTraitsConstruction<18>()();
        TestSubviewMemoryTraitsConstruction<19>()();
        TestSubviewMemoryTraitsConstruction<20>()();
        TestSubviewMemoryTraitsConstruction<21>()();
        TestSubviewMemoryTraitsConstruction<22>()();
        TestSubviewMemoryTraitsConstruction<23>()();
        TestSubviewMemoryTraitsConstruction<24>()();
        TestSubviewMemoryTraitsConstruction<25>()();
        TestSubviewMemoryTraitsConstruction<26>()();
        TestSubviewMemoryTraitsConstruction<27>()();
        TestSubviewMemoryTraitsConstruction<28>()();
        TestSubviewMemoryTraitsConstruction<29>()();
        TestSubviewMemoryTraitsConstruction<30>()();
        TestSubviewMemoryTraitsConstruction<31>()();
    }

//----------------------------------------------------------------------------

    template<class T>
    struct get_view_type;

    template<class T, class... Args>
    struct get_view_type<flare::View<T, Args...>> {
        using type = T;
    };

    template<class T>
    struct
    ___________________________________TYPE_DISPLAY________________________________________;
#define TYPE_DISPLAY(...)                                                                           \
  typename ___________________________________TYPE_DISPLAY________________________________________< \
      __VA_ARGS__>::type notdefined;

    template<class Space, class Layout>
    struct TestSubviewStaticSizes {
        flare::View<int *[10][5][2], Layout, Space> a;
        flare::View<int[6][7][8], Layout, Space> b;

        FLARE_INLINE_FUNCTION
        int operator()() const noexcept {
            /* Doesn't actually do anything; just static assertions */

            auto sub_a = flare::subview(a, 0, flare::ALL, flare::ALL, flare::ALL);
            typename static_expect_same<
                    /* expected */ int[10][5][2],
                    /*  actual  */ typename get_view_type<decltype(sub_a)>::type>::type
                    test_1 = 0;

            auto sub_a_2 = flare::subview(a, 0, 0, flare::ALL, flare::ALL);
            typename static_expect_same<
                    /* expected */ int[5][2],
                    /*  actual  */ typename get_view_type<decltype(sub_a_2)>::type>::type
                    test_2 = 0;

            auto sub_a_3 = flare::subview(a, 0, 0, flare::ALL, 0);
            typename static_expect_same<
                    /* expected */ int[5],
                    /*  actual  */ typename get_view_type<decltype(sub_a_3)>::type>::type
                    test_3 = 0;

            auto sub_a_4 = flare::subview(a, flare::ALL, 0, flare::ALL, flare::ALL);
            typename static_expect_same<
                    /* expected */ int *[5][2],
                    /*  actual  */ typename get_view_type<decltype(sub_a_4)>::type>::type
                    test_4 = 0;

            // TODO we'll need to update this test once we allow interleaving of static
            // and dynamic
            auto sub_a_5 = flare::subview(a, flare::ALL, 0, flare::ALL,
                                          flare::make_pair(0, 1));
            typename static_expect_same<
                    /* expected */ int ***,
                    /*  actual  */ typename get_view_type<decltype(sub_a_5)>::type>::type
                    test_5 = 0;

            auto sub_a_sub = flare::subview(sub_a_5, 0, flare::ALL, 0);
            typename static_expect_same<
                    /* expected */ int *,
                    /*  actual  */ typename get_view_type<decltype(sub_a_sub)>::type>::type
                    test_sub = 0;

            auto sub_a_7 = flare::subview(a, flare::ALL, 0, flare::make_pair(0, 1),
                                          flare::ALL);
            typename static_expect_same<
                    /* expected */ int **[2],
                    /*  actual  */ typename get_view_type<decltype(sub_a_7)>::type>::type
                    test_7 = 0;

            auto sub_a_8 =
                    flare::subview(a, flare::ALL, flare::ALL, flare::ALL, flare::ALL);
            typename static_expect_same<
                    /* expected */ int *[10][5][2],
                    /*  actual  */ typename get_view_type<decltype(sub_a_8)>::type>::type
                    test_8 = 0;

            auto sub_b = flare::subview(b, flare::ALL, flare::ALL, flare::ALL);
            typename static_expect_same<
                    /* expected */ int[6][7][8],
                    /*  actual  */ typename get_view_type<decltype(sub_b)>::type>::type
                    test_9 = 0;

            auto sub_b_2 = flare::subview(b, 0, flare::ALL, flare::ALL);
            typename static_expect_same<
                    /* expected */ int[7][8],
                    /*  actual  */ typename get_view_type<decltype(sub_b_2)>::type>::type
                    test_10 = 0;

            auto sub_b_3 =
                    flare::subview(b, flare::make_pair(2, 3), flare::ALL, flare::ALL);
            typename static_expect_same<
                    /* expected */ int *[7][8],
                    /*  actual  */ typename get_view_type<decltype(sub_b_3)>::type>::type
                    test_11 = 0;

            return test_1 + test_2 + test_3 + test_4 + test_5 + test_sub + test_7 +
                   test_8 + test_9 + test_10 + test_11;
        }

        TestSubviewStaticSizes() : a(flare::view_alloc("a"), 20), b("b") {}
    };

    template<class Space>
    struct TestExtentsStaticTests {
        using test1 = typename static_expect_same<
                /* expected */
                flare::experimental::Extents<flare::experimental::dynamic_extent,
                        flare::experimental::dynamic_extent, 1, 2,
                        3>,
                /* actual */
                typename flare::detail::ParseViewExtents<double **[1][2][3]>::type>::type;

        using test2 = typename static_expect_same<
                /* expected */
                flare::experimental::Extents<1, 2, 3>,
                /* actual */
                typename flare::detail::ParseViewExtents<double[1][2][3]>::type>::type;

        using test3 = typename static_expect_same<
                /* expected */
                flare::experimental::Extents<3>,
                /* actual */
                typename flare::detail::ParseViewExtents<double[3]>::type>::type;

        using test4 = typename static_expect_same<
                /* expected */
                flare::experimental::Extents<>,
                /* actual */
                typename flare::detail::ParseViewExtents<double>::type>::type;
    };

}  // namespace TestViewSubview

#endif  // TEST_VIEW_SUB_VIEW_H_
