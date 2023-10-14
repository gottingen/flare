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

namespace TestTensorSubtensor {

    template<class Layout, class Space>
    struct getTensor {
        static flare::Tensor<double **, Layout, Space> get(int n, int m) {
            return flare::Tensor<double **, Layout, Space>("G", n, m);
        }
    };

    template<class Space>
    struct getTensor<flare::LayoutStride, Space> {
        static flare::Tensor<double **, flare::LayoutStride, Space> get(int n, int m) {
            const int rank = 2;
            const int order[] = {0, 1};
            const unsigned dim[] = {unsigned(n), unsigned(m)};
            flare::LayoutStride stride =
                    flare::LayoutStride::order_dimensions(rank, order, dim);

            return flare::Tensor<double **, flare::LayoutStride, Space>("G", stride);
        }
    };

    template<class TensorType, class Space>
    struct fill_1D {
        using execution_space = typename Space::execution_space;
        using size_type = typename TensorType::size_type;

        TensorType a;
        double val;

        fill_1D(TensorType a_, double val_) : a(a_), val(val_) {}

        FLARE_INLINE_FUNCTION
        void operator()(const int i) const { a(i) = val; }
    };

    template<class TensorType, class Space>
    struct fill_2D {
        using execution_space = typename Space::execution_space;
        using size_type = typename TensorType::size_type;

        TensorType a;
        double val;

        fill_2D(TensorType a_, double val_) : a(a_), val(val_) {}

        FLARE_INLINE_FUNCTION
        void operator()(const int i) const {
            for (int j = 0; j < static_cast<int>(a.extent(1)); j++) {
                a(i, j) = val;
            }
        }
    };

    template<class Layout, class Space>
    void test_auto_1d() {
        using mv_type = flare::Tensor<double **, Layout, Space>;
        using execution_space = typename Space::execution_space;
        using size_type = typename mv_type::size_type;

        const double ZERO = 0.0;
        const double ONE = 1.0;
        const double TWO = 2.0;

        const size_type numRows = 10;
        const size_type numCols = 3;

        mv_type X = getTensor<Layout, Space>::get(numRows, numCols);
        typename mv_type::HostMirror X_h = flare::create_mirror_tensor(X);

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
            auto X_j = flare::subtensor(X, flare::ALL, j);

            fill_1D<decltype(X_j), Space> f4(X_j, ZERO);
            flare::parallel_for(
                    flare::RangePolicy<execution_space, Property>(0, X_j.extent(0)), f4);
            flare::fence();
            flare::deep_copy(X_h, X);
            for (size_type i = 0; i < numRows; ++i) {
                REQUIRE_EQ(X_h(i, j), ZERO);
            }

            for (size_type jj = 0; jj < numCols; ++jj) {
                auto X_jj = flare::subtensor(X, flare::ALL, jj);
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
        flare::Tensor<double **, LS, Space> l2d("l2d", n, m);

        int col = n > 2 ? 2 : 0;
        int row = m > 2 ? 2 : 0;

        if (flare::SpaceAccessibility<flare::HostSpace,
                typename Space::memory_space>::accessible) {
            if (a) {
                flare::Tensor<double *, LD, Space> l1da =
                        flare::subtensor(l2d, flare::ALL, row);
                REQUIRE_EQ(&l1da(0), &l2d(0, row));
                if (n > 1) {
                    REQUIRE_EQ(&l1da(1), &l2d(1, row));
                }
            }

            if (b && n > 13) {
                flare::Tensor<double *, LD, Space> l1db =
                        flare::subtensor(l2d, std::pair<unsigned, unsigned>(2, 13), row);
                REQUIRE_EQ(&l1db(0), &l2d(2, row));
                REQUIRE_EQ(&l1db(1), &l2d(3, row));
            }

            if (c) {
                flare::Tensor<double *, LD, Space> l1dc =
                        flare::subtensor(l2d, col, flare::ALL);
                REQUIRE_EQ(&l1dc(0), &l2d(col, 0));
                if (m > 1) {
                    REQUIRE_EQ(&l1dc(1), &l2d(col, 1));
                }
            }

            if (d && m > 13) {
                flare::Tensor<double *, LD, Space> l1dd =
                        flare::subtensor(l2d, col, std::pair<unsigned, unsigned>(2, 13));
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

    template<class NewTensor, class OrigTesnor, class... Args>
    void make_subtensor(bool use_constructor, NewTensor &v, OrigTesnor org,
                      Args... args) {
        if (use_constructor) {
            v = NewTensor(org, args...);
        } else {
            v = flare::subtensor(org, args...);
        }
    }

    template<class Space>
    void test_left_0(bool constr) {
        using tensor_static_8_type =
                flare::Tensor<int[2][3][4][5][2][3][4][5], flare::LayoutLeft, Space>;

        if (flare::SpaceAccessibility<flare::HostSpace,
                typename Space::memory_space>::accessible) {
            tensor_static_8_type x_static_8("x_static_left_8");

            REQUIRE(x_static_8.span_is_contiguous());

            flare::Tensor<int, flare::LayoutLeft, Space> x0;
            make_subtensor(constr, x0, x_static_8, 0, 0, 0, 0, 0, 0, 0, 0);

            REQUIRE(x0.span_is_contiguous());
            REQUIRE_EQ(x0.span(), 1u);
            REQUIRE_EQ(&x0(), &x_static_8(0, 0, 0, 0, 0, 0, 0, 0));

            flare::Tensor<int *, flare::LayoutLeft, Space> x1;
            make_subtensor(constr, x1, x_static_8, flare::pair<int, int>(0, 2), 1, 2, 3,
                         0, 1, 2, 3);

            REQUIRE(x1.span_is_contiguous());
            REQUIRE_EQ(x1.span(), 2u);
            REQUIRE_EQ(&x1(0), &x_static_8(0, 1, 2, 3, 0, 1, 2, 3));
            REQUIRE_EQ(&x1(1), &x_static_8(1, 1, 2, 3, 0, 1, 2, 3));

            flare::Tensor<int *, flare::LayoutLeft, Space> x_deg1;
            make_subtensor(constr, x_deg1, x_static_8, flare::pair<int, int>(0, 0), 1, 2,
                         3, 0, 1, 2, 3);

            REQUIRE(x_deg1.span_is_contiguous());
            REQUIRE_EQ(x_deg1.span(), 0u);
            REQUIRE_EQ(x_deg1.data(), &x_static_8(0, 1, 2, 3, 0, 1, 2, 3));

            flare::Tensor<int *, flare::LayoutLeft, Space> x_deg2;
            make_subtensor(constr, x_deg2, x_static_8, flare::pair<int, int>(2, 2), 2, 3,
                         4, 1, 2, 3, 4);

            REQUIRE(x_deg2.span_is_contiguous());
            REQUIRE_EQ(x_deg2.span(), 0u);
            REQUIRE_EQ(x_deg2.data(), x_static_8.data() + x_static_8.span());

            flare::Tensor<int **, flare::LayoutLeft, Space> x2;
            make_subtensor(constr, x2, x_static_8, flare::pair<int, int>(0, 2), 1, 2, 3,
                         flare::pair<int, int>(0, 2), 1, 2, 3);

            REQUIRE(!x2.span_is_contiguous());
            REQUIRE_EQ(&x2(0, 0), &x_static_8(0, 1, 2, 3, 0, 1, 2, 3));
            REQUIRE_EQ(&x2(1, 0), &x_static_8(1, 1, 2, 3, 0, 1, 2, 3));
            REQUIRE_EQ(&x2(0, 1), &x_static_8(0, 1, 2, 3, 1, 1, 2, 3));
            REQUIRE_EQ(&x2(1, 1), &x_static_8(1, 1, 2, 3, 1, 1, 2, 3));

            // flare::Tensor< int**, flare::LayoutLeft, Space > error_2 =
            flare::Tensor<int **, flare::LayoutStride, Space> sx2;
            make_subtensor(constr, sx2, x_static_8, 1, flare::pair<int, int>(0, 2), 2, 3,
                         flare::pair<int, int>(0, 2), 1, 2, 3);

            REQUIRE(!sx2.span_is_contiguous());
            REQUIRE_EQ(&sx2(0, 0), &x_static_8(1, 0, 2, 3, 0, 1, 2, 3));
            REQUIRE_EQ(&sx2(1, 0), &x_static_8(1, 1, 2, 3, 0, 1, 2, 3));
            REQUIRE_EQ(&sx2(0, 1), &x_static_8(1, 0, 2, 3, 1, 1, 2, 3));
            REQUIRE_EQ(&sx2(1, 1), &x_static_8(1, 1, 2, 3, 1, 1, 2, 3));

            flare::Tensor<int ****, flare::LayoutStride, Space> sx4;
            make_subtensor(constr, sx4, x_static_8, 0,
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
        using tensor_type =
                flare::Tensor<int ****[2][3][4][5], flare::LayoutLeft, Space>;

        if (flare::SpaceAccessibility<flare::HostSpace,
                typename Space::memory_space>::accessible) {
            tensor_type x8("x_left_8", 2, 3, 4, 5);

            REQUIRE(x8.span_is_contiguous());

            flare::Tensor<int, flare::LayoutLeft, Space> x0;
            make_subtensor(use_constr, x0, x8, 0, 0, 0, 0, 0, 0, 0, 0);

            REQUIRE(x0.span_is_contiguous());
            REQUIRE_EQ(&x0(), &x8(0, 0, 0, 0, 0, 0, 0, 0));

            flare::Tensor<int *, flare::LayoutLeft, Space> x1;
            make_subtensor(use_constr, x1, x8, flare::pair<int, int>(0, 2), 1, 2, 3, 0,
                         1, 2, 3);

            REQUIRE(x1.span_is_contiguous());
            REQUIRE_EQ(&x1(0), &x8(0, 1, 2, 3, 0, 1, 2, 3));
            REQUIRE_EQ(&x1(1), &x8(1, 1, 2, 3, 0, 1, 2, 3));

            flare::Tensor<int *, flare::LayoutLeft, Space> x1_deg1;
            make_subtensor(use_constr, x1_deg1, x8, flare::pair<int, int>(0, 0), 1, 2, 3,
                         0, 1, 2, 3);

            REQUIRE(x1_deg1.span_is_contiguous());
            REQUIRE_EQ(0u, x1_deg1.span());
            REQUIRE_EQ(x1_deg1.data(), &x8(0, 1, 2, 3, 0, 1, 2, 3));

            flare::Tensor<int *, flare::LayoutLeft, Space> x1_deg2;
            make_subtensor(use_constr, x1_deg2, x8, flare::pair<int, int>(2, 2), 2, 3, 4,
                         1, 2, 3, 4);

            REQUIRE_EQ(0u, x1_deg2.span());
            REQUIRE(x1_deg2.span_is_contiguous());
            REQUIRE_EQ(x1_deg2.data(), x8.data() + x8.span());

            flare::Tensor<int **, flare::LayoutLeft, Space> x2;
            make_subtensor(use_constr, x2, x8, flare::pair<int, int>(0, 2), 1, 2, 3,
                         flare::pair<int, int>(0, 2), 1, 2, 3);

            REQUIRE(!x2.span_is_contiguous());
            REQUIRE_EQ(&x2(0, 0), &x8(0, 1, 2, 3, 0, 1, 2, 3));
            REQUIRE_EQ(&x2(1, 0), &x8(1, 1, 2, 3, 0, 1, 2, 3));
            REQUIRE_EQ(&x2(0, 1), &x8(0, 1, 2, 3, 1, 1, 2, 3));
            REQUIRE_EQ(&x2(1, 1), &x8(1, 1, 2, 3, 1, 1, 2, 3));

            flare::Tensor<int **, flare::LayoutLeft, Space> x2_deg2;
            make_subtensor(use_constr, x2_deg2, x8, flare::pair<int, int>(2, 2), 2, 3, 4,
                         1, 2, flare::pair<int, int>(2, 3), 4);
            REQUIRE_EQ(0u, x2_deg2.span());

            // flare::Tensor< int**, flare::LayoutLeft, Space > error_2 =
            flare::Tensor<int **, flare::LayoutStride, Space> sx2;
            make_subtensor(use_constr, sx2, x8, 1, flare::pair<int, int>(0, 2), 2, 3,
                         flare::pair<int, int>(0, 2), 1, 2, 3);

            REQUIRE(!sx2.span_is_contiguous());
            REQUIRE_EQ(&sx2(0, 0), &x8(1, 0, 2, 3, 0, 1, 2, 3));
            REQUIRE_EQ(&sx2(1, 0), &x8(1, 1, 2, 3, 0, 1, 2, 3));
            REQUIRE_EQ(&sx2(0, 1), &x8(1, 0, 2, 3, 1, 1, 2, 3));
            REQUIRE_EQ(&sx2(1, 1), &x8(1, 1, 2, 3, 1, 1, 2, 3));

            flare::Tensor<int **, flare::LayoutStride, Space> sx2_deg;
            make_subtensor(use_constr, sx2, x8, 1, flare::pair<int, int>(0, 0), 2, 3,
                         flare::pair<int, int>(0, 2), 1, 2, 3);
            REQUIRE_EQ(0u, sx2_deg.span());

            flare::Tensor<int ****, flare::LayoutStride, Space> sx4;
            make_subtensor(use_constr, sx4, x8, 0,
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
        using tensor_type = flare::Tensor<int ****, flare::LayoutLeft, Space>;

        if (flare::SpaceAccessibility<flare::HostSpace,
                typename Space::memory_space>::accessible) {
            tensor_type x4("x4", 2, 3, 4, 5);

            REQUIRE(x4.span_is_contiguous());

            flare::Tensor<int, flare::LayoutLeft, Space> x0 =
                    flare::subtensor(x4, 0, 0, 0, 0);

            REQUIRE(x0.span_is_contiguous());
            REQUIRE_EQ(&x0(), &x4(0, 0, 0, 0));

            flare::Tensor<int *, flare::LayoutLeft, Space> x1 =
                    flare::subtensor(x4, flare::pair<int, int>(0, 2), 1, 2, 3);

            REQUIRE(x1.span_is_contiguous());
            REQUIRE_EQ(&x1(0), &x4(0, 1, 2, 3));
            REQUIRE_EQ(&x1(1), &x4(1, 1, 2, 3));

            flare::Tensor<int **, flare::LayoutLeft, Space> x2 = flare::subtensor(
                    x4, flare::pair<int, int>(0, 2), 1, flare::pair<int, int>(1, 3), 2);

            REQUIRE(!x2.span_is_contiguous());
            REQUIRE_EQ(&x2(0, 0), &x4(0, 1, 1, 2));
            REQUIRE_EQ(&x2(1, 0), &x4(1, 1, 1, 2));
            REQUIRE_EQ(&x2(0, 1), &x4(0, 1, 2, 2));
            REQUIRE_EQ(&x2(1, 1), &x4(1, 1, 2, 2));

            // flare::Tensor< int**, flare::LayoutLeft, Space > error_2 =
            flare::Tensor<int **, flare::LayoutStride, Space> sx2 = flare::subtensor(
                    x4, 1, flare::pair<int, int>(0, 2), 2, flare::pair<int, int>(1, 4));

            REQUIRE(!sx2.span_is_contiguous());
            REQUIRE_EQ(&sx2(0, 0), &x4(1, 0, 2, 1));
            REQUIRE_EQ(&sx2(1, 0), &x4(1, 1, 2, 1));
            REQUIRE_EQ(&sx2(0, 1), &x4(1, 0, 2, 2));
            REQUIRE_EQ(&sx2(1, 1), &x4(1, 1, 2, 2));
            REQUIRE_EQ(&sx2(0, 2), &x4(1, 0, 2, 3));
            REQUIRE_EQ(&sx2(1, 2), &x4(1, 1, 2, 3));

            flare::Tensor<int ****, flare::LayoutStride, Space> sx4 =
                    flare::subtensor(x4, flare::pair<int, int>(1, 2) /* of [2] */
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
        using tensor_type = flare::Tensor<int **, flare::LayoutLeft, Space>;

        if (flare::SpaceAccessibility<flare::HostSpace,
                typename Space::memory_space>::accessible) {
            tensor_type xm("x4", 10, 5);

            REQUIRE(xm.span_is_contiguous());

            flare::Tensor<int, flare::LayoutLeft, Space> x0 = flare::subtensor(xm, 5, 3);

            REQUIRE(x0.span_is_contiguous());
            REQUIRE_EQ(&x0(), &xm(5, 3));

            flare::Tensor<int *, flare::LayoutLeft, Space> x1 =
                    flare::subtensor(xm, flare::ALL, 3);

            REQUIRE(x1.span_is_contiguous());
            for (int i = 0; i < int(xm.extent(0)); ++i) {
                REQUIRE_EQ(&x1(i), &xm(i, 3));
            }

            flare::Tensor<int **, flare::LayoutLeft, Space> x2 =
                    flare::subtensor(xm, flare::pair<int, int>(1, 9), flare::ALL);

            REQUIRE(!x2.span_is_contiguous());
            for (int j = 0; j < int(x2.extent(1)); ++j)
                for (int i = 0; i < int(x2.extent(0)); ++i) {
                    REQUIRE_EQ(&x2(i, j), &xm(1 + i, j));
                }

            flare::Tensor<int **, flare::LayoutLeft, Space> x2c =
                    flare::subtensor(xm, flare::ALL, std::pair<int, int>(2, 4));

            REQUIRE(x2c.span_is_contiguous());
            for (int j = 0; j < int(x2c.extent(1)); ++j)
                for (int i = 0; i < int(x2c.extent(0)); ++i) {
                    REQUIRE_EQ(&x2c(i, j), &xm(i, 2 + j));
                }

            flare::Tensor<int **, flare::LayoutLeft, Space> x2_n1 =
                    flare::subtensor(xm, std::pair<int, int>(1, 1), flare::ALL);

            REQUIRE_EQ(x2_n1.extent(0), 0u);
            REQUIRE_EQ(x2_n1.extent(1), xm.extent(1));

            flare::Tensor<int **, flare::LayoutLeft, Space> x2_n2 =
                    flare::subtensor(xm, flare::ALL, std::pair<int, int>(1, 1));

            REQUIRE_EQ(x2_n2.extent(0), xm.extent(0));
            REQUIRE_EQ(x2_n2.extent(1), 0u);
        }
    }

//----------------------------------------------------------------------------

    template<class Space>
    void test_right_0(bool use_constr) {
        using tensor_static_8_type =
                flare::Tensor<int[2][3][4][5][2][3][4][5], flare::LayoutRight, Space>;

        if (flare::SpaceAccessibility<flare::HostSpace,
                typename Space::memory_space>::accessible) {
            tensor_static_8_type x_static_8("x_static_right_8");

            flare::Tensor<int, flare::LayoutRight, Space> x0;
            make_subtensor(use_constr, x0, x_static_8, 0, 0, 0, 0, 0, 0, 0, 0);

            REQUIRE_EQ(&x0(), &x_static_8(0, 0, 0, 0, 0, 0, 0, 0));

            flare::Tensor<int *, flare::LayoutRight, Space> x1;
            make_subtensor(use_constr, x1, x_static_8, 0, 1, 2, 3, 0, 1, 2,
                         flare::pair<int, int>(1, 3));

            REQUIRE_EQ(x1.extent(0), 2u);
            REQUIRE_EQ(&x1(0), &x_static_8(0, 1, 2, 3, 0, 1, 2, 1));
            REQUIRE_EQ(&x1(1), &x_static_8(0, 1, 2, 3, 0, 1, 2, 2));

            flare::Tensor<int **, flare::LayoutRight, Space> x2;
            make_subtensor(use_constr, x2, x_static_8, 0, 1, 2,
                         flare::pair<int, int>(1, 3), 0, 1, 2,
                         flare::pair<int, int>(1, 3));

            REQUIRE_EQ(x2.extent(0), 2u);
            REQUIRE_EQ(x2.extent(1), 2u);
            REQUIRE_EQ(&x2(0, 0), &x_static_8(0, 1, 2, 1, 0, 1, 2, 1));
            REQUIRE_EQ(&x2(1, 0), &x_static_8(0, 1, 2, 2, 0, 1, 2, 1));
            REQUIRE_EQ(&x2(0, 1), &x_static_8(0, 1, 2, 1, 0, 1, 2, 2));
            REQUIRE_EQ(&x2(1, 1), &x_static_8(0, 1, 2, 2, 0, 1, 2, 2));

            // flare::Tensor< int**, flare::LayoutRight, Space > error_2 =
            flare::Tensor<int **, flare::LayoutStride, Space> sx2;
            make_subtensor(use_constr, sx2, x_static_8, 1, flare::pair<int, int>(0, 2),
                         2, 3, flare::pair<int, int>(0, 2), 1, 2, 3);

            REQUIRE_EQ(sx2.extent(0), 2u);
            REQUIRE_EQ(sx2.extent(1), 2u);
            REQUIRE_EQ(&sx2(0, 0), &x_static_8(1, 0, 2, 3, 0, 1, 2, 3));
            REQUIRE_EQ(&sx2(1, 0), &x_static_8(1, 1, 2, 3, 0, 1, 2, 3));
            REQUIRE_EQ(&sx2(0, 1), &x_static_8(1, 0, 2, 3, 1, 1, 2, 3));
            REQUIRE_EQ(&sx2(1, 1), &x_static_8(1, 1, 2, 3, 1, 1, 2, 3));

            flare::Tensor<int ****, flare::LayoutStride, Space> sx4;
            make_subtensor(use_constr, sx4, x_static_8, 0,
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
        using tensor_type =
                flare::Tensor<int ****[2][3][4][5], flare::LayoutRight, Space>;

        if (flare::SpaceAccessibility<flare::HostSpace,
                typename Space::memory_space>::accessible) {
            tensor_type x8("x_right_8", 2, 3, 4, 5);

            flare::Tensor<int, flare::LayoutRight, Space> x0;
            make_subtensor(use_constr, x0, x8, 0, 0, 0, 0, 0, 0, 0, 0);

            REQUIRE_EQ(&x0(), &x8(0, 0, 0, 0, 0, 0, 0, 0));

            flare::Tensor<int *, flare::LayoutRight, Space> x1;
            make_subtensor(use_constr, x1, x8, 0, 1, 2, 3, 0, 1, 2,
                         flare::pair<int, int>(1, 3));

            REQUIRE_EQ(&x1(0), &x8(0, 1, 2, 3, 0, 1, 2, 1));
            REQUIRE_EQ(&x1(1), &x8(0, 1, 2, 3, 0, 1, 2, 2));

            flare::Tensor<int *, flare::LayoutRight, Space> x1_deg1;
            make_subtensor(use_constr, x1_deg1, x8, 0, 1, 2, 3, 0, 1, 2,
                         flare::pair<int, int>(3, 3));
            REQUIRE_EQ(0u, x1_deg1.span());

            flare::Tensor<int **, flare::LayoutRight, Space> x2;
            make_subtensor(use_constr, x2, x8, 0, 1, 2, flare::pair<int, int>(1, 3), 0,
                         1, 2, flare::pair<int, int>(1, 3));

            REQUIRE_EQ(&x2(0, 0), &x8(0, 1, 2, 1, 0, 1, 2, 1));
            REQUIRE_EQ(&x2(1, 0), &x8(0, 1, 2, 2, 0, 1, 2, 1));
            REQUIRE_EQ(&x2(0, 1), &x8(0, 1, 2, 1, 0, 1, 2, 2));
            REQUIRE_EQ(&x2(1, 1), &x8(0, 1, 2, 2, 0, 1, 2, 2));

            flare::Tensor<int **, flare::LayoutRight, Space> x2_deg2;
            make_subtensor(use_constr, x2_deg2, x8, 0, 1, 2, flare::pair<int, int>(1, 3),
                         0, 1, 2, flare::pair<int, int>(3, 3));
            REQUIRE_EQ(0u, x2_deg2.span());

            // flare::Tensor< int**, flare::LayoutRight, Space > error_2 =
            flare::Tensor<int **, flare::LayoutStride, Space> sx2;
            make_subtensor(use_constr, sx2, x8, 1, flare::pair<int, int>(0, 2), 2, 3,
                         flare::pair<int, int>(0, 2), 1, 2, 3);

            REQUIRE_EQ(&sx2(0, 0), &x8(1, 0, 2, 3, 0, 1, 2, 3));
            REQUIRE_EQ(&sx2(1, 0), &x8(1, 1, 2, 3, 0, 1, 2, 3));
            REQUIRE_EQ(&sx2(0, 1), &x8(1, 0, 2, 3, 1, 1, 2, 3));
            REQUIRE_EQ(&sx2(1, 1), &x8(1, 1, 2, 3, 1, 1, 2, 3));

            flare::Tensor<int **, flare::LayoutStride, Space> sx2_deg;
            make_subtensor(use_constr, sx2_deg, x8, 1, flare::pair<int, int>(0, 2), 2, 3,
                         1, 1, 2, flare::pair<int, int>(3, 3));
            REQUIRE_EQ(0u, sx2_deg.span());

            flare::Tensor<int ****, flare::LayoutStride, Space> sx4;
            make_subtensor(use_constr, sx4, x8, 0,
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
        using tensor_type = flare::Tensor<int **, flare::LayoutRight, Space>;

        if (flare::SpaceAccessibility<flare::HostSpace,
                typename Space::memory_space>::accessible) {
            tensor_type xm("x4", 10, 5);

            REQUIRE(xm.span_is_contiguous());

            flare::Tensor<int, flare::LayoutRight, Space> x0 =
                    flare::subtensor(xm, 5, 3);

            REQUIRE(x0.span_is_contiguous());
            REQUIRE_EQ(&x0(), &xm(5, 3));

            flare::Tensor<int *, flare::LayoutRight, Space> x1 =
                    flare::subtensor(xm, 3, flare::ALL);

            REQUIRE(x1.span_is_contiguous());
            for (int i = 0; i < int(xm.extent(1)); ++i) {
                REQUIRE_EQ(&x1(i), &xm(3, i));
            }

            flare::Tensor<int **, flare::LayoutRight, Space> x2c =
                    flare::subtensor(xm, flare::pair<int, int>(1, 9), flare::ALL);

            REQUIRE(x2c.span_is_contiguous());
            for (int j = 0; j < int(x2c.extent(1)); ++j)
                for (int i = 0; i < int(x2c.extent(0)); ++i) {
                    REQUIRE_EQ(&x2c(i, j), &xm(1 + i, j));
                }

            flare::Tensor<int **, flare::LayoutRight, Space> x2 =
                    flare::subtensor(xm, flare::ALL, std::pair<int, int>(2, 4));

            REQUIRE(!x2.span_is_contiguous());
            for (int j = 0; j < int(x2.extent(1)); ++j)
                for (int i = 0; i < int(x2.extent(0)); ++i) {
                    REQUIRE_EQ(&x2(i, j), &xm(i, 2 + j));
                }

            flare::Tensor<int **, flare::LayoutRight, Space> x2_n1 =
                    flare::subtensor(xm, std::pair<int, int>(1, 1), flare::ALL);

            REQUIRE_EQ(x2_n1.extent(0), 0u);
            REQUIRE_EQ(x2_n1.extent(1), xm.extent(1));

            flare::Tensor<int **, flare::LayoutRight, Space> x2_n2 =
                    flare::subtensor(xm, flare::ALL, std::pair<int, int>(1, 1));

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
        struct FillTensor_1D {
            using tensor_t = flare::Tensor<int *, Layout, Space>;
            tensor_t a;
            using policy_t = flare::RangePolicy<typename Space::execution_space>;

            FillTensor_1D(tensor_t a_) : a(a_) {}

            void run() {
                flare::parallel_for("FillTensor_1D", policy_t(0, a.extent(0)), *this);
            }

            FLARE_INLINE_FUNCTION
            void operator()(int i) const { a(i) = i; }
        };

        template<class Layout, class Space>
        struct FillTensor_3D {
            using exec_t = typename Space::execution_space;
            using tensor_t = flare::Tensor<int ***, Layout, Space>;
            using rank_t = flare::Rank<
                    tensor_t::rank,
                    std::is_same<Layout, flare::LayoutLeft>::value ? flare::Iterate::Left
                                                                   : flare::Iterate::Right,
                    std::is_same<Layout, flare::LayoutLeft>::value ? flare::Iterate::Left
                                                                   : flare::Iterate::Right>;
            using policy_t = flare::MDRangePolicy<exec_t, rank_t>;

            tensor_t a;

            FillTensor_3D(tensor_t a_) : a(a_) {}

            void run() {
                flare::parallel_for(
                        "FillTensor_3D",
                        policy_t({0, 0, 0}, {a.extent(0), a.extent(1), a.extent(2)}), *this);
            }

            FLARE_INLINE_FUNCTION
            void operator()(int i0, int i1, int i2) const {
                a(i0, i1, i2) = 1000000 * i0 + 1000 * i1 + i2;
            }
        };

        template<class Layout, class Space>
        struct FillTensor_4D {
            using exec_t = typename Space::execution_space;
            using tensor_t = flare::Tensor<int ****, Layout, Space>;
            using rank_t = flare::Rank<
                    tensor_t::rank,
                    std::is_same<Layout, flare::LayoutLeft>::value ? flare::Iterate::Left
                                                                   : flare::Iterate::Right,
                    std::is_same<Layout, flare::LayoutLeft>::value ? flare::Iterate::Left
                                                                   : flare::Iterate::Right>;
            using policy_t = flare::MDRangePolicy<exec_t, rank_t>;

            tensor_t a;

            FillTensor_4D(tensor_t a_) : a(a_) {}

            void run() {
                flare::parallel_for("FillTensor_4D",
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
        struct FillTensor_5D {
            using exec_t = typename Space::execution_space;
            using tensor_t = flare::Tensor<int *****, Layout, Space>;
            using rank_t = flare::Rank<
                    tensor_t::rank,
                    std::is_same<Layout, flare::LayoutLeft>::value ? flare::Iterate::Left
                                                                   : flare::Iterate::Right,
                    std::is_same<Layout, flare::LayoutLeft>::value ? flare::Iterate::Left
                                                                   : flare::Iterate::Right>;
            using policy_t = flare::MDRangePolicy<exec_t, rank_t>;

            tensor_t a;

            FillTensor_5D(tensor_t a_) : a(a_) {}

            void run() {
                flare::parallel_for(
                        "FillTensor_5D",
                        policy_t({0, 0, 0, 0, 0}, {a.extent(0), a.extent(1), a.extent(2),
                                                   a.extent(3), a.extent(4)}),
                        *this);
            }

            FLARE_INLINE_FUNCTION
            void operator()(int i0, int i1, int i2, int i3, int i4) const {
                a(i0, i1, i2, i3, i4) = 1000000 * i0 + 10000 * i1 + 100 * i2 + 10 * i3 + i4;
            }
        };

        template<class Tensor, class SubTensor>
        struct CheckSubtensorCorrectness_1D_1D {
            using policy_t = flare::RangePolicy<typename Tensor::execution_space>;
            Tensor a;
            SubTensor b;
            int offset;

            CheckSubtensorCorrectness_1D_1D(Tensor a_, SubTensor b_, int o)
                    : a(a_), b(b_), offset(o) {}

            void run() {
                int errors = 0;
                flare::parallel_reduce("CheckSubTensor_1D_1D", policy_t(0, b.size()), *this,
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

        template<class Tensor, class SubTensor>
        struct CheckSubtensorCorrectness_1D_2D {
            using policy_t = flare::RangePolicy<typename Tensor::execution_space>;
            Tensor a;
            SubTensor b;
            int i0;
            int offset;

            CheckSubtensorCorrectness_1D_2D(Tensor a_, SubTensor b_, int i0_, int o)
                    : a(a_), b(b_), i0(i0_), offset(o) {}

            void run() {
                int errors = 0;
                flare::parallel_reduce("CheckSubTensor_1D_2D", policy_t(0, b.size()), *this,
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

        template<class Tensor, class SubTensor>
        struct CheckSubtensorCorrectness_2D_3D {
            using policy_t = flare::RangePolicy<typename Tensor::execution_space>;
            using layout = typename Tensor::array_layout;
            Tensor a;
            SubTensor b;
            int i0;
            int offset_1;
            int offset_2;

            CheckSubtensorCorrectness_2D_3D(Tensor a_, SubTensor b_, int i0_, int o1, int o2)
                    : a(a_), b(b_), i0(i0_), offset_1(o1), offset_2(o2) {}

            void run() {
                int errors = 0;
                flare::parallel_reduce("CheckSubTensor_2D_3D", policy_t(0, b.size()), *this,
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

        template<class Tensor, class SubTensor>
        struct CheckSubtensorCorrectness_3D_3D {
            using policy_t = flare::RangePolicy<typename Tensor::execution_space>;
            using layout = typename Tensor::array_layout;
            Tensor a;
            SubTensor b;
            int offset_0;
            int offset_2;

            CheckSubtensorCorrectness_3D_3D(Tensor a_, SubTensor b_, int o0, int o2)
                    : a(a_), b(b_), offset_0(o0), offset_2(o2) {}

            void run() {
                int errors = 0;
                flare::parallel_reduce("CheckSubTensor_3D_3D", policy_t(0, b.size()), *this,
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

        template<class Tensor, class SubTensor>
        struct CheckSubtensorCorrectness_3D_4D {
            using policy_t = flare::RangePolicy<typename Tensor::execution_space>;
            using layout = typename Tensor::array_layout;
            Tensor a;
            SubTensor b;
            int index;
            int offset_0, offset_2;

            CheckSubtensorCorrectness_3D_4D(Tensor a_, SubTensor b_, int index_, int o0, int o2)
                    : a(a_), b(b_), index(index_), offset_0(o0), offset_2(o2) {}

            void run() {
                int errors = 0;
                flare::parallel_reduce("CheckSubTensor_3D_4D", policy_t(0, b.size()), *this,
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

        template<class Tensor, class SubTensor>
        struct CheckSubtensorCorrectness_3D_5D {
            using policy_t = flare::RangePolicy<typename Tensor::execution_space>;
            using layout = typename Tensor::array_layout;
            Tensor a;
            SubTensor b;
            int i0, i1;
            int offset_2, offset_3, offset_4;

            CheckSubtensorCorrectness_3D_5D(Tensor a_, SubTensor b_, int i0_, int i1_, int o2,
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
                flare::parallel_reduce("CheckSubTensor_3D_5D", policy_t(0, b.size()), *this,
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

        template<class SubTensor, class Tensor>
        void test_Check1D(SubTensor a, Tensor b, flare::pair<int, int> range) {
            CheckSubtensorCorrectness_1D_1D<Tensor, SubTensor> check(b, a, range.first);
            check.run();
        }

        template<class SubTensor, class Tensor>
        void test_Check1D2D(SubTensor a, Tensor b, int i0, std::pair<int, int> range) {
            CheckSubtensorCorrectness_1D_2D<Tensor, SubTensor> check(b, a, i0, range.first);
            check.run();
        }

        template<class SubTensor, class Tensor>
        void test_Check2D3D(SubTensor a, Tensor b, int i0, std::pair<int, int> range1,
                            std::pair<int, int> range2) {
            CheckSubtensorCorrectness_2D_3D<Tensor, SubTensor> check(b, a, i0, range1.first,
                                                               range2.first);
            check.run();
        }

        template<class SubTensor, class Tensor>
        void test_Check3D5D(SubTensor a, Tensor b, int i0, int i1,
                            flare::pair<int, int> range2,
                            flare::pair<int, int> range3,
                            flare::pair<int, int> range4) {
            CheckSubtensorCorrectness_3D_5D<Tensor, SubTensor> check(
                    b, a, i0, i1, range2.first, range3.first, range4.first);
            check.run();
        }

        template<class Space, class LayoutSub, class Layout, class LayoutOrg,
                class MemTraits>
        void test_1d_assign_impl() {
            {  // Breaks.
                flare::Tensor<int *, LayoutOrg, Space> a_org("A", N0);
                flare::Tensor<int *, LayoutOrg, Space, MemTraits> a(a_org);
                flare::fence();

                detail::FillTensor_1D<LayoutOrg, Space> fill(a_org);
                fill.run();

                flare::Tensor<int[N0], Layout, Space, MemTraits> a1(a);
                flare::fence();
                test_Check1D(a1, a, std::pair<int, int>(0, N0));

                flare::Tensor<int[N0], LayoutSub, Space, MemTraits> a2(a1);
                flare::fence();
                test_Check1D(a2, a, std::pair<int, int>(0, N0));
                a1 = a;
                test_Check1D(a1, a, std::pair<int, int>(0, N0));

                // Runtime Fail expected.
                // flare::Tensor< int[N1] > afail1( a );

                // Compile Time Fail expected.
                // flare::Tensor< int[N1] > afail2( a1 );
            }

            {  // Works.
                flare::Tensor<int[N0], LayoutOrg, Space, MemTraits> a("A");
                flare::Tensor<int *, Layout, Space, MemTraits> a1(a);
                flare::fence();
                test_Check1D(a1, a, std::pair<int, int>(0, N0));
                a1 = a;
                flare::fence();
                test_Check1D(a1, a, std::pair<int, int>(0, N0));
            }
        }

        template<class Space, class Type, class TypeSub, class LayoutSub, class Layout,
                class LayoutOrg, class MemTraits>
        void test_2d_subtensor_3d_impl_type() {
            flare::Tensor<int ***, LayoutOrg, Space> a_org("A", N0, N1, N2);
            flare::Tensor<Type, Layout, Space, MemTraits> a(a_org);

            detail::FillTensor_3D<LayoutOrg, Space> fill(a_org);
            fill.run();

            flare::Tensor<TypeSub, LayoutSub, Space, MemTraits> a1;
            a1 = flare::subtensor(a, 3, flare::ALL, flare::ALL);
            flare::fence();
            test_Check2D3D(a1, a, 3, std::pair<int, int>(0, N1),
                           std::pair<int, int>(0, N2));

            flare::Tensor<TypeSub, LayoutSub, Space, MemTraits> a2(a, 3, flare::ALL,
                                                                 flare::ALL);
            flare::fence();
            test_Check2D3D(a2, a, 3, std::pair<int, int>(0, N1),
                           std::pair<int, int>(0, N2));
        }

        template<class Space, class LayoutSub, class Layout, class LayoutOrg,
                class MemTraits>
        void test_2d_subtensor_3d_impl_layout() {
            test_2d_subtensor_3d_impl_type<Space, int[N0][N1][N2], int[N1][N2], LayoutSub,
                    Layout, LayoutOrg, MemTraits>();
            test_2d_subtensor_3d_impl_type<Space, int[N0][N1][N2], int *[N2], LayoutSub,
                    Layout, LayoutOrg, MemTraits>();
            test_2d_subtensor_3d_impl_type<Space, int[N0][N1][N2], int **, LayoutSub, Layout,
                    LayoutOrg, MemTraits>();

            test_2d_subtensor_3d_impl_type<Space, int *[N1][N2], int[N1][N2], LayoutSub,
                    Layout, LayoutOrg, MemTraits>();
            test_2d_subtensor_3d_impl_type<Space, int *[N1][N2], int *[N2], LayoutSub,
                    Layout, LayoutOrg, MemTraits>();
            test_2d_subtensor_3d_impl_type<Space, int *[N1][N2], int **, LayoutSub, Layout,
                    LayoutOrg, MemTraits>();

            test_2d_subtensor_3d_impl_type<Space, int **[N2], int[N1][N2], LayoutSub,
                    Layout, LayoutOrg, MemTraits>();
            test_2d_subtensor_3d_impl_type<Space, int **[N2], int *[N2], LayoutSub,
                    Layout, LayoutOrg, MemTraits>();
            test_2d_subtensor_3d_impl_type<Space, int **[N2], int **, LayoutSub, Layout,
                    LayoutOrg, MemTraits>();

            test_2d_subtensor_3d_impl_type<Space, int ***, int[N1][N2], LayoutSub, Layout,
                    LayoutOrg, MemTraits>();
            test_2d_subtensor_3d_impl_type<Space, int ***, int *[N2], LayoutSub, Layout,
                    LayoutOrg, MemTraits>();
            test_2d_subtensor_3d_impl_type<Space, int ***, int **, LayoutSub, Layout,
                    LayoutOrg, MemTraits>();

            test_2d_subtensor_3d_impl_type<Space, const int[N0][N1][N2], const int[N1][N2],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_2d_subtensor_3d_impl_type<Space, const int[N0][N1][N2], const int *[N2],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_2d_subtensor_3d_impl_type<Space, const int[N0][N1][N2], const int **,
                    LayoutSub, Layout, LayoutOrg, MemTraits>();

            test_2d_subtensor_3d_impl_type<Space, const int *[N1][N2], const int[N1][N2],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_2d_subtensor_3d_impl_type<Space, const int *[N1][N2], const int *[N2],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_2d_subtensor_3d_impl_type<Space, const int *[N1][N2], const int **,
                    LayoutSub, Layout, LayoutOrg, MemTraits>();

            test_2d_subtensor_3d_impl_type<Space, const int **[N2], const int[N1][N2],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_2d_subtensor_3d_impl_type<Space, const int **[N2], const int *[N2],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_2d_subtensor_3d_impl_type<Space, const int **[N2], const int **, LayoutSub,
                    Layout, LayoutOrg, MemTraits>();

            test_2d_subtensor_3d_impl_type<Space, const int ***, const int[N1][N2],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_2d_subtensor_3d_impl_type<Space, const int ***, const int *[N2], LayoutSub,
                    Layout, LayoutOrg, MemTraits>();
            test_2d_subtensor_3d_impl_type<Space, const int ***, const int **, LayoutSub,
                    Layout, LayoutOrg, MemTraits>();
        }

        template<class Space, class Type, class TypeSub, class LayoutSub, class Layout,
                class LayoutOrg, class MemTraits>
        void test_3d_subtensor_5d_impl_type() {
            flare::Tensor<int *****, LayoutOrg, Space> a_org("A", N0, N1, N2, N3, N4);
            flare::Tensor<Type, Layout, Space, MemTraits> a(a_org);

            detail::FillTensor_5D<LayoutOrg, Space> fill(a_org);
            fill.run();

            flare::Tensor<TypeSub, LayoutSub, Space, MemTraits> a1;
            a1 = flare::subtensor(a, 3, 5, flare::ALL, flare::ALL, flare::ALL);
            flare::fence();
            test_Check3D5D(a1, a, 3, 5, std::pair<int, int>(0, N2),
                           std::pair<int, int>(0, N3), std::pair<int, int>(0, N4));

            flare::Tensor<TypeSub, LayoutSub, Space, MemTraits> a2(
                    a, 3, 5, flare::ALL, flare::ALL, flare::ALL);
            flare::fence();
            test_Check3D5D(a2, a, 3, 5, std::pair<int, int>(0, N2),
                           std::pair<int, int>(0, N3), std::pair<int, int>(0, N4));
        }

        template<class Space, class LayoutSub, class Layout, class LayoutOrg,
                class MemTraits>
        void test_3d_subtensor_5d_impl_layout() {
            test_3d_subtensor_5d_impl_type<Space, int[N0][N1][N2][N3][N4], int[N2][N3][N4],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_3d_subtensor_5d_impl_type<Space, int[N0][N1][N2][N3][N4], int *[N3][N4],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_3d_subtensor_5d_impl_type<Space, int[N0][N1][N2][N3][N4], int **[N4],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_3d_subtensor_5d_impl_type<Space, int[N0][N1][N2][N3][N4], int ***,
                    LayoutSub, Layout, LayoutOrg, MemTraits>();

            test_3d_subtensor_5d_impl_type<Space, int *[N1][N2][N3][N4], int[N2][N3][N4],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_3d_subtensor_5d_impl_type<Space, int *[N1][N2][N3][N4], int *[N3][N4],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_3d_subtensor_5d_impl_type<Space, int *[N1][N2][N3][N4], int **[N4],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_3d_subtensor_5d_impl_type<Space, int *[N1][N2][N3][N4], int ***, LayoutSub,
                    Layout, LayoutOrg, MemTraits>();

            test_3d_subtensor_5d_impl_type<Space, int **[N2][N3][N4], int[N2][N3][N4],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_3d_subtensor_5d_impl_type<Space, int **[N2][N3][N4], int *[N3][N4],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_3d_subtensor_5d_impl_type<Space, int **[N2][N3][N4], int **[N4],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_3d_subtensor_5d_impl_type<Space, int **[N2][N3][N4], int ***, LayoutSub,
                    Layout, LayoutOrg, MemTraits>();

            test_3d_subtensor_5d_impl_type<Space, int ***[N3][N4], int[N2][N3][N4],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_3d_subtensor_5d_impl_type<Space, int ***[N3][N4], int *[N3][N4],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_3d_subtensor_5d_impl_type<Space, int ***[N3][N4], int **[N4], LayoutSub,
                    Layout, LayoutOrg, MemTraits>();
            test_3d_subtensor_5d_impl_type<Space, int ***[N3][N4], int ***, LayoutSub,
                    Layout, LayoutOrg, MemTraits>();

            test_3d_subtensor_5d_impl_type<Space, int ****[N4], int[N2][N3][N4], LayoutSub,
                    Layout, LayoutOrg, MemTraits>();
            test_3d_subtensor_5d_impl_type<Space, int ****[N4], int *[N3][N4], LayoutSub,
                    Layout, LayoutOrg, MemTraits>();
            test_3d_subtensor_5d_impl_type<Space, int ****[N4], int **[N4], LayoutSub,
                    Layout, LayoutOrg, MemTraits>();
            test_3d_subtensor_5d_impl_type<Space, int ****[N4], int ***, LayoutSub, Layout,
                    LayoutOrg, MemTraits>();

            test_3d_subtensor_5d_impl_type<Space, int *****, int[N2][N3][N4], LayoutSub,
                    Layout, LayoutOrg, MemTraits>();
            test_3d_subtensor_5d_impl_type<Space, int *****, int *[N3][N4], LayoutSub,
                    Layout, LayoutOrg, MemTraits>();
            test_3d_subtensor_5d_impl_type<Space, int *****, int **[N4], LayoutSub, Layout,
                    LayoutOrg, MemTraits>();
            test_3d_subtensor_5d_impl_type<Space, int *****, int ***, LayoutSub, Layout,
                    LayoutOrg, MemTraits>();

            test_3d_subtensor_5d_impl_type<Space, const int[N0][N1][N2][N3][N4],
                    const int[N2][N3][N4], LayoutSub, Layout,
                    LayoutOrg, MemTraits>();
            test_3d_subtensor_5d_impl_type<Space, const int[N0][N1][N2][N3][N4],
                    const int *[N3][N4], LayoutSub, Layout,
                    LayoutOrg, MemTraits>();
            test_3d_subtensor_5d_impl_type<Space, const int[N0][N1][N2][N3][N4],
                    const int **[N4], LayoutSub, Layout, LayoutOrg,
                    MemTraits>();
            test_3d_subtensor_5d_impl_type<Space, const int[N0][N1][N2][N3][N4],
                    const int ***, LayoutSub, Layout, LayoutOrg,
                    MemTraits>();

            test_3d_subtensor_5d_impl_type<Space, const int *[N1][N2][N3][N4],
                    const int[N2][N3][N4], LayoutSub, Layout,
                    LayoutOrg, MemTraits>();
            test_3d_subtensor_5d_impl_type<Space, const int *[N1][N2][N3][N4],
                    const int *[N3][N4], LayoutSub, Layout,
                    LayoutOrg, MemTraits>();
            test_3d_subtensor_5d_impl_type<Space, const int *[N1][N2][N3][N4],
                    const int **[N4], LayoutSub, Layout, LayoutOrg,
                    MemTraits>();
            test_3d_subtensor_5d_impl_type<Space, const int *[N1][N2][N3][N4],
                    const int ***, LayoutSub, Layout, LayoutOrg,
                    MemTraits>();

            test_3d_subtensor_5d_impl_type<Space, const int **[N2][N3][N4],
                    const int[N2][N3][N4], LayoutSub, Layout,
                    LayoutOrg, MemTraits>();
            test_3d_subtensor_5d_impl_type<Space, const int **[N2][N3][N4],
                    const int *[N3][N4], LayoutSub, Layout,
                    LayoutOrg, MemTraits>();
            test_3d_subtensor_5d_impl_type<Space, const int **[N2][N3][N4],
                    const int **[N4], LayoutSub, Layout, LayoutOrg,
                    MemTraits>();
            test_3d_subtensor_5d_impl_type<Space, const int **[N2][N3][N4], const int ***,
                    LayoutSub, Layout, LayoutOrg, MemTraits>();

            test_3d_subtensor_5d_impl_type<Space, const int ***[N3][N4],
                    const int[N2][N3][N4], LayoutSub, Layout,
                    LayoutOrg, MemTraits>();
            test_3d_subtensor_5d_impl_type<Space, const int ***[N3][N4],
                    const int *[N3][N4], LayoutSub, Layout,
                    LayoutOrg, MemTraits>();
            test_3d_subtensor_5d_impl_type<Space, const int ***[N3][N4], const int **[N4],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_3d_subtensor_5d_impl_type<Space, const int ***[N3][N4], const int ***,
                    LayoutSub, Layout, LayoutOrg, MemTraits>();

            test_3d_subtensor_5d_impl_type<Space, const int ****[N4],
                    const int[N2][N3][N4], LayoutSub, Layout,
                    LayoutOrg, MemTraits>();
            test_3d_subtensor_5d_impl_type<Space, const int ****[N4], const int *[N3][N4],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_3d_subtensor_5d_impl_type<Space, const int ****[N4], const int **[N4],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_3d_subtensor_5d_impl_type<Space, const int ****[N4], const int ***,
                    LayoutSub, Layout, LayoutOrg, MemTraits>();

            test_3d_subtensor_5d_impl_type<Space, const int *****, const int[N2][N3][N4],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_3d_subtensor_5d_impl_type<Space, const int *****, const int *[N3][N4],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_3d_subtensor_5d_impl_type<Space, const int *****, const int **[N4],
                    LayoutSub, Layout, LayoutOrg, MemTraits>();
            test_3d_subtensor_5d_impl_type<Space, const int *****, const int ***, LayoutSub,
                    Layout, LayoutOrg, MemTraits>();
        }

        inline void test_subtensor_legal_args_right() {
            REQUIRE_EQ(
                    0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, flare::ALL_t,
                    flare::ALL_t, flare::pair<int, int>, int, int>::value));
            REQUIRE_EQ(
                    0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, flare::ALL_t,
                    flare::ALL_t, flare::ALL_t, int, int>::value));
            REQUIRE_EQ(
                    0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, flare::ALL_t,
                    flare::pair<int, int>, flare::pair<int, int>, int, int>::value));
            REQUIRE_EQ(
                    0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, flare::ALL_t,
                    flare::pair<int, int>, flare::ALL_t, int, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0,
                    flare::pair<int, int>, flare::ALL_t,
                    flare::pair<int, int>, int, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0,
                    flare::pair<int, int>, flare::ALL_t, flare::ALL_t, int,
                    int>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0,
                    flare::pair<int, int>, flare::pair<int, int>,
                    flare::pair<int, int>, int, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0,
                    flare::pair<int, int>, flare::pair<int, int>,
                    flare::ALL_t, int, int>::value));

            REQUIRE_EQ(
                    0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, flare::ALL_t,
                    int, flare::ALL_t, flare::pair<int, int>, int>::value));
            REQUIRE_EQ(
                    0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, flare::ALL_t,
                    int, flare::ALL_t, flare::ALL_t, int>::value));
            REQUIRE_EQ(
                    0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, flare::ALL_t,
                    int, flare::pair<int, int>, flare::pair<int, int>, int>::value));
            REQUIRE_EQ(
                    0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, flare::ALL_t,
                    int, flare::pair<int, int>, flare::ALL_t, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0,
                    flare::pair<int, int>, int, flare::ALL_t,
                    flare::pair<int, int>, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0,
                    flare::pair<int, int>, int, flare::ALL_t, flare::ALL_t,
                    int>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0,
                    flare::pair<int, int>, int, flare::pair<int, int>,
                    flare::pair<int, int>, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0,
                    flare::pair<int, int>, int, flare::ALL_t,
                    flare::pair<int, int>, int>::value));

            REQUIRE_EQ(
                    0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, flare::ALL_t,
                    flare::ALL_t, int, flare::pair<int, int>, int>::value));
            REQUIRE_EQ(
                    0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, flare::ALL_t,
                    flare::ALL_t, int, flare::ALL_t, int>::value));
            REQUIRE_EQ(
                    0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, flare::ALL_t,
                    flare::pair<int, int>, int, flare::pair<int, int>, int>::value));
            REQUIRE_EQ(
                    0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, flare::ALL_t,
                    flare::pair<int, int>, int, flare::ALL_t, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0,
                    flare::pair<int, int>, flare::ALL_t, int,
                    flare::pair<int, int>, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0,
                    flare::pair<int, int>, flare::ALL_t, int, flare::ALL_t,
                    int>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0,
                    flare::pair<int, int>, flare::pair<int, int>, int,
                    flare::pair<int, int>, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0,
                    flare::pair<int, int>, flare::ALL_t, int,
                    flare::pair<int, int>, int>::value));

            REQUIRE_EQ(
                    0,
                    (flare::detail::SubtensorLegalArgsCompileTime<
                            flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int, flare::ALL_t,
                            flare::ALL_t, flare::pair<int, int>, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int,
                    flare::ALL_t, flare::ALL_t, flare::ALL_t, int>::value));
            REQUIRE_EQ(
                    0,
                    (flare::detail::SubtensorLegalArgsCompileTime<
                            flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int, flare::ALL_t,
                            flare::pair<int, int>, flare::pair<int, int>, int>::value));
            REQUIRE_EQ(
                    0,
                    (flare::detail::SubtensorLegalArgsCompileTime<
                            flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int, flare::ALL_t,
                            flare::pair<int, int>, flare::ALL_t, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int,
                    flare::pair<int, int>, flare::ALL_t,
                    flare::pair<int, int>, int>::value));
            REQUIRE_EQ(
                    0,
                    (flare::detail::SubtensorLegalArgsCompileTime<
                            flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int,
                            flare::pair<int, int>, flare::ALL_t, flare::ALL_t, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int,
                    flare::pair<int, int>, flare::pair<int, int>,
                    flare::pair<int, int>, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int,
                    flare::pair<int, int>, flare::pair<int, int>,
                    flare::ALL_t, int>::value));

            REQUIRE_EQ(
                    0,
                    (flare::detail::SubtensorLegalArgsCompileTime<
                            flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int, flare::ALL_t,
                            flare::ALL_t, int, flare::pair<int, int>>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int,
                    flare::ALL_t, flare::ALL_t, int, flare::ALL_t>::value));
            REQUIRE_EQ(
                    0,
                    (flare::detail::SubtensorLegalArgsCompileTime<
                            flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int, flare::ALL_t,
                            flare::pair<int, int>, int, flare::pair<int, int>>::value));
            REQUIRE_EQ(
                    0,
                    (flare::detail::SubtensorLegalArgsCompileTime<
                            flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int, flare::ALL_t,
                            flare::pair<int, int>, int, flare::ALL_t>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int,
                    flare::pair<int, int>, flare::ALL_t, int,
                    flare::pair<int, int>>::value));
            REQUIRE_EQ(
                    0,
                    (flare::detail::SubtensorLegalArgsCompileTime<
                            flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int,
                            flare::pair<int, int>, flare::ALL_t, int, flare::ALL_t>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int,
                    flare::pair<int, int>, flare::pair<int, int>, int,
                    flare::pair<int, int>>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int,
                    flare::pair<int, int>, flare::pair<int, int>, int,
                    flare::ALL_t>::value));

            REQUIRE_EQ(0,
                       (flare::detail::SubtensorLegalArgsCompileTime<
                               flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int, int,
                               flare::ALL_t, flare::ALL_t, flare::pair<int, int>>::value));
            REQUIRE_EQ(1, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int, int,
                    flare::ALL_t, flare::ALL_t, flare::ALL_t>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int, int,
                    flare::ALL_t, flare::pair<int, int>,
                    flare::pair<int, int>>::value));
            REQUIRE_EQ(0,
                       (flare::detail::SubtensorLegalArgsCompileTime<
                               flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int, int,
                               flare::ALL_t, flare::pair<int, int>, flare::ALL_t>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int, int,
                    flare::pair<int, int>, flare::ALL_t,
                    flare::pair<int, int>>::value));
            REQUIRE_EQ(1,
                       (flare::detail::SubtensorLegalArgsCompileTime<
                               flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int, int,
                               flare::pair<int, int>, flare::ALL_t, flare::ALL_t>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int, int,
                    flare::pair<int, int>, flare::pair<int, int>,
                    flare::pair<int, int>>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 5, 0, int, int,
                    flare::pair<int, int>, flare::pair<int, int>,
                    flare::ALL_t>::value));

            REQUIRE_EQ(1, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 3, 0,
                    flare::ALL_t, flare::ALL_t, flare::ALL_t>::value));
            REQUIRE_EQ(
                    0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 3, 0, flare::ALL_t,
                    flare::ALL_t, flare::pair<int, int>>::value));
            REQUIRE_EQ(1,
                       (flare::detail::SubtensorLegalArgsCompileTime<
                               flare::LayoutRight, flare::LayoutRight, 3, 3, 0,
                               flare::pair<int, int>, flare::ALL_t, flare::ALL_t>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 3, 0,
                    flare::pair<int, int>, flare::ALL_t,
                    flare::pair<int, int>>::value));
            REQUIRE_EQ(
                    0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 3, 0, flare::ALL_t,
                    flare::pair<int, int>, flare::ALL_t>::value));
            REQUIRE_EQ(
                    0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 3, 0, flare::ALL_t,
                    flare::pair<int, int>, flare::pair<int, int>>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 3, 0,
                    flare::pair<int, int>, flare::pair<int, int>,
                    flare::ALL_t>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutRight, flare::LayoutRight, 3, 3, 0,
                    flare::pair<int, int>, flare::pair<int, int>,
                    flare::pair<int, int>>::value));
        }

        inline void test_subtensor_legal_args_left() {
            REQUIRE_EQ(1,
                       (flare::detail::SubtensorLegalArgsCompileTime<
                               flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, flare::ALL_t,
                               flare::ALL_t, flare::pair<int, int>, int, int>::value));
            REQUIRE_EQ(1,
                       (flare::detail::SubtensorLegalArgsCompileTime<
                               flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, flare::ALL_t,
                               flare::ALL_t, flare::ALL_t, int, int>::value));
            REQUIRE_EQ(
                    0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, flare::ALL_t,
                    flare::pair<int, int>, flare::pair<int, int>, int, int>::value));
            REQUIRE_EQ(0,
                       (flare::detail::SubtensorLegalArgsCompileTime<
                               flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, flare::ALL_t,
                               flare::pair<int, int>, flare::ALL_t, int, int>::value));
            REQUIRE_EQ(1, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0,
                    flare::pair<int, int>, flare::ALL_t,
                    flare::pair<int, int>, int, int>::value));
            REQUIRE_EQ(1, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0,
                    flare::pair<int, int>, flare::ALL_t, flare::ALL_t, int,
                    int>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0,
                    flare::pair<int, int>, flare::pair<int, int>,
                    flare::pair<int, int>, int, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0,
                    flare::pair<int, int>, flare::pair<int, int>,
                    flare::ALL_t, int, int>::value));

            REQUIRE_EQ(0,
                       (flare::detail::SubtensorLegalArgsCompileTime<
                               flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, flare::ALL_t,
                               int, flare::ALL_t, flare::pair<int, int>, int>::value));
            REQUIRE_EQ(0,
                       (flare::detail::SubtensorLegalArgsCompileTime<
                               flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, flare::ALL_t,
                               int, flare::ALL_t, flare::ALL_t, int>::value));
            REQUIRE_EQ(
                    0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, flare::ALL_t,
                    int, flare::pair<int, int>, flare::pair<int, int>, int>::value));
            REQUIRE_EQ(0,
                       (flare::detail::SubtensorLegalArgsCompileTime<
                               flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, flare::ALL_t,
                               int, flare::pair<int, int>, flare::ALL_t, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0,
                    flare::pair<int, int>, int, flare::ALL_t,
                    flare::pair<int, int>, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0,
                    flare::pair<int, int>, int, flare::ALL_t, flare::ALL_t,
                    int>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0,
                    flare::pair<int, int>, int, flare::pair<int, int>,
                    flare::pair<int, int>, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0,
                    flare::pair<int, int>, int, flare::ALL_t,
                    flare::pair<int, int>, int>::value));

            REQUIRE_EQ(0,
                       (flare::detail::SubtensorLegalArgsCompileTime<
                               flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, flare::ALL_t,
                               flare::ALL_t, int, flare::pair<int, int>, int>::value));
            REQUIRE_EQ(0,
                       (flare::detail::SubtensorLegalArgsCompileTime<
                               flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, flare::ALL_t,
                               flare::ALL_t, int, flare::ALL_t, int>::value));
            REQUIRE_EQ(
                    0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, flare::ALL_t,
                    flare::pair<int, int>, int, flare::pair<int, int>, int>::value));
            REQUIRE_EQ(0,
                       (flare::detail::SubtensorLegalArgsCompileTime<
                               flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, flare::ALL_t,
                               flare::pair<int, int>, int, flare::ALL_t, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0,
                    flare::pair<int, int>, flare::ALL_t, int,
                    flare::pair<int, int>, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0,
                    flare::pair<int, int>, flare::ALL_t, int, flare::ALL_t,
                    int>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0,
                    flare::pair<int, int>, flare::pair<int, int>, int,
                    flare::pair<int, int>, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0,
                    flare::pair<int, int>, flare::ALL_t, int,
                    flare::pair<int, int>, int>::value));

            REQUIRE_EQ(
                    0,
                    (flare::detail::SubtensorLegalArgsCompileTime<
                            flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int, flare::ALL_t,
                            flare::ALL_t, flare::pair<int, int>, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int,
                    flare::ALL_t, flare::ALL_t, flare::ALL_t, int>::value));
            REQUIRE_EQ(
                    0,
                    (flare::detail::SubtensorLegalArgsCompileTime<
                            flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int, flare::ALL_t,
                            flare::pair<int, int>, flare::pair<int, int>, int>::value));
            REQUIRE_EQ(
                    0,
                    (flare::detail::SubtensorLegalArgsCompileTime<
                            flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int, flare::ALL_t,
                            flare::pair<int, int>, flare::ALL_t, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int,
                    flare::pair<int, int>, flare::ALL_t,
                    flare::pair<int, int>, int>::value));
            REQUIRE_EQ(
                    0,
                    (flare::detail::SubtensorLegalArgsCompileTime<
                            flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int,
                            flare::pair<int, int>, flare::ALL_t, flare::ALL_t, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int,
                    flare::pair<int, int>, flare::pair<int, int>,
                    flare::pair<int, int>, int>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int,
                    flare::pair<int, int>, flare::pair<int, int>,
                    flare::ALL_t, int>::value));

            REQUIRE_EQ(
                    0,
                    (flare::detail::SubtensorLegalArgsCompileTime<
                            flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int, flare::ALL_t,
                            flare::ALL_t, int, flare::pair<int, int>>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int,
                    flare::ALL_t, flare::ALL_t, int, flare::ALL_t>::value));
            REQUIRE_EQ(
                    0,
                    (flare::detail::SubtensorLegalArgsCompileTime<
                            flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int, flare::ALL_t,
                            flare::pair<int, int>, int, flare::pair<int, int>>::value));
            REQUIRE_EQ(
                    0,
                    (flare::detail::SubtensorLegalArgsCompileTime<
                            flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int, flare::ALL_t,
                            flare::pair<int, int>, int, flare::ALL_t>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int,
                    flare::pair<int, int>, flare::ALL_t, int,
                    flare::pair<int, int>>::value));
            REQUIRE_EQ(
                    0,
                    (flare::detail::SubtensorLegalArgsCompileTime<
                            flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int,
                            flare::pair<int, int>, flare::ALL_t, int, flare::ALL_t>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int,
                    flare::pair<int, int>, flare::pair<int, int>, int,
                    flare::pair<int, int>>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int,
                    flare::pair<int, int>, flare::pair<int, int>, int,
                    flare::ALL_t>::value));

            REQUIRE_EQ(0,
                       (flare::detail::SubtensorLegalArgsCompileTime<
                               flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int, int,
                               flare::ALL_t, flare::ALL_t, flare::pair<int, int>>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int, int,
                    flare::ALL_t, flare::ALL_t, flare::ALL_t>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int, int,
                    flare::ALL_t, flare::pair<int, int>,
                    flare::pair<int, int>>::value));
            REQUIRE_EQ(0,
                       (flare::detail::SubtensorLegalArgsCompileTime<
                               flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int, int,
                               flare::ALL_t, flare::pair<int, int>, flare::ALL_t>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int, int,
                    flare::pair<int, int>, flare::ALL_t,
                    flare::pair<int, int>>::value));
            REQUIRE_EQ(0,
                       (flare::detail::SubtensorLegalArgsCompileTime<
                               flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int, int,
                               flare::pair<int, int>, flare::ALL_t, flare::ALL_t>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int, int,
                    flare::pair<int, int>, flare::pair<int, int>,
                    flare::pair<int, int>>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 5, 0, int, int,
                    flare::pair<int, int>, flare::pair<int, int>,
                    flare::ALL_t>::value));

            REQUIRE_EQ(1,
                       (flare::detail::SubtensorLegalArgsCompileTime<
                               flare::LayoutLeft, flare::LayoutLeft, 3, 3, 0, flare::ALL_t,
                               flare::ALL_t, flare::pair<int, int>>::value));
            REQUIRE_EQ(1, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 3, 0,
                    flare::ALL_t, flare::ALL_t, flare::ALL_t>::value));
            REQUIRE_EQ(1, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 3, 0,
                    flare::pair<int, int>, flare::ALL_t,
                    flare::pair<int, int>>::value));
            REQUIRE_EQ(1,
                       (flare::detail::SubtensorLegalArgsCompileTime<
                               flare::LayoutLeft, flare::LayoutLeft, 3, 3, 0,
                               flare::pair<int, int>, flare::ALL_t, flare::ALL_t>::value));
            REQUIRE_EQ(0,
                       (flare::detail::SubtensorLegalArgsCompileTime<
                               flare::LayoutLeft, flare::LayoutLeft, 3, 3, 0, flare::ALL_t,
                               flare::pair<int, int>, flare::ALL_t>::value));
            REQUIRE_EQ(0,
                       (flare::detail::SubtensorLegalArgsCompileTime<
                               flare::LayoutLeft, flare::LayoutLeft, 3, 3, 0, flare::ALL_t,
                               flare::pair<int, int>, flare::pair<int, int>>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
                    flare::LayoutLeft, flare::LayoutLeft, 3, 3, 0,
                    flare::pair<int, int>, flare::pair<int, int>,
                    flare::ALL_t>::value));
            REQUIRE_EQ(0, (flare::detail::SubtensorLegalArgsCompileTime<
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
    void test_2d_subtensor_3d() {
        detail::test_2d_subtensor_3d_impl_layout<Space, flare::LayoutRight,
                flare::LayoutRight, flare::LayoutRight,
                MemTraits>();
        detail::test_2d_subtensor_3d_impl_layout<Space, flare::LayoutStride,
                flare::LayoutRight, flare::LayoutRight,
                MemTraits>();
        detail::test_2d_subtensor_3d_impl_layout<Space, flare::LayoutStride,
                flare::LayoutStride,
                flare::LayoutRight, MemTraits>();
        detail::test_2d_subtensor_3d_impl_layout<Space, flare::LayoutStride,
                flare::LayoutLeft, flare::LayoutLeft,
                MemTraits>();
        detail::test_2d_subtensor_3d_impl_layout<Space, flare::LayoutStride,
                flare::LayoutStride, flare::LayoutLeft,
                MemTraits>();
    }

    template<class Space, class MemTraits = void>
    void test_3d_subtensor_5d_right() {
        detail::test_3d_subtensor_5d_impl_layout<Space, flare::LayoutStride,
                flare::LayoutRight, flare::LayoutRight,
                MemTraits>();
        detail::test_3d_subtensor_5d_impl_layout<Space, flare::LayoutStride,
                flare::LayoutStride,
                flare::LayoutRight, MemTraits>();
    }

    template<class Space, class MemTraits = void>
    void test_3d_subtensor_5d_left() {
        detail::test_3d_subtensor_5d_impl_layout<Space, flare::LayoutStride,
                flare::LayoutLeft, flare::LayoutLeft,
                MemTraits>();
        detail::test_3d_subtensor_5d_impl_layout<Space, flare::LayoutStride,
                flare::LayoutStride, flare::LayoutLeft,
                MemTraits>();
    }

    template<class Space, class MemTraits = void>
    void test_layoutleft_to_layoutleft() {
        detail::test_subtensor_legal_args_left();

        using tensor3D_t = flare::Tensor<int ***, flare::LayoutLeft, Space>;
        using tensor4D_t = flare::Tensor<int ****, flare::LayoutLeft, Space>;
        {
            tensor3D_t a("A", 100, 4, 3);
            tensor3D_t b(a, flare::pair<int, int>(16, 32), flare::ALL, flare::ALL);

            detail::FillTensor_3D<flare::LayoutLeft, Space> fill(a);
            fill.run();

            detail::CheckSubtensorCorrectness_3D_3D<tensor3D_t, tensor3D_t> check(a, b, 16, 0);
            check.run();
        }

        {
            tensor3D_t a("A", 100, 4, 5);
            tensor3D_t b(a, flare::pair<int, int>(16, 32), flare::ALL,
                       flare::pair<int, int>(1, 3));

            detail::FillTensor_3D<flare::LayoutLeft, Space> fill(a);
            fill.run();

            detail::CheckSubtensorCorrectness_3D_3D<tensor3D_t, tensor3D_t> check(a, b, 16, 1);
            check.run();
        }

        {
            tensor4D_t a("A", 100, 4, 5, 3);
            tensor3D_t b(a, flare::pair<int, int>(16, 32), flare::ALL,
                       flare::pair<int, int>(1, 3), 1);

            detail::FillTensor_4D<flare::LayoutLeft, Space> fill(a);
            fill.run();

            detail::CheckSubtensorCorrectness_3D_4D<tensor4D_t, tensor3D_t> check(a, b, 1, 16,
                                                                            1);
            check.run();
        }
    }

    template<class Space, class MemTraits = void>
    void test_layoutright_to_layoutright() {
        detail::test_subtensor_legal_args_right();

        using tensor3D_t = flare::Tensor<int ***, flare::LayoutRight, Space>;
        using tensor4D_t = flare::Tensor<int ****, flare::LayoutRight, Space>;
        {
            tensor3D_t a("A", 100, 4, 3);
            tensor3D_t b(a, flare::pair<int, int>(16, 32), flare::ALL, flare::ALL);

            detail::FillTensor_3D<flare::LayoutRight, Space> fill(a);
            fill.run();

            detail::CheckSubtensorCorrectness_3D_3D<tensor3D_t, tensor3D_t> check(a, b, 16, 0);
            check.run();
        }
        {
            tensor4D_t a("A", 3, 4, 5, 100);
            tensor3D_t b(a, 1, flare::pair<int, int>(1, 3), flare::ALL, flare::ALL);

            detail::FillTensor_4D<flare::LayoutRight, Space> fill(a);
            fill.run();

            detail::CheckSubtensorCorrectness_3D_4D<tensor4D_t, tensor3D_t> check(a, b, 1, 1,
                                                                            0);
            check.run();
        }
    }
//----------------------------------------------------------------------------

    template<class Space>
    struct TestUnmanagedSubtensorReset {
        flare::Tensor<int ****, Space> a;

        FLARE_INLINE_FUNCTION
        void operator()(int) const noexcept {
            auto sub_a = flare::subtensor(a, 0, flare::ALL, flare::ALL, flare::ALL);

            for (int i = 0; i < int(a.extent(0)); ++i) {
                sub_a.assign_data(&a(i, 0, 0, 0));
                if (&sub_a(1, 1, 1) != &a(i, 1, 1, 1)) {
                    flare::abort("TestUnmanagedSubtensorReset");
                }
            }
        }

        TestUnmanagedSubtensorReset() : a(flare::tensor_alloc(), 20, 10, 5, 2) {}
    };

    template<class Space>
    void test_unmanaged_subtensor_reset() {
        flare::parallel_for(
                flare::RangePolicy<typename Space::execution_space>(0, 1),
                TestUnmanagedSubtensorReset<Space>());
    }

//----------------------------------------------------------------------------

    template<std::underlying_type_t<flare::MemoryTraitsFlags> MTF>
    struct TestSubtensorMemoryTraitsConstruction {
        void operator()() const noexcept {
            using memory_traits_type = flare::MemoryTraits<MTF>;
            using tensor_type =
                    flare::Tensor<double *, flare::HostSpace, memory_traits_type>;
            using size_type = typename tensor_type::size_type;

            // Create a managed Tensor first and then apply the desired memory traits to
            // an unmanaged version of it since a managed Tensor can't use the Unmanaged
            // trait.
            flare::Tensor<double *, flare::HostSpace> v_original("v", 7);
            tensor_type v(v_original.data(), v_original.size());
            for (size_type i = 0; i != v.size(); ++i) v[i] = static_cast<double>(i);

            std::pair<int, int> range(3, 5);
            auto sv = flare::subtensor(v, range);

            // check that the subtensor memory traits are the same as the original tensor
            // (with the Aligned trait stripped).
            using tensor_memory_traits = typename decltype(v)::memory_traits;
            using subtensor_memory_traits = typename decltype(sv)::memory_traits;
            static_assert(tensor_memory_traits::impl_value ==
                          memory_traits_type::impl_value);
            if constexpr (memory_traits_type::is_aligned)
                static_assert(subtensor_memory_traits::impl_value + flare::Aligned ==
                              memory_traits_type::impl_value);
            else
                static_assert(subtensor_memory_traits::impl_value ==
                              memory_traits_type::impl_value);

            REQUIRE_EQ(2u, sv.size());
            REQUIRE_EQ(3., sv[0]);
            REQUIRE_EQ(4., sv[1]);
        }
    };

    inline void test_subtensor_memory_traits_construction() {
        // Test all combinations of MemoryTraits:
        // Unmanaged (1)
        // RandomAccess (2)
        // Atomic (4)
        // Restricted (8)
        // Aligned (16)
        TestSubtensorMemoryTraitsConstruction<0>()();
        TestSubtensorMemoryTraitsConstruction<1>()();
        TestSubtensorMemoryTraitsConstruction<2>()();
        TestSubtensorMemoryTraitsConstruction<3>()();
        TestSubtensorMemoryTraitsConstruction<4>()();
        TestSubtensorMemoryTraitsConstruction<5>()();
        TestSubtensorMemoryTraitsConstruction<6>()();
        TestSubtensorMemoryTraitsConstruction<7>()();
        TestSubtensorMemoryTraitsConstruction<8>()();
        TestSubtensorMemoryTraitsConstruction<9>()();
        TestSubtensorMemoryTraitsConstruction<10>()();
        TestSubtensorMemoryTraitsConstruction<11>()();
        TestSubtensorMemoryTraitsConstruction<12>()();
        TestSubtensorMemoryTraitsConstruction<13>()();
        TestSubtensorMemoryTraitsConstruction<14>()();
        TestSubtensorMemoryTraitsConstruction<15>()();
        TestSubtensorMemoryTraitsConstruction<16>()();
        TestSubtensorMemoryTraitsConstruction<17>()();
        TestSubtensorMemoryTraitsConstruction<18>()();
        TestSubtensorMemoryTraitsConstruction<19>()();
        TestSubtensorMemoryTraitsConstruction<20>()();
        TestSubtensorMemoryTraitsConstruction<21>()();
        TestSubtensorMemoryTraitsConstruction<22>()();
        TestSubtensorMemoryTraitsConstruction<23>()();
        TestSubtensorMemoryTraitsConstruction<24>()();
        TestSubtensorMemoryTraitsConstruction<25>()();
        TestSubtensorMemoryTraitsConstruction<26>()();
        TestSubtensorMemoryTraitsConstruction<27>()();
        TestSubtensorMemoryTraitsConstruction<28>()();
        TestSubtensorMemoryTraitsConstruction<29>()();
        TestSubtensorMemoryTraitsConstruction<30>()();
        TestSubtensorMemoryTraitsConstruction<31>()();
    }

//----------------------------------------------------------------------------

    template<class T>
    struct get_tensor_type;

    template<class T, class... Args>
    struct get_tensor_type<flare::Tensor<T, Args...>> {
        using type = T;
    };

    template<class T>
    struct
    ___________________________________TYPE_DISPLAY________________________________________;
#define TYPE_DISPLAY(...)                                                                           \
  typename ___________________________________TYPE_DISPLAY________________________________________< \
      __VA_ARGS__>::type notdefined;

    template<class Space, class Layout>
    struct TestSubtensorStaticSizes {
        flare::Tensor<int *[10][5][2], Layout, Space> a;
        flare::Tensor<int[6][7][8], Layout, Space> b;

        FLARE_INLINE_FUNCTION
        int operator()() const noexcept {
            /* Doesn't actually do anything; just static assertions */

            auto sub_a = flare::subtensor(a, 0, flare::ALL, flare::ALL, flare::ALL);
            typename static_expect_same<
                    /* expected */ int[10][5][2],
                    /*  actual  */ typename get_tensor_type<decltype(sub_a)>::type>::type
                    test_1 = 0;

            auto sub_a_2 = flare::subtensor(a, 0, 0, flare::ALL, flare::ALL);
            typename static_expect_same<
                    /* expected */ int[5][2],
                    /*  actual  */ typename get_tensor_type<decltype(sub_a_2)>::type>::type
                    test_2 = 0;

            auto sub_a_3 = flare::subtensor(a, 0, 0, flare::ALL, 0);
            typename static_expect_same<
                    /* expected */ int[5],
                    /*  actual  */ typename get_tensor_type<decltype(sub_a_3)>::type>::type
                    test_3 = 0;

            auto sub_a_4 = flare::subtensor(a, flare::ALL, 0, flare::ALL, flare::ALL);
            typename static_expect_same<
                    /* expected */ int *[5][2],
                    /*  actual  */ typename get_tensor_type<decltype(sub_a_4)>::type>::type
                    test_4 = 0;

            // TODO we'll need to update this test once we allow interleaving of static
            // and dynamic
            auto sub_a_5 = flare::subtensor(a, flare::ALL, 0, flare::ALL,
                                          flare::make_pair(0, 1));
            typename static_expect_same<
                    /* expected */ int ***,
                    /*  actual  */ typename get_tensor_type<decltype(sub_a_5)>::type>::type
                    test_5 = 0;

            auto sub_a_sub = flare::subtensor(sub_a_5, 0, flare::ALL, 0);
            typename static_expect_same<
                    /* expected */ int *,
                    /*  actual  */ typename get_tensor_type<decltype(sub_a_sub)>::type>::type
                    test_sub = 0;

            auto sub_a_7 = flare::subtensor(a, flare::ALL, 0, flare::make_pair(0, 1),
                                          flare::ALL);
            typename static_expect_same<
                    /* expected */ int **[2],
                    /*  actual  */ typename get_tensor_type<decltype(sub_a_7)>::type>::type
                    test_7 = 0;

            auto sub_a_8 =
                    flare::subtensor(a, flare::ALL, flare::ALL, flare::ALL, flare::ALL);
            typename static_expect_same<
                    /* expected */ int *[10][5][2],
                    /*  actual  */ typename get_tensor_type<decltype(sub_a_8)>::type>::type
                    test_8 = 0;

            auto sub_b = flare::subtensor(b, flare::ALL, flare::ALL, flare::ALL);
            typename static_expect_same<
                    /* expected */ int[6][7][8],
                    /*  actual  */ typename get_tensor_type<decltype(sub_b)>::type>::type
                    test_9 = 0;

            auto sub_b_2 = flare::subtensor(b, 0, flare::ALL, flare::ALL);
            typename static_expect_same<
                    /* expected */ int[7][8],
                    /*  actual  */ typename get_tensor_type<decltype(sub_b_2)>::type>::type
                    test_10 = 0;

            auto sub_b_3 =
                    flare::subtensor(b, flare::make_pair(2, 3), flare::ALL, flare::ALL);
            typename static_expect_same<
                    /* expected */ int *[7][8],
                    /*  actual  */ typename get_tensor_type<decltype(sub_b_3)>::type>::type
                    test_11 = 0;

            return test_1 + test_2 + test_3 + test_4 + test_5 + test_sub + test_7 +
                   test_8 + test_9 + test_10 + test_11;
        }

        TestSubtensorStaticSizes() : a(flare::tensor_alloc("a"), 20), b("b") {}
    };

    template<class Space>
    struct TestExtentsStaticTests {
        using test1 = typename static_expect_same<
                /* expected */
                flare::experimental::Extents<flare::experimental::dynamic_extent,
                        flare::experimental::dynamic_extent, 1, 2,
                        3>,
                /* actual */
                typename flare::detail::ParseTensorExtents<double **[1][2][3]>::type>::type;

        using test2 = typename static_expect_same<
                /* expected */
                flare::experimental::Extents<1, 2, 3>,
                /* actual */
                typename flare::detail::ParseTensorExtents<double[1][2][3]>::type>::type;

        using test3 = typename static_expect_same<
                /* expected */
                flare::experimental::Extents<3>,
                /* actual */
                typename flare::detail::ParseTensorExtents<double[3]>::type>::type;

        using test4 = typename static_expect_same<
                /* expected */
                flare::experimental::Extents<>,
                /* actual */
                typename flare::detail::ParseTensorExtents<double>::type>::type;
    };

}  // namespace TestTensorSubtensor
