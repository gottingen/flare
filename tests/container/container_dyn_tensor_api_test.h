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
#include <flare/dyn_rank_tensor.h>

namespace Test {

    template<class T, class... P>
    size_t allocation_count(const flare::DynRankTensor<T, P...> &tensor) {
        const size_t card = tensor.size();
        const size_t alloc = tensor.span();

        return card <= alloc ? alloc : 0;
    }

    template<typename T, class DeviceType>
    struct TestTensorOperator {
        using execution_space = DeviceType;

        static const unsigned N = 100;
        static const unsigned D = 3;

        using tensor_type = flare::DynRankTensor<T, execution_space>;

        const tensor_type v1;
        const tensor_type v2;

        TestTensorOperator() : v1("v1", N, D), v2("v2", N, D) {}

        static void testit() { flare::parallel_for(N, TestTensorOperator()); }

        FLARE_INLINE_FUNCTION
        void operator()(const unsigned i) const {
            const unsigned X = 0;
            const unsigned Y = 1;
            const unsigned Z = 2;

            v2(i, X) = v1(i, X);
            v2(i, Y) = v1(i, Y);
            v2(i, Z) = v1(i, Z);
        }
    };

    template<class DataType, class DeviceType, unsigned Rank>
    struct TestTensorOperator_LeftAndRight;

    template<class DataType, class DeviceType>
    struct TestTensorOperator_LeftAndRight<DataType, DeviceType, 7> {
        using execution_space = DeviceType;
        using memory_space = typename execution_space::memory_space;
        using size_type = typename execution_space::size_type;

        using value_type = int;

        FLARE_INLINE_FUNCTION
        static void join(value_type &update, const value_type &input) {
            update |= input;
        }

        FLARE_INLINE_FUNCTION
        static void init(value_type &update) { update = 0; }

        using left_tensor =
                flare::DynRankTensor<DataType, flare::LayoutLeft, execution_space>;

        using right_tensor =
                flare::DynRankTensor<DataType, flare::LayoutRight, execution_space>;

        left_tensor left;
        right_tensor right;
        long left_alloc;
        long right_alloc;

        TestTensorOperator_LeftAndRight(unsigned N0, unsigned N1, unsigned N2,
                                      unsigned N3, unsigned N4, unsigned N5,
                                      unsigned N6)
                : left("left", N0, N1, N2, N3, N4, N5, N6),
                  right("right", N0, N1, N2, N3, N4, N5, N6),
                  left_alloc(allocation_count(left)),
                  right_alloc(allocation_count(right)) {}

        static void testit(unsigned N0, unsigned N1, unsigned N2, unsigned N3,
                           unsigned N4, unsigned N5, unsigned N6) {
            TestTensorOperator_LeftAndRight driver(N0, N1, N2, N3, N4, N5, N6);

            int error_flag = 0;

            flare::parallel_reduce(1, driver, error_flag);

            REQUIRE_EQ(error_flag, 0);
        }

        FLARE_INLINE_FUNCTION
        void operator()(const size_type, value_type &update) const {
            long offset;

            offset = -1;
            for (unsigned i6 = 0; i6 < unsigned(left.extent(6)); ++i6)
                for (unsigned i5 = 0; i5 < unsigned(left.extent(5)); ++i5)
                    for (unsigned i4 = 0; i4 < unsigned(left.extent(4)); ++i4)
                        for (unsigned i3 = 0; i3 < unsigned(left.extent(3)); ++i3)
                            for (unsigned i2 = 0; i2 < unsigned(left.extent(2)); ++i2)
                                for (unsigned i1 = 0; i1 < unsigned(left.extent(1)); ++i1)
                                    for (unsigned i0 = 0; i0 < unsigned(left.extent(0)); ++i0) {
                                        const long j = &left(i0, i1, i2, i3, i4, i5, i6) -
                                                       &left(0, 0, 0, 0, 0, 0, 0);
                                        if (j <= offset || left_alloc <= j) {
                                            update |= 1;
                                        }
                                        offset = j;
                                    }

            offset = -1;
            for (unsigned i0 = 0; i0 < unsigned(right.extent(0)); ++i0)
                for (unsigned i1 = 0; i1 < unsigned(right.extent(1)); ++i1)
                    for (unsigned i2 = 0; i2 < unsigned(right.extent(2)); ++i2)
                        for (unsigned i3 = 0; i3 < unsigned(right.extent(3)); ++i3)
                            for (unsigned i4 = 0; i4 < unsigned(right.extent(4)); ++i4)
                                for (unsigned i5 = 0; i5 < unsigned(right.extent(5)); ++i5)
                                    for (unsigned i6 = 0; i6 < unsigned(right.extent(6)); ++i6) {
                                        const long j = &right(i0, i1, i2, i3, i4, i5, i6) -
                                                       &right(0, 0, 0, 0, 0, 0, 0);
                                        if (j <= offset || right_alloc <= j) {
                                            update |= 2;
                                        }
                                        offset = j;
                                    }
        }
    };

    template<class DataType, class DeviceType>
    struct TestTensorOperator_LeftAndRight<DataType, DeviceType, 6> {
        using execution_space = DeviceType;
        using memory_space = typename execution_space::memory_space;
        using size_type = typename execution_space::size_type;

        using value_type = int;

        FLARE_INLINE_FUNCTION
        static void join(value_type &update, const value_type &input) {
            update |= input;
        }

        FLARE_INLINE_FUNCTION
        static void init(value_type &update) { update = 0; }

        using left_tensor =
                flare::DynRankTensor<DataType, flare::LayoutLeft, execution_space>;

        using right_tensor =
                flare::DynRankTensor<DataType, flare::LayoutRight, execution_space>;

        left_tensor left;
        right_tensor right;
        long left_alloc;
        long right_alloc;

        TestTensorOperator_LeftAndRight(unsigned N0, unsigned N1, unsigned N2,
                                      unsigned N3, unsigned N4, unsigned N5)
                : left("left", N0, N1, N2, N3, N4, N5),
                  right("right", N0, N1, N2, N3, N4, N5),
                  left_alloc(allocation_count(left)),
                  right_alloc(allocation_count(right)) {}

        static void testit(unsigned N0, unsigned N1, unsigned N2, unsigned N3,
                           unsigned N4, unsigned N5) {
            TestTensorOperator_LeftAndRight driver(N0, N1, N2, N3, N4, N5);

            int error_flag = 0;

            flare::parallel_reduce(1, driver, error_flag);

            REQUIRE_EQ(error_flag, 0);
        }

        FLARE_INLINE_FUNCTION
        void operator()(const size_type, value_type &update) const {
            long offset;

            offset = -1;
            for (unsigned i5 = 0; i5 < unsigned(left.extent(5)); ++i5)
                for (unsigned i4 = 0; i4 < unsigned(left.extent(4)); ++i4)
                    for (unsigned i3 = 0; i3 < unsigned(left.extent(3)); ++i3)
                        for (unsigned i2 = 0; i2 < unsigned(left.extent(2)); ++i2)
                            for (unsigned i1 = 0; i1 < unsigned(left.extent(1)); ++i1)
                                for (unsigned i0 = 0; i0 < unsigned(left.extent(0)); ++i0) {
                                    const long j =
                                            &left(i0, i1, i2, i3, i4, i5) - &left(0, 0, 0, 0, 0, 0);
                                    if (j <= offset || left_alloc <= j) {
                                        update |= 1;
                                    }
                                    offset = j;
                                }

            offset = -1;
            for (unsigned i0 = 0; i0 < unsigned(right.extent(0)); ++i0)
                for (unsigned i1 = 0; i1 < unsigned(right.extent(1)); ++i1)
                    for (unsigned i2 = 0; i2 < unsigned(right.extent(2)); ++i2)
                        for (unsigned i3 = 0; i3 < unsigned(right.extent(3)); ++i3)
                            for (unsigned i4 = 0; i4 < unsigned(right.extent(4)); ++i4)
                                for (unsigned i5 = 0; i5 < unsigned(right.extent(5)); ++i5) {
                                    const long j =
                                            &right(i0, i1, i2, i3, i4, i5) - &right(0, 0, 0, 0, 0, 0);
                                    if (j <= offset || right_alloc <= j) {
                                        update |= 2;
                                    }
                                    offset = j;
                                }
        }
    };

    template<class DataType, class DeviceType>
    struct TestTensorOperator_LeftAndRight<DataType, DeviceType, 5> {
        using execution_space = DeviceType;
        using memory_space = typename execution_space::memory_space;
        using size_type = typename execution_space::size_type;

        using value_type = int;

        FLARE_INLINE_FUNCTION
        static void join(value_type &update, const value_type &input) {
            update |= input;
        }

        FLARE_INLINE_FUNCTION
        static void init(value_type &update) { update = 0; }

        using left_tensor =
                flare::DynRankTensor<DataType, flare::LayoutLeft, execution_space>;

        using right_tensor =
                flare::DynRankTensor<DataType, flare::LayoutRight, execution_space>;

        using stride_tensor =
                flare::DynRankTensor<DataType, flare::LayoutStride, execution_space>;

        left_tensor left;
        right_tensor right;
        stride_tensor left_stride;
        stride_tensor right_stride;
        long left_alloc;
        long right_alloc;

        TestTensorOperator_LeftAndRight(unsigned N0, unsigned N1, unsigned N2,
                                      unsigned N3, unsigned N4)
                : left("left", N0, N1, N2, N3, N4),
                  right("right", N0, N1, N2, N3, N4),
                  left_stride(left),
                  right_stride(right),
                  left_alloc(allocation_count(left)),
                  right_alloc(allocation_count(right)) {}

        static void testit(unsigned N0, unsigned N1, unsigned N2, unsigned N3,
                           unsigned N4) {
            TestTensorOperator_LeftAndRight driver(N0, N1, N2, N3, N4);

            int error_flag = 0;

            flare::parallel_reduce(1, driver, error_flag);

            REQUIRE_EQ(error_flag, 0);
        }

        FLARE_INLINE_FUNCTION
        void operator()(const size_type, value_type &update) const {
            long offset;

            offset = -1;
            for (unsigned i4 = 0; i4 < unsigned(left.extent(4)); ++i4)
                for (unsigned i3 = 0; i3 < unsigned(left.extent(3)); ++i3)
                    for (unsigned i2 = 0; i2 < unsigned(left.extent(2)); ++i2)
                        for (unsigned i1 = 0; i1 < unsigned(left.extent(1)); ++i1)
                            for (unsigned i0 = 0; i0 < unsigned(left.extent(0)); ++i0) {
                                const long j = &left(i0, i1, i2, i3, i4) - &left(0, 0, 0, 0, 0);
                                if (j <= offset || left_alloc <= j) {
                                    update |= 1;
                                }
                                offset = j;

                                if (&left(i0, i1, i2, i3, i4) !=
                                    &left_stride(i0, i1, i2, i3, i4)) {
                                    update |= 4;
                                }
                            }

            offset = -1;
            for (unsigned i0 = 0; i0 < unsigned(right.extent(0)); ++i0)
                for (unsigned i1 = 0; i1 < unsigned(right.extent(1)); ++i1)
                    for (unsigned i2 = 0; i2 < unsigned(right.extent(2)); ++i2)
                        for (unsigned i3 = 0; i3 < unsigned(right.extent(3)); ++i3)
                            for (unsigned i4 = 0; i4 < unsigned(right.extent(4)); ++i4) {
                                const long j = &right(i0, i1, i2, i3, i4) - &right(0, 0, 0, 0, 0);
                                if (j <= offset || right_alloc <= j) {
                                    update |= 2;
                                }
                                offset = j;

                                if (&right(i0, i1, i2, i3, i4) !=
                                    &right_stride(i0, i1, i2, i3, i4)) {
                                    update |= 8;
                                }
                            }
        }
    };

    template<class DataType, class DeviceType>
    struct TestTensorOperator_LeftAndRight<DataType, DeviceType, 4> {
        using execution_space = DeviceType;
        using memory_space = typename execution_space::memory_space;
        using size_type = typename execution_space::size_type;

        using value_type = int;

        FLARE_INLINE_FUNCTION
        static void join(value_type &update, const value_type &input) {
            update |= input;
        }

        FLARE_INLINE_FUNCTION
        static void init(value_type &update) { update = 0; }

        using left_tensor =
                flare::DynRankTensor<DataType, flare::LayoutLeft, execution_space>;

        using right_tensor =
                flare::DynRankTensor<DataType, flare::LayoutRight, execution_space>;

        left_tensor left;
        right_tensor right;
        long left_alloc;
        long right_alloc;

        TestTensorOperator_LeftAndRight(unsigned N0, unsigned N1, unsigned N2,
                                      unsigned N3)
                : left("left", N0, N1, N2, N3),
                  right("right", N0, N1, N2, N3),
                  left_alloc(allocation_count(left)),
                  right_alloc(allocation_count(right)) {}

        static void testit(unsigned N0, unsigned N1, unsigned N2, unsigned N3) {
            TestTensorOperator_LeftAndRight driver(N0, N1, N2, N3);

            int error_flag = 0;

            flare::parallel_reduce(1, driver, error_flag);

            REQUIRE_EQ(error_flag, 0);
        }

        FLARE_INLINE_FUNCTION
        void operator()(const size_type, value_type &update) const {
            long offset;

            offset = -1;
            for (unsigned i3 = 0; i3 < unsigned(left.extent(3)); ++i3)
                for (unsigned i2 = 0; i2 < unsigned(left.extent(2)); ++i2)
                    for (unsigned i1 = 0; i1 < unsigned(left.extent(1)); ++i1)
                        for (unsigned i0 = 0; i0 < unsigned(left.extent(0)); ++i0) {
                            const long j = &left(i0, i1, i2, i3) - &left(0, 0, 0, 0);
                            if (j <= offset || left_alloc <= j) {
                                update |= 1;
                            }
                            offset = j;
                        }

            offset = -1;
            for (unsigned i0 = 0; i0 < unsigned(right.extent(0)); ++i0)
                for (unsigned i1 = 0; i1 < unsigned(right.extent(1)); ++i1)
                    for (unsigned i2 = 0; i2 < unsigned(right.extent(2)); ++i2)
                        for (unsigned i3 = 0; i3 < unsigned(right.extent(3)); ++i3) {
                            const long j = &right(i0, i1, i2, i3) - &right(0, 0, 0, 0);
                            if (j <= offset || right_alloc <= j) {
                                update |= 2;
                            }
                            offset = j;
                        }
        }
    };

    template<class DataType, class DeviceType>
    struct TestTensorOperator_LeftAndRight<DataType, DeviceType, 3> {
        using execution_space = DeviceType;
        using memory_space = typename execution_space::memory_space;
        using size_type = typename execution_space::size_type;

        using value_type = int;

        FLARE_INLINE_FUNCTION
        static void join(value_type &update, const value_type &input) {
            update |= input;
        }

        FLARE_INLINE_FUNCTION
        static void init(value_type &update) { update = 0; }

        using left_tensor =
                flare::DynRankTensor<DataType, flare::LayoutLeft, execution_space>;

        using right_tensor =
                flare::DynRankTensor<DataType, flare::LayoutRight, execution_space>;

        using stride_tensor =
                flare::DynRankTensor<DataType, flare::LayoutStride, execution_space>;

        left_tensor left;
        right_tensor right;
        stride_tensor left_stride;
        stride_tensor right_stride;
        long left_alloc;
        long right_alloc;

        TestTensorOperator_LeftAndRight(unsigned N0, unsigned N1, unsigned N2)
                : left(std::string("left"), N0, N1, N2),
                  right(std::string("right"), N0, N1, N2),
                  left_stride(left),
                  right_stride(right),
                  left_alloc(allocation_count(left)),
                  right_alloc(allocation_count(right)) {}

        static void testit(unsigned N0, unsigned N1, unsigned N2) {
            TestTensorOperator_LeftAndRight driver(N0, N1, N2);

            int error_flag = 0;

            flare::parallel_reduce(1, driver, error_flag);

            REQUIRE_EQ(error_flag, 0);
        }

        FLARE_INLINE_FUNCTION
        void operator()(const size_type, value_type &update) const {
            long offset;

            offset = -1;
            for (unsigned i2 = 0; i2 < unsigned(left.extent(2)); ++i2)
                for (unsigned i1 = 0; i1 < unsigned(left.extent(1)); ++i1)
                    for (unsigned i0 = 0; i0 < unsigned(left.extent(0)); ++i0) {
                        const long j = &left(i0, i1, i2) - &left(0, 0, 0);
                        if (j <= offset || left_alloc <= j) {
                            update |= 1;
                        }
                        offset = j;

                        if (&left(i0, i1, i2) != &left_stride(i0, i1, i2)) {
                            update |= 4;
                        }
                    }

            offset = -1;
            for (unsigned i0 = 0; i0 < unsigned(right.extent(0)); ++i0)
                for (unsigned i1 = 0; i1 < unsigned(right.extent(1)); ++i1)
                    for (unsigned i2 = 0; i2 < unsigned(right.extent(2)); ++i2) {
                        const long j = &right(i0, i1, i2) - &right(0, 0, 0);
                        if (j <= offset || right_alloc <= j) {
                            update |= 2;
                        }
                        offset = j;

                        if (&right(i0, i1, i2) != &right_stride(i0, i1, i2)) {
                            update |= 8;
                        }
                    }

            for (unsigned i0 = 0; i0 < unsigned(left.extent(0)); ++i0)
                for (unsigned i1 = 0; i1 < unsigned(left.extent(1)); ++i1)
                    for (unsigned i2 = 0; i2 < unsigned(left.extent(2)); ++i2) {
                        if (&left(i0, i1, i2) != &left(i0, i1, i2, 0, 0, 0, 0)) {
                            update |= 3;
                        }
                        if (&right(i0, i1, i2) != &right(i0, i1, i2, 0, 0, 0, 0)) {
                            update |= 3;
                        }
                    }
        }
    };

    template<class DataType, class DeviceType>
    struct TestTensorOperator_LeftAndRight<DataType, DeviceType, 2> {
        using execution_space = DeviceType;
        using memory_space = typename execution_space::memory_space;
        using size_type = typename execution_space::size_type;

        using value_type = int;

        FLARE_INLINE_FUNCTION
        static void join(value_type &update, const value_type &input) {
            update |= input;
        }

        FLARE_INLINE_FUNCTION
        static void init(value_type &update) { update = 0; }

        using left_tensor =
                flare::DynRankTensor<DataType, flare::LayoutLeft, execution_space>;

        using right_tensor =
                flare::DynRankTensor<DataType, flare::LayoutRight, execution_space>;

        left_tensor left;
        right_tensor right;
        long left_alloc;
        long right_alloc;

        TestTensorOperator_LeftAndRight(unsigned N0, unsigned N1)
                : left("left", N0, N1),
                  right("right", N0, N1),
                  left_alloc(allocation_count(left)),
                  right_alloc(allocation_count(right)) {}

        static void testit(unsigned N0, unsigned N1) {
            TestTensorOperator_LeftAndRight driver(N0, N1);

            int error_flag = 0;

            flare::parallel_reduce(1, driver, error_flag);

            REQUIRE_EQ(error_flag, 0);
        }

        FLARE_INLINE_FUNCTION
        void operator()(const size_type, value_type &update) const {
            long offset;

            offset = -1;
            for (unsigned i1 = 0; i1 < unsigned(left.extent(1)); ++i1)
                for (unsigned i0 = 0; i0 < unsigned(left.extent(0)); ++i0) {
                    const long j = &left(i0, i1) - &left(0, 0);
                    if (j <= offset || left_alloc <= j) {
                        update |= 1;
                    }
                    offset = j;
                }

            offset = -1;
            for (unsigned i0 = 0; i0 < unsigned(right.extent(0)); ++i0)
                for (unsigned i1 = 0; i1 < unsigned(right.extent(1)); ++i1) {
                    const long j = &right(i0, i1) - &right(0, 0);
                    if (j <= offset || right_alloc <= j) {
                        update |= 2;
                    }
                    offset = j;
                }

            for (unsigned i0 = 0; i0 < unsigned(left.extent(0)); ++i0)
                for (unsigned i1 = 0; i1 < unsigned(left.extent(1)); ++i1) {
                    if (&left(i0, i1) != &left(i0, i1, 0, 0, 0, 0, 0)) {
                        update |= 3;
                    }
                    if (&right(i0, i1) != &right(i0, i1, 0, 0, 0, 0, 0)) {
                        update |= 3;
                    }
                }
        }
    };

    template<class DataType, class DeviceType>
    struct TestTensorOperator_LeftAndRight<DataType, DeviceType, 1> {
        using execution_space = DeviceType;
        using memory_space = typename execution_space::memory_space;
        using size_type = typename execution_space::size_type;

        using value_type = int;

        FLARE_INLINE_FUNCTION
        static void join(value_type &update, const value_type &input) {
            update |= input;
        }

        FLARE_INLINE_FUNCTION
        static void init(value_type &update) { update = 0; }

        using left_tensor =
                flare::DynRankTensor<DataType, flare::LayoutLeft, execution_space>;

        using right_tensor =
                flare::DynRankTensor<DataType, flare::LayoutRight, execution_space>;

        using stride_tensor =
                flare::DynRankTensor<DataType, flare::LayoutStride, execution_space>;

        left_tensor left;
        right_tensor right;
        stride_tensor left_stride;
        stride_tensor right_stride;
        long left_alloc;
        long right_alloc;

        TestTensorOperator_LeftAndRight(unsigned N0)
                : left("left", N0),
                  right("right", N0),
                  left_stride(left),
                  right_stride(right),
                  left_alloc(allocation_count(left)),
                  right_alloc(allocation_count(right)) {}

        static void testit(unsigned N0) {
            TestTensorOperator_LeftAndRight driver(N0);

            int error_flag = 0;

            flare::parallel_reduce(1, driver, error_flag);

            REQUIRE_EQ(error_flag, 0);
        }

        FLARE_INLINE_FUNCTION
        void operator()(const size_type, value_type &update) const {
            for (unsigned i0 = 0; i0 < unsigned(left.extent(0)); ++i0) {
                if (&left(i0) != &left(i0, 0, 0, 0, 0, 0, 0)) {
                    update |= 3;
                }
                if (&right(i0) != &right(i0, 0, 0, 0, 0, 0, 0)) {
                    update |= 3;
                }
                if (&left(i0) != &left_stride(i0)) {
                    update |= 4;
                }
                if (&right(i0) != &right_stride(i0)) {
                    update |= 8;
                }
            }
        }
    };

/*--------------------------------------------------------------------------*/

    template<typename T, class DeviceType>
    class TestDynTensorAPI {
    public:
        using device = DeviceType;

        enum {
            N0 = 1000, N1 = 3, N2 = 5, N3 = 7
        };

        using DTensor0 = flare::DynRankTensor<T, device>;
        using const_DTensor0 = flare::DynRankTensor<const T, device>;

        using DTensor0_unmanaged =
                flare::DynRankTensor<T, device, flare::MemoryUnmanaged>;
        using host_drv_space = typename DTensor0::host_mirror_space;

        using Tensor0 = flare::Tensor<T, device>;
        using Tensor1 = flare::Tensor<T *, device>;
        using Tensor2 = flare::Tensor<T **, device>;
        using Tensor3 = flare::Tensor<T ***, device>;
        using Tensor4 = flare::Tensor<T ****, device>;
        using Tensor5 = flare::Tensor<T *****, device>;
        using Tensor6 = flare::Tensor<T ******, device>;
        using Tensor7 = flare::Tensor<T *******, device>;

        using host_tensor_space = typename Tensor0::host_mirror_space;

        static void run_tests() {
            run_test_resize_realloc<false>();
            run_test_resize_realloc<true>();
            run_test_mirror();
            run_test_mirror_and_copy();
            run_test_scalar();
            run_test();
            run_test_allocated();
            run_test_const();
            run_test_subtensor();
            run_test_subtensor_strided();
            run_test_vector();
            run_test_as_tensor_of_rank_n();
            run_test_layout();
        }

        static void run_operator_test_rank12345() {
            TestTensorOperator<T, device>::testit();
            TestTensorOperator_LeftAndRight<int, device, 5>::testit(2, 3, 4, 2, 3);
            TestTensorOperator_LeftAndRight<int, device, 4>::testit(2, 3, 4, 2);
            TestTensorOperator_LeftAndRight<int, device, 3>::testit(2, 3, 4);
            TestTensorOperator_LeftAndRight<int, device, 2>::testit(2, 3);
            TestTensorOperator_LeftAndRight<int, device, 1>::testit(2);
        }

        static void run_operator_test_rank67() {
            TestTensorOperator_LeftAndRight<int, device, 7>::testit(2, 3, 4, 2, 3, 4, 2);
            TestTensorOperator_LeftAndRight<int, device, 6>::testit(2, 3, 4, 2, 3, 4);
        }

        template<bool Initialize>
        static void run_test_resize_realloc() {
            DTensor0 drv0("drv0", 10, 20, 30);
            REQUIRE_EQ(drv0.rank(), 3u);

            if (Initialize)
                flare::resize(flare::WithoutInitializing, drv0, 5, 10);
            else
                flare::resize(drv0, 5, 10);
            REQUIRE_EQ(drv0.rank(), 2u);
            REQUIRE_EQ(drv0.extent(0), 5u);
            REQUIRE_EQ(drv0.extent(1), 10u);
            REQUIRE_EQ(drv0.extent(2), 1u);

            if (Initialize)
                flare::realloc(flare::WithoutInitializing, drv0, 10, 20);
            else
                flare::realloc(drv0, 10, 20);
            REQUIRE_EQ(drv0.rank(), 2u);
            REQUIRE_EQ(drv0.extent(0), 10u);
            REQUIRE_EQ(drv0.extent(1), 20u);
            REQUIRE_EQ(drv0.extent(2), 1u);
        }

        static void run_test_mirror() {
            using tensor_type = flare::DynRankTensor<int, host_drv_space>;
            using mirror_type = typename tensor_type::HostMirror;
            tensor_type a("a");
            mirror_type am = flare::create_mirror_tensor(a);
            mirror_type ax = flare::create_mirror(a);
            REQUIRE_EQ(&a(), &am());
            REQUIRE_EQ(a.rank(), am.rank());
            REQUIRE_EQ(ax.rank(), am.rank());

            {
                flare::DynRankTensor<double, flare::LayoutLeft, flare::HostSpace> a_h(
                        "A", 1000);
                auto a_h2 = flare::create_mirror(flare::HostSpace(), a_h);
                auto a_d = flare::create_mirror(typename device::memory_space(), a_h);

                int equal_ptr_h_h2 = (a_h.data() == a_h2.data()) ? 1 : 0;
                int equal_ptr_h_d = (a_h.data() == a_d.data()) ? 1 : 0;
                int equal_ptr_h2_d = (a_h2.data() == a_d.data()) ? 1 : 0;

                REQUIRE_EQ(equal_ptr_h_h2, 0);
                REQUIRE_EQ(equal_ptr_h_d, 0);
                REQUIRE_EQ(equal_ptr_h2_d, 0);

                REQUIRE_EQ(a_h.extent(0), a_h2.extent(0));
                REQUIRE_EQ(a_h.extent(0), a_d.extent(0));

                REQUIRE_EQ(a_h.rank(), a_h2.rank());
                REQUIRE_EQ(a_h.rank(), a_d.rank());
            }
            {
                flare::DynRankTensor<double, flare::LayoutRight, flare::HostSpace> a_h(
                        "A", 1000);
                auto a_h2 = flare::create_mirror(flare::HostSpace(), a_h);
                auto a_d = flare::create_mirror(typename device::memory_space(), a_h);

                int equal_ptr_h_h2 = (a_h.data() == a_h2.data()) ? 1 : 0;
                int equal_ptr_h_d = (a_h.data() == a_d.data()) ? 1 : 0;
                int equal_ptr_h2_d = (a_h2.data() == a_d.data()) ? 1 : 0;

                REQUIRE_EQ(equal_ptr_h_h2, 0);
                REQUIRE_EQ(equal_ptr_h_d, 0);
                REQUIRE_EQ(equal_ptr_h2_d, 0);

                REQUIRE_EQ(a_h.extent(0), a_h2.extent(0));
                REQUIRE_EQ(a_h.extent(0), a_d.extent(0));

                REQUIRE_EQ(a_h.rank(), a_h2.rank());
                REQUIRE_EQ(a_h.rank(), a_d.rank());
            }

            {
                flare::DynRankTensor<double, flare::LayoutLeft, flare::HostSpace> a_h(
                        "A", 1000);
                auto a_h2 = flare::create_mirror_tensor(flare::HostSpace(), a_h);
                auto a_d =
                        flare::create_mirror_tensor(typename device::memory_space(), a_h);

                int equal_ptr_h_h2 = a_h.data() == a_h2.data() ? 1 : 0;
                int equal_ptr_h_d = a_h.data() == a_d.data() ? 1 : 0;
                int equal_ptr_h2_d = a_h2.data() == a_d.data() ? 1 : 0;

                int is_same_memspace =
                        std::is_same<flare::HostSpace, typename device::memory_space>::value
                        ? 1
                        : 0;
                REQUIRE_EQ(equal_ptr_h_h2, 1);
                REQUIRE_EQ(equal_ptr_h_d, is_same_memspace);
                REQUIRE_EQ(equal_ptr_h2_d, is_same_memspace);

                REQUIRE_EQ(a_h.extent(0), a_h2.extent(0));
                REQUIRE_EQ(a_h.extent(0), a_d.extent(0));

                REQUIRE_EQ(a_h.rank(), a_h2.rank());
                REQUIRE_EQ(a_h.rank(), a_d.rank());
            }
            {
                flare::DynRankTensor<double, flare::LayoutRight, flare::HostSpace> a_h(
                        "A", 1000);
                auto a_h2 = flare::create_mirror_tensor(flare::HostSpace(), a_h);
                auto a_d =
                        flare::create_mirror_tensor(typename device::memory_space(), a_h);

                int equal_ptr_h_h2 = a_h.data() == a_h2.data() ? 1 : 0;
                int equal_ptr_h_d = a_h.data() == a_d.data() ? 1 : 0;
                int equal_ptr_h2_d = a_h2.data() == a_d.data() ? 1 : 0;

                int is_same_memspace =
                        std::is_same<flare::HostSpace, typename device::memory_space>::value
                        ? 1
                        : 0;
                REQUIRE_EQ(equal_ptr_h_h2, 1);
                REQUIRE_EQ(equal_ptr_h_d, is_same_memspace);
                REQUIRE_EQ(equal_ptr_h2_d, is_same_memspace);

                REQUIRE_EQ(a_h.extent(0), a_h2.extent(0));
                REQUIRE_EQ(a_h.extent(0), a_d.extent(0));

                REQUIRE_EQ(a_h.rank(), a_h2.rank());
                REQUIRE_EQ(a_h.rank(), a_d.rank());
            }
            {
                using tensor_stride_type =
                        flare::DynRankTensor<int, flare::LayoutStride, flare::HostSpace>;
                unsigned order[] = {6, 5, 4, 3, 2, 1, 0},
                        dimen[] = {N0, N1, N2, 2, 2, 2, 2};  // LayoutRight equivalent
                tensor_stride_type a_h(
                        "a", flare::LayoutStride::order_dimensions(7, order, dimen));
                auto a_h2 = flare::create_mirror_tensor(flare::HostSpace(), a_h);
                auto a_d =
                        flare::create_mirror_tensor(typename device::memory_space(), a_h);

                int equal_ptr_h_h2 = a_h.data() == a_h2.data() ? 1 : 0;
                int equal_ptr_h_d = a_h.data() == a_d.data() ? 1 : 0;
                int equal_ptr_h2_d = a_h2.data() == a_d.data() ? 1 : 0;

                int is_same_memspace =
                        std::is_same<flare::HostSpace, typename device::memory_space>::value
                        ? 1
                        : 0;
                REQUIRE_EQ(equal_ptr_h_h2, 1);
                REQUIRE_EQ(equal_ptr_h_d, is_same_memspace);
                REQUIRE_EQ(equal_ptr_h2_d, is_same_memspace);

                REQUIRE_EQ(a_h.extent(0), a_h2.extent(0));
                REQUIRE_EQ(a_h.extent(0), a_d.extent(0));

                REQUIRE_EQ(a_h.rank(), a_h2.rank());
                REQUIRE_EQ(a_h.rank(), a_d.rank());
            }
        }

        static void run_test_mirror_and_copy() {
            // LayoutLeft
            {
                flare::DynRankTensor<double, flare::LayoutLeft, flare::HostSpace> a_org(
                        "A", 10);
                a_org(5) = 42.0;
                flare::DynRankTensor<double, flare::LayoutLeft, flare::HostSpace> a_h =
                        a_org;
                auto a_h2 = flare::create_mirror_tensor_and_copy(flare::HostSpace(), a_h);
                auto a_d = flare::create_mirror_tensor_and_copy(DeviceType(), a_h);
                auto a_h3 = flare::create_mirror_tensor_and_copy(flare::HostSpace(), a_d);

                int equal_ptr_h_h2 = a_h.data() == a_h2.data() ? 1 : 0;
                int equal_ptr_h_d = a_h.data() == a_d.data() ? 1 : 0;
                int equal_ptr_h2_d = a_h2.data() == a_d.data() ? 1 : 0;
                int equal_ptr_h3_d = a_h3.data() == a_d.data() ? 1 : 0;

                int is_same_memspace =
                        std::is_same<flare::HostSpace,
                                typename DeviceType::memory_space>::value
                        ? 1
                        : 0;
                REQUIRE_EQ(equal_ptr_h_h2, 1);
                REQUIRE_EQ(equal_ptr_h_d, is_same_memspace);
                REQUIRE_EQ(equal_ptr_h2_d, is_same_memspace);
                REQUIRE_EQ(equal_ptr_h3_d, is_same_memspace);

                REQUIRE_EQ(a_h.extent(0), a_h3.extent(0));
                REQUIRE_EQ(a_h.extent(0), a_h2.extent(0));
                REQUIRE_EQ(a_h.extent(0), a_d.extent(0));
                REQUIRE_EQ(a_h.extent(0), a_h3.extent(0));
                REQUIRE_EQ(a_h.rank(), a_org.rank());
                REQUIRE_EQ(a_h.rank(), a_h2.rank());
                REQUIRE_EQ(a_h.rank(), a_h3.rank());
                REQUIRE_EQ(a_h.rank(), a_d.rank());
                REQUIRE_EQ(a_org(5), a_h3(5));
            }
            // LayoutRight
            {
                flare::DynRankTensor<double, flare::LayoutRight, flare::HostSpace> a_org(
                        "A", 10);
                a_org(5) = 42.0;
                flare::DynRankTensor<double, flare::LayoutRight, flare::HostSpace> a_h =
                        a_org;
                auto a_h2 = flare::create_mirror_tensor_and_copy(flare::HostSpace(), a_h);
                auto a_d = flare::create_mirror_tensor_and_copy(DeviceType(), a_h);
                auto a_h3 = flare::create_mirror_tensor_and_copy(flare::HostSpace(), a_d);

                int equal_ptr_h_h2 = a_h.data() == a_h2.data() ? 1 : 0;
                int equal_ptr_h_d = a_h.data() == a_d.data() ? 1 : 0;
                int equal_ptr_h2_d = a_h2.data() == a_d.data() ? 1 : 0;
                int equal_ptr_h3_d = a_h3.data() == a_d.data() ? 1 : 0;

                int is_same_memspace =
                        std::is_same<flare::HostSpace,
                                typename DeviceType::memory_space>::value
                        ? 1
                        : 0;
                REQUIRE_EQ(equal_ptr_h_h2, 1);
                REQUIRE_EQ(equal_ptr_h_d, is_same_memspace);
                REQUIRE_EQ(equal_ptr_h2_d, is_same_memspace);
                REQUIRE_EQ(equal_ptr_h3_d, is_same_memspace);

                REQUIRE_EQ(a_h.extent(0), a_h3.extent(0));
                REQUIRE_EQ(a_h.extent(0), a_h2.extent(0));
                REQUIRE_EQ(a_h.extent(0), a_d.extent(0));
                REQUIRE_EQ(a_h.rank(), a_org.rank());
                REQUIRE_EQ(a_h.rank(), a_h2.rank());
                REQUIRE_EQ(a_h.rank(), a_h3.rank());
                REQUIRE_EQ(a_h.rank(), a_d.rank());
                REQUIRE_EQ(a_org(5), a_h3(5));
            }
        }

        static void run_test_as_tensor_of_rank_n() {
            flare::Tensor<int, flare::HostSpace> error_flag_host("error_flag");
            error_flag_host() = 0;
            auto error_flag =
                    flare::create_mirror_tensor_and_copy(DeviceType(), error_flag_host);

            DTensor0 d("d");

#if defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)

            // Rank 0
            flare::resize(d);

            auto policy0 = flare::RangePolicy<DeviceType>(DeviceType(), 0, 1);

            Tensor0 v0 = flare::detail::as_tensor_of_rank_n<0>(d);
            // Assign values after calling as_tensor_of_rank_n() function under
            // test to ensure aliasing
            flare::parallel_for(
                    policy0, FLARE_LAMBDA(int) { d() = 13; });
            REQUIRE_EQ(v0.size(), d.size());
            REQUIRE_EQ(v0.data(), d.data());
            flare::parallel_for(
                    policy0, FLARE_LAMBDA(int) {
                        if (d() != v0()) error_flag() = 1;
                    });
            flare::deep_copy(error_flag_host, error_flag);
            REQUIRE_EQ(error_flag_host(), 0);

            // Rank 1
            flare::resize(d, 1);

            auto policy1 =
                    flare::RangePolicy<DeviceType>(DeviceType(), 0, d.extent(0));

            Tensor1 v1 = flare::detail::as_tensor_of_rank_n<1>(d);
            flare::parallel_for(
                    policy1, FLARE_LAMBDA(int i0) { d(i0) = i0; });
            for (unsigned int rank = 0; rank < d.rank(); ++rank)
                REQUIRE_EQ(v1.extent(rank), d.extent(rank));
            REQUIRE_EQ(v1.data(), d.data());
            flare::parallel_for(
                    policy1, FLARE_LAMBDA(int i0) {
                        if (d(i0) != v1(i0)) error_flag() = 1;
                    });
            flare::deep_copy(error_flag_host, error_flag);
            REQUIRE_EQ(error_flag_host(), 0);

            // Rank 2
            flare::resize(d, 1, 2);

            auto policy2 = flare::MDRangePolicy<DeviceType, flare::Rank<2>>(
                    {0, 0}, {d.extent(0), d.extent(1)});

            Tensor2 v2 = flare::detail::as_tensor_of_rank_n<2>(d);
            flare::parallel_for(
                    policy2, FLARE_LAMBDA(int i0, int i1) { d(i0, i1) = i0 + 10 * i1; });
            for (unsigned int rank = 0; rank < d.rank(); ++rank)
                REQUIRE_EQ(v2.extent(rank), d.extent(rank));
            REQUIRE_EQ(v2.data(), d.data());
            flare::parallel_for(
                    policy2, FLARE_LAMBDA(int i0, int i1) {
                        if (d(i0, i1) != v2(i0, i1)) error_flag() = 1;
                    });
            flare::deep_copy(error_flag_host, error_flag);
            REQUIRE_EQ(error_flag_host(), 0);

            // Rank 3
            flare::resize(d, 1, 2, 3);

            auto policy3 = flare::MDRangePolicy<DeviceType, flare::Rank<3>>(
                    {0, 0, 0}, {d.extent(0), d.extent(1), d.extent(2)});

            Tensor3 v3 = flare::detail::as_tensor_of_rank_n<3>(d);
            flare::parallel_for(
                    policy3, FLARE_LAMBDA(int i0, int i1, int i2) {
                        d(i0, i1, i2) = i0 + 10 * i1 + 100 * i2;
                    });
            for (unsigned int rank = 0; rank < d.rank(); ++rank)
                REQUIRE_EQ(v3.extent(rank), d.extent(rank));
            REQUIRE_EQ(v3.data(), d.data());
            flare::parallel_for(
                    policy3, FLARE_LAMBDA(int i0, int i1, int i2) {
                        if (d(i0, i1, i2) != v3(i0, i1, i2)) error_flag() = 1;
                    });
            flare::deep_copy(error_flag_host, error_flag);
            REQUIRE_EQ(error_flag_host(), 0);

            // Rank 4
            flare::resize(d, 1, 2, 3, 4);

            auto policy4 = flare::MDRangePolicy<DeviceType, flare::Rank<4>>(
                    {0, 0, 0, 0}, {d.extent(0), d.extent(1), d.extent(2), d.extent(3)});

            Tensor4 v4 = flare::detail::as_tensor_of_rank_n<4>(d);
            flare::parallel_for(
                    policy4, FLARE_LAMBDA(int i0, int i1, int i2, int i3) {
                        d(i0, i1, i2, i3) = i0 + 10 * i1 + 100 * i2 + 1000 * i3;
                    });
            for (unsigned int rank = 0; rank < d.rank(); ++rank)
                REQUIRE_EQ(v4.extent(rank), d.extent(rank));
            REQUIRE_EQ(v4.data(), d.data());
            flare::parallel_for(
                    policy4, FLARE_LAMBDA(int i0, int i1, int i2, int i3) {
                        if (d(i0, i1, i2, i3) != v4(i0, i1, i2, i3)) error_flag() = 1;
                    });
            flare::deep_copy(error_flag_host, error_flag);
            REQUIRE_EQ(error_flag_host(), 0);

            // Rank 5
            flare::resize(d, 1, 2, 3, 4, 5);

            auto policy5 = flare::MDRangePolicy<DeviceType, flare::Rank<5>>(
                    {0, 0, 0, 0, 0},
                    {d.extent(0), d.extent(1), d.extent(2), d.extent(3), d.extent(4)});

            Tensor5 v5 = flare::detail::as_tensor_of_rank_n<5>(d);
            flare::parallel_for(
                    policy5, FLARE_LAMBDA(int i0, int i1, int i2, int i3, int i4) {
                        d(i0, i1, i2, i3, i4) =
                                i0 + 10 * i1 + 100 * i2 + 1000 * i3 + 10000 * i4;
                    });
            for (unsigned int rank = 0; rank < d.rank(); ++rank)
                REQUIRE_EQ(v5.extent(rank), d.extent(rank));
            REQUIRE_EQ(v5.data(), d.data());
            flare::parallel_for(
                    policy5, FLARE_LAMBDA(int i0, int i1, int i2, int i3, int i4) {
                        if (d(i0, i1, i2, i3, i4) != v5(i0, i1, i2, i3, i4)) error_flag() = 1;
                    });
            flare::deep_copy(error_flag_host, error_flag);
            REQUIRE_EQ(error_flag_host(), 0);

            // Rank 6
            flare::resize(d, 1, 2, 3, 4, 5, 6);

            auto policy6 = flare::MDRangePolicy<DeviceType, flare::Rank<6>>(
                    {0, 0, 0, 0, 0, 0}, {d.extent(0), d.extent(1), d.extent(2), d.extent(3),
                                         d.extent(4), d.extent(5)});

            Tensor6 v6 = flare::detail::as_tensor_of_rank_n<6>(d);
            flare::parallel_for(
                    policy6, FLARE_LAMBDA(int i0, int i1, int i2, int i3, int i4, int i5) {
                        d(i0, i1, i2, i3, i4, i5) =
                                i0 + 10 * i1 + 100 * i2 + 1000 * i3 + 10000 * i4 + 100000 * i5;
                    });
            for (unsigned int rank = 0; rank < d.rank(); ++rank)
                REQUIRE_EQ(v6.extent(rank), d.extent(rank));
            REQUIRE_EQ(v6.data(), d.data());
            flare::parallel_for(
                    policy6, FLARE_LAMBDA(int i0, int i1, int i2, int i3, int i4, int i5) {
                        if (d(i0, i1, i2, i3, i4, i5) != v6(i0, i1, i2, i3, i4, i5))
                            error_flag() = 1;
                    });
            flare::deep_copy(error_flag_host, error_flag);
            REQUIRE_EQ(error_flag_host(), 0);

            // Rank 7
            flare::resize(d, 1, 2, 3, 4, 5, 6, 7);

            // MDRangePolicy only accepts Rank < 7
#if 0
            auto policy7 = flare::MDRangePolicy<DeviceType, flare::Rank<7>>(
                {0, 0, 0, 0, 0, 0, 0},
                {d.extent(0), d.extent(1), d.extent(2), d.extent(3), d.extent(4),
                 d.extent(5), d.extent(6)});

            Tensor7 v7 = flare::detail::as_tensor_of_rank_n<7>(d);
            flare::parallel_for(
                policy7,
                FLARE_LAMBDA(int i0, int i1, int i2, int i3, int i4, int i5, int i6) {
                  d(i0, i1, i2, i3, i4, i5, i6) = i0 + 10 * i1 + 100 * i2 + 1000 * i3 +
                                                  10000 * i4 + 100000 * i5 +
                                                  1000000 * i6;
                });
            for (unsigned int rank = 0; rank < d.rank(); ++rank)
              REQUIRE_EQ(v7.extent(rank), d.extent(rank));
            REQUIRE_EQ(v7.data(), d.data());
            flare::parallel_for(
                policy7,
                FLARE_LAMBDA(int i0, int i1, int i2, int i3, int i4, int i5, int i6) {
                  if (d(i0, i1, i2, i3, i4, i5, i6) != v7(i0, i1, i2, i3, i4, i5, i6))
                    error_flag() = 1;
                });
            flare::deep_copy(error_flag_host, error_flag);
            REQUIRE_EQ(error_flag_host(), 0);
#endif  // MDRangePolict Rank < 7

#endif  // defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)
        }

        static void run_test_scalar() {
            using hTensor0 = typename DTensor0::HostMirror;  // HostMirror of DynRankTensor is
            // a DynRankTensor

            DTensor0 dx, dy;
            hTensor0 hx, hy;

            dx = DTensor0("dx");
            dy = DTensor0("dy");

            hx = flare::create_mirror(dx);
            hy = flare::create_mirror(dy);

            hx() = 1;

            flare::deep_copy(dx, hx);
            flare::deep_copy(dy, dx);
            flare::deep_copy(hy, dy);

            REQUIRE_EQ(hx(), hy());
            REQUIRE_EQ(dx.rank(), hx.rank());
            REQUIRE_EQ(dy.rank(), hy.rank());

            // Tensor - DynRankTensor Interoperability tests
            // deep_copy DynRankTensor to Tensor
            Tensor0 vx("vx");
            flare::deep_copy(vx, dx);
            REQUIRE_EQ(rank(dx), rank(vx));

            Tensor0 vy("vy");
            flare::deep_copy(vy, dy);
            REQUIRE_EQ(rank(dy), rank(vy));

            // deep_copy Tensor to DynRankTensor
            DTensor0 dxx("dxx");
            flare::deep_copy(dxx, vx);
            REQUIRE_EQ(rank(dxx), rank(vx));

            Tensor7 vcast = dx.ConstDownCast();
            REQUIRE_EQ(dx.extent(0), vcast.extent(0));
            REQUIRE_EQ(dx.extent(1), vcast.extent(1));
            REQUIRE_EQ(dx.extent(2), vcast.extent(2));
            REQUIRE_EQ(dx.extent(3), vcast.extent(3));
            REQUIRE_EQ(dx.extent(4), vcast.extent(4));

            Tensor7 vcast1(dy.ConstDownCast());
            REQUIRE_EQ(dy.extent(0), vcast1.extent(0));
            REQUIRE_EQ(dy.extent(1), vcast1.extent(1));
            REQUIRE_EQ(dy.extent(2), vcast1.extent(2));
            REQUIRE_EQ(dy.extent(3), vcast1.extent(3));
            REQUIRE_EQ(dy.extent(4), vcast1.extent(4));

            // Tensor - DynRankTensor Interoperability tests
            // copy Tensor to DynRankTensor
            DTensor0 dfromvx(vx);
            auto hmx = flare::create_mirror_tensor(dfromvx);
            flare::deep_copy(hmx, dfromvx);
            auto hvx = flare::create_mirror_tensor(vx);
            flare::deep_copy(hvx, vx);
            REQUIRE_EQ(rank(hvx), rank(hmx));
            REQUIRE_EQ(hvx.extent(0), hmx.extent(0));
            REQUIRE_EQ(hvx.extent(1), hmx.extent(1));

            // copy-assign Tensor to DynRankTensor
            DTensor0 dfromvy = vy;
            auto hmy = flare::create_mirror_tensor(dfromvy);
            flare::deep_copy(hmy, dfromvy);
            auto hvy = flare::create_mirror_tensor(vy);
            flare::deep_copy(hvy, vy);
            REQUIRE_EQ(rank(hvy), rank(hmy));
            REQUIRE_EQ(hvy.extent(0), hmy.extent(0));
            REQUIRE_EQ(hvy.extent(1), hmy.extent(1));

            Tensor7 vtest1("vtest1", 2, 2, 2, 2, 2, 2, 2);
            DTensor0 dfromv1(vtest1);
            REQUIRE_EQ(dfromv1.rank(), vtest1.rank);
            REQUIRE_EQ(dfromv1.extent(0), vtest1.extent(0));
            REQUIRE_EQ(dfromv1.extent(1), vtest1.extent(1));
            REQUIRE_EQ(dfromv1.use_count(), vtest1.use_count());

            DTensor0 dfromv2(vcast);
            REQUIRE_EQ(dfromv2.rank(), vcast.rank);
            REQUIRE_EQ(dfromv2.extent(0), vcast.extent(0));
            REQUIRE_EQ(dfromv2.extent(1), vcast.extent(1));
            REQUIRE_EQ(dfromv2.use_count(), vcast.use_count());

            DTensor0 dfromv3 = vcast1;
            REQUIRE_EQ(dfromv3.rank(), vcast1.rank);
            REQUIRE_EQ(dfromv3.extent(0), vcast1.extent(0));
            REQUIRE_EQ(dfromv3.extent(1), vcast1.extent(1));
            REQUIRE_EQ(dfromv3.use_count(), vcast1.use_count());
        }

        static void run_test() {
            // mfh 14 Feb 2014: This test doesn't actually create instances of
            // these types.  In order to avoid "unused type alias"
            // warnings, we declare empty instances of these types, with the
            // usual "(void)" marker to avoid compiler warnings for unused
            // variables.

            using hTensor0 = typename DTensor0::HostMirror;

            {
                hTensor0 thing;
                (void) thing;
            }

            DTensor0 d_uninitialized(
                    flare::tensor_alloc(flare::WithoutInitializing, "uninit"), 10, 20);
            REQUIRE_NE(d_uninitialized.data(), nullptr);
            REQUIRE_EQ(d_uninitialized.rank(), 2u);
            REQUIRE_EQ(d_uninitialized.extent(0), 10u);
            REQUIRE_EQ(d_uninitialized.extent(1), 20u);
            REQUIRE_EQ(d_uninitialized.extent(2), 1u);

            DTensor0 dx, dy, dz;
            hTensor0 hx, hy, hz;

            REQUIRE(flare::is_dyn_rank_tensor<DTensor0>::value);
            REQUIRE_FALSE(flare::is_dyn_rank_tensor<flare::Tensor<double>>::value);

            REQUIRE_EQ(dx.data(), nullptr);  // Okay with UVM
            REQUIRE_EQ(dy.data(), nullptr);  // Okay with UVM
            REQUIRE_EQ(dz.data(), nullptr);  // Okay with UVM
            REQUIRE_EQ(hx.data(), nullptr);
            REQUIRE_EQ(hy.data(), nullptr);
            REQUIRE_EQ(hz.data(), nullptr);
            REQUIRE_EQ(dx.extent(0), 0u);  // Okay with UVM
            REQUIRE_EQ(dy.extent(0), 0u);  // Okay with UVM
            REQUIRE_EQ(dz.extent(0), 0u);  // Okay with UVM
            REQUIRE_EQ(hx.extent(0), 0u);
            REQUIRE_EQ(hy.extent(0), 0u);
            REQUIRE_EQ(hz.extent(0), 0u);
            REQUIRE_EQ(dx.rank(), 0u);  // Okay with UVM
            REQUIRE_EQ(hx.rank(), 0u);

            dx = DTensor0("dx", N1, N2, N3);
            dy = DTensor0("dy", N1, N2, N3);

            hx = hTensor0("hx", N1, N2, N3);
            hy = hTensor0("hy", N1, N2, N3);

            REQUIRE_EQ(dx.extent(0), unsigned(N1));  // Okay with UVM
            REQUIRE_EQ(dy.extent(0), unsigned(N1));  // Okay with UVM
            REQUIRE_EQ(hx.extent(0), unsigned(N1));
            REQUIRE_EQ(hy.extent(0), unsigned(N1));
            REQUIRE_EQ(dx.rank(), 3u);  // Okay with UVM
            REQUIRE_EQ(hx.rank(), 3u);

            dx = DTensor0("dx", N0, N1, N2, N3);
            dy = DTensor0("dy", N0, N1, N2, N3);
            hx = hTensor0("hx", N0, N1, N2, N3);
            hy = hTensor0("hy", N0, N1, N2, N3);

            REQUIRE_EQ(dx.extent(0), unsigned(N0));
            REQUIRE_EQ(dy.extent(0), unsigned(N0));
            REQUIRE_EQ(hx.extent(0), unsigned(N0));
            REQUIRE_EQ(hy.extent(0), unsigned(N0));
            REQUIRE_EQ(dx.rank(), 4u);
            REQUIRE_EQ(dy.rank(), 4u);
            REQUIRE_EQ(hx.rank(), 4u);
            REQUIRE_EQ(hy.rank(), 4u);

            REQUIRE_EQ(dx.use_count(), 1);

            DTensor0_unmanaged unmanaged_dx = dx;
            REQUIRE_EQ(dx.use_count(), 1);

            DTensor0_unmanaged unmanaged_from_ptr_dx = DTensor0_unmanaged(
                    dx.data(), dx.extent(0), dx.extent(1), dx.extent(2), dx.extent(3));

            {
                // Destruction of this tensor should be harmless
                const_DTensor0 unmanaged_from_ptr_const_dx(
                        dx.data(), dx.extent(0), dx.extent(1), dx.extent(2), dx.extent(3));
            }

            const_DTensor0 const_dx = dx;
            REQUIRE_EQ(dx.use_count(), 2);

            {
                const_DTensor0 const_dx2;
                const_dx2 = const_dx;
                REQUIRE_EQ(dx.use_count(), 3);

                const_dx2 = dy;
                REQUIRE_EQ(dx.use_count(), 2);

                const_DTensor0 const_dx3(dx);
                REQUIRE_EQ(dx.use_count(), 3);

                DTensor0_unmanaged dx4_unmanaged(dx);
                REQUIRE_EQ(dx.use_count(), 3);
            }

            REQUIRE_EQ(dx.use_count(), 2);

            REQUIRE_NE(dx.data(), nullptr);
            REQUIRE_NE(const_dx.data(), nullptr);
            REQUIRE_NE(unmanaged_dx.data(), nullptr);
            REQUIRE_NE(unmanaged_from_ptr_dx.data(), nullptr);
            REQUIRE_NE(dy.data(), nullptr);
            REQUIRE_NE(dx, dy);

            REQUIRE_EQ(dx.extent(0), unsigned(N0));
            REQUIRE_EQ(dx.extent(1), unsigned(N1));
            REQUIRE_EQ(dx.extent(2), unsigned(N2));
            REQUIRE_EQ(dx.extent(3), unsigned(N3));

            REQUIRE_EQ(dy.extent(0), unsigned(N0));
            REQUIRE_EQ(dy.extent(1), unsigned(N1));
            REQUIRE_EQ(dy.extent(2), unsigned(N2));
            REQUIRE_EQ(dy.extent(3), unsigned(N3));

            REQUIRE_EQ(unmanaged_from_ptr_dx.span(),
                       unsigned(N0) *unsigned(N1) * unsigned(N2) * unsigned(N3));

            hx = flare::create_mirror(dx);
            hy = flare::create_mirror(dy);

            REQUIRE_EQ(hx.rank(), dx.rank());
            REQUIRE_EQ(hy.rank(), dy.rank());

            REQUIRE_EQ(hx.extent(0), unsigned(N0));
            REQUIRE_EQ(hx.extent(1), unsigned(N1));
            REQUIRE_EQ(hx.extent(2), unsigned(N2));
            REQUIRE_EQ(hx.extent(3), unsigned(N3));

            REQUIRE_EQ(hy.extent(0), unsigned(N0));
            REQUIRE_EQ(hy.extent(1), unsigned(N1));
            REQUIRE_EQ(hy.extent(2), unsigned(N2));
            REQUIRE_EQ(hy.extent(3), unsigned(N3));

            // T v1 = hx() ;    // Generates compile error as intended
            // T v2 = hx(0,0) ; // Generates compile error as intended
            // hx(0,0) = v2 ;   // Generates compile error as intended

#if 0 /* Asynchronous deep copies not implemented for dynamic rank tensor */
            // Testing with asynchronous deep copy with respect to device
            {
              size_t count = 0 ;
              for ( size_t ip = 0 ; ip < N0 ; ++ip ) {
              for ( size_t i1 = 0 ; i1 < hx.extent(1) ; ++i1 ) {
              for ( size_t i2 = 0 ; i2 < hx.extent(2) ; ++i2 ) {
              for ( size_t i3 = 0 ; i3 < hx.extent(3) ; ++i3 ) {
                hx(ip,i1,i2,i3) = ++count ;
              }}}}


              flare::deep_copy(typename hTensor0::execution_space(), dx , hx );
              flare::deep_copy(typename hTensor0::execution_space(), dy , dx );
              flare::deep_copy(typename hTensor0::execution_space(), hy , dy );

              for ( size_t ip = 0 ; ip < N0 ; ++ip ) {
              for ( size_t i1 = 0 ; i1 < N1 ; ++i1 ) {
              for ( size_t i2 = 0 ; i2 < N2 ; ++i2 ) {
              for ( size_t i3 = 0 ; i3 < N3 ; ++i3 ) {
                { REQUIRE_EQ( hx(ip,i1,i2,i3) , hy(ip,i1,i2,i3) ); }
              }}}}

              flare::deep_copy(typename hTensor0::execution_space(), dx , T(0) );
              flare::deep_copy(typename hTensor0::execution_space(), hx , dx );

              for ( size_t ip = 0 ; ip < N0 ; ++ip ) {
              for ( size_t i1 = 0 ; i1 < N1 ; ++i1 ) {
              for ( size_t i2 = 0 ; i2 < N2 ; ++i2 ) {
              for ( size_t i3 = 0 ; i3 < N3 ; ++i3 ) {
                { REQUIRE_EQ( hx(ip,i1,i2,i3) , T(0) ); }
              }}}}
            }

            // Testing with asynchronous deep copy with respect to host
            {
              size_t count = 0 ;
              for ( size_t ip = 0 ; ip < N0 ; ++ip ) {
              for ( size_t i1 = 0 ; i1 < hx.extent(1) ; ++i1 ) {
              for ( size_t i2 = 0 ; i2 < hx.extent(2) ; ++i2 ) {
              for ( size_t i3 = 0 ; i3 < hx.extent(3) ; ++i3 ) {
                hx(ip,i1,i2,i3) = ++count ;
              }}}}

              flare::deep_copy(typename DTensor0::execution_space(), dx , hx );
              flare::deep_copy(typename DTensor0::execution_space(), dy , dx );
              flare::deep_copy(typename DTensor0::execution_space(), hy , dy );

              for ( size_t ip = 0 ; ip < N0 ; ++ip ) {
              for ( size_t i1 = 0 ; i1 < N1 ; ++i1 ) {
              for ( size_t i2 = 0 ; i2 < N2 ; ++i2 ) {
              for ( size_t i3 = 0 ; i3 < N3 ; ++i3 ) {
                { REQUIRE_EQ( hx(ip,i1,i2,i3) , hy(ip,i1,i2,i3) ); }
              }}}}

              flare::deep_copy(typename DTensor0::execution_space(), dx , T(0) );
              flare::deep_copy(typename DTensor0::execution_space(), hx , dx );

              for ( size_t ip = 0 ; ip < N0 ; ++ip ) {
              for ( size_t i1 = 0 ; i1 < N1 ; ++i1 ) {
              for ( size_t i2 = 0 ; i2 < N2 ; ++i2 ) {
              for ( size_t i3 = 0 ; i3 < N3 ; ++i3 ) {
                { REQUIRE_EQ( hx(ip,i1,i2,i3) , T(0) ); }
              }}}}
            }
#endif

            // Testing with synchronous deep copy
            {
                size_t count = 0;
                for (size_t ip = 0; ip < N0; ++ip) {
                    for (size_t i1 = 0; i1 < hx.extent(1); ++i1) {
                        for (size_t i2 = 0; i2 < hx.extent(2); ++i2) {
                            for (size_t i3 = 0; i3 < hx.extent(3); ++i3) {
                                hx(ip, i1, i2, i3) = ++count;
                            }
                        }
                    }
                }

                flare::deep_copy(dx, hx);
                flare::deep_copy(dy, dx);
                flare::deep_copy(hy, dy);
                flare::fence();

                for (size_t ip = 0; ip < N0; ++ip) {
                    for (size_t i1 = 0; i1 < N1; ++i1) {
                        for (size_t i2 = 0; i2 < N2; ++i2) {
                            for (size_t i3 = 0; i3 < N3; ++i3) {
                                {
                                    REQUIRE_EQ(hx(ip, i1, i2, i3), hy(ip, i1, i2, i3));
                                }
                            }
                        }
                    }
                }

                flare::deep_copy(dx, T(0));
                flare::deep_copy(hx, dx);
                flare::fence();

                for (size_t ip = 0; ip < N0; ++ip) {
                    for (size_t i1 = 0; i1 < N1; ++i1) {
                        for (size_t i2 = 0; i2 < N2; ++i2) {
                            for (size_t i3 = 0; i3 < N3; ++i3) {
                                {
                                    REQUIRE_EQ(hx(ip, i1, i2, i3), T(0));
                                }
                            }
                        }
                    }
                }
                //    REQUIRE_EQ( hx(0,0,0,0,0,0,0,0) , T(0) ); //Test rank8 op behaves
                //    properly - if implemented
            }

            dz = dx;
            REQUIRE_EQ(dx, dz);
            REQUIRE_NE(dy, dz);
            dz = dy;
            REQUIRE_EQ(dy, dz);
            REQUIRE_NE(dx, dz);

            dx = DTensor0();
            REQUIRE_EQ(dx.data(), nullptr);
            REQUIRE_NE(dy.data(), nullptr);
            REQUIRE_NE(dz.data(), nullptr);
            dy = DTensor0();
            REQUIRE_EQ(dx.data(), nullptr);
            REQUIRE_EQ(dy.data(), nullptr);
            REQUIRE_NE(dz.data(), nullptr);
            dz = DTensor0();
            REQUIRE_EQ(dx.data(), nullptr);
            REQUIRE_EQ(dy.data(), nullptr);
            REQUIRE_EQ(dz.data(), nullptr);

            // Tensor - DynRankTensor Interoperability tests
            // deep_copy from tensor to dynranktensor
            constexpr size_t testdim = 4;
            DTensor0 dxx("dxx", testdim);
            Tensor1 vxx("vxx", testdim);
            auto hvxx = flare::create_mirror_tensor(vxx);
            for (size_t i = 0; i < testdim; ++i) {
                hvxx(i) = i;
            }
            flare::deep_copy(vxx, hvxx);
            flare::deep_copy(dxx, vxx);
            auto hdxx = flare::create_mirror_tensor(dxx);
            flare::deep_copy(hdxx, dxx);
            for (size_t i = 0; i < testdim; ++i) {
                REQUIRE_EQ(hvxx(i), hdxx(i));
            }

            REQUIRE_EQ(rank(hdxx), rank(hvxx));
            REQUIRE_EQ(hdxx.extent(0), testdim);
            REQUIRE_EQ(hdxx.extent(0), hvxx.extent(0));

            Tensor1 vdxx("vdxx", testdim);
            auto hvdxx = flare::create_mirror_tensor(vdxx);
            flare::deep_copy(hvdxx, hdxx);
            REQUIRE_EQ(rank(hdxx), rank(hvdxx));
            REQUIRE_EQ(hvdxx.extent(0), testdim);
            REQUIRE_EQ(hdxx.extent(0), hvdxx.extent(0));
            for (size_t i = 0; i < testdim; ++i) {
                REQUIRE_EQ(hvxx(i), hvdxx(i));
            }
        }

        using DataType = T;

        static void check_auto_conversion_to_const(
                const flare::DynRankTensor<const DataType, device> &arg_const,
                const flare::DynRankTensor<DataType, device> &arg) {
            REQUIRE_EQ(arg_const, arg);
        }

        static void run_test_allocated() {
            using device_type = flare::DynRankTensor<DataType, device>;

            const int N1 = 100;
            const int N2 = 10;

            device_type d1;
            REQUIRE_FALSE(d1.is_allocated());

            d1 = device_type("d1", N1, N2);
            device_type d2(d1);
            device_type d3("d3", N1);
            REQUIRE(d1.is_allocated());
            REQUIRE(d2.is_allocated());
            REQUIRE(d3.is_allocated());
        }

        static void run_test_const() {
            using typeX = flare::DynRankTensor<DataType, device>;
            using const_typeX = flare::DynRankTensor<const DataType, device>;
            using const_typeR =
                    flare::DynRankTensor<const DataType, device, flare::MemoryRandomAccess>;
            typeX x("X", 2);
            const_typeX xc = x;
            const_typeR xr = x;

            REQUIRE_EQ(xc, x);
            REQUIRE_EQ(x, xc);

            // For CUDA the constant random access Tensor does not return
            // an lvalue reference due to retrieving through texture cache
            // therefore not allowed to query the underlying pointer.
#if defined(FLARE_ON_CUDA_DEVICE)
            if (!std::is_same<typename device::execution_space, flare::Cuda>::value)
#endif
            {
                REQUIRE_EQ(x.data(), xr.data());
            }

            // typeX xf = xc ; // setting non-const from const must not compile

            check_auto_conversion_to_const(x, x);
        }

        static void run_test_subtensor() {
            using cDTensor = flare::DynRankTensor<const T, device>;
            using DTensor = flare::DynRankTensor<T, device>;
            // LayoutStride required for all returned DynRankTensor subdynranktensor's
            using sDTensor = flare::DynRankTensor<T, flare::LayoutStride, device>;

            DTensor0 d0("d0");
            cDTensor s0 = d0;

            //  N0 = 1000,N1 = 3,N2 = 5,N3 = 7
            unsigned order[] = {6, 5, 4, 3, 2, 1, 0},
                    dimen[] = {N0, N1, N2, 2, 2, 2, 2};  // LayoutRight equivalent
            sDTensor d7("d7", flare::LayoutStride::order_dimensions(7, order, dimen));
            REQUIRE_EQ(d7.rank(), 7u);

            sDTensor ds0 = flare::subdynranktensor(d7, 1, 1, 1, 1, 1, 1, 1);
            REQUIRE_EQ(ds0.rank(), 0u);

            // Basic test - ALL
            sDTensor dsALL = flare::subdynranktensor(
                    d7, flare::ALL(), flare::ALL(), flare::ALL(), flare::ALL(),
                    flare::ALL(), flare::ALL(), flare::ALL());
            REQUIRE_EQ(dsALL.rank(), 7u);

            //  Send a value to final rank returning rank 6 subtensor
            sDTensor dsm1 =
                    flare::subdynranktensor(d7, flare::ALL(), flare::ALL(), flare::ALL(),
                                          flare::ALL(), flare::ALL(), flare::ALL(), 1);
            REQUIRE_EQ(dsm1.rank(), 6u);

            //  Send a std::pair as argument to a rank
            sDTensor dssp = flare::subdynranktensor(
                    d7, flare::ALL(), flare::ALL(), flare::ALL(), flare::ALL(),
                    flare::ALL(), flare::ALL(), std::pair<unsigned, unsigned>(1, 2));
            REQUIRE_EQ(dssp.rank(), 7u);

            //  Send a flare::pair as argument to a rank; take default layout as input
            DTensor0 dd0("dd0", N0, N1, N2, 2, 2, 2, 2);  // default layout
            REQUIRE_EQ(dd0.rank(), 7u);
            sDTensor dtkp = flare::subdynranktensor(
                    dd0, flare::ALL(), flare::ALL(), flare::ALL(), flare::ALL(),
                    flare::ALL(), flare::ALL(), flare::pair<unsigned, unsigned>(0, 1));
            REQUIRE_EQ(dtkp.rank(), 7u);

            // Return rank 7 subtensor, taking a pair as one argument, layout stride input
            sDTensor ds7 = flare::subdynranktensor(
                    d7, flare::ALL(), flare::ALL(), flare::ALL(), flare::ALL(),
                    flare::ALL(), flare::ALL(), flare::pair<unsigned, unsigned>(0, 1));
            REQUIRE_EQ(ds7.rank(), 7u);

            // Default Layout DynRankTensor
            DTensor dv6("dv6", N0, N1, N2, N3, 2, 2);
            REQUIRE_EQ(dv6.rank(), 6u);

            // DynRankTensor with LayoutRight
            using drTensor = flare::DynRankTensor<T, flare::LayoutRight, device>;
            drTensor dr5("dr5", N0, N1, N2, 2, 2);
            REQUIRE_EQ(dr5.rank(), 5u);

            // LayoutStride but arranged as LayoutRight
            // NOTE: unused arg_layout dimensions must be set toFLARE_INVALID_INDEX so
            // that
            //  rank deduction can properly take place
            unsigned order5[] = {4, 3, 2, 1, 0}, dimen5[] = {N0, N1, N2, 2, 2};
            flare::LayoutStride ls =
                    flare::LayoutStride::order_dimensions(5, order5, dimen5);
            ls.dimension[5] = FLARE_INVALID_INDEX;
            ls.dimension[6] = FLARE_INVALID_INDEX;
            ls.dimension[7] = FLARE_INVALID_INDEX;
            sDTensor d5("d5", ls);
            REQUIRE_EQ(d5.rank(), 5u);

            //  LayoutStride arranged as LayoutRight - commented out as example that
            //  fails unit test
            //    unsigned order5[] = { 4,3,2,1,0 }, dimen5[] = { N0, N1, N2, 2, 2 };
            //    sDTensor d5( "d5" , flare::LayoutStride::order_dimensions(5, order5,
            //    dimen5) );
            //
            //  Fails the following unit test:
            //    REQUIRE_EQ( d5.rank() , dr5.rank() );
            //
            //  Explanation: In construction of the flare::LayoutStride below, since
            //  the
            //   remaining dimensions are not specified, they will default to values of
            //   0 rather thanFLARE_INVALID_INDEX.
            //  When passed to the DynRankTensor constructor the default dimensions (of 0)
            //   will be counted toward the dynamic rank and returning an incorrect
            //   value (i.e. rank 7 rather than 5).

            // Check LayoutRight dr5 and LayoutStride d5 dimensions agree (as they
            // should)
            REQUIRE_EQ(d5.extent(0), dr5.extent(0));
            REQUIRE_EQ(d5.extent(1), dr5.extent(1));
            REQUIRE_EQ(d5.extent(2), dr5.extent(2));
            REQUIRE_EQ(d5.extent(3), dr5.extent(3));
            REQUIRE_EQ(d5.extent(4), dr5.extent(4));
            REQUIRE_EQ(d5.extent(5), dr5.extent(5));
            REQUIRE_EQ(d5.rank(), dr5.rank());

            // Rank 5 subtensor of rank 5 dynamic rank tensor, layout stride input
            sDTensor ds5 = flare::subdynranktensor(d5, flare::ALL(), flare::ALL(),
                                               flare::ALL(), flare::ALL(),
                                               flare::pair<unsigned, unsigned>(0, 1));
            REQUIRE_EQ(ds5.rank(), 5u);

            // Pass in extra ALL arguments beyond the rank of the DynRank Tensor.
            // This behavior is allowed - ignore the extra ALL arguments when
            //  the src.rank() < number of arguments, but be careful!
            sDTensor ds5plus = flare::subdynranktensor(
                    d5, flare::ALL(), flare::ALL(), flare::ALL(), flare::ALL(),
                    flare::pair<unsigned, unsigned>(0, 1), flare::ALL());

            REQUIRE_EQ(ds5.rank(), ds5plus.rank());
            REQUIRE_EQ(ds5.extent(0), ds5plus.extent(0));
            REQUIRE_EQ(ds5.extent(4), ds5plus.extent(4));
            REQUIRE_EQ(ds5.extent(5), ds5plus.extent(5));

#if !defined(FLARE_ON_CUDA_DEVICE)
            REQUIRE_EQ(&ds5(1, 1, 1, 1, 0) - &ds5plus(1, 1, 1, 1, 0), 0);
            REQUIRE_EQ(&ds5(1, 1, 1, 1, 0, 0) - &ds5plus(1, 1, 1, 1, 0, 0),
                      0);  // passing argument to rank beyond the tensor's rank is allowed
                           // iff it is a 0.
#endif

            // Similar test to rank 5 above, but create rank 4 subtensor
            // Check that the rank contracts (ds4 and ds4plus) and that subdynranktensor
            // can accept extra args (ds4plus)
            sDTensor ds4 = flare::subdynranktensor(d5, flare::ALL(), flare::ALL(),
                                               flare::ALL(), flare::ALL(), 0);
            sDTensor ds4plus =
                    flare::subdynranktensor(d5, flare::ALL(), flare::ALL(), flare::ALL(),
                                          flare::ALL(), 0, flare::ALL());

            REQUIRE_EQ(ds4.rank(), ds4plus.rank());
            REQUIRE_EQ(ds4.rank(), 4u);
            REQUIRE_EQ(ds4.extent(0), ds4plus.extent(0));
            REQUIRE_EQ(ds4.extent(4), ds4plus.extent(4));
            REQUIRE_EQ(ds4.extent(5), ds4plus.extent(5));
        }

        static void run_test_subtensor_strided() {
            using drtensor_left =
                    flare::DynRankTensor<int, flare::LayoutLeft, host_drv_space>;
            using drtensor_right =
                    flare::DynRankTensor<int, flare::LayoutRight, host_drv_space>;
            using drtensor_stride =
                    flare::DynRankTensor<int, flare::LayoutStride, host_drv_space>;

            drtensor_left xl2("xl2", 100, 200);
            drtensor_right xr2("xr2", 100, 200);
            drtensor_stride yl1 = flare::subdynranktensor(xl2, 0, flare::ALL());
            drtensor_stride yl2 = flare::subdynranktensor(xl2, 1, flare::ALL());
            drtensor_stride ys1 = flare::subdynranktensor(xr2, 0, flare::ALL());
            drtensor_stride ys2 = flare::subdynranktensor(xr2, 1, flare::ALL());
            drtensor_stride yr1 = flare::subdynranktensor(xr2, 0, flare::ALL());
            drtensor_stride yr2 = flare::subdynranktensor(xr2, 1, flare::ALL());

            REQUIRE_EQ(yl1.extent(0), xl2.extent(1));
            REQUIRE_EQ(yl2.extent(0), xl2.extent(1));

            REQUIRE_EQ(yr1.extent(0), xr2.extent(1));
            REQUIRE_EQ(yr2.extent(0), xr2.extent(1));

            REQUIRE_EQ(&yl1(0) - &xl2(0, 0), 0);
            REQUIRE_EQ(&yl2(0) - &xl2(1, 0), 0);
            REQUIRE_EQ(&yr1(0) - &xr2(0, 0), 0);
            REQUIRE_EQ(&yr2(0) - &xr2(1, 0), 0);

            drtensor_left xl4("xl4", 10, 20, 30, 40);
            drtensor_right xr4("xr4", 10, 20, 30, 40);

            // Replace subdynranktensor with subtensor - test
            drtensor_stride yl4 =
                    flare::subtensor(xl4, 1, flare::ALL(), 2, flare::ALL());
            drtensor_stride yr4 =
                    flare::subtensor(xr4, 1, flare::ALL(), 2, flare::ALL());

            REQUIRE_EQ(yl4.extent(0), xl4.extent(1));
            REQUIRE_EQ(yl4.extent(1), xl4.extent(3));
            REQUIRE_EQ(yr4.extent(0), xr4.extent(1));
            REQUIRE_EQ(yr4.extent(1), xr4.extent(3));
            REQUIRE_EQ(yl4.rank(), 2u);
            REQUIRE_EQ(yr4.rank(), 2u);

            REQUIRE_EQ(&yl4(4, 4) - &xl4(1, 4, 2, 4), 0);
            REQUIRE_EQ(&yr4(4, 4) - &xr4(1, 4, 2, 4), 0);
        }

        static void run_test_vector() {
            static const unsigned Length = 1000, Count = 8;

            using multivector_type =
                    typename flare::DynRankTensor<T, flare::LayoutLeft, host_drv_space>;

            using multivector_right_type =
                    typename flare::DynRankTensor<T, flare::LayoutRight, host_drv_space>;

            multivector_type mv = multivector_type("mv", Length, Count);
            multivector_right_type mv_right =
                    multivector_right_type("mv", Length, Count);

            using svector_type =
                    typename flare::DynRankTensor<T, flare::LayoutStride, host_drv_space>;
            using smultivector_type =
                    typename flare::DynRankTensor<T, flare::LayoutStride, host_drv_space>;
            using const_svector_right_type =
                    typename flare::DynRankTensor<const T, flare::LayoutStride,
                            host_drv_space>;
            using const_svector_type =
                    typename flare::DynRankTensor<const T, flare::LayoutStride,
                            host_drv_space>;
            using const_smultivector_type =
                    typename flare::DynRankTensor<const T, flare::LayoutStride,
                            host_drv_space>;

            svector_type v1 = flare::subdynranktensor(mv, flare::ALL(), 0);
            svector_type v2 = flare::subdynranktensor(mv, flare::ALL(), 1);
            svector_type v3 = flare::subdynranktensor(mv, flare::ALL(), 2);

            svector_type rv1 = flare::subdynranktensor(mv_right, 0, flare::ALL());
            svector_type rv2 = flare::subdynranktensor(mv_right, 1, flare::ALL());
            svector_type rv3 = flare::subdynranktensor(mv_right, 2, flare::ALL());

            smultivector_type mv1 = flare::subdynranktensor(mv, std::make_pair(1, 998),
                                                          std::make_pair(2, 5));

            smultivector_type mvr1 = flare::subdynranktensor(
                    mv_right, std::make_pair(1, 998), std::make_pair(2, 5));

            const_svector_type cv1 = flare::subdynranktensor(mv, flare::ALL(), 0);
            const_svector_type cv2 = flare::subdynranktensor(mv, flare::ALL(), 1);
            const_svector_type cv3 = flare::subdynranktensor(mv, flare::ALL(), 2);

            svector_type vr1 = flare::subdynranktensor(mv, flare::ALL(), 0);
            svector_type vr2 = flare::subdynranktensor(mv, flare::ALL(), 1);
            svector_type vr3 = flare::subdynranktensor(mv, flare::ALL(), 2);

            const_svector_right_type cvr1 =
                    flare::subdynranktensor(mv, flare::ALL(), 0);
            const_svector_right_type cvr2 =
                    flare::subdynranktensor(mv, flare::ALL(), 1);
            const_svector_right_type cvr3 =
                    flare::subdynranktensor(mv, flare::ALL(), 2);

            REQUIRE_EQ(&v1[0], &v1(0));
            REQUIRE_EQ(&v1[0], &mv(0, 0));
            REQUIRE_EQ(&v2[0], &mv(0, 1));
            REQUIRE_EQ(&v3[0], &mv(0, 2));

            REQUIRE_EQ(&cv1[0], &mv(0, 0));
            REQUIRE_EQ(&cv2[0], &mv(0, 1));
            REQUIRE_EQ(&cv3[0], &mv(0, 2));

            REQUIRE_EQ(&vr1[0], &mv(0, 0));
            REQUIRE_EQ(&vr2[0], &mv(0, 1));
            REQUIRE_EQ(&vr3[0], &mv(0, 2));

            REQUIRE_EQ(&cvr1[0], &mv(0, 0));
            REQUIRE_EQ(&cvr2[0], &mv(0, 1));
            REQUIRE_EQ(&cvr3[0], &mv(0, 2));

            REQUIRE_EQ(&mv1(0, 0), &mv(1, 2));
            REQUIRE_EQ(&mv1(1, 1), &mv(2, 3));
            REQUIRE_EQ(&mv1(3, 2), &mv(4, 4));
            REQUIRE_EQ(&mvr1(0, 0), &mv_right(1, 2));
            REQUIRE_EQ(&mvr1(1, 1), &mv_right(2, 3));
            REQUIRE_EQ(&mvr1(3, 2), &mv_right(4, 4));

            const_svector_type c_cv1(v1);
            typename svector_type::const_type c_cv2(v2);
            typename const_svector_type::const_type c_ccv2(v2);

            const_smultivector_type cmv(mv);
            typename smultivector_type::const_type cmvX(cmv);
            typename const_smultivector_type::const_type ccmvX(cmv);
        }

        static void run_test_layout() {
            flare::DynRankTensor<double> d("source", 1, 2, 3, 4);
            flare::DynRankTensor<double> e("dest");

            auto props = flare::tensor_alloc(flare::WithoutInitializing, d.label());
            e = flare::DynRankTensor<double>(props, d.layout());

            REQUIRE_EQ(d.rank(), 4u);
            REQUIRE_EQ(e.rank(), 4u);
            REQUIRE_EQ(e.label(), "source");

            auto ulayout = e.layout();
            REQUIRE_EQ(ulayout.dimension[0], 1u);
            REQUIRE_EQ(ulayout.dimension[1], 2u);
            REQUIRE_EQ(ulayout.dimension[2], 3u);
            REQUIRE_EQ(ulayout.dimension[3], 4u);
            REQUIRE_EQ(ulayout.dimension[4], FLARE_INVALID_INDEX);
            REQUIRE_EQ(ulayout.dimension[5], FLARE_INVALID_INDEX);
            REQUIRE_EQ(ulayout.dimension[6], FLARE_INVALID_INDEX);
            REQUIRE_EQ(ulayout.dimension[7], FLARE_INVALID_INDEX);
        }
    };

}  // namespace Test

/*--------------------------------------------------------------------------*/
