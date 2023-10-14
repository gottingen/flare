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

#include <flare/core.h>
#include <cstddef>
#include <doctest.h>

namespace Test {

#ifdef FLARE_ENABLE_CXX11_DISPATCH_LAMBDA
    namespace detail {
        template<class MemorySpaceA, class MemorySpaceB>
        struct TestDeepCopy {
            using a_base_t = flare::Tensor<double *, flare::LayoutRight, MemorySpaceA>;
            using b_base_t = flare::Tensor<double *, flare::LayoutRight, MemorySpaceB>;
            using a_char_t = flare::Tensor<char *, flare::LayoutRight, MemorySpaceA>;
            using b_char_t = flare::Tensor<char *, flare::LayoutRight, MemorySpaceB>;

            using policyA_t = flare::RangePolicy<typename MemorySpaceA::execution_space>;
            using policyB_t = flare::RangePolicy<typename MemorySpaceB::execution_space>;

            static void reset_a_copy_and_b(
                    flare::Tensor<char *, flare::LayoutRight, MemorySpaceA> a_char_copy,
                    flare::Tensor<char *, flare::LayoutRight, MemorySpaceB> b_char) {
                const int N = b_char.extent_int(0);
                flare::parallel_for(
                        "TestDeepCopy: FillA_copy", policyA_t(0, N),
                        FLARE_LAMBDA(const int &i) { a_char_copy(i) = char(0); });
                flare::parallel_for(
                        "TestDeepCopy: FillB", policyB_t(0, N),
                        FLARE_LAMBDA(const int &i) { b_char(i) = char(0); });
            }

            static int compare_equal(
                    flare::Tensor<char *, flare::LayoutRight, MemorySpaceA> a_char_copy,
                    flare::Tensor<char *, flare::LayoutRight, MemorySpaceA> a_char) {
                const int N = a_char.extent_int(0);
                int errors;
                flare::parallel_reduce(
                        "TestDeepCopy: FillA_copy", policyA_t(0, N),
                        FLARE_LAMBDA(const int &i, int &lsum) {
                            if (a_char_copy(i) != a_char(i)) lsum++;
                        },
                        errors);
                return errors;
            }

            static void run_test(int num_bytes) {
                a_base_t a_base("test_space_to_space", (num_bytes + 128) / 8);
                a_base_t a_base_copy("test_space_to_space", (num_bytes + 128) / 8);
                flare::Tensor<double *, flare::LayoutRight, MemorySpaceB> b_base(
                        "test_space_to_space", (num_bytes + 128) / 8);

                flare::Tensor<char *, flare::LayoutRight, MemorySpaceA> a_char(
                        (char *) a_base.data(), a_base.extent(0) * 8);
                flare::Tensor<char *, flare::LayoutRight, MemorySpaceA> a_char_copy(
                        (char *) a_base_copy.data(), a_base.extent(0) * 8);
                flare::Tensor<char *, flare::LayoutRight, MemorySpaceB> b_char(
                        (char *) b_base.data(), b_base.extent(0) * 8);

                flare::parallel_for(
                        "TestDeepCopy: FillA", policyA_t(0, a_char.extent(0)),
                        FLARE_LAMBDA(const int &i) {
                            a_char(i) = static_cast<char>(i % 97) + 1;
                        });

                reset_a_copy_and_b(a_char_copy, b_char);

                {
                    size_t check = compare_equal(a_char_copy, a_char);
                    REQUIRE_EQ(check, a_char.extent(0));
                }

                // (a.data()%8, (a.data()+a.extent(0))%8, b.data()%8,
                // (b.data()+b.extent(0))%8 (0,0,0,0)
                {
                    int a_begin = 0;
                    int a_end = 0;
                    int b_begin = 0;
                    int b_end = 0;
                    auto a = flare::subtensor(
                            a_char, std::pair<int, int>(a_begin, a_char.extent_int(0) - a_end));
                    auto b = flare::subtensor(
                            b_char, std::pair<int, int>(b_begin, b_char.extent_int(0) - b_end));
                    auto a_copy = flare::subtensor(
                            a_char_copy,
                            std::pair<int, int>(a_begin, a_char_copy.extent_int(0) - a_end));
                    flare::deep_copy(b, a);
                    flare::deep_copy(a_copy, b);
                    int check = compare_equal(a_copy, a);
                    REQUIRE_EQ(check, 0);
                }

                {
                    int a_begin = 0;
                    int a_end = 5;
                    int b_begin = 0;
                    int b_end = 5;
                    auto a = flare::subtensor(
                            a_char, std::pair<int, int>(a_begin, a_char.extent_int(0) - a_end));
                    auto b = flare::subtensor(
                            b_char, std::pair<int, int>(b_begin, b_char.extent_int(0) - b_end));
                    auto a_copy = flare::subtensor(
                            a_char_copy,
                            std::pair<int, int>(a_begin, a_char_copy.extent_int(0) - a_end));
                    flare::deep_copy(b, a);
                    flare::deep_copy(a_copy, b);
                    int check = compare_equal(a_copy, a);
                    REQUIRE_EQ(check, 0);
                }

                {
                    int a_begin = 3;
                    int a_end = 0;
                    int b_begin = 3;
                    int b_end = 0;
                    auto a = flare::subtensor(
                            a_char, std::pair<int, int>(a_begin, a_char.extent_int(0) - a_end));
                    auto b = flare::subtensor(
                            b_char, std::pair<int, int>(b_begin, b_char.extent_int(0) - b_end));
                    auto a_copy = flare::subtensor(
                            a_char_copy,
                            std::pair<int, int>(a_begin, a_char_copy.extent_int(0) - a_end));
                    flare::deep_copy(b, a);
                    flare::deep_copy(a_copy, b);
                    int check = compare_equal(a_copy, a);
                    REQUIRE_EQ(check, 0);
                }

                {
                    int a_begin = 3;
                    int a_end = 6;
                    int b_begin = 3;
                    int b_end = 6;
                    auto a = flare::subtensor(
                            a_char, std::pair<int, int>(a_begin, a_char.extent_int(0) - a_end));
                    auto b = flare::subtensor(
                            b_char, std::pair<int, int>(b_begin, b_char.extent_int(0) - b_end));
                    auto a_copy = flare::subtensor(
                            a_char_copy,
                            std::pair<int, int>(a_begin, a_char_copy.extent_int(0) - a_end));
                    flare::deep_copy(b, a);
                    flare::deep_copy(a_copy, b);
                    int check = compare_equal(a_copy, a);
                    REQUIRE_EQ(check, 0);
                }

                {
                    int a_begin = 5;
                    int a_end = 4;
                    int b_begin = 3;
                    int b_end = 6;
                    auto a = flare::subtensor(
                            a_char, std::pair<int, int>(a_begin, a_char.extent_int(0) - a_end));
                    auto b = flare::subtensor(
                            b_char, std::pair<int, int>(b_begin, b_char.extent_int(0) - b_end));
                    auto a_copy = flare::subtensor(
                            a_char_copy,
                            std::pair<int, int>(a_begin, a_char_copy.extent_int(0) - a_end));
                    flare::deep_copy(b, a);
                    flare::deep_copy(a_copy, b);
                    int check = compare_equal(a_copy, a);
                    REQUIRE_EQ(check, 0);
                }

                {
                    int a_begin = 0;
                    int a_end = 8;
                    int b_begin = 2;
                    int b_end = 6;
                    auto a = flare::subtensor(
                            a_char, std::pair<int, int>(a_begin, a_char.extent_int(0) - a_end));
                    auto b = flare::subtensor(
                            b_char, std::pair<int, int>(b_begin, b_char.extent_int(0) - b_end));
                    auto a_copy = flare::subtensor(
                            a_char_copy,
                            std::pair<int, int>(a_begin, a_char_copy.extent_int(0) - a_end));
                    flare::deep_copy(b, a);
                    flare::deep_copy(a_copy, b);
                    int check = compare_equal(a_copy, a);
                    REQUIRE_EQ(check, 0);
                }

                {
                    int a_begin = 2;
                    int a_end = 6;
                    int b_begin = 0;
                    int b_end = 8;
                    auto a = flare::subtensor(
                            a_char, std::pair<int, int>(a_begin, a_char.extent_int(0) - a_end));
                    auto b = flare::subtensor(
                            b_char, std::pair<int, int>(b_begin, b_char.extent_int(0) - b_end));
                    auto a_copy = flare::subtensor(
                            a_char_copy,
                            std::pair<int, int>(a_begin, a_char_copy.extent_int(0) - a_end));
                    flare::deep_copy(b, a);
                    flare::deep_copy(a_copy, b);
                    int check = compare_equal(a_copy, a);
                    REQUIRE_EQ(check, 0);
                }
            }
        };
    }  // namespace detail

    TEST_CASE("TEST_CATEGORY, deep_copy_alignment") {
        {
            detail::TestDeepCopy<TEST_EXECSPACE::memory_space,
                    TEST_EXECSPACE::memory_space>::run_test(100000);
        }
        {
            detail::TestDeepCopy<flare::HostSpace,
                    TEST_EXECSPACE::memory_space>::run_test(100000);
        }
        {
            detail::TestDeepCopy<TEST_EXECSPACE::memory_space,
                    flare::HostSpace>::run_test(100000);
        }
    }

#endif

    namespace detail {
        template<class Scalar1, class Scalar2, class Layout1, class Layout2>
        struct TestDeepCopyScalarConversion {
            struct TagFill {
            };
            struct TagCompare {
            };

            using tensor_type_s1_1d = flare::Tensor<Scalar1 *, Layout1, TEST_EXECSPACE>;
            using tensor_type_s2_1d = flare::Tensor<Scalar2 *, Layout2, TEST_EXECSPACE>;
            using tensor_type_s1_2d = flare::Tensor<Scalar1 **, Layout1, TEST_EXECSPACE>;
            using tensor_type_s2_2d = flare::Tensor<Scalar2 **, Layout2, TEST_EXECSPACE>;

            using base_layout1 =
                    std::conditional_t<std::is_same<Layout1, flare::LayoutStride>::value,
                            flare::LayoutLeft, Layout1>;
            using base_layout2 =
                    std::conditional_t<std::is_same<Layout2, flare::LayoutStride>::value,
                            flare::LayoutLeft, Layout2>;

            using base_type_s1_1d = flare::Tensor<Scalar1 *, base_layout1, TEST_EXECSPACE>;
            using base_type_s2_1d = flare::Tensor<Scalar2 *, base_layout2, TEST_EXECSPACE>;
            using base_type_s1_2d = flare::Tensor<Scalar1 **, base_layout1, TEST_EXECSPACE>;
            using base_type_s2_2d = flare::Tensor<Scalar2 **, base_layout2, TEST_EXECSPACE>;

            tensor_type_s1_1d tensor_s1_1d;
            tensor_type_s2_1d tensor_s2_1d;
            tensor_type_s1_2d tensor_s1_2d;
            tensor_type_s2_2d tensor_s2_2d;

            flare::Tensor<int64_t, TEST_EXECSPACE> error_count;

            void create_tensors(int64_t N0, int64_t N1) {
                base_type_s1_1d b_s1_1d("TestDeepCopyConversion::b_s1_1d", N0);
                base_type_s2_1d b_s2_1d("TestDeepCopyConversion::b_s2_1d", N0);
                base_type_s1_2d b_s1_2d("TestDeepCopyConversion::b_s1_2d", N0, N1);
                base_type_s2_2d b_s2_2d("TestDeepCopyConversion::b_s2_2d", N0, N1);

                tensor_s1_1d = tensor_type_s1_1d(b_s1_1d, flare::ALL);
                tensor_s2_1d = tensor_type_s2_1d(b_s2_1d, flare::ALL);
                tensor_s1_2d = tensor_type_s1_2d(b_s1_2d, flare::ALL, flare::ALL);
                tensor_s2_2d = tensor_type_s2_2d(b_s2_2d, flare::ALL, flare::ALL);

                error_count = flare::Tensor<int64_t, TEST_EXECSPACE>(
                        "TestDeepCopyConversion::error_count");
            }

            FLARE_FUNCTION
            void operator()(TagFill, const int64_t i) const {
                tensor_s2_1d(i) = static_cast<Scalar2>(i + 1);
                for (int64_t j = 0; j < static_cast<int64_t>(tensor_s2_2d.extent(1)); j++)
                    tensor_s2_2d(i, j) = static_cast<Scalar2>((i + 1) * 1000 + j + 1);
            }

            FLARE_FUNCTION
            void operator()(TagCompare, const int64_t i) const {
                int64_t errors = 0;
                if (tensor_s1_1d(i) != static_cast<Scalar1>(static_cast<Scalar2>(i + 1)))
                    errors++;
                for (int64_t j = 0; j < static_cast<int64_t>(tensor_s1_2d.extent(1)); j++) {
                    if (tensor_s1_2d(i, j) !=
                        static_cast<Scalar1>(static_cast<Scalar2>((i + 1) * 1000 + j + 1)))
                        errors++;
                }
                if (errors > 0) flare::atomic_add(&error_count(), errors);
            }

            void run_tests(int64_t N0, int64_t N1) {
                create_tensors(N0, N1);

                flare::parallel_for("TestDeepCopyConversion::Fill",
                                    flare::RangePolicy<TEST_EXECSPACE, TagFill,
                                            flare::IndexType<int64_t>>(0, N0),
                                    *this);

                flare::deep_copy(tensor_s1_1d, tensor_s2_1d);
                flare::deep_copy(tensor_s1_2d, tensor_s2_2d);

                flare::parallel_for("TestDeepCopyConversion::Compare",
                                    flare::RangePolicy<TEST_EXECSPACE, TagCompare,
                                            flare::IndexType<int64_t>>(0, N0),
                                    *this);

                int64_t errors = 0;
                flare::deep_copy(errors, error_count);
                REQUIRE_EQ(errors, 0);

                flare::deep_copy(tensor_s1_1d, static_cast<Scalar1>(0));
                flare::deep_copy(tensor_s1_2d, static_cast<Scalar1>(0));

                flare::parallel_for("TestDeepCopyConversion::Compare",
                                    flare::RangePolicy<TEST_EXECSPACE, TagCompare,
                                            flare::IndexType<int64_t>>(0, N0),
                                    *this);
                flare::deep_copy(errors, error_count);
                REQUIRE_GT(errors, 0);

                flare::deep_copy(error_count, 0);
                flare::deep_copy(TEST_EXECSPACE(), tensor_s1_1d, tensor_s2_1d);
                flare::deep_copy(TEST_EXECSPACE(), tensor_s1_2d, tensor_s2_2d);

                flare::parallel_for("TestDeepCopyConversion::Compare",
                                    flare::RangePolicy<TEST_EXECSPACE, TagCompare,
                                            flare::IndexType<int64_t>>(0, N0),
                                    *this);

                flare::deep_copy(errors, error_count);
                REQUIRE_EQ(errors, 0);
            }
        };
    }  // namespace detail

    TEST_CASE("TEST_CATEGORY, deep_copy_conversion") {
#ifdef FLARE_IMPL_32BIT
        GTEST_SKIP() << "Failing FLARE_IMPL_32BIT";  // FIXME_32BIT
#endif
        int64_t N0 = 19381;
        int64_t N1 = 17;

        using right = flare::LayoutRight;
        using left = flare::LayoutLeft;
        using stride = flare::LayoutStride;

        detail::TestDeepCopyScalarConversion<double, double, right, right>().run_tests(
                N0, N1);
        detail::TestDeepCopyScalarConversion<double, double, right, left>().run_tests(
                N0, N1);
        detail::TestDeepCopyScalarConversion<double, double, left, right>().run_tests(
                N0, N1);
        detail::TestDeepCopyScalarConversion<double, double, stride, right>().run_tests(
                N0, N1);
        detail::TestDeepCopyScalarConversion<double, double, right, stride>().run_tests(
                N0, N1);

        detail::TestDeepCopyScalarConversion<double, float, right, right>().run_tests(
                N0, N1);
        detail::TestDeepCopyScalarConversion<double, float, right, left>().run_tests(
                N0, N1);
        detail::TestDeepCopyScalarConversion<double, float, left, right>().run_tests(
                N0, N1);
        detail::TestDeepCopyScalarConversion<double, float, stride, right>().run_tests(
                N0, N1);
        detail::TestDeepCopyScalarConversion<double, float, right, stride>().run_tests(
                N0, N1);
    }
}  // namespace Test
