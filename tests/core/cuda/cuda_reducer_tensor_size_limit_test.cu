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

#include <cuda_category_test.h>
#include <flare/core.h>
#include <doctest.h>

namespace Test {

    using ValueType = double;
    using MemSpace = flare::CudaSpace;
    using Matrix2D = flare::Tensor<ValueType **, MemSpace>;
    using Matrix3D = flare::Tensor<ValueType ***, MemSpace>;
    using Vector = flare::Tensor<ValueType *, MemSpace>;

    namespace detail {

        struct ArrayReduceFunctor {
            using value_type = ValueType[];

            int value_count;
            Matrix2D m;

            ArrayReduceFunctor(const Matrix2D &m_) : value_count(m_.extent(1)), m(m_) {}

            FLARE_INLINE_FUNCTION void operator()(const int i, value_type sum) const {
                const int numVecs = value_count;
                for (int j = 0; j < numVecs; ++j) {
                    sum[j] += m(i, j);
                }
            }

            FLARE_INLINE_FUNCTION void init(value_type update) const {
                const int numVecs = value_count;
                for (int j = 0; j < numVecs; ++j) {
                    update[j] = 0.0;
                }
            }

            FLARE_INLINE_FUNCTION void join(value_type update,
                                            const value_type source) const {
                const int numVecs = value_count;
                for (int j = 0; j < numVecs; ++j) {
                    update[j] += source[j];
                }
            }

            FLARE_INLINE_FUNCTION void final(value_type) const {}
        };

        struct MDArrayReduceFunctor {
            using value_type = ValueType[];

            int value_count;
            Matrix3D m;

            MDArrayReduceFunctor(const Matrix3D &m_) : value_count(m_.extent(2)), m(m_) {}

            FLARE_INLINE_FUNCTION void operator()(const int i, const int j,
                                                  value_type sum) const {
                const int numVecs = value_count;
                for (int k = 0; k < numVecs; ++k) {
                    sum[k] += m(i, j, k);
                }
            }

            FLARE_INLINE_FUNCTION void init(value_type update) const {
                const int numVecs = value_count;
                for (int j = 0; j < numVecs; ++j) {
                    update[j] = 0.0;
                }
            }

            FLARE_INLINE_FUNCTION void final(value_type) const {}
        };

        struct ReduceTensorSizeLimitTester {
            const ValueType initValue = 3;
            const size_t nGlobalEntries = 100;
            const int testTensorSize = 200;
            const size_t expectedInitShmemLimit = 373584;
            const unsigned initBlockSize = flare::detail::CudaTraits::WarpSize * 8;

            void run_test_range() {
                Matrix2D matrix;
                Vector sum;

                for (int i = 0; i < testTensorSize; ++i) {
                    size_t sumInitShmemSize = (initBlockSize + 2) * sizeof(ValueType) * i;

                    flare::resize(flare::WithoutInitializing, sum, i);
                    flare::resize(flare::WithoutInitializing, matrix, nGlobalEntries, i);
                    flare::deep_copy(matrix, initValue);

                    auto policy = flare::RangePolicy<TEST_EXECSPACE>(0, nGlobalEntries);
                    auto functor = ArrayReduceFunctor(matrix);

                    if (sumInitShmemSize < expectedInitShmemLimit) {
                        REQUIRE_NOTHROW(flare::parallel_reduce(policy, functor, sum));
                    } else {
                        REQUIRE_THROWS_AS(flare::parallel_reduce(policy, functor, sum),
                                     std::runtime_error);
                    }
                }
            }

            void run_test_md_range_2D() {
                Matrix3D matrix;
                Vector sum;

                for (int i = 0; i < testTensorSize; ++i) {
                    size_t sumInitShmemSize = (initBlockSize + 2) * sizeof(ValueType) * i;

                    flare::resize(flare::WithoutInitializing, sum, i);
                    flare::resize(flare::WithoutInitializing, matrix, nGlobalEntries,
                                  nGlobalEntries, i);
                    flare::deep_copy(matrix, initValue);

                    auto policy = flare::MDRangePolicy<flare::Rank<2>>(
                            {0, 0}, {nGlobalEntries, nGlobalEntries});
                    auto functor = MDArrayReduceFunctor(matrix);

                    if (sumInitShmemSize < expectedInitShmemLimit) {
                        REQUIRE_NOTHROW(flare::parallel_reduce(policy, functor, sum));
                    } else {
                        REQUIRE_THROWS_AS(flare::parallel_reduce(policy, functor, sum),
                                     std::runtime_error);
                    }
                }
            }
        };

    }  // namespace detail

    TEST_CASE("cuda, reduceRangePolicyTensorSizeLimit") {
        detail::ReduceTensorSizeLimitTester reduceTensorSizeLimitTester;

        reduceTensorSizeLimitTester.run_test_range();
    }

    TEST_CASE("cuda, reduceMDRangePolicyTensorSizeLimit") {
        detail::ReduceTensorSizeLimitTester reduceTensorSizeLimitTester;

        reduceTensorSizeLimitTester.run_test_md_range_2D();
    }

}  // namespace Test
