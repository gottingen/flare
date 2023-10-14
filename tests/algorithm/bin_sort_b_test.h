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

#ifndef FLARE_ALGORITHMS_BIN_SORT_B_TEST_H_
#define FLARE_ALGORITHMS_BIN_SORT_B_TEST_H_

#include <doctest.h>
#include <flare/core.h>
#include <flare/random.h>
#include <flare/sort.h>
#include <flare/algorithm.h>
#include <std_algorithms_helper_functors_test.h>
#include <random>
#include <numeric>  //needed for iota

namespace Test {
    namespace BinSortSetB {

        template<class TensorTypeFrom, class TensorTypeTo>
        struct CopyFunctorRank2 {
            TensorTypeFrom m_tensor_from;
            TensorTypeTo m_tensor_to;

            CopyFunctorRank2() = delete;

            CopyFunctorRank2(const TensorTypeFrom tensor_from, const TensorTypeTo tensor_to)
                    : m_tensor_from(tensor_from), m_tensor_to(tensor_to) {}

            FLARE_INLINE_FUNCTION
            void operator()(int k) const {
                const auto i = k / m_tensor_from.extent(1);
                const auto j = k % m_tensor_from.extent(1);
                m_tensor_to(i, j) = m_tensor_from(i, j);
            }
        };

        template<class ValueType, class... Props>
        auto create_deep_copyable_compatible_tensor_with_same_extent(
                flare::Tensor<ValueType *, Props...> tensor) {
            using tensor_type = flare::Tensor<ValueType *, Props...>;
            using tensor_value_type = typename tensor_type::value_type;
            using tensor_exespace = typename tensor_type::execution_space;
            const std::size_t ext0 = tensor.extent(0);
            using tensor_deep_copyable_t = flare::Tensor<tensor_value_type *, tensor_exespace>;
            return tensor_deep_copyable_t{"tensor_dc", ext0};
        }

        template<class ValueType, class... Props>
        auto create_deep_copyable_compatible_tensor_with_same_extent(
                flare::Tensor<ValueType **, Props...> tensor) {
            using tensor_type = flare::Tensor<ValueType **, Props...>;
            using tensor_value_type = typename tensor_type::value_type;
            using tensor_exespace = typename tensor_type::execution_space;
            using tensor_deep_copyable_t = flare::Tensor<tensor_value_type **, tensor_exespace>;
            const std::size_t ext0 = tensor.extent(0);
            const std::size_t ext1 = tensor.extent(1);
            return tensor_deep_copyable_t{"tensor_dc", ext0, ext1};
        }

        template<class TensorType>
        auto create_deep_copyable_compatible_clone(TensorType tensor) {
            static_assert(TensorType::rank <= 2);

            auto tensor_dc = create_deep_copyable_compatible_tensor_with_same_extent(tensor);
            using tensor_dc_t = decltype(tensor_dc);
            if constexpr (TensorType::rank == 1) {
                Test::stdalgos::CopyFunctor<TensorType, tensor_dc_t> F1(tensor, tensor_dc);
                flare::parallel_for("copy", tensor.extent(0), F1);
            } else {
                static_assert(TensorType::rank == 2, "Only rank 1 or 2 supported.");
                CopyFunctorRank2<TensorType, tensor_dc_t> F1(tensor, tensor_dc);
                flare::parallel_for("copy", tensor.extent(0) * tensor.extent(1), F1);
            }
            return tensor_dc;
        }

        template<class TensorType>
        auto create_host_space_copy(TensorType tensor) {
            auto tensor_dc = create_deep_copyable_compatible_clone(tensor);
            return create_mirror_tensor_and_copy(flare::HostSpace(), tensor_dc);
        }

        template<class KeyType, class ExecutionSpace>
        auto create_rank1_dev_and_host_tensors_of_keys(const ExecutionSpace &exec,
                                                     int N) {
            namespace KE = flare::experimental;
            flare::DefaultHostExecutionSpace defaultHostExeSpace;

            using KeyTensorType = flare::Tensor<KeyType *, ExecutionSpace>;
            KeyTensorType keys("keys", N);
            auto keys_h = flare::create_mirror_tensor(keys);
            std::iota(KE::begin(keys_h), KE::end(keys_h), KeyType(0));
            KE::reverse(defaultHostExeSpace, keys_h);
            // keys now is = [N-1,N-2,...,2,1,0], shuffle it for avoid trivial case
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(KE::begin(keys_h), KE::end(keys_h), g);
            flare::deep_copy(exec, keys, keys_h);

            return std::make_pair(keys, keys_h);
        }

        template<class ExecutionSpace, class ValueType, int ValuesTensorRank,
                std::enable_if_t<ValuesTensorRank == 1, int> = 0>
        auto create_strided_tensor(std::size_t numRows, std::size_t /*numCols*/) {
            flare::LayoutStride layout{numRows, 2};
            using v_t = flare::Tensor<ValueType *, flare::LayoutStride, ExecutionSpace>;
            v_t v("v", layout);
            return v;
        }

        template<class ExecutionSpace, class ValueType, int ValuesTensorRank,
                std::enable_if_t<ValuesTensorRank == 2, int> = 0>
        auto create_strided_tensor(std::size_t numRows, std::size_t numCols) {
            flare::LayoutStride layout{numRows, 2, numCols, numRows * 2};
            using v_t = flare::Tensor<ValueType **, flare::LayoutStride, ExecutionSpace>;
            v_t v("v", layout);
            return v;
        }

        template<class ExecutionSpace, class KeyType, class ValueType, int ValuesTensorRank>
        void
        test_on_tensor_with_stride(std::size_t numRows, std::size_t indB, std::size_t indE, std::size_t numCols = 1) {
            ExecutionSpace exec;
            flare::DefaultHostExecutionSpace defaultHostExeSpace;
            namespace KE = flare::experimental;

            // 1. generate 1D tensor of keys
            auto [keys, keys_h] =
                    create_rank1_dev_and_host_tensors_of_keys<KeyType>(exec, numRows);
            using KeyTensorType = decltype(keys);

            // need this map key->row to use later for checking
            std::unordered_map<KeyType, std::size_t> keyToRowBeforeSort;
            for (std::size_t i = 0; i < numRows; ++i) {
                keyToRowBeforeSort[keys_h(i)] = i;
            }

            // 2. create binOp
            using BinOp = flare::BinOp1D<KeyTensorType>;
            auto itB = KE::cbegin(keys_h) + indB;
            auto itE = itB + indE - indB;
            auto it = KE::minmax_element(defaultHostExeSpace, itB, itE);
            // seems like the behavior is odd when we use # buckets = # keys
            // so use +5 for using more buckets than keys.
            // This is something to investigate.
            BinOp binner(indE - indB + 5, *it.first, *it.second);

            // 3. create sorter
            flare::BinSort<KeyTensorType, BinOp> sorter(keys, indB, indE, binner, false);
            sorter.create_permute_vector(exec);
            sorter.sort(exec, keys, indB, indE);
            flare::deep_copy(exec, keys_h, keys);

            auto v = create_strided_tensor<ExecutionSpace, ValueType, ValuesTensorRank>(
                    numRows, numCols);

            flare::Random_XorShift64_Pool<ExecutionSpace> pool(73931);
            flare::fill_random(v, pool, ValueType(545));
            auto v_before_sort_h = create_host_space_copy(v);
            sorter.sort(exec, v, indB, indE);
            auto v_after_sort_h = create_host_space_copy(v);

            for (size_t i = 0; i < v.extent(0); ++i) {
                // if i within [indB,indE), the sorting was done
                // so we need to do proper checking since rows have changed
                if (i >= size_t(indB) && i < size_t(indE)) {
                    const KeyType key = keys_h(i);
                    if constexpr (ValuesTensorRank == 1) {
                        REQUIRE(v_before_sort_h(keyToRowBeforeSort.at(key)) ==
                                v_after_sort_h(i));
                    } else {
                        for (size_t j = 0; j < v.extent(1); ++j) {
                            REQUIRE(v_before_sort_h(keyToRowBeforeSort.at(key), j) ==
                                    v_after_sort_h(i, j));
                        }
                    }
                } else { // outside the target bounds, then the i-th row remains unchanged
                    if constexpr (ValuesTensorRank == 1) {
                        REQUIRE(v_before_sort_h(i) == v_after_sort_h(i));
                    } else {
                        for (size_t j = 0; j < v.extent(1); ++j) {
                            REQUIRE(v_before_sort_h(i, j) == v_after_sort_h(i, j));
                        }
                    }
                }
            }
        }

        template<class ExecutionSpace, class KeyType, class ValueType>
        void run_for_rank1() {
            constexpr int rank = 1;

            // trivial case
            test_on_tensor_with_stride<ExecutionSpace, KeyType, ValueType, rank>(1, 0, 1);
            test_on_tensor_with_stride<ExecutionSpace, KeyType, ValueType, rank>(4, 0, 4);
            // nontrivial cases
            /*
            for (std::size_t N: {311, 710017}) {
                // various cases for bounds
                test_on_tensor_with_stride<ExecutionSpace, KeyType, ValueType, rank>(N, 0, N);
                test_on_tensor_with_stride<ExecutionSpace, KeyType, ValueType, rank>(N, 3, N);
                test_on_tensor_with_stride<ExecutionSpace, KeyType, ValueType, rank>(N, 0,N - 4);
                test_on_tensor_with_stride<ExecutionSpace, KeyType, ValueType, rank>(N, 4,N - 3);
            }
             */
        }

        template<class ExecutionSpace, class KeyType, class ValueType>
        void run_for_rank2() {
            constexpr int rank = 2;

            // trivial case
            test_on_tensor_with_stride<ExecutionSpace, KeyType, ValueType, rank>(1, 0, 1, 1);

            // nontrivial cases
            /*
            for (std::size_t Nr: {11, 1157, 710017}) {
                for (std::size_t Nc: {3, 51}) {
                    // various cases for bounds
                    test_on_tensor_with_stride<ExecutionSpace, KeyType, ValueType, rank>(
                            Nr, 0, Nr, Nc);
                    test_on_tensor_with_stride<ExecutionSpace, KeyType, ValueType, rank>(
                            Nr, 3, Nr, Nc);
                    test_on_tensor_with_stride<ExecutionSpace, KeyType, ValueType, rank>(
                            Nr, 0, Nr - 4, Nc);
                    test_on_tensor_with_stride<ExecutionSpace, KeyType, ValueType, rank>(
                            Nr, 4, Nr - 3, Nc);
                }
            }
             */
        }

    }  // namespace BinSortSetB

    TEST_CASE("TEST_CATEGORY, BinSortUnsignedKeyLayoutStrideValues") {
        using ExeSpace = TEST_EXECSPACE;
        using key_type = unsigned;
        BinSortSetB::run_for_rank1<ExeSpace, key_type, int>();
        BinSortSetB::run_for_rank1<ExeSpace, key_type, double>();

        BinSortSetB::run_for_rank2<ExeSpace, key_type, int>();
        BinSortSetB::run_for_rank2<ExeSpace, key_type, double>();
    }

}  // namespace Test
#endif  // FLARE_ALGORITHMS_BIN_SORT_B_TEST_H_
