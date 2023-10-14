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

#ifndef FLARE_ALGORITHMS_NESTED_SORT_TEST_H_
#define FLARE_ALGORITHMS_NESTED_SORT_TEST_H_

#include <doctest.h>
#include <unordered_set>
#include <random>
#include <flare/random.h>
#include <flare/nested_sort.h>

namespace Test {
    namespace NestedSortImpl {

// Comparator for sorting in descending order
        template<typename Key>
        struct GreaterThan {
            FLARE_FUNCTION constexpr bool operator()(const Key &lhs,
                                                     const Key &rhs) const {
                return lhs > rhs;
            }
        };

// Functor to test sort_team: each team responsible for sorting one array
        template<typename ExecSpace, typename KeyTensorType, typename OffsetTensorType>
        struct TeamSortFunctor {
            using TeamMem = typename flare::TeamPolicy<ExecSpace>::member_type;
            using SizeType = typename KeyTensorType::size_type;
            using KeyType = typename KeyTensorType::non_const_value_type;

            TeamSortFunctor(const KeyTensorType &keys_, const OffsetTensorType &offsets_,
                            bool sortDescending_)
                    : keys(keys_), offsets(offsets_), sortDescending(sortDescending_) {}

            FLARE_INLINE_FUNCTION void operator()(const TeamMem &t) const {
                int i = t.league_rank();
                SizeType begin = offsets(i);
                SizeType end = offsets(i + 1);
                if (sortDescending)
                    flare::experimental::sort_team(
                            t, flare::subtensor(keys, flare::make_pair(begin, end)),
                            GreaterThan<KeyType>());
                else
                    flare::experimental::sort_team(
                            t, flare::subtensor(keys, flare::make_pair(begin, end)));
            }

            KeyTensorType keys;
            OffsetTensorType offsets;
            bool sortDescending;
        };

// Functor to test sort_by_key_team: each team responsible for sorting one array
        template<typename ExecSpace, typename KeyTensorType, typename ValueTensorType,
                typename OffsetTensorType>
        struct TeamSortByKeyFunctor {
            using TeamMem = typename flare::TeamPolicy<ExecSpace>::member_type;
            using SizeType = typename KeyTensorType::size_type;
            using KeyType = typename KeyTensorType::non_const_value_type;

            TeamSortByKeyFunctor(const KeyTensorType &keys_, const ValueTensorType &values_,
                                 const OffsetTensorType &offsets_, bool sortDescending_)
                    : keys(keys_),
                      values(values_),
                      offsets(offsets_),
                      sortDescending(sortDescending_) {}

            FLARE_INLINE_FUNCTION void operator()(const TeamMem &t) const {
                int i = t.league_rank();
                SizeType begin = offsets(i);
                SizeType end = offsets(i + 1);
                if (sortDescending) {
                    flare::experimental::sort_by_key_team(
                            t, flare::subtensor(keys, flare::make_pair(begin, end)),
                            flare::subtensor(values, flare::make_pair(begin, end)),
                            GreaterThan<KeyType>());
                } else {
                    flare::experimental::sort_by_key_team(
                            t, flare::subtensor(keys, flare::make_pair(begin, end)),
                            flare::subtensor(values, flare::make_pair(begin, end)));
                }
            }

            KeyTensorType keys;
            ValueTensorType values;
            OffsetTensorType offsets;
            bool sortDescending;
        };

// Functor to test sort_thread: each thread (multiple vector lanes) responsible
// for sorting one array
        template<typename ExecSpace, typename KeyTensorType, typename OffsetTensorType>
        struct ThreadSortFunctor {
            using TeamMem = typename flare::TeamPolicy<ExecSpace>::member_type;
            using SizeType = typename KeyTensorType::size_type;
            using KeyType = typename KeyTensorType::non_const_value_type;

            ThreadSortFunctor(const KeyTensorType &keys_, const OffsetTensorType &offsets_,
                              bool sortDescending_)
                    : keys(keys_), offsets(offsets_), sortDescending(sortDescending_) {}

            FLARE_INLINE_FUNCTION void operator()(const TeamMem &t) const {
                int i = t.league_rank() * t.team_size() + t.team_rank();
                // Number of arrays to sort doesn't have to be divisible by team size, so
                // some threads may be idle.
                if (i < offsets.extent_int(0) - 1) {
                    SizeType begin = offsets(i);
                    SizeType end = offsets(i + 1);
                    if (sortDescending)
                        flare::experimental::sort_thread(
                                t, flare::subtensor(keys, flare::make_pair(begin, end)),
                                GreaterThan<KeyType>());
                    else
                        flare::experimental::sort_thread(
                                t, flare::subtensor(keys, flare::make_pair(begin, end)));
                }
            }

            KeyTensorType keys;
            OffsetTensorType offsets;
            bool sortDescending;
        };

// Functor to test sort_by_key_thread
        template<typename ExecSpace, typename KeyTensorType, typename ValueTensorType,
                typename OffsetTensorType>
        struct ThreadSortByKeyFunctor {
            using TeamMem = typename flare::TeamPolicy<ExecSpace>::member_type;
            using SizeType = typename KeyTensorType::size_type;
            using KeyType = typename KeyTensorType::non_const_value_type;

            ThreadSortByKeyFunctor(const KeyTensorType &keys_, const ValueTensorType &values_,
                                   const OffsetTensorType &offsets_, bool sortDescending_)
                    : keys(keys_),
                      values(values_),
                      offsets(offsets_),
                      sortDescending(sortDescending_) {}

            FLARE_INLINE_FUNCTION void operator()(const TeamMem &t) const {
                int i = t.league_rank() * t.team_size() + t.team_rank();
                // Number of arrays to sort doesn't have to be divisible by team size, so
                // some threads may be idle.
                if (i < offsets.extent_int(0) - 1) {
                    SizeType begin = offsets(i);
                    SizeType end = offsets(i + 1);
                    if (sortDescending) {
                        flare::experimental::sort_by_key_thread(
                                t, flare::subtensor(keys, flare::make_pair(begin, end)),
                                flare::subtensor(values, flare::make_pair(begin, end)),
                                GreaterThan<KeyType>());
                    } else {
                        flare::experimental::sort_by_key_thread(
                                t, flare::subtensor(keys, flare::make_pair(begin, end)),
                                flare::subtensor(values, flare::make_pair(begin, end)));
                    }
                }
            }

            KeyTensorType keys;
            ValueTensorType values;
            OffsetTensorType offsets;
            bool sortDescending;
        };

// Generate the offsets tensor for a set of n packed arrays, each with uniform
// random length in [0,k]. Array i will occupy the indices [offsets(i),
// offsets(i+1)), like a row in a CRS graph. Returns the total length of all the
// arrays.
        template<typename OffsetTensorType>
        size_t randomPackedArrayOffsets(unsigned n, unsigned k,
                                        OffsetTensorType &offsets) {
            offsets = OffsetTensorType("Offsets", n + 1);
            auto offsetsHost = flare::create_mirror_tensor(flare::HostSpace(), offsets);
            std::mt19937 gen;
            std::uniform_int_distribution<> distrib(0, k);
            // This will leave offsetsHost(n) == 0.
            std::generate(offsetsHost.data(), offsetsHost.data() + n,
                          [&]() { return distrib(gen); });
            // Exclusive prefix-sum to get offsets
            size_t accum = 0;
            for (unsigned i = 0; i <= n; i++) {
                size_t num = offsetsHost(i);
                offsetsHost(i) = accum;
                accum += num;
            }
            flare::deep_copy(offsets, offsetsHost);
            return offsetsHost(n);
        }

        template<typename ValueTensorType>
        ValueTensorType uniformRandomTensorFill(size_t totalLength,
                                            typename ValueTensorType::value_type minVal,
                                            typename ValueTensorType::value_type maxVal) {
            ValueTensorType vals("vals", totalLength);
            flare::Random_XorShift64_Pool<typename ValueTensorType::execution_space> g(
                    1931);
            flare::fill_random(vals, g, minVal, maxVal);
            return vals;
        }

        template<class ExecutionSpace, typename KeyType>
        void test_nested_sort_impl(unsigned narray, unsigned n, bool useTeams,
                                   bool customCompare, KeyType minKey, KeyType maxKey) {
            using KeyTensorType = flare::Tensor<KeyType *, ExecutionSpace>;
            using OffsetTensorType = flare::Tensor<unsigned *, ExecutionSpace>;
            using TeamPol = flare::TeamPolicy<ExecutionSpace>;
            OffsetTensorType offsets;
            size_t totalLength = randomPackedArrayOffsets(narray, n, offsets);
            KeyTensorType keys =
                    uniformRandomTensorFill<KeyTensorType>(totalLength, minKey, maxKey);
            // note: doing create_mirror because we always want this to be a separate
            // copy, even if keys is already host-accessible. keysHost becomes the correct
            // result to compare against.
            auto keysHost = flare::create_mirror(flare::HostSpace(), keys);
            flare::deep_copy(keysHost, keys);
            auto offsetsHost =
                    flare::create_mirror_tensor_and_copy(flare::HostSpace(), offsets);
            // Sort the same arrays on host to compare against
            for (unsigned i = 0; i < narray; i++) {
                KeyType *begin = keysHost.data() + offsetsHost(i);
                KeyType *end = keysHost.data() + offsetsHost(i + 1);
                if (customCompare)
                    std::sort(begin, end,
                              [](const KeyType &a, const KeyType &b) { return a > b; });
                else
                    std::sort(begin, end);
            }
            if (useTeams) {
                int vectorLen = std::min<int>(4, TeamPol::vector_length_max());
                TeamPol policy(narray, flare::AUTO(), vectorLen);
                flare::parallel_for(
                        policy, TeamSortFunctor<ExecutionSpace, KeyTensorType, OffsetTensorType>(
                                keys, offsets, customCompare));
            } else {
                ThreadSortFunctor<ExecutionSpace, KeyTensorType, OffsetTensorType> functor(
                        keys, offsets, customCompare);
                int vectorLen = std::min<int>(4, TeamPol::vector_length_max());
                TeamPol dummy(1, flare::AUTO(), vectorLen);
                int teamSize =
                        dummy.team_size_recommended(functor, flare::ParallelForTag());
                int numTeams = (narray + teamSize - 1) / teamSize;
                flare::parallel_for(TeamPol(numTeams, teamSize, vectorLen), functor);
            }
            auto keysOut = flare::create_mirror_tensor_and_copy(flare::HostSpace(), keys);
            std::string testLabel = useTeams ? "sort_team" : "sort_thread";
            for (unsigned i = 0; i < keys.extent(0); i++) {
                REQUIRE_EQ(keysOut(i), keysHost(i));
            }
        }

        template<class ExecutionSpace, typename KeyType, typename ValueType>
        void test_nested_sort_by_key_impl(unsigned narray, unsigned n, bool useTeams,
                                          bool customCompare, KeyType minKey,
                                          KeyType maxKey, ValueType minVal,
                                          ValueType maxVal) {
            using KeyTensorType = flare::Tensor<KeyType *, ExecutionSpace>;
            using ValueTensorType = flare::Tensor<ValueType *, ExecutionSpace>;
            using OffsetTensorType = flare::Tensor<unsigned *, ExecutionSpace>;
            using TeamPol = flare::TeamPolicy<ExecutionSpace>;
            OffsetTensorType offsets;
            size_t totalLength = randomPackedArrayOffsets(narray, n, offsets);
            KeyTensorType keys =
                    uniformRandomTensorFill<KeyTensorType>(totalLength, minKey, maxKey);
            ValueTensorType values =
                    uniformRandomTensorFill<ValueTensorType>(totalLength, minVal, maxVal);
            // note: doing create_mirror because we always want this to be a separate
            // copy, even if keys/vals are already host-accessible. keysHost and valsHost
            // becomes the correct result to compare against.
            auto keysHost = flare::create_mirror(flare::HostSpace(), keys);
            auto valuesHost = flare::create_mirror(flare::HostSpace(), values);
            flare::deep_copy(keysHost, keys);
            flare::deep_copy(valuesHost, values);
            auto offsetsHost =
                    flare::create_mirror_tensor_and_copy(flare::HostSpace(), offsets);
            // Sort the same arrays on host to compare against
            for (unsigned i = 0; i < narray; i++) {
                // std:: doesn't have a sort_by_key, so sort a vector of key-value pairs
                // instead
                using KV = std::pair<KeyType, ValueType>;
                std::vector<KV> keysAndValues(offsetsHost(i + 1) - offsetsHost(i));
                for (unsigned j = 0; j < keysAndValues.size(); j++) {
                    keysAndValues[j].first = keysHost(offsetsHost(i) + j);
                    keysAndValues[j].second = valuesHost(offsetsHost(i) + j);
                }
                if (customCompare) {
                    std::sort(keysAndValues.begin(), keysAndValues.end(),
                              [](const KV &a, const KV &b) { return a.first > b.first; });
                } else {
                    std::sort(keysAndValues.begin(), keysAndValues.end(),
                              [](const KV &a, const KV &b) { return a.first < b.first; });
                }
                // Copy back from pairs to tensors
                for (unsigned j = 0; j < keysAndValues.size(); j++) {
                    keysHost(offsetsHost(i) + j) = keysAndValues[j].first;
                    valuesHost(offsetsHost(i) + j) = keysAndValues[j].second;
                }
            }
            if (useTeams) {
                int vectorLen = std::min<int>(4, TeamPol::vector_length_max());
                TeamPol policy(narray, flare::AUTO(), vectorLen);
                flare::parallel_for(
                        policy, TeamSortByKeyFunctor<ExecutionSpace, KeyTensorType, ValueTensorType,
                                OffsetTensorType>(keys, values, offsets,
                                                customCompare));
            } else {
                ThreadSortByKeyFunctor<ExecutionSpace, KeyTensorType, ValueTensorType,
                        OffsetTensorType>
                        functor(keys, values, offsets, customCompare);
                int vectorLen = std::min<int>(4, TeamPol::vector_length_max());
                TeamPol dummy(1, flare::AUTO(), vectorLen);
                int teamSize =
                        dummy.team_size_recommended(functor, flare::ParallelForTag());
                int numTeams = (narray + teamSize - 1) / teamSize;
                flare::parallel_for(TeamPol(numTeams, teamSize, vectorLen), functor);
            }
            auto keysOut = flare::create_mirror_tensor_and_copy(flare::HostSpace(), keys);
            auto valuesOut =
                    flare::create_mirror_tensor_and_copy(flare::HostSpace(), values);
            std::string testLabel = useTeams ? "sort_by_key_team" : "sort_by_key_thread";
            // First, compare keys since they will always match exactly
            for (unsigned i = 0; i < keys.extent(0); i++) {
                REQUIRE_EQ(keysOut(i), keysHost(i));
            }
            // flare::sort_by_key_X is not stable, so if a key happens to
            // appear more than once, the order of the values may not match exactly.
            // But the set of values for a given key should be identical.
            unsigned keyStart = 0;
            while (keyStart < keys.extent(0)) {
                KeyType key = keysHost(keyStart);
                unsigned keyEnd = keyStart + 1;
                while (keyEnd < keys.extent(0) && keysHost(keyEnd) == key) keyEnd++;
                std::unordered_multiset < ValueType > correctVals;
                std::unordered_multiset < ValueType > outputVals;
                for (unsigned i = keyStart; i < keyEnd; i++) {
                    correctVals.insert(valuesHost(i));
                    outputVals.insert(valuesOut(i));
                }
                // Check one value at a time that they match
                for (auto it = correctVals.begin(); it != correctVals.end(); it++) {
                    ValueType val = *it;
                    REQUIRE(outputVals.find(val) != outputVals.end());
                    REQUIRE_EQ(correctVals.count(val), outputVals.count(val));
                }
                keyStart = keyEnd;
            }
        }

        template<class ExecutionSpace, typename KeyType>
        void test_nested_sort(unsigned int N, KeyType minKey, KeyType maxKey) {
            // 2nd arg: true = team-level, false = thread-level.
            // 3rd arg: true = custom comparator, false = default comparator.
            test_nested_sort_impl<ExecutionSpace, KeyType>(N, N, true, false, minKey,
                                                           maxKey);
            test_nested_sort_impl<ExecutionSpace, KeyType>(N, N, true, true, minKey,
                                                           maxKey);
            test_nested_sort_impl<ExecutionSpace, KeyType>(N, N, false, false, minKey,
                                                           maxKey);
            test_nested_sort_impl<ExecutionSpace, KeyType>(N, N, false, true, minKey,
                                                           maxKey);
        }

        template<class ExecutionSpace, typename KeyType, typename ValueType>
        void test_nested_sort_by_key(unsigned int N, KeyType minKey, KeyType maxKey,
                                     ValueType minVal, ValueType maxVal) {
            // 2nd arg: true = team-level, false = thread-level.
            // 3rd arg: true = custom comparator, false = default comparator.
            test_nested_sort_by_key_impl<ExecutionSpace, KeyType, ValueType>(
                    N, N, true, false, minKey, maxKey, minVal, maxVal);
            test_nested_sort_by_key_impl<ExecutionSpace, KeyType, ValueType>(
                    N, N, true, true, minKey, maxKey, minVal, maxVal);
            test_nested_sort_by_key_impl<ExecutionSpace, KeyType, ValueType>(
                    N, N, false, false, minKey, maxKey, minVal, maxVal);
            test_nested_sort_by_key_impl<ExecutionSpace, KeyType, ValueType>(
                    N, N, false, true, minKey, maxKey, minVal, maxVal);
        }
    }  // namespace NestedSortImpl

    TEST_CASE("TEST_CATEGORY, NestedSort") {
        using ExecutionSpace = TEST_EXECSPACE;
        NestedSortImpl::test_nested_sort<ExecutionSpace, unsigned>(171, 0U, UINT_MAX);
        NestedSortImpl::test_nested_sort<ExecutionSpace, float>(42, -1e6f, 1e6f);
        NestedSortImpl::test_nested_sort<ExecutionSpace, char>(67, CHAR_MIN,
                                                               CHAR_MAX);
    }

    TEST_CASE("TEST_CATEGORY, NestedSortByKey") {
        using ExecutionSpace = TEST_EXECSPACE;

        // Second/third template arguments are key and value respectively.
        // In sort_by_key_X functions, a key tensor and a value tensor are both permuted
        // to make the keys sorted. This means that the value type doesn't need to be
        // ordered, unlike key
        NestedSortImpl::test_nested_sort_by_key<ExecutionSpace, unsigned, unsigned>(
                161, 0U, UINT_MAX, 0U, UINT_MAX);
        NestedSortImpl::test_nested_sort_by_key<ExecutionSpace, float, char>(
                267, -1e6f, 1e6f, CHAR_MIN, CHAR_MAX);
        NestedSortImpl::test_nested_sort_by_key<ExecutionSpace, char, double>(
                11, CHAR_MIN, CHAR_MAX, 2.718, 3.14);
    }

}  // namespace Test
#endif  // FLARE_ALGORITHMS_NESTED_SORT_TEST_H_
