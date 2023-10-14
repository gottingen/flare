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

#include <std_algorithms_common_test.h>

namespace Test::stdalgos::TeamSearchN {

    namespace KE = flare::experimental;

    template<class ValueType>
    struct EqualFunctor {
        FLARE_INLINE_FUNCTION
        bool operator()(const ValueType &lhs, const ValueType &rhs) const {
            return lhs == rhs;
        }
    };

    template<class DatATensorType, class SearchedValuesTensorType,
            class DistancesTensorType, class BinaryPredType>
    struct TestFunctorA {
        DatATensorType m_dataTensor;
        std::size_t m_seqSize;
        SearchedValuesTensorType m_searchedValuesTensor;
        DistancesTensorType m_distancesTensor;
        BinaryPredType m_binaryPred;
        int m_apiPick;

        TestFunctorA(const DatATensorType dataTensor, std::size_t seqSize,
                     const SearchedValuesTensorType searchedValuesTensor,
                     const DistancesTensorType distancesTensor, BinaryPredType binaryPred,
                     int apiPick)
                : m_dataTensor(dataTensor),
                  m_seqSize(seqSize),
                  m_searchedValuesTensor(searchedValuesTensor),
                  m_distancesTensor(distancesTensor),
                  m_binaryPred(binaryPred),
                  m_apiPick(apiPick) {}

        template<class MemberType>
        FLARE_INLINE_FUNCTION void operator()(const MemberType &member) const {
            const auto myRowIndex = member.league_rank();
            auto myRowTensorFrom = flare::subtensor(m_dataTensor, myRowIndex, flare::ALL());
            auto rowFromBegin = KE::begin(myRowTensorFrom);
            auto rowFromEnd = KE::end(myRowTensorFrom);
            const auto searchedValue = m_searchedValuesTensor(myRowIndex);

            switch (m_apiPick) {
                case 0: {
                    const auto it = KE::search_n(member, rowFromBegin, rowFromEnd,
                                                 m_seqSize, searchedValue);
                    flare::single(flare::PerTeam(member), [=, *this]() {
                        m_distancesTensor(myRowIndex) = KE::distance(rowFromBegin, it);
                    });

                    break;
                }

                case 1: {
                    const auto it =
                            KE::search_n(member, myRowTensorFrom, m_seqSize, searchedValue);
                    flare::single(flare::PerTeam(member), [=, *this]() {
                        m_distancesTensor(myRowIndex) = KE::distance(rowFromBegin, it);
                    });

                    break;
                }

                case 2: {
                    const auto it = KE::search_n(member, rowFromBegin, rowFromEnd,
                                                 m_seqSize, searchedValue, m_binaryPred);
                    flare::single(flare::PerTeam(member), [=, *this]() {
                        m_distancesTensor(myRowIndex) = KE::distance(rowFromBegin, it);
                    });

                    break;
                }

                case 3: {
                    const auto it = KE::search_n(member, myRowTensorFrom, m_seqSize,
                                                 searchedValue, m_binaryPred);
                    flare::single(flare::PerTeam(member), [=, *this]() {
                        m_distancesTensor(myRowIndex) = KE::distance(rowFromBegin, it);
                    });

                    break;
                }
            }
        }
    };

    template<class LayoutTag, class ValueType>
    void test_A(const bool sequencesExist, std::size_t numTeams,
                std::size_t numCols, int apiId) {
        /* description:
           use a rank-2 tensor randomly filled with values,
           and run a team-level search_n
         */

        // -----------------------------------------------
        // prepare data
        // -----------------------------------------------
        // create a tensor in the memory space associated with default exespace
        // with as many rows as the number of teams and fill it with random
        // values from an arbitrary range.
        constexpr ValueType lowerBound = 5;
        constexpr ValueType upperBound = 523;
        const auto bounds = make_bounds(lowerBound, upperBound);

        auto [dataTensor, dataTensorBeforeOp_h] = create_random_tensor_and_host_clone(
                LayoutTag{}, numTeams, numCols, bounds, "dataTensor");

        // If sequencesExist == true we need to inject some sequence of count test
        // value into dataTensor. If sequencesExist == false we set searchedVal to a
        // value that is not present in dataTensor

        const std::size_t halfCols = (numCols > 1) ? ((numCols + 1) / 2) : (1);
        const std::size_t seqSize = (numCols > 1) ? (std::log2(numCols)) : (1);

        flare::Tensor<ValueType *> searchedValuesTensor("searchedValuesTensor", numTeams);
        auto searchedValuesTensor_h = create_host_space_copy(searchedValuesTensor);

        // dataTensor might not deep copyable (e.g. strided layout) so to prepare it
        // correclty, we make a new tensor that is for sure deep copyable, modify it
        // on the host, deep copy to device and then launch a kernel to copy to
        // dataTensor
        auto dataTensor_dc =
                create_deep_copyable_compatible_tensor_with_same_extent(dataTensor);
        auto dataTensor_dc_h = create_mirror_tensor(flare::HostSpace(), dataTensor_dc);

        if (sequencesExist) {
            const std::size_t dataBegin = halfCols - seqSize;
            for (std::size_t i = 0; i < searchedValuesTensor.extent(0); ++i) {
                const ValueType searchedVal = dataTensor_dc_h(i, dataBegin);
                searchedValuesTensor_h(i) = searchedVal;

                for (std::size_t j = dataBegin + 1; j < seqSize; ++j) {
                    dataTensor_dc_h(i, j) = searchedVal;
                }
            }

            // copy to dataTensor_dc and then to dataTensor
            flare::deep_copy(dataTensor_dc, dataTensor_dc_h);

            CopyFunctorRank2 cpFun(dataTensor_dc, dataTensor);
            flare::parallel_for("copy", dataTensor.extent(0) * dataTensor.extent(1),
                                cpFun);
        } else {
            using rand_pool =
                    flare::Random_XorShift64_Pool<flare::DefaultHostExecutionSpace>;
            rand_pool pool(lowerBound * upperBound);
            flare::fill_random(searchedValuesTensor_h, pool, upperBound, upperBound * 2);
        }

        flare::deep_copy(searchedValuesTensor, searchedValuesTensor_h);

        // -----------------------------------------------
        // launch flare kernel
        // -----------------------------------------------
        using space_t = flare::DefaultExecutionSpace;
        flare::TeamPolicy<space_t> policy(numTeams, flare::AUTO());

        // search_n returns an iterator so to verify that it is correct each team
        // stores the distance of the returned iterator from the beginning of the
        // interval that team operates on and then we check that these distances match
        // the std result
        flare::Tensor<std::size_t *> distancesTensor("distancesTensor", numTeams);

        EqualFunctor<ValueType> binaryPred;

        // use CTAD for functor
        TestFunctorA fnc(dataTensor, seqSize, searchedValuesTensor, distancesTensor,
                         binaryPred, apiId);
        flare::parallel_for(policy, fnc);

        // -----------------------------------------------
        // run cpp-std kernel and check
        // -----------------------------------------------
        auto distancesTensor_h = create_host_space_copy(distancesTensor);

        for (std::size_t i = 0; i < dataTensor.extent(0); ++i) {
            auto rowFrom = flare::subtensor(dataTensor_dc_h, i, flare::ALL());

            const auto rowFromBegin = KE::cbegin(rowFrom);
            const auto rowFromEnd = KE::cend(rowFrom);

            const ValueType searchedVal = searchedValuesTensor_h(i);

            const std::size_t beginEndDist = KE::distance(rowFromBegin, rowFromEnd);

            switch (apiId) {
                case 0:
                case 1: {
                    const auto it =
                            std::search_n(rowFromBegin, rowFromEnd, seqSize, searchedVal);
                    const std::size_t stdDistance = KE::distance(rowFromBegin, it);

                    if (sequencesExist) {
                        REQUIRE_LT(distancesTensor_h(i), beginEndDist);
                    } else {
                        REQUIRE_EQ(distancesTensor_h(i), beginEndDist);
                    }

                    REQUIRE_EQ(stdDistance, distancesTensor_h(i));

                    break;
                }

                case 2:
                case 3: {
                    const auto it = std::search_n(rowFromBegin, rowFromEnd, seqSize,
                                                  searchedVal, binaryPred);
                    const std::size_t stdDistance = KE::distance(rowFromBegin, it);

                    if (sequencesExist) {
                        REQUIRE_LT(distancesTensor_h(i), beginEndDist);
                    } else {
                        REQUIRE_EQ(distancesTensor_h(i), beginEndDist);
                    }

                    REQUIRE_EQ(stdDistance, distancesTensor_h(i));

                    break;
                }
            }
        }
    }

    template<class LayoutTag, class ValueType>
    void run_all_scenarios(const bool sequencesExist) {
        for (int numTeams: teamSizesToTest) {
            for (const auto &numCols: {2, 13, 101, 1444, 8153}) {
                for (int apiId: {0, 1}) {
                    test_A<LayoutTag, ValueType>(sequencesExist, numTeams, numCols, apiId);
                }
            }
        }
    }

    TEST_CASE("std_algorithms_search_n_team_test, sequences_of_equal_elements_exist") {
        constexpr bool sequencesExist = true;

        run_all_scenarios<DynamicTag, double>(sequencesExist);
        run_all_scenarios<StridedTwoRowsTag, int>(sequencesExist);
        run_all_scenarios<StridedThreeRowsTag, unsigned>(sequencesExist);
    }

    TEST_CASE("std_algorithms_search_n_team_test, sequences_of_equal_elements_probably_does_not_exist") {
        constexpr bool sequencesExist = false;

        run_all_scenarios<DynamicTag, double>(sequencesExist);
        run_all_scenarios<StridedTwoRowsTag, int>(sequencesExist);
        run_all_scenarios<StridedThreeRowsTag, unsigned>(sequencesExist);
    }

}  // namespace Test::stdalgos::TeamSearchN
