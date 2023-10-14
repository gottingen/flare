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

#include <sstream>
#include <iostream>
#include <time.h>

#include <flare/core.h>

namespace Test {

    template<typename ExecSpace, typename TensorType>
    void impl_test_local_deepcopy_teampolicy_rank_1(const int N) {
        // Allocate matrices on device.
        TensorType A("A", N, N, N, N, N, N, N, N);
        TensorType B("B", N, N, N, N, N, N, N, N);

        // Create host mirrors of device tensors.
        typename TensorType::HostMirror h_A = flare::create_mirror_tensor(A);
        typename TensorType::HostMirror h_B = flare::create_mirror_tensor(B);

        // Initialize A matrix.
        auto subA =
                flare::subtensor(A, 1, 1, 1, 1, 1, 1, flare::ALL(), flare::ALL());
        flare::deep_copy(subA, 10.0);

        using team_policy = flare::TeamPolicy<ExecSpace>;
        using member_type = typename flare::TeamPolicy<ExecSpace>::member_type;

        // Deep Copy
        flare::parallel_for(
                team_policy(N, flare::AUTO),
                FLARE_LAMBDA(const member_type &teamMember) {
                    int lid = teamMember.league_rank();  // returns a number between 0 and N
                    auto subSrc = flare::subtensor(A, 1, 1, 1, 1, 1, 1, lid, flare::ALL());
                    auto subDst = flare::subtensor(B, 1, 1, 1, 1, 1, 1, lid, flare::ALL());
                    flare::experimental::local_deep_copy(teamMember, subDst, subSrc);
                });

        flare::deep_copy(h_A, A);
        flare::deep_copy(h_B, B);

        bool test = true;
        for (size_t i = 0; i < A.span(); i++) {
            if (h_A.data()[i] != h_B.data()[i]) {
                test = false;
                break;
            }
        }

        REQUIRE_EQ(test, true);

        // Fill
        flare::deep_copy(B, 0.0);

        flare::parallel_for(
                team_policy(N, flare::AUTO),
                FLARE_LAMBDA(const member_type &teamMember) {
                    int lid = teamMember.league_rank();  // returns a number between 0 and N
                    auto subDst = flare::subtensor(B, 1, 1, 1, 1, 1, 1, lid, flare::ALL());
                    flare::experimental::local_deep_copy(teamMember, subDst, 20.0);
                });

        flare::deep_copy(h_B, B);

        double sum_all = 0.0;
        for (size_t i = 0; i < B.span(); i++) {
            sum_all += h_B.data()[i];
        }

        REQUIRE_EQ(sum_all, 20.0 * N * N);
    }

//-------------------------------------------------------------------------------------------------------------
    template<typename ExecSpace, typename TensorType>
    void impl_test_local_deepcopy_teampolicy_rank_2(const int N) {
        // Allocate matrices on device.
        TensorType A("A", N, N, N, N, N, N, N, N);
        TensorType B("B", N, N, N, N, N, N, N, N);

        // Create host mirrors of device tensors.
        typename TensorType::HostMirror h_A = flare::create_mirror_tensor(A);
        typename TensorType::HostMirror h_B = flare::create_mirror_tensor(B);

        // Initialize A matrix.
        auto subA = flare::subtensor(A, 1, 1, 1, 1, 1, flare::ALL(), flare::ALL(),
                                   flare::ALL());
        flare::deep_copy(subA, 10.0);

        using team_policy = flare::TeamPolicy<ExecSpace>;
        using member_type = typename flare::TeamPolicy<ExecSpace>::member_type;

        // Deep Copy
        flare::parallel_for(
                team_policy(N, flare::AUTO),
                FLARE_LAMBDA(const member_type &teamMember) {
                    int lid = teamMember.league_rank();  // returns a number between 0 and N
                    auto subSrc = flare::subtensor(A, 1, 1, 1, 1, 1, lid, flare::ALL(),
                                                 flare::ALL());
                    auto subDst = flare::subtensor(B, 1, 1, 1, 1, 1, lid, flare::ALL(),
                                                 flare::ALL());
                    flare::experimental::local_deep_copy(teamMember, subDst, subSrc);
                });

        flare::deep_copy(h_A, A);
        flare::deep_copy(h_B, B);

        bool test = true;
        for (size_t i = 0; i < A.span(); i++) {
            if (h_A.data()[i] != h_B.data()[i]) {
                test = false;
                break;
            }
        }

        REQUIRE_EQ(test, true);

        // Fill
        flare::deep_copy(B, 0.0);

        flare::parallel_for(
                team_policy(N, flare::AUTO),
                FLARE_LAMBDA(const member_type &teamMember) {
                    int lid = teamMember.league_rank();  // returns a number between 0 and N
                    auto subDst = flare::subtensor(B, 1, 1, 1, 1, 1, lid, flare::ALL(),
                                                 flare::ALL());
                    flare::experimental::local_deep_copy(teamMember, subDst, 20.0);
                });

        flare::deep_copy(h_B, B);

        double sum_all = 0.0;
        for (size_t i = 0; i < B.span(); i++) {
            sum_all += h_B.data()[i];
        }

        REQUIRE_EQ(sum_all, 20.0 * N * N * N);
    }

//-------------------------------------------------------------------------------------------------------------
    template<typename ExecSpace, typename TensorType>
    void impl_test_local_deepcopy_teampolicy_rank_3(const int N) {
        // Allocate matrices on device.
        TensorType A("A", N, N, N, N, N, N, N, N);
        TensorType B("B", N, N, N, N, N, N, N, N);

        // Create host mirrors of device tensors.
        typename TensorType::HostMirror h_A = flare::create_mirror_tensor(A);
        typename TensorType::HostMirror h_B = flare::create_mirror_tensor(B);

        // Initialize A matrix.
        auto subA = flare::subtensor(A, 1, 1, 1, 1, flare::ALL(), flare::ALL(),
                                   flare::ALL(), flare::ALL());
        flare::deep_copy(subA, 10.0);

        using team_policy = flare::TeamPolicy<ExecSpace>;
        using member_type = typename flare::TeamPolicy<ExecSpace>::member_type;

        // Deep Copy
        flare::parallel_for(
                team_policy(N, flare::AUTO),
                FLARE_LAMBDA(const member_type &teamMember) {
                    int lid = teamMember.league_rank();  // returns a number between 0 and N
                    auto subSrc = flare::subtensor(A, 1, 1, 1, 1, lid, flare::ALL(),
                                                 flare::ALL(), flare::ALL());
                    auto subDst = flare::subtensor(B, 1, 1, 1, 1, lid, flare::ALL(),
                                                 flare::ALL(), flare::ALL());
                    flare::experimental::local_deep_copy(teamMember, subDst, subSrc);
                });

        flare::deep_copy(h_A, A);
        flare::deep_copy(h_B, B);

        bool test = true;
        for (size_t i = 0; i < A.span(); i++) {
            if (h_A.data()[i] != h_B.data()[i]) {
                test = false;
                break;
            }
        }

        REQUIRE_EQ(test, true);

        // Fill
        flare::deep_copy(B, 0.0);

        flare::parallel_for(
                team_policy(N, flare::AUTO),
                FLARE_LAMBDA(const member_type &teamMember) {
                    int lid = teamMember.league_rank();  // returns a number between 0 and N
                    auto subDst = flare::subtensor(B, 1, 1, 1, 1, lid, flare::ALL(),
                                                 flare::ALL(), flare::ALL());
                    flare::experimental::local_deep_copy(teamMember, subDst, 20.0);
                });

        flare::deep_copy(h_B, B);

        double sum_all = 0.0;
        for (size_t i = 0; i < B.span(); i++) {
            sum_all += h_B.data()[i];
        }

        REQUIRE_EQ(sum_all, 20.0 * N * N * N * N);
    }

//-------------------------------------------------------------------------------------------------------------
    template<typename ExecSpace, typename TensorType>
    void impl_test_local_deepcopy_teampolicy_rank_4(const int N) {
        // Allocate matrices on device.
        TensorType A("A", N, N, N, N, N, N, N, N);
        TensorType B("B", N, N, N, N, N, N, N, N);

        // Create host mirrors of device tensors.
        typename TensorType::HostMirror h_A = flare::create_mirror_tensor(A);
        typename TensorType::HostMirror h_B = flare::create_mirror_tensor(B);

        // Initialize A matrix.
        auto subA = flare::subtensor(A, 1, 1, 1, flare::ALL(), flare::ALL(),
                                   flare::ALL(), flare::ALL(), flare::ALL());
        flare::deep_copy(subA, 10.0);

        using team_policy = flare::TeamPolicy<ExecSpace>;
        using member_type = typename flare::TeamPolicy<ExecSpace>::member_type;

        // Deep Copy
        flare::parallel_for(
                team_policy(N, flare::AUTO),
                FLARE_LAMBDA(const member_type &teamMember) {
                    int lid = teamMember.league_rank();  // returns a number between 0 and N
                    auto subSrc =
                            flare::subtensor(A, 1, 1, 1, lid, flare::ALL(), flare::ALL(),
                                           flare::ALL(), flare::ALL());
                    auto subDst =
                            flare::subtensor(B, 1, 1, 1, lid, flare::ALL(), flare::ALL(),
                                           flare::ALL(), flare::ALL());
                    flare::experimental::local_deep_copy(teamMember, subDst, subSrc);
                });

        flare::deep_copy(h_A, A);
        flare::deep_copy(h_B, B);

        bool test = true;
        for (size_t i = 0; i < A.span(); i++) {
            if (h_A.data()[i] != h_B.data()[i]) {
                test = false;
                break;
            }
        }

        REQUIRE_EQ(test, true);

        // Fill
        flare::deep_copy(B, 0.0);

        flare::parallel_for(
                team_policy(N, flare::AUTO),
                FLARE_LAMBDA(const member_type &teamMember) {
                    int lid = teamMember.league_rank();  // returns a number between 0 and N
                    auto subDst =
                            flare::subtensor(B, 1, 1, 1, lid, flare::ALL(), flare::ALL(),
                                           flare::ALL(), flare::ALL());
                    flare::experimental::local_deep_copy(teamMember, subDst, 20.0);
                });

        flare::deep_copy(h_B, B);

        double sum_all = 0.0;
        for (size_t i = 0; i < B.span(); i++) {
            sum_all += h_B.data()[i];
        }

        REQUIRE_EQ(sum_all, 20.0 * N * N * N * N * N);
    }

//-------------------------------------------------------------------------------------------------------------
    template<typename ExecSpace, typename TensorType>
    void impl_test_local_deepcopy_teampolicy_rank_5(const int N) {
        // Allocate matrices on device.
        TensorType A("A", N, N, N, N, N, N, N, N);
        TensorType B("B", N, N, N, N, N, N, N, N);

        // Create host mirrors of device tensors.
        typename TensorType::HostMirror h_A = flare::create_mirror_tensor(A);
        typename TensorType::HostMirror h_B = flare::create_mirror_tensor(B);

        // Initialize A matrix.
        auto subA =
                flare::subtensor(A, 1, 1, flare::ALL(), flare::ALL(), flare::ALL(),
                               flare::ALL(), flare::ALL(), flare::ALL());
        flare::deep_copy(subA, 10.0);

        using team_policy = flare::TeamPolicy<ExecSpace>;
        using member_type = typename flare::TeamPolicy<ExecSpace>::member_type;

        // Deep Copy
        flare::parallel_for(
                team_policy(N, flare::AUTO),
                FLARE_LAMBDA(const member_type &teamMember) {
                    int lid = teamMember.league_rank();  // returns a number between 0 and N
                    auto subSrc =
                            flare::subtensor(A, 1, 1, lid, flare::ALL(), flare::ALL(),
                                           flare::ALL(), flare::ALL(), flare::ALL());
                    auto subDst =
                            flare::subtensor(B, 1, 1, lid, flare::ALL(), flare::ALL(),
                                           flare::ALL(), flare::ALL(), flare::ALL());
                    flare::experimental::local_deep_copy(teamMember, subDst, subSrc);
                });

        flare::deep_copy(h_A, A);
        flare::deep_copy(h_B, B);

        bool test = true;
        for (size_t i = 0; i < A.span(); i++) {
            if (h_A.data()[i] != h_B.data()[i]) {
                test = false;
                break;
            }
        }

        REQUIRE_EQ(test, true);

        // Fill
        flare::deep_copy(B, 0.0);

        flare::parallel_for(
                team_policy(N, flare::AUTO),
                FLARE_LAMBDA(const member_type &teamMember) {
                    int lid = teamMember.league_rank();  // returns a number between 0 and N
                    auto subDst =
                            flare::subtensor(B, 1, 1, lid, flare::ALL(), flare::ALL(),
                                           flare::ALL(), flare::ALL(), flare::ALL());
                    flare::experimental::local_deep_copy(teamMember, subDst, 20.0);
                });

        flare::deep_copy(h_B, B);

        double sum_all = 0.0;
        for (size_t i = 0; i < B.span(); i++) {
            sum_all += h_B.data()[i];
        }

        REQUIRE_EQ(sum_all, 20.0 * N * N * N * N * N * N);
    }

//-------------------------------------------------------------------------------------------------------------
    template<typename ExecSpace, typename TensorType>
    void impl_test_local_deepcopy_teampolicy_rank_6(const int N) {
        // Allocate matrices on device.
        TensorType A("A", N, N, N, N, N, N, N, N);
        TensorType B("B", N, N, N, N, N, N, N, N);

        // Create host mirrors of device tensors.
        typename TensorType::HostMirror h_A = flare::create_mirror_tensor(A);
        typename TensorType::HostMirror h_B = flare::create_mirror_tensor(B);

        // Initialize A matrix.
        auto subA = flare::subtensor(A, 1, flare::ALL(), flare::ALL(), flare::ALL(),
                                   flare::ALL(), flare::ALL(), flare::ALL(),
                                   flare::ALL());
        flare::deep_copy(subA, 10.0);

        using team_policy = flare::TeamPolicy<ExecSpace>;
        using member_type = typename flare::TeamPolicy<ExecSpace>::member_type;

        // Deep Copy
        flare::parallel_for(
                team_policy(N, flare::AUTO),
                FLARE_LAMBDA(const member_type &teamMember) {
                    int lid = teamMember.league_rank();  // returns a number between 0 and N
                    auto subSrc = flare::subtensor(A, 1, lid, flare::ALL(), flare::ALL(),
                                                 flare::ALL(), flare::ALL(),
                                                 flare::ALL(), flare::ALL());
                    auto subDst = flare::subtensor(B, 1, lid, flare::ALL(), flare::ALL(),
                                                 flare::ALL(), flare::ALL(),
                                                 flare::ALL(), flare::ALL());
                    flare::experimental::local_deep_copy(teamMember, subDst, subSrc);
                });

        flare::deep_copy(h_A, A);
        flare::deep_copy(h_B, B);

        bool test = true;
        for (size_t i = 0; i < A.span(); i++) {
            if (h_A.data()[i] != h_B.data()[i]) {
                test = false;
                break;
            }
        }

        REQUIRE_EQ(test, true);

        // Fill
        flare::deep_copy(B, 0.0);

        flare::parallel_for(
                team_policy(N, flare::AUTO),
                FLARE_LAMBDA(const member_type &teamMember) {
                    int lid = teamMember.league_rank();  // returns a number between 0 and N
                    auto subDst = flare::subtensor(B, 1, lid, flare::ALL(), flare::ALL(),
                                                 flare::ALL(), flare::ALL(),
                                                 flare::ALL(), flare::ALL());
                    flare::experimental::local_deep_copy(teamMember, subDst, 20.0);
                });

        flare::deep_copy(h_B, B);

        double sum_all = 0.0;
        for (size_t i = 0; i < B.span(); i++) {
            sum_all += h_B.data()[i];
        }

        REQUIRE_EQ(sum_all, 20.0 * N * N * N * N * N * N * N);
    }

//-------------------------------------------------------------------------------------------------------------
    template<typename ExecSpace, typename TensorType>
    void impl_test_local_deepcopy_teampolicy_rank_7(const int N) {
        // Allocate matrices on device.
        TensorType A("A", N, N, N, N, N, N, N, N);
        TensorType B("B", N, N, N, N, N, N, N, N);

        // Create host mirrors of device tensors.
        typename TensorType::HostMirror h_A = flare::create_mirror_tensor(A);
        typename TensorType::HostMirror h_B = flare::create_mirror_tensor(B);

        // Initialize A matrix.
        flare::deep_copy(A, 10.0);

        using team_policy = flare::TeamPolicy<ExecSpace>;
        using member_type = typename flare::TeamPolicy<ExecSpace>::member_type;

        // Deep Copy
        flare::parallel_for(
                team_policy(N, flare::AUTO),
                FLARE_LAMBDA(const member_type &teamMember) {
                    int lid = teamMember.league_rank();  // returns a number between 0 and N
                    auto subSrc = flare::subtensor(
                            A, lid, flare::ALL(), flare::ALL(), flare::ALL(), flare::ALL(),
                            flare::ALL(), flare::ALL(), flare::ALL());
                    auto subDst = flare::subtensor(
                            B, lid, flare::ALL(), flare::ALL(), flare::ALL(), flare::ALL(),
                            flare::ALL(), flare::ALL(), flare::ALL());
                    flare::experimental::local_deep_copy(teamMember, subDst, subSrc);
                });

        flare::deep_copy(h_A, A);
        flare::deep_copy(h_B, B);

        bool test = true;
        for (size_t i = 0; i < A.span(); i++) {
            if (h_A.data()[i] != h_B.data()[i]) {
                test = false;
                break;
            }
        }

        REQUIRE_EQ(test, true);

        // Fill
        flare::deep_copy(B, 0.0);

        flare::parallel_for(
                team_policy(N, flare::AUTO),
                FLARE_LAMBDA(const member_type &teamMember) {
                    int lid = teamMember.league_rank();  // returns a number between 0 and N
                    auto subDst = flare::subtensor(
                            B, lid, flare::ALL(), flare::ALL(), flare::ALL(), flare::ALL(),
                            flare::ALL(), flare::ALL(), flare::ALL());
                    flare::experimental::local_deep_copy(teamMember, subDst, 20.0);
                });

        flare::deep_copy(h_B, B);

        double sum_all = 0.0;
        for (size_t i = 0; i < B.span(); i++) {
            sum_all += h_B.data()[i];
        }

        REQUIRE_EQ(sum_all, 20.0 * N * N * N * N * N * N * N * N);
    }

//-------------------------------------------------------------------------------------------------------------
    template<typename ExecSpace, typename TensorType>
    void impl_test_local_deepcopy_rangepolicy_rank_1(const int N) {
        // Allocate matrices on device.
        TensorType A("A", N, N, N, N, N, N, N, N);
        TensorType B("B", N, N, N, N, N, N, N, N);

        // Create host mirrors of device tensors.
        typename TensorType::HostMirror h_A = flare::create_mirror_tensor(A);
        typename TensorType::HostMirror h_B = flare::create_mirror_tensor(B);

        // Initialize A matrix.
        auto subA =
                flare::subtensor(A, 1, 1, 1, 1, 1, 1, flare::ALL(), flare::ALL());
        flare::deep_copy(subA, 10.0);

        // Deep Copy
        flare::parallel_for(
                flare::RangePolicy<ExecSpace>(0, N), FLARE_LAMBDA(const int &i) {
                    auto subSrc = flare::subtensor(A, 1, 1, 1, 1, 1, 1, i, flare::ALL());
                    auto subDst = flare::subtensor(B, 1, 1, 1, 1, 1, 1, i, flare::ALL());
                    flare::experimental::local_deep_copy(subDst, subSrc);
                });

        flare::deep_copy(h_A, A);
        flare::deep_copy(h_B, B);

        bool test = true;
        for (size_t i = 0; i < A.span(); i++) {
            if (h_A.data()[i] != h_B.data()[i]) {
                test = false;
                break;
            }
        }

        REQUIRE_EQ(test, true);

        // Fill
        flare::deep_copy(B, 0.0);

        flare::parallel_for(
                flare::RangePolicy<ExecSpace>(0, N), FLARE_LAMBDA(const int &i) {
                    auto subDst = flare::subtensor(B, 1, 1, 1, 1, 1, 1, i, flare::ALL());
                    flare::experimental::local_deep_copy(subDst, 20.0);
                });

        flare::deep_copy(h_B, B);

        double sum_all = 0.0;
        for (size_t i = 0; i < B.span(); i++) {
            sum_all += h_B.data()[i];
        }

        REQUIRE_EQ(sum_all, 20.0 * N * N);
    }

//-------------------------------------------------------------------------------------------------------------
    template<typename ExecSpace, typename TensorType>
    void impl_test_local_deepcopy_rangepolicy_rank_2(const int N) {
        // Allocate matrices on device.
        TensorType A("A", N, N, N, N, N, N, N, N);
        TensorType B("B", N, N, N, N, N, N, N, N);

        // Create host mirrors of device tensors.
        typename TensorType::HostMirror h_A = flare::create_mirror_tensor(A);
        typename TensorType::HostMirror h_B = flare::create_mirror_tensor(B);

        // Initialize A matrix.
        auto subA = flare::subtensor(A, 1, 1, 1, 1, 1, flare::ALL(), flare::ALL(),
                                   flare::ALL());
        flare::deep_copy(subA, 10.0);

        // Deep Copy
        flare::parallel_for(
                flare::RangePolicy<ExecSpace>(0, N), FLARE_LAMBDA(const int &i) {
                    auto subSrc =
                            flare::subtensor(A, 1, 1, 1, 1, 1, i, flare::ALL(), flare::ALL());
                    auto subDst =
                            flare::subtensor(B, 1, 1, 1, 1, 1, i, flare::ALL(), flare::ALL());
                    flare::experimental::local_deep_copy(subDst, subSrc);
                });

        flare::deep_copy(h_A, A);
        flare::deep_copy(h_B, B);

        bool test = true;
        for (size_t i = 0; i < A.span(); i++) {
            if (h_A.data()[i] != h_B.data()[i]) {
                test = false;
                break;
            }
        }

        REQUIRE_EQ(test, true);

        // Fill
        flare::deep_copy(B, 0.0);

        flare::parallel_for(
                flare::RangePolicy<ExecSpace>(0, N), FLARE_LAMBDA(const int &i) {
                    auto subDst =
                            flare::subtensor(B, 1, 1, 1, 1, 1, i, flare::ALL(), flare::ALL());
                    flare::experimental::local_deep_copy(subDst, 20.0);
                });

        flare::deep_copy(h_B, B);

        double sum_all = 0.0;
        for (size_t i = 0; i < B.span(); i++) {
            sum_all += h_B.data()[i];
        }

        REQUIRE_EQ(sum_all, 20.0 * N * N * N);
    }

//-------------------------------------------------------------------------------------------------------------
    template<typename ExecSpace, typename TensorType>
    void impl_test_local_deepcopy_rangepolicy_rank_3(const int N) {
        // Allocate matrices on device.
        TensorType A("A", N, N, N, N, N, N, N, N);
        TensorType B("B", N, N, N, N, N, N, N, N);

        // Create host mirrors of device tensors.
        typename TensorType::HostMirror h_A = flare::create_mirror_tensor(A);
        typename TensorType::HostMirror h_B = flare::create_mirror_tensor(B);

        // Initialize A matrix.
        auto subA = flare::subtensor(A, 1, 1, 1, 1, flare::ALL(), flare::ALL(),
                                   flare::ALL(), flare::ALL());
        flare::deep_copy(subA, 10.0);

        // Deep Copy
        flare::parallel_for(
                flare::RangePolicy<ExecSpace>(0, N), FLARE_LAMBDA(const int &i) {
                    auto subSrc = flare::subtensor(A, 1, 1, 1, 1, i, flare::ALL(),
                                                 flare::ALL(), flare::ALL());
                    auto subDst = flare::subtensor(B, 1, 1, 1, 1, i, flare::ALL(),
                                                 flare::ALL(), flare::ALL());
                    flare::experimental::local_deep_copy(subDst, subSrc);
                });

        flare::deep_copy(h_A, A);
        flare::deep_copy(h_B, B);

        bool test = true;
        for (size_t i = 0; i < A.span(); i++) {
            if (h_A.data()[i] != h_B.data()[i]) {
                test = false;
                break;
            }
        }

        REQUIRE_EQ(test, true);

        // Fill
        flare::deep_copy(B, 0.0);

        flare::parallel_for(
                flare::RangePolicy<ExecSpace>(0, N), FLARE_LAMBDA(const int &i) {
                    auto subDst = flare::subtensor(B, 1, 1, 1, 1, i, flare::ALL(),
                                                 flare::ALL(), flare::ALL());
                    flare::experimental::local_deep_copy(subDst, 20.0);
                });

        flare::deep_copy(h_B, B);

        double sum_all = 0.0;
        for (size_t i = 0; i < B.span(); i++) {
            sum_all += h_B.data()[i];
        }

        REQUIRE_EQ(sum_all, 20.0 * N * N * N * N);
    }

//-------------------------------------------------------------------------------------------------------------
    template<typename ExecSpace, typename TensorType>
    void impl_test_local_deepcopy_rangepolicy_rank_4(const int N) {
        // Allocate matrices on device.
        TensorType A("A", N, N, N, N, N, N, N, N);
        TensorType B("B", N, N, N, N, N, N, N, N);

        // Create host mirrors of device tensors.
        typename TensorType::HostMirror h_A = flare::create_mirror_tensor(A);
        typename TensorType::HostMirror h_B = flare::create_mirror_tensor(B);

        // Initialize A matrix.
        auto subA = flare::subtensor(A, 1, 1, 1, flare::ALL(), flare::ALL(),
                                   flare::ALL(), flare::ALL(), flare::ALL());
        flare::deep_copy(subA, 10.0);

        // Deep Copy
        flare::parallel_for(
                flare::RangePolicy<ExecSpace>(0, N), FLARE_LAMBDA(const int &i) {
                    auto subSrc =
                            flare::subtensor(A, 1, 1, 1, i, flare::ALL(), flare::ALL(),
                                           flare::ALL(), flare::ALL());
                    auto subDst =
                            flare::subtensor(B, 1, 1, 1, i, flare::ALL(), flare::ALL(),
                                           flare::ALL(), flare::ALL());
                    flare::experimental::local_deep_copy(subDst, subSrc);
                });

        flare::deep_copy(h_A, A);
        flare::deep_copy(h_B, B);

        bool test = true;
        for (size_t i = 0; i < A.span(); i++) {
            if (h_A.data()[i] != h_B.data()[i]) {
                test = false;
                break;
            }
        }

        REQUIRE_EQ(test, true);

        // Fill
        flare::deep_copy(B, 0.0);

        flare::parallel_for(
                flare::RangePolicy<ExecSpace>(0, N), FLARE_LAMBDA(const int &i) {
                    auto subDst =
                            flare::subtensor(B, 1, 1, 1, i, flare::ALL(), flare::ALL(),
                                           flare::ALL(), flare::ALL());
                    flare::experimental::local_deep_copy(subDst, 20.0);
                });

        flare::deep_copy(h_B, B);

        double sum_all = 0.0;
        for (size_t i = 0; i < B.span(); i++) {
            sum_all += h_B.data()[i];
        }

        REQUIRE_EQ(sum_all, 20.0 * N * N * N * N * N);
    }

//-------------------------------------------------------------------------------------------------------------
    template<typename ExecSpace, typename TensorType>
    void impl_test_local_deepcopy_rangepolicy_rank_5(const int N) {
        // Allocate matrices on device.
        TensorType A("A", N, N, N, N, N, N, N, N);
        TensorType B("B", N, N, N, N, N, N, N, N);

        // Create host mirrors of device tensors.
        typename TensorType::HostMirror h_A = flare::create_mirror_tensor(A);
        typename TensorType::HostMirror h_B = flare::create_mirror_tensor(B);

        // Initialize A matrix.
        auto subA =
                flare::subtensor(A, 1, 1, flare::ALL(), flare::ALL(), flare::ALL(),
                               flare::ALL(), flare::ALL(), flare::ALL());
        flare::deep_copy(subA, 10.0);

        // Deep Copy
        flare::parallel_for(
                flare::RangePolicy<ExecSpace>(0, N), FLARE_LAMBDA(const int &i) {
                    auto subSrc =
                            flare::subtensor(A, 1, 1, i, flare::ALL(), flare::ALL(),
                                           flare::ALL(), flare::ALL(), flare::ALL());
                    auto subDst =
                            flare::subtensor(B, 1, 1, i, flare::ALL(), flare::ALL(),
                                           flare::ALL(), flare::ALL(), flare::ALL());
                    flare::experimental::local_deep_copy(subDst, subSrc);
                });

        flare::deep_copy(h_A, A);
        flare::deep_copy(h_B, B);

        bool test = true;
        for (size_t i = 0; i < A.span(); i++) {
            if (h_A.data()[i] != h_B.data()[i]) {
                test = false;
                break;
            }
        }

        REQUIRE_EQ(test, true);

        // Fill
        flare::deep_copy(B, 0.0);

        flare::parallel_for(
                flare::RangePolicy<ExecSpace>(0, N), FLARE_LAMBDA(const int &i) {
                    auto subDst =
                            flare::subtensor(B, 1, 1, i, flare::ALL(), flare::ALL(),
                                           flare::ALL(), flare::ALL(), flare::ALL());
                    flare::experimental::local_deep_copy(subDst, 20.0);
                });

        flare::deep_copy(h_B, B);

        double sum_all = 0.0;
        for (size_t i = 0; i < B.span(); i++) {
            sum_all += h_B.data()[i];
        }

        REQUIRE_EQ(sum_all, 20.0 * N * N * N * N * N * N);
    }

//-------------------------------------------------------------------------------------------------------------
    template<typename ExecSpace, typename TensorType>
    void impl_test_local_deepcopy_rangepolicy_rank_6(const int N) {
        // Allocate matrices on device.
        TensorType A("A", N, N, N, N, N, N, N, N);
        TensorType B("B", N, N, N, N, N, N, N, N);

        // Create host mirrors of device tensors.
        typename TensorType::HostMirror h_A = flare::create_mirror_tensor(A);
        typename TensorType::HostMirror h_B = flare::create_mirror_tensor(B);

        // Initialize A matrix.
        auto subA = flare::subtensor(A, 1, flare::ALL(), flare::ALL(), flare::ALL(),
                                   flare::ALL(), flare::ALL(), flare::ALL(),
                                   flare::ALL());
        flare::deep_copy(subA, 10.0);

        // Deep Copy
        flare::parallel_for(
                flare::RangePolicy<ExecSpace>(0, N), FLARE_LAMBDA(const int &i) {
                    auto subSrc = flare::subtensor(A, 1, i, flare::ALL(), flare::ALL(),
                                                 flare::ALL(), flare::ALL(),
                                                 flare::ALL(), flare::ALL());
                    auto subDst = flare::subtensor(B, 1, i, flare::ALL(), flare::ALL(),
                                                 flare::ALL(), flare::ALL(),
                                                 flare::ALL(), flare::ALL());
                    flare::experimental::local_deep_copy(subDst, subSrc);
                });

        flare::deep_copy(h_A, A);
        flare::deep_copy(h_B, B);

        bool test = true;
        for (size_t i = 0; i < A.span(); i++) {
            if (h_A.data()[i] != h_B.data()[i]) {
                test = false;
                break;
            }
        }

        REQUIRE_EQ(test, true);

        // Fill
        flare::deep_copy(B, 0.0);

        flare::parallel_for(
                flare::RangePolicy<ExecSpace>(0, N), FLARE_LAMBDA(const int &i) {
                    auto subDst = flare::subtensor(B, 1, i, flare::ALL(), flare::ALL(),
                                                 flare::ALL(), flare::ALL(),
                                                 flare::ALL(), flare::ALL());
                    flare::experimental::local_deep_copy(subDst, 20.0);
                });

        flare::deep_copy(h_B, B);

        double sum_all = 0.0;
        for (size_t i = 0; i < B.span(); i++) {
            sum_all += h_B.data()[i];
        }

        REQUIRE_EQ(sum_all, 20.0 * N * N * N * N * N * N * N);
    }

//-------------------------------------------------------------------------------------------------------------
    template<typename ExecSpace, typename TensorType>
    void impl_test_local_deepcopy_rangepolicy_rank_7(const int N) {
        // Allocate matrices on device.
        TensorType A("A", N, N, N, N, N, N, N, N);
        TensorType B("B", N, N, N, N, N, N, N, N);

        // Create host mirrors of device tensors.
        typename TensorType::HostMirror h_A = flare::create_mirror_tensor(A);
        typename TensorType::HostMirror h_B = flare::create_mirror_tensor(B);

        // Initialize A matrix.
        flare::deep_copy(A, 10.0);

        // Deep Copy
        flare::parallel_for(
                flare::RangePolicy<ExecSpace>(0, N), FLARE_LAMBDA(const int &i) {
                    auto subSrc = flare::subtensor(
                            A, i, flare::ALL(), flare::ALL(), flare::ALL(), flare::ALL(),
                            flare::ALL(), flare::ALL(), flare::ALL());
                    auto subDst = flare::subtensor(
                            B, i, flare::ALL(), flare::ALL(), flare::ALL(), flare::ALL(),
                            flare::ALL(), flare::ALL(), flare::ALL());
                    flare::experimental::local_deep_copy(subDst, subSrc);
                });

        flare::deep_copy(h_A, A);
        flare::deep_copy(h_B, B);

        bool test = true;
        for (size_t i = 0; i < A.span(); i++) {
            if (h_A.data()[i] != h_B.data()[i]) {
                test = false;
                break;
            }
        }

        REQUIRE_EQ(test, true);

        // Fill
        flare::deep_copy(B, 0.0);

        flare::parallel_for(
                flare::RangePolicy<ExecSpace>(0, N), FLARE_LAMBDA(const int &i) {
                    auto subDst = flare::subtensor(
                            B, i, flare::ALL(), flare::ALL(), flare::ALL(), flare::ALL(),
                            flare::ALL(), flare::ALL(), flare::ALL());
                    flare::experimental::local_deep_copy(subDst, 20.0);
                });

        flare::deep_copy(h_B, B);

        double sum_all = 0.0;
        for (size_t i = 0; i < B.span(); i++) {
            sum_all += h_B.data()[i];
        }

        REQUIRE_EQ(sum_all, 20.0 * N * N * N * N * N * N * N * N);
    }
//-------------------------------------------------------------------------------------------------------------

#if defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)

    TEST_CASE("TEST_CATEGORY, local_deepcopy_teampolicy_layoutleft") {
        using ExecSpace = TEST_EXECSPACE;
#if defined(FLARE_ON_CUDA_DEVICE) && \
    defined(FLARE_COMPILER_NVHPC)  // FIXME_NVHPC 23.7
        if (std::is_same_v<ExecSpace, flare::Cuda>)
          GTEST_SKIP()
              << "FIXME_NVHPC : Compiler bug affecting subtensors of high rank Tensors";
#endif
        using TensorType = flare::Tensor<double ********, flare::LayoutLeft, ExecSpace>;

        {  // Rank-1
            impl_test_local_deepcopy_teampolicy_rank_1<ExecSpace, TensorType>(8);
        }
        {  // Rank-2
            impl_test_local_deepcopy_teampolicy_rank_2<ExecSpace, TensorType>(8);
        }
        {  // Rank-3
            impl_test_local_deepcopy_teampolicy_rank_3<ExecSpace, TensorType>(8);
        }
        {  // Rank-4
            impl_test_local_deepcopy_teampolicy_rank_4<ExecSpace, TensorType>(8);
        }
        {  // Rank-5
            impl_test_local_deepcopy_teampolicy_rank_5<ExecSpace, TensorType>(8);
        }
        {  // Rank-6
            impl_test_local_deepcopy_teampolicy_rank_6<ExecSpace, TensorType>(8);
        }
        {  // Rank-7
            impl_test_local_deepcopy_teampolicy_rank_7<ExecSpace, TensorType>(8);
        }
    }
//-------------------------------------------------------------------------------------------------------------
    TEST_CASE("TEST_CATEGORY, local_deepcopy_rangepolicy_layoutleft") {
        using ExecSpace = TEST_EXECSPACE;
#if defined(FLARE_ON_CUDA_DEVICE) && \
    defined(FLARE_COMPILER_NVHPC)  // FIXME_NVHPC 23.7
        if (std::is_same_v<ExecSpace, flare::Cuda>) {
          INFO("FIXME_NVHPC : Compiler bug affecting subtensors of high rank Tensors");
          return;
          }

#endif
        using TensorType = flare::Tensor<double ********, flare::LayoutLeft, ExecSpace>;

        {  // Rank-1
            impl_test_local_deepcopy_rangepolicy_rank_1<ExecSpace, TensorType>(8);
        }
        {  // Rank-2
            impl_test_local_deepcopy_rangepolicy_rank_2<ExecSpace, TensorType>(8);
        }
        {  // Rank-3
            impl_test_local_deepcopy_rangepolicy_rank_3<ExecSpace, TensorType>(8);
        }
        {  // Rank-4
            impl_test_local_deepcopy_rangepolicy_rank_4<ExecSpace, TensorType>(8);
        }
        {  // Rank-5
            impl_test_local_deepcopy_rangepolicy_rank_5<ExecSpace, TensorType>(8);
        }
        {  // Rank-6
            impl_test_local_deepcopy_rangepolicy_rank_6<ExecSpace, TensorType>(8);
        }
        {  // Rank-7
            impl_test_local_deepcopy_rangepolicy_rank_7<ExecSpace, TensorType>(8);
        }
    }
//-------------------------------------------------------------------------------------------------------------
    TEST_CASE("TEST_CATEGORY, local_deepcopy_teampolicy_layoutright") {
        using ExecSpace = TEST_EXECSPACE;
#if defined(FLARE_ON_CUDA_DEVICE) && \
    defined(FLARE_COMPILER_NVHPC)  // FIXME_NVHPC 23.7
        if (std::is_same_v<ExecSpace, flare::Cuda>)
          GTEST_SKIP()
              << "FIXME_NVHPC : Compiler bug affecting subtensors of high rank Tensors";
#endif
        using TensorType = flare::Tensor<double ********, flare::LayoutRight, ExecSpace>;

        {  // Rank-1
            impl_test_local_deepcopy_teampolicy_rank_1<ExecSpace, TensorType>(8);
        }
        {  // Rank-2
            impl_test_local_deepcopy_teampolicy_rank_2<ExecSpace, TensorType>(8);
        }
        {  // Rank-3
            impl_test_local_deepcopy_teampolicy_rank_3<ExecSpace, TensorType>(8);
        }
        {  // Rank-4
            impl_test_local_deepcopy_teampolicy_rank_4<ExecSpace, TensorType>(8);
        }
        {  // Rank-5
            impl_test_local_deepcopy_teampolicy_rank_5<ExecSpace, TensorType>(8);
        }
        {  // Rank-6
            impl_test_local_deepcopy_teampolicy_rank_6<ExecSpace, TensorType>(8);
        }
        {  // Rank-7
            impl_test_local_deepcopy_teampolicy_rank_7<ExecSpace, TensorType>(8);
        }
    }
//-------------------------------------------------------------------------------------------------------------
    TEST_CASE("TEST_CATEGORY, local_deepcopy_rangepolicy_layoutright") {
        using ExecSpace = TEST_EXECSPACE;
#if defined(FLARE_ON_CUDA_DEVICE) && \
    defined(FLARE_COMPILER_NVHPC)  // FIXME_NVHPC 23.7
        if (std::is_same_v<ExecSpace, flare::Cuda>)
          GTEST_SKIP()
              << "FIXME_NVHPC : Compiler bug affecting subtensors of high rank Tensors";
#endif

        using TensorType = flare::Tensor<double ********, flare::LayoutRight, ExecSpace>;

        {  // Rank-1
            impl_test_local_deepcopy_rangepolicy_rank_1<ExecSpace, TensorType>(8);
        }
        {  // Rank-2
            impl_test_local_deepcopy_rangepolicy_rank_2<ExecSpace, TensorType>(8);
        }
        {  // Rank-3
            impl_test_local_deepcopy_rangepolicy_rank_3<ExecSpace, TensorType>(8);
        }
        {  // Rank-4
            impl_test_local_deepcopy_rangepolicy_rank_4<ExecSpace, TensorType>(8);
        }
        {  // Rank-5
            impl_test_local_deepcopy_rangepolicy_rank_5<ExecSpace, TensorType>(8);
        }
        {  // Rank-6
            impl_test_local_deepcopy_rangepolicy_rank_6<ExecSpace, TensorType>(8);
        }
        {  // Rank-7
            impl_test_local_deepcopy_rangepolicy_rank_7<ExecSpace, TensorType>(8);
        }
    }

#endif

    namespace detail {
        template<typename T, typename SHMEMTYPE>
        using ShMemTensor =
                flare::Tensor<T, flare::LayoutRight, SHMEMTYPE, flare::MemoryUnmanaged>;

        struct DeepCopyScratchFunctor {
            DeepCopyScratchFunctor(
                    flare::Tensor<double *, TEST_EXECSPACE::memory_space> check_tensor_1,
                    flare::Tensor<double *, TEST_EXECSPACE::memory_space> check_tensor_2)
                    : check_tensor_1_(check_tensor_1),
                      check_tensor_2_(check_tensor_2),
                      N_(check_tensor_1.extent(0)) {}

            FLARE_INLINE_FUNCTION void operator()(
                    flare::TeamPolicy<TEST_EXECSPACE,
                            flare::Schedule<flare::Dynamic>>::member_type team)
            const {
                using ShmemType = TEST_EXECSPACE::scratch_memory_space;
                auto shtensor =
                        detail::ShMemTensor<double **, ShmemType>(team.team_scratch(1), N_, 1);

                flare::parallel_for(
                        flare::TeamThreadRange(team, N_), FLARE_LAMBDA(const size_t &index) {
                            auto thread_shtensor = flare::subtensor(shtensor, index, flare::ALL());
                            flare::experimental::local_deep_copy(thread_shtensor, index);
                        });
                flare::experimental::local_deep_copy(
                        team, check_tensor_1_, flare::subtensor(shtensor, flare::ALL(), 0));

                flare::experimental::local_deep_copy(team, shtensor, 6.);
                flare::experimental::local_deep_copy(
                        team, check_tensor_2_, flare::subtensor(shtensor, flare::ALL(), 0));
            }

            flare::Tensor<double *, TEST_EXECSPACE::memory_space> check_tensor_1_;
            flare::Tensor<double *, TEST_EXECSPACE::memory_space> check_tensor_2_;
            int const N_;
        };
    }  // namespace detail

    TEST_CASE("TEST_CATEGORY, deep_copy_scratch") {
        using TestDeviceTeamPolicy = flare::TeamPolicy<TEST_EXECSPACE>;

        const int N = 8;
        const int bytes_per_team =
                detail::ShMemTensor<double **,
                        TEST_EXECSPACE::scratch_memory_space>::shmem_size(N, 1);

        TestDeviceTeamPolicy policy(1, flare::AUTO);
        auto team_exec = policy.set_scratch_size(1, flare::PerTeam(bytes_per_team));

        flare::Tensor<double *, TEST_EXECSPACE::memory_space> check_tensor_1("check_1",
                                                                         N);
        flare::Tensor<double *, TEST_EXECSPACE::memory_space> check_tensor_2("check_2",
                                                                         N);

        flare::parallel_for(
                team_exec, detail::DeepCopyScratchFunctor{check_tensor_1, check_tensor_2});
        auto host_copy_1 =
                flare::create_mirror_tensor_and_copy(flare::HostSpace(), check_tensor_1);
        auto host_copy_2 =
                flare::create_mirror_tensor_and_copy(flare::HostSpace(), check_tensor_2);

        for (unsigned int i = 0; i < N; ++i) {
            REQUIRE_EQ(host_copy_1(i), i);
            REQUIRE_EQ(host_copy_2(i), 6.0);
        }
    }
}  // namespace Test
