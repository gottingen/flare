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

    namespace detail {

        struct CudaStreamScratchTestFunctor {
            using team_t = flare::TeamPolicy<flare::Cuda>::member_type;
            using scratch_t = flare::View<int64_t *, flare::Cuda::scratch_memory_space>;

            flare::View<int64_t, flare::CudaSpace, flare::MemoryTraits<flare::Atomic>>
                    counter;
            int N, M;

            CudaStreamScratchTestFunctor(
                    flare::View<int64_t, flare::CudaSpace> counter_, int N_, int M_)
                    : counter(counter_), N(N_), M(M_) {}

            FLARE_FUNCTION
            void operator()(const team_t &team) const {
                scratch_t scr(team.team_scratch(1), M);
                flare::parallel_for(flare::TeamThreadRange(team, 0, M),
                                    [&](int i) { scr[i] = 0; });
                team.team_barrier();
                for (int i = 0; i < N; i++) {
                    flare::parallel_for(flare::TeamThreadRange(team, 0, M),
                                        [&](int j) { scr[j] += 1; });
                }
                team.team_barrier();
                flare::parallel_for(flare::TeamThreadRange(team, 0, M), [&](int i) {
                    if (scr[i] != N) counter()++;
                });
            }
        };

        void cuda_stream_scratch_test_one(
                int N, int T, int M_base, flare::View<int64_t, flare::CudaSpace> counter,
                flare::Cuda cuda, int tid) {
            int M = M_base + tid * 5;
            flare::TeamPolicy<flare::Cuda> p(cuda, T, 64);
            using scratch_t = flare::View<int64_t *, flare::Cuda::scratch_memory_space>;

            int bytes = scratch_t::shmem_size(M);

            for (int r = 0; r < 15; r++) {
                flare::parallel_for("Run", p.set_scratch_size(1, flare::PerTeam(bytes)),
                                    CudaStreamScratchTestFunctor(counter, N, M));
            }
        }

        void cuda_stream_scratch_test(
                int N, int T, int M_base,
                flare::View<int64_t, flare::CudaSpace> counter) {
            int K = 4;
            cudaStream_t stream[4];
            flare::Cuda cuda[4];
            for (int i = 0; i < K; i++) {
                cudaStreamCreate(&stream[i]);
                cuda[i] = flare::Cuda(stream[i]);
            }
            // Test that growing scratch size in subsequent calls doesn't crash things
#if defined(FLARE_ENABLE_OPENMP)
#pragma omp parallel
            {
                int tid = omp_get_thread_num();
                // Limit how many threads submit
                if (tid < 4) {
                    cuda_stream_scratch_test_one(N, T, M_base, counter, cuda[tid], tid);
                }
            }
#else
            for (int tid = 0; tid < K; tid++) {
              cuda_stream_scratch_test_one(N, T, M_base, counter, cuda[tid], tid);
            }
#endif
            // Test that if everything is large enough, multiple launches with different
            // scratch sizes don't step on each other
            for (int tid = K - 1; tid >= 0; tid--) {
                cuda_stream_scratch_test_one(N, T, M_base, counter, cuda[tid], tid);
            }

            flare::fence();
            for (int i = 0; i < K; i++) {
                cuda[i] = flare::Cuda();
                cudaStreamDestroy(stream[i]);
            }
        }
    }  // namespace detail

    TEST_CASE("cuda, team_scratch_1_streams") {
        int N = 1000000;
        int T = 10;
        int M_base = 150;

        flare::View<int64_t, flare::CudaSpace> counter("C");

        detail::cuda_stream_scratch_test(N, T, M_base, counter);

        int64_t result;
        flare::deep_copy(result, counter);
        REQUIRE_EQ(0, result);
    }
}  // namespace Test
