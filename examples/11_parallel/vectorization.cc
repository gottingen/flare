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
#include <flare/random.h>
#include <cstdio>

// The TeamPolicy actually supports 3D parallelism: Teams, Threads, Vector
// flare::parallel_{for/reduce/scan} calls can be completely free nested.
// The execution policies for the nested layers are TeamThreadRange and
// ThreadVectorRange.
// The only restriction on nesting is that a given level can only be nested in a
// higher one. e.g. a ThreadVectorRange can be nested inside a TeamPolicy
// operator and inside a TeamThreadRange, but you can not nest a
// ThreadVectorRange or a TeamThreadRange inside another ThreadVectorRange. As
// with the 2D execution of TeamPolicy the operator has to be considered as a
// parallel region even with respect to VectorLanes. That means even outside a
// TeamThread or VectorThread loop all threads of a team and all vector lanes of
// a thread execute every line of the operator as long as there are no
// restricitons on them. Code lines can be restricted using flare::single to
// either execute once PerThread or execute once PerTeam.
using team_member = typename flare::TeamPolicy<>::member_type;

struct SomeCorrelation {
    using value_type = int;  // Specify value type for reduction target, sum
    using shared_space = flare::DefaultExecutionSpace::scratch_memory_space;
    using shared_1d_int =
            flare::View<int *, shared_space, flare::MemoryUnmanaged>;

    flare::View<const int ***, flare::LayoutRight> data;
    flare::View<int> gsum;

    SomeCorrelation(flare::View<int ***, flare::LayoutRight> data_in,
                    flare::View<int> sum)
            : data(data_in), gsum(sum) {}

    FLARE_INLINE_FUNCTION
    void operator()(const team_member &thread) const {
        int i = thread.league_rank();

        // Allocate a shared array for the team.
        shared_1d_int count(thread.team_shmem(), data.extent(1));

        // With each team run a parallel_for with its threads
        flare::parallel_for(
                flare::TeamThreadRange(thread, data.extent(1)),
                [=, *this](const int &j) {
                    int tsum;
                    // Run a vector loop reduction over the inner dimension of data
                    // Count how many values are multiples of 4
                    // Every vector lane gets the same reduction value (tsum) back, it is
                    // broadcast to all vector lanes
                    flare::parallel_reduce(
                            flare::ThreadVectorRange(thread, data.extent(2)),
                            [=, *this](const int &k, int &vsum) {
                                vsum += (data(i, j, k) % 4 == 0) ? 1 : 0;
                            },
                            tsum);

                    // Make sure only one vector lane adds the reduction value to the
                    // shared array, i.e. execute the next line only once PerThread
                    flare::single(flare::PerThread(thread), [=]() { count(j) = tsum; });
                });

        // Wait for all threads to finish the parallel_for so that all shared memory
        // writes are done
        thread.team_barrier();

        // Check with one vector lane from each thread how many consecutive
        // data segments have the same number of values divisible by 4
        // The team reduction value is again broadcast to every team member (and
        // every vector lane)
        int team_sum = 0;
        flare::parallel_reduce(
                flare::TeamThreadRange(thread, data.extent(1) - 1),
                [=](const int &j, int &thread_sum) {
                    // It is not valid to directly add to thread_sum
                    // Use a single function with broadcast instead
                    // team_sum will be used as input to the operator (i.e. it is used to
                    // initialize sum) the end value of sum will be broadcast to all
                    // vector lanes in the thread.
                    flare::single(
                            flare::PerThread(thread),
                            [=](int &sum) {
                                if (count(j) == count(j + 1)) sum++;
                            },
                            thread_sum);
                },
                team_sum);

        // Add with one thread and vectorlane of the team the team_sum to the global
        // value
        flare::single(flare::PerTeam(thread),
                      [=, *this]() { flare::atomic_add(&gsum(), team_sum); });
    }

    // The functor needs to define how much shared memory it requests given a
    // team_size.
    size_t team_shmem_size(int /*team_size*/) const {
        return shared_1d_int::shmem_size(data.extent(1));
    }
};

int main(int narg, char *args[]) {
    flare::initialize(narg, args);

    {
        // Produce some 3D random data (see Algorithms/01_random_numbers for more
        // info)
        flare::View<int ***, flare::LayoutRight> data("Data", 512, 512, 32);
        flare::Random_XorShift64_Pool<> rand_pool64(5374857);
        flare::fill_random(data, rand_pool64, 100);

        // A global value to put the result in
        flare::View<int> gsum("Sum");

        // Each team handles a slice of the data
        // Set up TeamPolicy with 512 teams with maximum number of threads per team
        // and 16 vector lanes. flare::AUTO will determine the number of threads
        // The maximum vector length is hardware dependent but can always be smaller
        // than the hardware allows. The vector length must be a power of 2.

        const flare::TeamPolicy<> policy(512, flare::AUTO, 16);

        flare::parallel_for(policy, SomeCorrelation(data, gsum));

        flare::fence();

        // Copy result value back
        int sum = 0;
        flare::deep_copy(sum, gsum);
        printf("Result %i\n", sum);
    }

    flare::finalize();
}
