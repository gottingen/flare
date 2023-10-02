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
#include <flare/dual_view.h>
#include <flare/timer.h>
#include <cstdio>
#include <cstdlib>

using Device = flare::DefaultExecutionSpace;
using Host = flare::HostSpace::execution_space;

using team_policy = flare::TeamPolicy<Device>;
using team_member = team_policy::member_type;

static const int TEAM_SIZE = 16;

struct find_2_tuples {
    int chunk_size;
    flare::View<const int *> data;
    flare::View<int **> histogram;

    find_2_tuples(int chunk_size_, flare::DualView<int *> data_,
                  flare::DualView<int **> histogram_)
            : chunk_size(chunk_size_),
              data(data_.d_view),
              histogram(histogram_.d_view) {
        data_.sync<Device>();
        histogram_.sync<Device>();
        histogram_.modify<Device>();
    }

    FLARE_INLINE_FUNCTION void operator()(const team_member &dev) const {
        flare::View<int **, flare::MemoryUnmanaged> l_histogram(
                dev.team_shmem(), TEAM_SIZE, TEAM_SIZE);
        flare::View<int *, flare::MemoryUnmanaged> l_data(dev.team_shmem(),
                                                          chunk_size + 1);

        const int i = dev.league_rank() * chunk_size;
        for (int j = dev.team_rank(); j < chunk_size + 1; j += dev.team_size())
            l_data(j) = data(i + j);

        for (int k = dev.team_rank(); k < TEAM_SIZE; k += dev.team_size())
            for (int l = 0; l < TEAM_SIZE; l++) l_histogram(k, l) = 0;
        dev.team_barrier();

        for (int j = 0; j < chunk_size; j++) {
            for (int k = dev.team_rank(); k < TEAM_SIZE; k += dev.team_size())
                for (int l = 0; l < TEAM_SIZE; l++) {
                    if ((l_data(j) == k) && (l_data(j + 1) == l)) l_histogram(k, l)++;
                }
        }

        for (int k = dev.team_rank(); k < TEAM_SIZE; k += dev.team_size())
            for (int l = 0; l < TEAM_SIZE; l++) {
                flare::atomic_fetch_add(&histogram(k, l), l_histogram(k, l));
            }
        dev.team_barrier();
    }

    size_t team_shmem_size(int team_size) const {
        return flare::View<int **, flare::MemoryUnmanaged>::shmem_size(TEAM_SIZE,
                                                                       TEAM_SIZE) +
               flare::View<int *, flare::MemoryUnmanaged>::shmem_size(chunk_size +
                                                                      1);
    }
};

int main(int narg, char *args[]) {
    flare::initialize(narg, args);

    {
        int chunk_size = 1024;
        int nchunks = 100000;  // 1024*1024;
        flare::DualView<int *> data("data", nchunks * chunk_size + 1);

        srand(1231093);

        for (int i = 0; i < (int) data.extent(0); i++) {
            data.h_view(i) = rand() % TEAM_SIZE;
        }
        data.modify<Host>();
        data.sync<Device>();

        flare::DualView<int **> histogram("histogram", TEAM_SIZE, TEAM_SIZE);

        flare::Timer timer;
        // threads/team is automatically limited to maximum supported by the device.
        int const concurrency = Device::execution_space().concurrency();
        int team_size = TEAM_SIZE;
        if (team_size > concurrency) team_size = concurrency;
        flare::parallel_for(team_policy(nchunks, team_size),
                            find_2_tuples(chunk_size, data, histogram));
        flare::fence();
        double time = timer.seconds();

        histogram.sync<Host>();

        printf("Time: %f \n\n", time);
        int sum = 0;
        for (int k = 0; k < TEAM_SIZE; k++) {
            for (int l = 0; l < TEAM_SIZE; l++) {
                printf("%i ", histogram.h_view(k, l));
                sum += histogram.h_view(k, l);
            }
            printf("\n");
        }
        printf("Result: %i %i\n", sum, chunk_size * nchunks);
    }
    flare::finalize();
}
