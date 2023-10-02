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
#include <openmp_category_test.h>
#include <omp.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

namespace Test {

// Test whether allocations survive flare initialize/finalize if done via Raw
// Cuda.
    TEST_CASE("openmp, raw_openmp_interop") {
        int count = 0;
        int num_threads, concurrency;
#pragma omp parallel
        {
#pragma omp atomic
            count++;
            if (omp_get_thread_num() == 0) num_threads = omp_get_num_threads();
        }

        REQUIRE_EQ(count, num_threads);

        flare::initialize();

        count = 0;
#pragma omp parallel
        {
#pragma omp atomic
            count++;
        }

        concurrency = flare::OpenMP().concurrency();
        REQUIRE_EQ(count, concurrency);

        flare::finalize();

        count = 0;
#pragma omp parallel
        {
#pragma omp atomic
            count++;
        }

        REQUIRE_EQ(count, concurrency);
    }
}  // namespace Test
