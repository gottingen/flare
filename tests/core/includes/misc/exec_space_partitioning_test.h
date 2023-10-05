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

#include <cstdio>
#include <sstream>
#include <iostream>
#include <thread>
#include <doctest.h>
#include <flare/core.h>

namespace Test {
namespace {
struct SumFunctor {
  FLARE_INLINE_FUNCTION
  void operator()(int i, int& lsum) const { lsum += i; }
};

template <class ExecSpace>
void check_distinctive(ExecSpace, ExecSpace) {}

#ifdef FLARE_ON_CUDA_DEVICE
void check_distinctive(flare::Cuda exec1, flare::Cuda exec2) {
  REQUIRE_NE(exec1.cuda_stream(), exec2.cuda_stream());
}
#endif

#ifdef FLARE_ENABLE_OPENMP
void check_distinctive(flare::OpenMP exec1, flare::OpenMP exec2) {
  REQUIRE_NE(exec1, exec2);
}
#endif
}  // namespace

#ifdef FLARE_ENABLE_OPENMP
template <class Lambda1, class Lambda2>
void run_threaded_test(const Lambda1 l1, const Lambda2 l2) {
#pragma omp parallel num_threads(2)
  {
    if (omp_get_thread_num() == 0) l1();
    if (omp_get_thread_num() == 1) l2();
  }
}
#elif !defined(FLARE_ENABLE_THREADS)
template <class Lambda1, class Lambda2>
void run_threaded_test(const Lambda1 l1, const Lambda2 l2) {
  std::thread t1(std::move(l1));
  std::thread t2(std::move(l2));
  t1.join();
  t2.join();
}
#else
template <class Lambda1, class Lambda2>
void run_threaded_test(const Lambda1 l1, const Lambda2 l2) {
  l1();
  l2();
}
#endif

void test_partitioning(std::vector<TEST_EXECSPACE>& instances) {
  check_distinctive(instances[0], instances[1]);
  int sum1, sum2;
  int N = 3910;
  run_threaded_test(
      [&]() {
        flare::parallel_reduce(
            flare::RangePolicy<TEST_EXECSPACE>(instances[0], 0, N),
            SumFunctor(), sum1);
      },
      [&]() {
        flare::parallel_reduce(
            flare::RangePolicy<TEST_EXECSPACE>(instances[1], 0, N),
            SumFunctor(), sum2);
      });
  REQUIRE_EQ(sum1, sum2);
  REQUIRE_EQ(sum1, N * (N - 1) / 2);

#if defined(FLARE_ON_CUDA_DEVICE) || defined(FLARE_ENABLE_OPENMP)
  // Eliminate unused function warning
  // (i.e. when compiling for Serial and CUDA, during Serial compilation the
  // Cuda overload is unused ...)
  if (sum1 != sum2) {
#ifdef FLARE_ON_CUDA_DEVICE
    check_distinctive(flare::Cuda(), flare::Cuda());
#endif
#ifdef FLARE_ENABLE_OPENMP
    check_distinctive(flare::OpenMP(), flare::OpenMP());
#endif
  }
#endif
}

TEST_CASE("TEST_CATEGORY, partitioning_by_args") {
  auto instances =
      flare::experimental::partition_space(TEST_EXECSPACE(), 1, 1);
  REQUIRE_EQ(int(instances.size()), 2);
  test_partitioning(instances);
}

TEST_CASE("TEST_CATEGORY, partitioning_by_vector") {
  // Make sure we can use a temporary as argument for weights
  auto instances = flare::experimental::partition_space(
      TEST_EXECSPACE(), std::vector<int> /*weights*/ {1, 1});
  REQUIRE_EQ(int(instances.size()), 2);
  test_partitioning(instances);
}
}  // namespace Test
