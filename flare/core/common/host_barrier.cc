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

#include <flare/core/defines.h>

#include <flare/core/common/host_barrier.h>
#include <flare/core/common/bit_ops.h>

#include <thread>
#if defined(_WIN32)
#include <process.h>
#include <winsock2.h>
#include <windows.h>
#endif

namespace flare {
namespace detail {

void HostBarrier::impl_backoff_wait_until_equal(
    int* ptr, const int v, const bool active_wait) noexcept {
  unsigned count = 0u;

  while (!test_equal(ptr, v)) {
    const int c = int_log2(++count);
    if (!active_wait || c > log2_iterations_till_sleep) {
      std::this_thread::sleep_for(
          std::chrono::nanoseconds(c < 16 ? 256 * c : 4096));
    } else if (c > log2_iterations_till_yield) {
      std::this_thread::yield();
    }
#if defined(FLARE_ENABLE_ASM)
#if defined(__PPC64__)
    for (int j = 0; j < num_nops; ++j) {
      asm volatile("nop\n");
    }
    asm volatile("or 27, 27, 27" ::: "memory");
#elif defined(__amd64) || defined(__amd64__) || defined(__x86_64) || \
    defined(__x86_64__)
    for (int j = 0; j < num_nops; ++j) {
      asm volatile("nop\n");
    }
    asm volatile("pause\n" ::: "memory");
#endif
#endif
  }
}
}  // namespace detail
}  // namespace flare
