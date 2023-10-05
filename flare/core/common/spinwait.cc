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

#include <flare/core/atomic.h>
#include <flare/core/common/spinwait.h>
#include <flare/core/common/bit_ops.h>

#include <thread>

#if defined(_WIN32)
#include <process.h>
#include <winsock2.h>
#include <windows.h>
#endif

namespace flare::detail {

    void host_thread_yield(const uint32_t i, const WaitMode mode) {
        static constexpr uint32_t sleep_limit = 1 << 13;
        static constexpr uint32_t yield_limit = 1 << 12;

        const int c = int_log2(i);

        if (WaitMode::ROOT != mode) {
            if (sleep_limit < i) {
                // Attempt to put the thread to sleep for 'c' microseconds
                std::this_thread::yield();
                std::this_thread::sleep_for(std::chrono::microseconds(c));
            } else if (mode == WaitMode::PASSIVE || yield_limit < i) {
                // Attempt to yield thread resources to runtime
                std::this_thread::yield();
            }
#if defined(FLARE_ENABLE_ASM)

            else if ((1u << 4) < i) {

                // Insert a few no-ops to quiet the thread:

                for (int k = 0; k < c; ++k) {
#if defined(__amd64) || defined(__amd64__) || defined(__x86_64) || \
    defined(__x86_64__)
#if !defined(_WIN32) /* IS NOT Microsoft Windows */
                    asm volatile("nop\n");
#else
                    __asm__ __volatile__("nop\n");
#endif
#elif defined(__PPC64__)
                    asm volatile("nop\n");
#endif
                }
            }
#endif /* defined( FLARE_ENABLE_ASM ) */
        }
#if defined(FLARE_ENABLE_ASM)
        else if ((1u << 3) < i) {
            // no-ops for root thread
            for (int k = 0; k < c; ++k) {
#if defined(__amd64) || defined(__amd64__) || defined(__x86_64) || \
    defined(__x86_64__)
#if !defined(_WIN32) /* IS NOT Microsoft Windows */
                asm volatile("nop\n");
#else
                __asm__ __volatile__("nop\n");
#endif
#elif defined(__PPC64__)
                asm volatile("nop\n");
#endif
            }
        }

        {
            // Insert memory pause
#if defined(__amd64) || defined(__amd64__) || defined(__x86_64) || \
    defined(__x86_64__)
#if !defined(_WIN32) /* IS NOT Microsoft Windows */
            asm volatile("pause\n":: : "memory");
#else
            __asm__ __volatile__("pause\n" ::: "memory");
#endif
#elif defined(__PPC64__)
            asm volatile("or 27, 27, 27" ::: "memory");
#endif
        }

#endif /* defined( FLARE_ENABLE_ASM ) */
    }

}  // namespace flare::detail
