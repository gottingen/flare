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

#ifndef FLARE_CORE_COMMON_CLOCKTIC_H_
#define FLARE_CORE_COMMON_CLOCKTIC_H_

#include <flare/core/defines.h>
#include <stdint.h>
#include <chrono>

namespace flare::detail {

    /**\brief  Quick query of clock register tics
     *
     *  Primary use case is to, with low overhead,
     *  obtain a integral value that consistently varies
     *  across concurrent threads of execution within
     *  a parallel algorithm.
     *  This value is often used to "randomly" seed an
     *  attempt to acquire an indexed resource (e.g., bit)
     *  from an array of resources (e.g., bitset) such that
     *  concurrent threads will have high likelihood of
     *  having different index-seed values.
     */

    FLARE_IMPL_DEVICE_FUNCTION inline uint64_t clock_tic_device() noexcept {
#if defined(__CUDACC__)

        // Return value of 64-bit hi-res clock register.
        return clock64();
#else

        return 0;

#endif
    }

    FLARE_IMPL_HOST_FUNCTION inline uint64_t clock_tic_host() noexcept {
#if defined(__i386__) || defined(__x86_64)

        // Return value of 64-bit hi-res clock register.

        unsigned a = 0, d = 0;

        __asm__ volatile("rdtsc" : "=a"(a), "=d"(d));

        return ((uint64_t) a) | (((uint64_t) d) << 32);

#elif defined(__powerpc64__) || defined(__ppc64__)

        unsigned long cycles = 0;

        asm volatile("mftb %0" : "=r"(cycles));

        return (uint64_t)cycles;

#elif defined(__ppc__)
        // see : pages.cs.wisc.edu/~legault/miniproj-736.pdf or
        // cmssdt.cern.ch/lxr/source/FWCore/Utilities/interface/HRRealTime.h

        uint64_t result = 0;
        uint32_t upper, lower, tmp;

        __asm__ volatile(
            "0: \n"
            "\tmftbu %0     \n"
            "\tmftb  %1     \n"
            "\tmftbu %2     \n"
            "\tcmpw  %2, %0 \n"
            "\tbne   0b     \n"
            : "=r"(upper), "=r"(lower), "=r"(tmp));
        result = upper;
        result = result << 32;
        result = result | lower;

        return (result);

#else

        return std::chrono::high_resolution_clock::now().time_since_epoch().count();

#endif
    }

    FLARE_FORCEINLINE_FUNCTION
    uint64_t clock_tic() noexcept {
        FLARE_IF_ON_DEVICE((return clock_tic_device();))
        FLARE_IF_ON_HOST((return clock_tic_host();))
    }

}  // namespace flare::detail

#endif  // FLARE_CORE_COMMON_CLOCKTIC_H_
