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

#ifndef FLARE_TIMER_H_
#define FLARE_TIMER_H_

#include <flare/core/defines.h>

#if defined(FLARE_COMPILER_GNU) && (FLARE_COMPILER_GNU == 1030) && \
    defined(FLARE_COMPILER_NVCC)
#include <sys/time.h>
#else

#include <chrono>

#endif

namespace flare {

/** \brief  Time since construction */

#if defined(FLARE_COMPILER_GNU) && (FLARE_COMPILER_GNU == 1030) && \
    defined(FLARE_COMPILER_NVCC)
    class Timer {
     private:
      struct timeval m_old;

     public:
      inline void reset() { gettimeofday(&m_old, nullptr); }

      inline ~Timer() = default;

      inline Timer() { reset(); }

      Timer(const Timer&) = delete;
      Timer& operator=(const Timer&) = delete;

      inline double seconds() const {
        struct timeval m_new;

        gettimeofday(&m_new, nullptr);

        return ((double)(m_new.tv_sec - m_old.tv_sec)) +
               ((double)(m_new.tv_usec - m_old.tv_usec) * 1.0e-6);
      }
    };
#else

    class Timer {
    private:
        std::chrono::high_resolution_clock::time_point m_old;

    public:
        inline void reset() { m_old = std::chrono::high_resolution_clock::now(); }

        inline ~Timer() = default;

        inline Timer() { reset(); }

        Timer(const Timer &);

        Timer &operator=(const Timer &);

        inline double seconds() const {
            std::chrono::high_resolution_clock::time_point m_new =
                    std::chrono::high_resolution_clock::now();
            return std::chrono::duration_cast<std::chrono::duration<double> >(m_new -
                                                                              m_old)
                    .count();
        }
    };

#endif

}  // namespace flare

#endif  // FLARE_TIMER_H_
