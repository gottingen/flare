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
#ifndef FLARE_RUNTIME_ALGORITHM_CRITICAL_H_
#define FLARE_RUNTIME_ALGORITHM_CRITICAL_H_

#include <flare/runtime/core/task.h>


namespace flare::rt {

    // ----------------------------------------------------------------------------
    // CriticalSection
    // ----------------------------------------------------------------------------

    /**
    @class CriticalSection

    @brief class to create a critical region of limited workers to run tasks

    flare::rt::CriticalSection is a warpper over flare::rt::Semaphore and is specialized for
    limiting the maximum concurrency over a set of tasks.
    A critical section starts with an initial count representing that limit.
    When a task is added to the critical section,
    the task acquires and releases the semaphore internal to the critical section.
    This design avoids explicit call of flare::rt::Task::acquire and flare::rt::Task::release.
    The following example creates a critical section of one worker and adds
    the five tasks to the critical section.

    @code{.cpp}
    flare::rt::Executor executor(8);   // create an executor of 8 workers
    flare::rt::Taskflow taskflow;

    // create a critical section of 1 worker
    flare::rt::CriticalSection critical_section(1);

    flare::rt::Task A = taskflow.emplace([](){ std::cout << "A" << std::endl; });
    flare::rt::Task B = taskflow.emplace([](){ std::cout << "B" << std::endl; });
    flare::rt::Task C = taskflow.emplace([](){ std::cout << "C" << std::endl; });
    flare::rt::Task D = taskflow.emplace([](){ std::cout << "D" << std::endl; });
    flare::rt::Task E = taskflow.emplace([](){ std::cout << "E" << std::endl; });

    critical_section.add(A, B, C, D, E);

    executor.run(taskflow).wait();
    @endcode

    */
    class CriticalSection : public Semaphore {

    public:

        /**
        @brief constructs a critical region of a limited number of workers
        */
        explicit CriticalSection(size_t max_workers = 1);

        /**
        @brief adds a task into the critical region
        */
        template<typename... Tasks>
        void add(Tasks...tasks);
    };

    inline CriticalSection::CriticalSection(size_t max_workers) :
            Semaphore{max_workers} {
    }

    template<typename... Tasks>
    void CriticalSection::add(Tasks... tasks) {
        (tasks.acquire(*this), ...);
        (tasks.release(*this), ...);
    }


}  // namespace flare::rt
#endif  // FLARE_RUNTIME_ALGORITHM_CRITICAL_H_

