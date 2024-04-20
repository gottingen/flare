// Copyright 2023 The EA Authors.
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
#pragma once

#include <Param.hpp>
#include <common/util.hpp>
#include <memory.hpp>

#include <algorithm>

// FIXME: Is there a better way to check for std::future not being supported ?
#if defined(FLY_DISABLE_CPU_ASYNC) || \
    (defined(__GNUC__) &&            \
     (__GCC_ATOMIC_INT_LOCK_FREE < 2 || __GCC_ATOMIC_POINTER_LOCK_FREE < 2))

#include <functional>
using std::function;
#include <err_cpu.hpp>
#define __SYNCHRONOUS_ARCH 1
class queue_impl {
   public:
    template<typename F, typename... Args>
    void enqueue(const F func, Args... args) const {
        FLY_ERROR("Incorrectly configured", FLY_ERR_INTERNAL);
    }

    void sync() const { FLY_ERROR("Incorrectly configured", FLY_ERR_INTERNAL); }

    bool is_worker() const {
        FLY_ERROR("Incorrectly configured", FLY_ERR_INTERNAL);
        return false;
    }
};

#else

#include <fly/threads/async_queue.hpp>
#include <fly/threads/event.hpp>
#define __SYNCHRONOUS_ARCH 0
using queue_impl = threads::async_queue;
using event_impl = threads::event;

#endif

namespace flare {
namespace cpu {

/// Wraps the async_queue class
class queue {
   public:
    queue()
        : count(0)
        , sync_calls(__SYNCHRONOUS_ARCH == 1 ||
                     common::getEnvVar("FLY_SYNCHRONOUS_CALLS") == "1") {}

    template<typename F, typename... Args>
    void enqueue(const F func, Args &&...args) {
        count++;
        if (sync_calls) {
            func(toParam(std::forward<Args>(args))...);
        } else {
            aQueue.enqueue(func, toParam(std::forward<Args>(args))...);
        }
#ifndef NDEBUG
        sync();
#else
        if (getMemoryPressure() >= getMemoryPressureThreshold() ||
            count >= 25) {
            sync();
        }
#endif
    }

    void sync() {
        count = 0;
        if (!sync_calls) aQueue.sync();
    }

    bool is_worker() const {
        return (!sync_calls) ? aQueue.is_worker() : false;
    }

    friend class queue_event;

   private:
    int count;
    const bool sync_calls;
    queue_impl aQueue;
};

class queue_event {
    event_impl event_;

   public:
    queue_event() = default;
    queue_event(int val) : event_(val) {}

    int create() { return event_.create(); }

    int mark(queue &q) { return event_.mark(q.aQueue); }
    int wait(queue &q) { return event_.wait(q.aQueue); }
    int sync() noexcept { return event_.sync(); }
    operator bool() const noexcept { return event_; }
};
}  // namespace cpu
}  // namespace flare
