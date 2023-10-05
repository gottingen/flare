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

#ifndef FLARE_CORE_MEMORY_HOST_SHARED_PTR_H_
#define FLARE_CORE_MEMORY_HOST_SHARED_PTR_H_

#include <flare/core/defines.h>
#include <flare/core/atomic.h>
#include <flare/core/common/error.h>

#include <functional>

namespace flare::detail {

    template<typename T>
    class HostSharedPtr {
    public:
        using element_type = T;

        FLARE_DEFAULTED_FUNCTION constexpr HostSharedPtr() = default;

        FLARE_FUNCTION constexpr HostSharedPtr(std::nullptr_t) {}

        explicit HostSharedPtr(T *element_ptr)
                : HostSharedPtr(element_ptr, [](T *const t) { delete t; }) {}

        template<class Deleter>
        HostSharedPtr(T *element_ptr, const Deleter &deleter)
                : m_element_ptr(element_ptr) {
            static_assert(std::is_invocable_v<Deleter, T *> &&
                          std::is_copy_constructible_v<Deleter>);
            if (element_ptr) {
                try {
                    m_control = new Control{deleter, 1};
                } catch (...) {
                    deleter(element_ptr);
                    throw;
                }
            }
        }

        FLARE_FUNCTION HostSharedPtr(HostSharedPtr &&other) noexcept
                : m_element_ptr(other.m_element_ptr), m_control(other.m_control) {
            other.m_element_ptr = nullptr;
            other.m_control = nullptr;
        }

        FLARE_FUNCTION HostSharedPtr(const HostSharedPtr &other) noexcept
                : m_element_ptr(other.m_element_ptr), m_control(other.m_control) {
            FLARE_IF_ON_HOST(
            (if (m_control) flare::atomic_add(&(m_control->m_counter), 1);))
            FLARE_IF_ON_DEVICE(m_control = nullptr;)
        }

        FLARE_FUNCTION HostSharedPtr &operator=(HostSharedPtr &&other) noexcept {
            if (&other != this) {
                cleanup();
                m_element_ptr = other.m_element_ptr;
                other.m_element_ptr = nullptr;
                m_control = other.m_control;
                other.m_control = nullptr;
            }
            return *this;
        }

        FLARE_FUNCTION HostSharedPtr &operator=(
                const HostSharedPtr &other) noexcept {
            if (&other != this) {
                cleanup();
                m_element_ptr = other.m_element_ptr;
                m_control = other.m_control;
                FLARE_IF_ON_HOST(
                (if (m_control) flare::atomic_add(&(m_control->m_counter), 1);))
                FLARE_IF_ON_DEVICE(m_control = nullptr;)
            }
            return *this;
        }

        FLARE_FUNCTION ~HostSharedPtr() { cleanup(); }

        // returns the stored pointer
        FLARE_FUNCTION T *get() const noexcept { return m_element_ptr; }
        // dereferences the stored pointer
        FLARE_FUNCTION T &operator*() const noexcept {
            FLARE_EXPECTS(bool(*this));
            return *get();
        }
        // dereferences the stored pointer
        FLARE_FUNCTION T *operator->() const noexcept {
            FLARE_EXPECTS(bool(*this));
            return get();
        }

        // checks if the stored pointer is not null
        FLARE_FUNCTION explicit operator bool() const noexcept {
            return get() != nullptr;
        }

        // returns the number of HostSharedPtr instances managing the current object
        // or 0 if there is no managed object.
        int use_count() const noexcept {
            return m_control ? m_control->m_counter : 0;
        }

    private:
        FLARE_FUNCTION void cleanup() noexcept {
            FLARE_IF_ON_HOST((
                                     // If m_counter is set, then this instance is responsible for managing
                                     // the object pointed to by m_counter and m_element_ptr.
                                     if (m_control) {
                                     int const count =
                                     flare::atomic_fetch_sub(&(m_control->m_counter), 1);
                                     // atomic_fetch_sub might have memory order relaxed, so we need to
                                     // force synchronization to avoid multiple threads doing the cleanup.
                                     flare::memory_fence();
                                     if (count == 1) {
                             (m_control->m_deleter)(m_element_ptr);
                                     m_element_ptr = nullptr;
                                     delete m_control;
                                     m_control = nullptr;
                             }
                             }))
        }

        struct Control {
            std::function<void(T *)> m_deleter;
            int m_counter;
        };

        T *m_element_ptr = nullptr;
        Control *m_control = nullptr;
    };
}  // namespace flare::detail

#endif  // FLARE_CORE_MEMORY_HOST_SHARED_PTR_H_
