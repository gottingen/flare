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
#include <flare/backend/serial/serial.h>
#include <flare/core/common/traits.h>
#include <flare/core/common/error.h>
#include <flare/core/common/exec_space_manager.h>
#include <flare/core/memory/shared_alloc.h>
#include <cstdlib>
#include <iostream>
#include <sstream>

namespace flare::detail {

    bool SerialInternal::is_initialized() { return m_is_initialized; }

    void SerialInternal::initialize() {
        if (is_initialized()) return;

        detail::SharedAllocationRecord<void, void>::tracking_enable();

        m_is_initialized = true;
    }

    void SerialInternal::finalize() {
        if (m_thread_team_data.scratch_buffer()) {
            m_thread_team_data.disband_team();
            m_thread_team_data.disband_pool();

            flare::HostSpace space;

            space.deallocate(m_thread_team_data.scratch_buffer(),
                             m_thread_team_data.scratch_bytes());

            m_thread_team_data.scratch_assign(nullptr, 0, 0, 0, 0, 0);
        }

        flare::Profiling::finalize();

        m_is_initialized = false;
    }

    SerialInternal &SerialInternal::singleton() {
        static SerialInternal *self = nullptr;
        if (!self) {
            self = new SerialInternal();
        }
        return *self;
    }

    // Resize thread team data scratch memory
    void SerialInternal::resize_thread_team_data(size_t pool_reduce_bytes,
                                                 size_t team_reduce_bytes,
                                                 size_t team_shared_bytes,
                                                 size_t thread_local_bytes) {
        if (pool_reduce_bytes < 512) pool_reduce_bytes = 512;
        if (team_reduce_bytes < 512) team_reduce_bytes = 512;

        const size_t old_pool_reduce = m_thread_team_data.pool_reduce_bytes();
        const size_t old_team_reduce = m_thread_team_data.team_reduce_bytes();
        const size_t old_team_shared = m_thread_team_data.team_shared_bytes();
        const size_t old_thread_local = m_thread_team_data.thread_local_bytes();
        const size_t old_alloc_bytes = m_thread_team_data.scratch_bytes();

        // Allocate if any of the old allocation is tool small:

        const bool allocate = (old_pool_reduce < pool_reduce_bytes) ||
                              (old_team_reduce < team_reduce_bytes) ||
                              (old_team_shared < team_shared_bytes) ||
                              (old_thread_local < thread_local_bytes);

        if (allocate) {
            flare::HostSpace space;

            if (old_alloc_bytes) {
                m_thread_team_data.disband_team();
                m_thread_team_data.disband_pool();

                space.deallocate("flare::Serial::scratch_mem",
                                 m_thread_team_data.scratch_buffer(),
                                 m_thread_team_data.scratch_bytes());
            }

            if (pool_reduce_bytes < old_pool_reduce) {
                pool_reduce_bytes = old_pool_reduce;
            }
            if (team_reduce_bytes < old_team_reduce) {
                team_reduce_bytes = old_team_reduce;
            }
            if (team_shared_bytes < old_team_shared) {
                team_shared_bytes = old_team_shared;
            }
            if (thread_local_bytes < old_thread_local) {
                thread_local_bytes = old_thread_local;
            }

            const size_t alloc_bytes =
                    HostThreadTeamData::scratch_size(pool_reduce_bytes, team_reduce_bytes,
                                                     team_shared_bytes, thread_local_bytes);

            void *ptr = nullptr;
            try {
                ptr = space.allocate("flare::Serial::scratch_mem", alloc_bytes);
            } catch (flare::experimental::RawMemoryAllocationFailure const &failure) {
                // For now, just rethrow the error message the existing way
                flare::detail::throw_runtime_exception(failure.get_error_message());
            }

            m_thread_team_data.scratch_assign(static_cast<char *>(ptr), alloc_bytes,
                                              pool_reduce_bytes, team_reduce_bytes,
                                              team_shared_bytes, thread_local_bytes);

            HostThreadTeamData *pool[1] = {&m_thread_team_data};

            m_thread_team_data.organize_pool(pool, 1);
            m_thread_team_data.organize_team(1);
        }
    }
}  // namespace flare::detail

namespace flare {
    Serial::Serial()
            : m_space_instance(&detail::SerialInternal::singleton(),
                               [](detail::SerialInternal *) {}) {}

    void Serial::print_configuration(std::ostream &os, bool /*verbose*/) const {
        os << "Host Serial Execution Space:\n";
        os << "  FLARE_ENABLE_SERIAL: yes\n";

#ifdef FLARE_INTERNAL_NOT_PARALLEL
        os << "flare atomics disabled\n";
#endif

        os << "\nSerial Runtime Configuration:\n";
    }

    bool Serial::impl_is_initialized() {
        return detail::SerialInternal::singleton().is_initialized();
    }

    void Serial::impl_initialize(InitializationSettings const &) {
        detail::SerialInternal::singleton().initialize();
    }

    void Serial::impl_finalize() { detail::SerialInternal::singleton().finalize(); }

    const char *Serial::name() { return "Serial"; }
}
namespace flare::detail {

    int g_serial_space_factory_initialized =
            initialize_space_factory<Serial>("100_Serial");

}  // namespace flare::detail
