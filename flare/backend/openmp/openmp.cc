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

#include <flare/backend/openmp/openmp.h>
#include <flare/backend/openmp/openmp_instance.h>

#include <flare/core/common/exec_space_manager.h>

namespace flare {

OpenMP::OpenMP()
    : m_space_instance(&detail::OpenMPInternal::singleton(),
                       [](detail::OpenMPInternal *) {}) {
  detail::OpenMPInternal::singleton().verify_is_initialized(
      "OpenMP instance constructor");
}

OpenMP::OpenMP(int pool_size)
    : m_space_instance(new detail::OpenMPInternal(pool_size),
                       [](detail::OpenMPInternal *ptr) {
                         ptr->finalize();
                         delete ptr;
                       }) {
  detail::OpenMPInternal::singleton().verify_is_initialized(
      "OpenMP instance constructor");
}

int OpenMP::impl_get_current_max_threads() noexcept {
  return detail::OpenMPInternal::get_current_max_threads();
}

void OpenMP::impl_initialize(InitializationSettings const &settings) {
  detail::OpenMPInternal::singleton().initialize(
      settings.has_num_threads() ? settings.get_num_threads() : -1);
}

void OpenMP::impl_finalize() { detail::OpenMPInternal::singleton().finalize(); }

void OpenMP::print_configuration(std::ostream &os, bool /*verbose*/) const {
  os << "Host Parallel Execution Space:\n";
  os << "  FLARE_ENABLE_OPENMP: yes\n";

  os << "\nOpenMP Runtime Configuration:\n";

  m_space_instance->print_configuration(os);
}

int OpenMP::concurrency() const { return impl_thread_pool_size(); }

void OpenMP::fence(const std::string &name) const {
  flare::Tools::experimental::detail::profile_fence_event<flare::OpenMP>(
      name, flare::Tools::experimental::detail::DirectFenceIDHandle{1}, []() {});
}

bool OpenMP::impl_is_initialized() noexcept {
  return detail::OpenMPInternal::singleton().is_initialized();
}

bool OpenMP::in_parallel(OpenMP const &exec_space) noexcept {
  return exec_space.impl_internal_space_instance()->m_level < omp_get_level();
}

int OpenMP::impl_thread_pool_size() const noexcept {
  return OpenMP::in_parallel(*this)
             ? omp_get_num_threads()
             : impl_internal_space_instance()->m_pool_size;
}

int OpenMP::impl_max_hardware_threads() noexcept {
  return detail::g_openmp_hardware_max_threads;
}

namespace detail {

int g_openmp_space_factory_initialized =
    initialize_space_factory<OpenMP>("050_OpenMP");

}  // namespace detail

}  // namespace flare
