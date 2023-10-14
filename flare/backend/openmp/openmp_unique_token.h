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

#ifndef FLARE_BACKEND_OPENMP_OPENMP_UNIQUE_TOKEN_H_
#define FLARE_BACKEND_OPENMP_OPENMP_UNIQUE_TOKEN_H_

#include <flare/core/parallel/unique_token.h>

namespace flare::experimental {
template <>
class UniqueToken<OpenMP, UniqueTokenScope::Instance> {
 public:
  using execution_space = OpenMP;
  using size_type       = int;

 private:
  using buffer_type = flare::Tensor<uint32_t*, flare::HostSpace>;
  execution_space m_exec;
  size_type m_count;
  buffer_type m_buffer_tensor;
  uint32_t volatile* m_buffer;

 public:
  /// \brief create object size for concurrency on the given instance
  ///
  /// This object should not be shared between instances
  UniqueToken(execution_space const& exec = execution_space()) noexcept
      : m_exec(exec),
        m_count(m_exec.impl_thread_pool_size()),
        m_buffer_tensor(buffer_type()),
        m_buffer(nullptr) {}

  UniqueToken(size_type max_size,
              execution_space const& exec = execution_space())
      : m_exec(exec),
        m_count(max_size),
        m_buffer_tensor("UniqueToken::m_buffer_tensor",
                      ::flare::detail::concurrent_bitset::buffer_bound(m_count)),
        m_buffer(m_buffer_tensor.data()) {}

  /// \brief upper bound for acquired values, i.e. 0 <= value < size()
  FLARE_INLINE_FUNCTION
  int size() const noexcept {
    FLARE_IF_ON_HOST((return m_count;))

    FLARE_IF_ON_DEVICE((return 0;))
  }

  /// \brief acquire value such that 0 <= value < size()
  FLARE_INLINE_FUNCTION
  int acquire() const noexcept {
    FLARE_IF_ON_HOST(
        (if (m_count >= m_exec.impl_thread_pool_size()) return m_exec
             .impl_thread_pool_rank();
         const ::flare::pair<int, int> result =
             ::flare::detail::concurrent_bitset::acquire_bounded(
                 m_buffer, m_count, ::flare::detail::clock_tic() % m_count);

         if (result.first < 0) {
           ::flare::abort(
               "UniqueToken<OpenMP> failure to acquire tokens, no tokens "
               "available");
         }

         return result.first;))

    FLARE_IF_ON_DEVICE((return 0;))
  }

  /// \brief release a value acquired by generate
  FLARE_INLINE_FUNCTION
  void release(int i) const noexcept {
    FLARE_IF_ON_HOST((if (m_count < m_exec.impl_thread_pool_size()) {
      ::flare::detail::concurrent_bitset::release(m_buffer, i);
    }))

    FLARE_IF_ON_DEVICE(((void)i;))
  }
};

template <>
class UniqueToken<OpenMP, UniqueTokenScope::Global> {
 public:
  using execution_space = OpenMP;
  using size_type       = int;

  /// \brief create object size for concurrency on the given instance
  ///
  /// This object should not be shared between instances
  UniqueToken(execution_space const& = execution_space()) noexcept {}

  /// \brief upper bound for acquired values, i.e. 0 <= value < size()
  FLARE_INLINE_FUNCTION
  int size() const noexcept {
    FLARE_IF_ON_HOST((return flare::detail::g_openmp_hardware_max_threads;))

    FLARE_IF_ON_DEVICE((return 0;))
  }

  /// \brief acquire value such that 0 <= value < size()
  // FIXME this is wrong when using nested parallelism. In that case multiple
  // threads have the same thread ID.
  FLARE_INLINE_FUNCTION
  int acquire() const noexcept {
    FLARE_IF_ON_HOST((return omp_get_thread_num();))

    FLARE_IF_ON_DEVICE((return 0;))
  }

  /// \brief release a value acquired by generate
  FLARE_INLINE_FUNCTION
  void release(int) const noexcept {}
};
}  // namespace flare::experimental

#endif  // FLARE_BACKEND_OPENMP_OPENMP_UNIQUE_TOKEN_H_
