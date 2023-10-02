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

#ifndef FLARE_BACKEND_THREADS_THREADS_UNIQUE_TOKEN_H_
#define FLARE_BACKEND_THREADS_THREADS_UNIQUE_TOKEN_H_

#include <flare/core/parallel/unique_token.h>

namespace flare {
namespace experimental {

template <>
class UniqueToken<Threads, UniqueTokenScope::Instance> {
 private:
  using buffer_type = flare::View<uint32_t *, flare::HostSpace>;
  int m_count;
  buffer_type m_buffer_view;
  uint32_t volatile *m_buffer;

 public:
  using execution_space = Threads;
  using size_type       = int;

  /// \brief create object size for concurrency on the given instance
  ///
  /// This object should not be shared between instances
  UniqueToken(execution_space const & = execution_space()) noexcept
      : m_count(::flare::Threads::impl_thread_pool_size()),
        m_buffer_view(buffer_type()),
        m_buffer(nullptr) {}

  UniqueToken(size_type max_size, execution_space const & = execution_space())
      : m_count(max_size > ::flare::Threads::impl_thread_pool_size()
                    ? ::flare::Threads::impl_thread_pool_size()
                    : max_size),
        m_buffer_view(
            max_size > ::flare::Threads::impl_thread_pool_size()
                ? buffer_type()
                : buffer_type("UniqueToken::m_buffer_view",
                              ::flare::detail::concurrent_bitset::buffer_bound(
                                  m_count))),
        m_buffer(m_buffer_view.data()) {}

  /// \brief upper bound for acquired values, i.e. 0 <= value < size()
  FLARE_INLINE_FUNCTION
  int size() const noexcept { return m_count; }

  /// \brief acquire value such that 0 <= value < size()
  FLARE_INLINE_FUNCTION
  int acquire() const noexcept {
    FLARE_IF_ON_HOST((
        if (m_buffer == nullptr) {
          return Threads::impl_thread_pool_rank();
        } else {
          const ::flare::pair<int, int> result =
              ::flare::detail::concurrent_bitset::acquire_bounded(
                  m_buffer, m_count, ::flare::detail::clock_tic() % m_count);

          if (result.first < 0) {
            ::flare::abort(
                "UniqueToken<Threads> failure to acquire tokens, no tokens "
                "available");
          }
          return result.first;
        }))

    FLARE_IF_ON_DEVICE((return 0;))
  }

  /// \brief release a value acquired by generate
  FLARE_INLINE_FUNCTION
  void release(int i) const noexcept {
    FLARE_IF_ON_HOST((if (m_buffer != nullptr) {
      ::flare::detail::concurrent_bitset::release(m_buffer, i);
    }))

    FLARE_IF_ON_DEVICE(((void)i;))
  }
};

template <>
class UniqueToken<Threads, UniqueTokenScope::Global> {
 public:
  using execution_space = Threads;
  using size_type       = int;

  /// \brief create object size for concurrency on the given instance
  ///
  /// This object should not be shared between instances
  UniqueToken(execution_space const & = execution_space()) noexcept {}

  /// \brief upper bound for acquired values, i.e. 0 <= value < size()
  FLARE_INLINE_FUNCTION
  int size() const noexcept {
    FLARE_IF_ON_HOST((return Threads::impl_thread_pool_size();))

    FLARE_IF_ON_DEVICE((return 0;))
  }

  /// \brief acquire value such that 0 <= value < size()
  FLARE_INLINE_FUNCTION
  int acquire() const noexcept {
    FLARE_IF_ON_HOST((return Threads::impl_thread_pool_rank();))

    FLARE_IF_ON_DEVICE((return 0;))
  }

  /// \brief release a value acquired by generate
  FLARE_INLINE_FUNCTION
  void release(int) const noexcept {}
};

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_BACKEND_THREADS_THREADS_UNIQUE_TOKEN_H_
