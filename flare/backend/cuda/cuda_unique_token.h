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

#ifndef FLARE_BACKEND_CUDA_CUDA_UNIQUE_TOKEN_H_
#define FLARE_BACKEND_CUDA_CUDA_UNIQUE_TOKEN_H_

#include <flare/core/defines.h>
#ifdef FLARE_ON_CUDA_DEVICE

#include <flare/backend/cuda/cuda_space.h>
#include <flare/core/parallel/unique_token.h>
#include <flare/core/memory/shared_alloc.h>

namespace flare {

namespace detail {
flare::Tensor<uint32_t*, flare::CudaSpace> cuda_global_unique_token_locks(
    bool deallocate = false);
}

namespace experimental {
// both global and instance Unique Tokens are implemented in the same way
// the global version has one shared static lock array underneath
// but it can't be a static member variable since we need to acces it on device
// and we share the implementation with the instance version
template <>
class UniqueToken<Cuda, UniqueTokenScope::Global> {
 protected:
  flare::Tensor<uint32_t*, flare::CudaSpace> m_locks;

 public:
  using execution_space = Cuda;
  using size_type       = int32_t;

  explicit UniqueToken(execution_space const& = Cuda())
      : m_locks(flare::detail::cuda_global_unique_token_locks()) {}

 protected:
  // These are constructors for the Instance version
  UniqueToken(size_type max_size) {
    m_locks = flare::Tensor<uint32_t*, flare::CudaSpace>(
        "flare::UniqueToken::m_locks", max_size);
  }
  UniqueToken(size_type max_size, execution_space const& exec) {
    m_locks = flare::Tensor<uint32_t*, flare::CudaSpace>(
        flare::tensor_alloc(exec, "flare::UniqueToken::m_locks"), max_size);
  }

 public:
  FLARE_DEFAULTED_FUNCTION
  UniqueToken(const UniqueToken&) = default;

  FLARE_DEFAULTED_FUNCTION
  UniqueToken(UniqueToken&&) = default;

  FLARE_DEFAULTED_FUNCTION
  UniqueToken& operator=(const UniqueToken&) = default;

  FLARE_DEFAULTED_FUNCTION
  UniqueToken& operator=(UniqueToken&&) = default;

  /// \brief upper bound for acquired values, i.e. 0 <= value < size()
  FLARE_INLINE_FUNCTION
  size_type size() const noexcept { return m_locks.extent(0); }

 private:
  __device__ size_type impl_acquire() const {
    int idx = blockIdx.x * (blockDim.x * blockDim.y) +
              threadIdx.y * blockDim.x + threadIdx.x;
    idx = idx % size();
#if defined(FLARE_ARCH_KEPLER) || defined(FLARE_ARCH_PASCAL) || \
    defined(FLARE_ARCH_MAXWELL)
    unsigned int mask        = __activemask();
    unsigned int active      = __ballot_sync(mask, 1);
    unsigned int done_active = 0;
    bool done                = false;
    while (active != done_active) {
      if (!done) {
        if (flare::atomic_compare_exchange(&m_locks(idx), 0, 1) == 0) {
          done = true;
        } else {
          idx += blockDim.y * blockDim.x + 1;
          idx = idx % size();
        }
      }
      done_active = __ballot_sync(mask, done ? 1 : 0);
    }
#else
    while (flare::atomic_compare_exchange(&m_locks(idx), 0, 1) == 1) {
      idx += blockDim.y * blockDim.x + 1;
      idx = idx % size();
    }
#endif
    // Make sure that all writes in the previous lock owner are visible to me
    flare::atomic_thread_fence(flare::MemoryOrderAcquire(),
                               flare::MemoryScopeDevice());
    return idx;
  }

 public:
  /// \brief acquire value such that 0 <= value < size()
  FLARE_INLINE_FUNCTION
  size_type acquire() const {
    FLARE_IF_ON_DEVICE(return impl_acquire();)
    FLARE_IF_ON_HOST(return 0;)
  }

  /// \brief release an acquired value
  FLARE_INLINE_FUNCTION
  void release(size_type idx) const noexcept {
    // Make sure my writes are visible to the next lock owner
    flare::atomic_thread_fence(flare::MemoryOrderRelease(),
                               flare::MemoryScopeDevice());
    (void)flare::atomic_exchange(&m_locks(idx), 0);
  }
};

template <>
class UniqueToken<Cuda, UniqueTokenScope::Instance>
    : public UniqueToken<Cuda, UniqueTokenScope::Global> {
 public:
  // The instance version will forward to protected constructor which creates
  // a lock array per instance
  UniqueToken()
      : UniqueToken<Cuda, UniqueTokenScope::Global>(
            flare::Cuda().concurrency()) {}
  explicit UniqueToken(execution_space const& arg)
      : UniqueToken<Cuda, UniqueTokenScope::Global>(
            flare::Cuda().concurrency(), arg) {}
  explicit UniqueToken(size_type max_size)
      : UniqueToken<Cuda, UniqueTokenScope::Global>(max_size) {}
  UniqueToken(size_type max_size, execution_space const& arg)
      : UniqueToken<Cuda, UniqueTokenScope::Global>(max_size, arg) {}
};

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ON_CUDA_DEVICE
#endif  // FLARE_BACKEND_CUDA_CUDA_UNIQUE_TOKEN_H_
