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


#ifndef FLARE_CORE_ATOMIC_ADAPT_CXX_H_
#define FLARE_CORE_ATOMIC_ADAPT_CXX_H_

#include <atomic>
#include <flare/core/atomic/common.h>

namespace flare {
namespace detail {

template <class MemoryOrderDesul>
struct CXXMemoryOrder;

template <>
struct CXXMemoryOrder<MemoryOrderRelaxed> {
  static constexpr std::memory_order value = std::memory_order_relaxed;
};

template <>
struct CXXMemoryOrder<MemoryOrderAcquire> {
  static constexpr std::memory_order value = std::memory_order_acquire;
};

template <>
struct CXXMemoryOrder<MemoryOrderRelease> {
  static constexpr std::memory_order value = std::memory_order_release;
};

template <>
struct CXXMemoryOrder<MemoryOrderAcqRel> {
  static constexpr std::memory_order value = std::memory_order_acq_rel;
};

template <>
struct CXXMemoryOrder<MemoryOrderSeqCst> {
  static constexpr std::memory_order value = std::memory_order_seq_cst;
};

}  // namespace detail
}  // namespace flare

#endif  // FLARE_CORE_ATOMIC_ADAPT_CXX_H_
