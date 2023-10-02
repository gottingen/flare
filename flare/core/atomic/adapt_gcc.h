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


#ifndef FLARE_CORE_ATOMIC_ADAPT_GCC_H_
#define FLARE_CORE_ATOMIC_ADAPT_GCC_H_

#include <flare/core/atomic/common.h>

namespace flare {
namespace detail {

template <class MemoryOrder>
struct GCCMemoryOrder;

template <>
struct GCCMemoryOrder<MemoryOrderRelaxed> {
  static constexpr int value = __ATOMIC_RELAXED;
};

template <>
struct GCCMemoryOrder<MemoryOrderAcquire> {
  static constexpr int value = __ATOMIC_ACQUIRE;
};

template <>
struct GCCMemoryOrder<MemoryOrderRelease> {
  static constexpr int value = __ATOMIC_RELEASE;
};

template <>
struct GCCMemoryOrder<MemoryOrderAcqRel> {
  static constexpr int value = __ATOMIC_ACQ_REL;
};

template <>
struct GCCMemoryOrder<MemoryOrderSeqCst> {
  static constexpr int value = __ATOMIC_SEQ_CST;
};

}  // namespace detail
}  // namespace flare

#endif  // FLARE_CORE_ATOMIC_ADAPT_GCC_H_
