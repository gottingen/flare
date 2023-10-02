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


#ifndef FLARE_CORE_ATOMIC_COMMON_H_
#define FLARE_CORE_ATOMIC_COMMON_H_

#include <cstdint>
#include <flare/core/defines.h>
#include <type_traits>

namespace flare {
struct alignas(16) Dummy16ByteValue {
  int64_t value1;
  int64_t value2;
  bool operator!=(Dummy16ByteValue v) const {
    return (value1 != v.value1) || (value2 != v.value2);
  }
  bool operator==(Dummy16ByteValue v) const {
    return (value1 == v.value1) && (value2 == v.value2);
  }
};
}  // namespace flare

namespace flare {
// Memory order sequential consistent
struct MemoryOrderSeqCst {};
// Memory order acquire release
struct MemoryOrderAcqRel {};
// Memory order acquire
struct MemoryOrderAcquire {};
// Memory order release
struct MemoryOrderRelease {};
// Memory order relaxed
struct MemoryOrderRelaxed {};
}  // namespace flare

namespace flare {
// Entire machine scope (e.g. for global arrays)
struct MemoryScopeSystem {};
// Node level
struct MemoryScopeNode {};
// Device or socket scope (i.e. a CPU socket, a single GPU)
struct MemoryScopeDevice {};
// Core scoped (i.e. a shared Level 1 cache)
struct MemoryScopeCore {};
// Caller scoped (i.e. NOT atomic!)
struct MemoryScopeCaller {};
}  // namespace flare

namespace flare {
namespace detail {
template <class MemoryOrder>
struct CmpExchFailureOrder {
  using memory_order = std::conditional_t<
      std::is_same<MemoryOrder, MemoryOrderAcqRel>{},
      MemoryOrderAcquire,
      std::conditional_t<std::is_same<MemoryOrder, MemoryOrderRelease>{},
                         MemoryOrderRelaxed,
                         MemoryOrder>>;
};
template <class MemoryOrder>
using cmpexch_failure_memory_order =
    typename CmpExchFailureOrder<MemoryOrder>::memory_order;
}  // namespace detail
}  // namespace flare

// We should in principle use std::numeric_limits, but that requires constexpr function
// support on device Currently that is still considered experimental on CUDA and
// sometimes not reliable.
namespace flare {
namespace detail {
template <class T>
struct numeric_limits_max;

template <>
struct numeric_limits_max<uint32_t> {
  static constexpr auto value = static_cast<uint32_t>(-1);
};
template <>
struct numeric_limits_max<uint64_t> {
  static constexpr auto value = static_cast<uint64_t>(-1);
};

constexpr bool atomic_always_lock_free(std::size_t size) {
  return size == 4 || size == 8
#if defined(FLARE_HAVE_16BYTE_COMPARE_AND_SWAP)
         || size == 16
#endif
      ;
}

template <std::size_t Size, std::size_t Align>
FLARE_INLINE_FUNCTION bool atomic_is_lock_free() noexcept {
  return Size == 4 || Size == 8
#if defined(FLARE_HAVE_16BYTE_COMPARE_AND_SWAP)
         || Size == 16
#endif
      ;
}

template <std::size_t Bytes>
struct atomic_compare_exchange_helper;

template <>
struct atomic_compare_exchange_helper<4> {
  using type = int32_t;
};

template <>
struct atomic_compare_exchange_helper<8> {
  using type = int64_t;
};

template <>
struct atomic_compare_exchange_helper<16> {
  using type = Dummy16ByteValue;
};

template <class T>
using atomic_compare_exchange_t =
    typename atomic_compare_exchange_helper<sizeof(T)>::type;

template <class T>
struct dont_deduce_this_parameter {
  using type = T;
};

template <class T>
using dont_deduce_this_parameter_t = typename dont_deduce_this_parameter<T>::type;

}  // namespace detail
}  // namespace flare

#endif  // FLARE_CORE_ATOMIC_COMMON_H_
