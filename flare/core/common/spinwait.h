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

#ifndef FLARE_SPINWAIT_HPP
#define FLARE_SPINWAIT_HPP

#include <flare/core/defines.h>
#include <flare/core/atomic.h>

#include <cstdint>

#include <type_traits>

namespace flare {
namespace detail {

enum class WaitMode : int {
  ACTIVE  // Used for tight loops to keep threads active longest
  ,
  PASSIVE  // Used to quickly yield the thread to quite down the system
  ,
  ROOT  // Never sleep or yield the root thread
};

void host_thread_yield(const uint32_t i, const WaitMode mode);

template <typename T>
std::enable_if_t<std::is_integral<T>::value, void> root_spinwait_while_equal(
    T const volatile& flag, const T value) {
  flare::store_fence();
  uint32_t i = 0;
  while (value == flag) {
    host_thread_yield(++i, WaitMode::ROOT);
  }
  flare::load_fence();
}

template <typename T>
std::enable_if_t<std::is_integral<T>::value, void> root_spinwait_until_equal(
    T const volatile& flag, const T value) {
  flare::store_fence();
  uint32_t i = 0;
  while (value != flag) {
    host_thread_yield(++i, WaitMode::ROOT);
  }
  flare::load_fence();
}

template <typename T>
std::enable_if_t<std::is_integral<T>::value, void> spinwait_while_equal(
    T const volatile& flag, const T value) {
  flare::store_fence();
  uint32_t i = 0;
  while (value == flag) {
    host_thread_yield(++i, WaitMode::ACTIVE);
  }
  flare::load_fence();
}

template <typename T>
std::enable_if_t<std::is_integral<T>::value, void> yield_while_equal(
    T const volatile& flag, const T value) {
  flare::store_fence();
  uint32_t i = 0;
  while (value == flag) {
    host_thread_yield(++i, WaitMode::PASSIVE);
  }
  flare::load_fence();
}

template <typename T>
std::enable_if_t<std::is_integral<T>::value, void> spinwait_until_equal(
    T const volatile& flag, const T value) {
  flare::store_fence();
  uint32_t i = 0;
  while (value != flag) {
    host_thread_yield(++i, WaitMode::ACTIVE);
  }
  flare::load_fence();
}

template <typename T>
std::enable_if_t<std::is_integral<T>::value, void> yield_until_equal(
    T const volatile& flag, const T value) {
  flare::store_fence();
  uint32_t i = 0;
  while (value != flag) {
    host_thread_yield(++i, WaitMode::PASSIVE);
  }
  flare::load_fence();
}

} /* namespace detail */
} /* namespace flare */

#endif /* #ifndef FLARE_SPINWAIT_HPP */
