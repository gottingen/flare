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

// Experimental unified task-data parallel manycore LDRD

#ifndef FLARE_CORE_COMMON_OPTIONAL_REF_H_
#define FLARE_CORE_COMMON_OPTIONAL_REF_H_

#include <flare/core/defines.h>

#include <flare/core_fwd.h>

#include <flare/core/memory/pointer_ownership.h>
#include <flare/core/common/error.h>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
namespace flare {
namespace detail {

struct InPlaceTag {};

template <class T>
struct OptionalRef {
 private:
  ObservingRawPtr<T> m_value = nullptr;

 public:
  using value_type = T;

  FLARE_DEFAULTED_FUNCTION
  OptionalRef() = default;

  FLARE_DEFAULTED_FUNCTION
  OptionalRef(OptionalRef const&) = default;

  FLARE_DEFAULTED_FUNCTION
  OptionalRef(OptionalRef&&) = default;

  FLARE_INLINE_FUNCTION
  // MSVC requires that this copy constructor is not defaulted
  // if there exists a (non-defaulted) volatile one.
  OptionalRef& operator=(OptionalRef const& other) noexcept {
    m_value = other.m_value;
    return *this;
  }

  FLARE_INLINE_FUNCTION
  // Can't return a reference to volatile OptionalRef, since GCC issues a
  // warning about reference to volatile not accessing the underlying value
  void operator=(OptionalRef const volatile& other) volatile noexcept {
    m_value = other.m_value;
  }

  FLARE_DEFAULTED_FUNCTION
  OptionalRef& operator=(OptionalRef&&) = default;

  FLARE_DEFAULTED_FUNCTION
  ~OptionalRef() = default;

  FLARE_INLINE_FUNCTION
  explicit OptionalRef(T& arg_value) : m_value(&arg_value) {}

  FLARE_INLINE_FUNCTION
  explicit OptionalRef(std::nullptr_t) : m_value(nullptr) {}

  FLARE_INLINE_FUNCTION
  OptionalRef& operator=(T& arg_value) {
    m_value = &arg_value;
    return *this;
  }

  FLARE_INLINE_FUNCTION
  OptionalRef& operator=(std::nullptr_t) {
    m_value = nullptr;
    return *this;
  }

  //----------------------------------------

  FLARE_INLINE_FUNCTION
  OptionalRef<std::add_volatile_t<T>> as_volatile() volatile noexcept {
    return OptionalRef<std::add_volatile_t<T>>(*(*this));
  }

  FLARE_INLINE_FUNCTION
  OptionalRef<std::add_volatile_t<std::add_const_t<T>>> as_volatile() const
      volatile noexcept {
    return OptionalRef<std::add_volatile_t<std::add_const_t<T>>>(*(*this));
  }

  //----------------------------------------

  FLARE_INLINE_FUNCTION
  T& operator*() & {
    FLARE_EXPECTS(this->has_value());
    return *m_value;
  }

  FLARE_INLINE_FUNCTION
  T const& operator*() const& {
    FLARE_EXPECTS(this->has_value());
    return *m_value;
  }

  FLARE_INLINE_FUNCTION
  T volatile& operator*() volatile& {
    FLARE_EXPECTS(this->has_value());
    return *m_value;
  }

  FLARE_INLINE_FUNCTION
  T const volatile& operator*() const volatile& {
    FLARE_EXPECTS(this->has_value());
    return *m_value;
  }

  FLARE_INLINE_FUNCTION
  T&& operator*() && {
    FLARE_EXPECTS(this->has_value());
    return std::move(*m_value);
  }

  FLARE_INLINE_FUNCTION
  T* operator->() {
    FLARE_EXPECTS(this->has_value());
    return m_value;
  }

  FLARE_INLINE_FUNCTION
  T const* operator->() const {
    FLARE_EXPECTS(this->has_value());
    return m_value;
  }

  FLARE_INLINE_FUNCTION
  T volatile* operator->() volatile {
    FLARE_EXPECTS(this->has_value());
    return m_value;
  }

  FLARE_INLINE_FUNCTION
  T const volatile* operator->() const volatile {
    FLARE_EXPECTS(this->has_value());
    return m_value;
  }

  FLARE_INLINE_FUNCTION
  T* get() { return m_value; }

  FLARE_INLINE_FUNCTION
  T const* get() const { return m_value; }

  FLARE_INLINE_FUNCTION
  T volatile* get() volatile { return m_value; }

  FLARE_INLINE_FUNCTION
  T const volatile* get() const volatile { return m_value; }

  //----------------------------------------

  FLARE_INLINE_FUNCTION
  operator bool() { return m_value != nullptr; }

  FLARE_INLINE_FUNCTION
  operator bool() const { return m_value != nullptr; }

  FLARE_INLINE_FUNCTION
  operator bool() volatile { return m_value != nullptr; }

  FLARE_INLINE_FUNCTION
  operator bool() const volatile { return m_value != nullptr; }

  FLARE_INLINE_FUNCTION
  bool has_value() { return m_value != nullptr; }

  FLARE_INLINE_FUNCTION
  bool has_value() const { return m_value != nullptr; }

  FLARE_INLINE_FUNCTION
  bool has_value() volatile { return m_value != nullptr; }

  FLARE_INLINE_FUNCTION
  bool has_value() const volatile { return m_value != nullptr; }
};

}  // end namespace detail
}  // end namespace flare

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif  // FLARE_CORE_COMMON_OPTIONAL_REF_H_
