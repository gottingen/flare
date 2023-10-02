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

#ifndef FLARE_CORE_ARRAY_H_
#define FLARE_CORE_ARRAY_H_

#include <flare/core/defines.h>
#include <flare/core/common/error.h>
#include <flare/core/common/string_manipulation.h>

#include <type_traits>
#include <algorithm>
#include <utility>
#include <limits>
#include <cstddef>

namespace flare {

#ifdef FLARE_ENABLE_DEBUG_BOUNDS_CHECK
    namespace detail {
    template <typename Integral, bool Signed = std::is_signed<Integral>::value>
    struct ArrayBoundsCheck;

    template <typename Integral>
    struct ArrayBoundsCheck<Integral, true> {
      FLARE_INLINE_FUNCTION
      constexpr ArrayBoundsCheck(Integral i, size_t N) {
        if (i < 0) {
          char err[128] = "flare::Array: index ";
          to_chars_i(err + strlen(err), err + 128, i);
          strcat(err, " < 0");
          flare::abort(err);
        }
        ArrayBoundsCheck<Integral, false>(i, N);
      }
    };

    template <typename Integral>
    struct ArrayBoundsCheck<Integral, false> {
      FLARE_INLINE_FUNCTION
      constexpr ArrayBoundsCheck(Integral i, size_t N) {
        if (size_t(i) >= N) {
          char err[128] = "flare::Array: index ";
          to_chars_i(err + strlen(err), err + 128, i);
          strcat(err, " >= ");
          to_chars_i(err + strlen(err), err + 128, N);
          flare::abort(err);
        }
      }
    };
    }  // end namespace detail

#define FLARE_ARRAY_BOUNDS_CHECK(i, N) \
  flare::detail::ArrayBoundsCheck<decltype(i)>(i, N)

#else  // !defined( FLARE_ENABLE_DEBUG_BOUNDS_CHECK )

#define FLARE_ARRAY_BOUNDS_CHECK(i, N) (void)0

#endif  // !defined( FLARE_ENABLE_DEBUG_BOUNDS_CHECK )

    /**\brief  Derived from the C++17 'std::array'.
     *         Dropping the iterator interface.
     */
    template<class T = void, size_t N = FLARE_INVALID_INDEX, class Proxy = void>
    struct Array {
    public:
        /**
         * The elements of this C array shall not be accessed directly. The data
         * member has to be declared public to enable aggregate initialization as for
         * std::array. We mark it as private in the documentation.
         * @private
         */
        T m_internal_implementation_private_member_data[N];

    public:
        using reference = T &;
        using const_reference = std::add_const_t<T> &;
        using size_type = size_t;
        using difference_type = ptrdiff_t;
        using value_type = T;
        using pointer = T *;
        using const_pointer = std::add_const_t<T> *;

        FLARE_INLINE_FUNCTION static constexpr size_type size() { return N; }

        FLARE_INLINE_FUNCTION static constexpr bool empty() { return false; }

        FLARE_INLINE_FUNCTION constexpr size_type max_size() const { return N; }

        template<typename iType>
        FLARE_INLINE_FUNCTION constexpr reference operator[](const iType &i) {
            static_assert(
                    (std::is_integral<iType>::value || std::is_enum<iType>::value),
                    "Must be integral argument");
            FLARE_ARRAY_BOUNDS_CHECK(i, N);
            return m_internal_implementation_private_member_data[i];
        }

        template<typename iType>
        FLARE_INLINE_FUNCTION constexpr const_reference operator[](
                const iType &i) const {
            static_assert(
                    (std::is_integral<iType>::value || std::is_enum<iType>::value),
                    "Must be integral argument");
            FLARE_ARRAY_BOUNDS_CHECK(i, N);
            return m_internal_implementation_private_member_data[i];
        }

        FLARE_INLINE_FUNCTION constexpr pointer data() {
            return &m_internal_implementation_private_member_data[0];
        }

        FLARE_INLINE_FUNCTION constexpr const_pointer data() const {
            return &m_internal_implementation_private_member_data[0];
        }
    };

    template<class T, class Proxy>
    struct Array<T, 0, Proxy> {
    public:
        using reference = T &;
        using const_reference = std::add_const_t<T> &;
        using size_type = size_t;
        using difference_type = ptrdiff_t;
        using value_type = T;
        using pointer = T *;
        using const_pointer = std::add_const_t<T> *;

        FLARE_INLINE_FUNCTION static constexpr size_type size() { return 0; }

        FLARE_INLINE_FUNCTION static constexpr bool empty() { return true; }

        FLARE_INLINE_FUNCTION constexpr size_type max_size() const { return 0; }

        template<typename iType>
        FLARE_INLINE_FUNCTION reference operator[](const iType &) {
            static_assert(
                    (std::is_integral<iType>::value || std::is_enum<iType>::value),
                    "Must be integer argument");
            flare::abort("Unreachable code");
            return *reinterpret_cast<pointer>(-1);
        }

        template<typename iType>
        FLARE_INLINE_FUNCTION const_reference operator[](const iType &) const {
            static_assert(
                    (std::is_integral<iType>::value || std::is_enum<iType>::value),
                    "Must be integer argument");
            flare::abort("Unreachable code");
            return *reinterpret_cast<const_pointer>(-1);
        }

        FLARE_INLINE_FUNCTION pointer data() { return pointer(0); }

        FLARE_INLINE_FUNCTION const_pointer data() const { return const_pointer(0); }

        FLARE_DEFAULTED_FUNCTION ~Array() = default;

        FLARE_DEFAULTED_FUNCTION Array() = default;

        FLARE_DEFAULTED_FUNCTION Array(const Array &) = default;

        FLARE_DEFAULTED_FUNCTION Array &operator=(const Array &) = default;

        // Some supported compilers are not sufficiently C++11 compliant
        // for default move constructor and move assignment operator.
        // Array( Array && ) = default ;
        // Array & operator = ( Array && ) = default ;
    };

    template<>
    struct Array<void, FLARE_INVALID_INDEX, void> {
        struct contiguous {
        };
        struct strided {
        };
    };

    template<class T>
    struct Array<T, FLARE_INVALID_INDEX, Array<>::contiguous> {
    private:
        T *m_elem;
        size_t m_size;

    public:
        using reference = T &;
        using const_reference = std::add_const_t<T> &;
        using size_type = size_t;
        using difference_type = ptrdiff_t;
        using value_type = T;
        using pointer = T *;
        using const_pointer = std::add_const_t<T> *;

        FLARE_INLINE_FUNCTION constexpr size_type size() const { return m_size; }

        FLARE_INLINE_FUNCTION constexpr bool empty() const { return 0 != m_size; }

        FLARE_INLINE_FUNCTION constexpr size_type max_size() const { return m_size; }

        template<typename iType>
        FLARE_INLINE_FUNCTION reference operator[](const iType &i) {
            static_assert(
                    (std::is_integral<iType>::value || std::is_enum<iType>::value),
                    "Must be integral argument");
            FLARE_ARRAY_BOUNDS_CHECK(i, m_size);
            return m_elem[i];
        }

        template<typename iType>
        FLARE_INLINE_FUNCTION const_reference operator[](const iType &i) const {
            static_assert(
                    (std::is_integral<iType>::value || std::is_enum<iType>::value),
                    "Must be integral argument");
            FLARE_ARRAY_BOUNDS_CHECK(i, m_size);
            return m_elem[i];
        }

        FLARE_INLINE_FUNCTION pointer data() { return m_elem; }

        FLARE_INLINE_FUNCTION const_pointer data() const { return m_elem; }

        FLARE_DEFAULTED_FUNCTION ~Array() = default;

        FLARE_INLINE_FUNCTION_DELETED Array() = delete;

        FLARE_INLINE_FUNCTION_DELETED Array(const Array &rhs) = delete;

        // Some supported compilers are not sufficiently C++11 compliant
        // for default move constructor and move assignment operator.
        // Array( Array && rhs ) = default ;
        // Array & operator = ( Array && rhs ) = delete ;

        FLARE_INLINE_FUNCTION
        Array &operator=(const Array &rhs) {
            const size_t n = std::min(m_size, rhs.size());
            for (size_t i = 0; i < n; ++i) m_elem[i] = rhs[i];
            return *this;
        }

        template<size_t N, class P>
        FLARE_INLINE_FUNCTION Array &operator=(const Array<T, N, P> &rhs) {
            const size_t n = std::min(m_size, rhs.size());
            for (size_t i = 0; i < n; ++i) m_elem[i] = rhs[i];
            return *this;
        }

        FLARE_INLINE_FUNCTION constexpr Array(pointer arg_ptr, size_type arg_size,
                                              size_type = 0)
                : m_elem(arg_ptr), m_size(arg_size) {}
    };

    template<class T>
    struct Array<T, FLARE_INVALID_INDEX, Array<>::strided> {
    private:
        T *m_elem;
        size_t m_size;
        size_t m_stride;

    public:
        using reference = T &;
        using const_reference = std::add_const_t<T> &;
        using size_type = size_t;
        using difference_type = ptrdiff_t;
        using value_type = T;
        using pointer = T *;
        using const_pointer = std::add_const_t<T> *;

        FLARE_INLINE_FUNCTION constexpr size_type size() const { return m_size; }

        FLARE_INLINE_FUNCTION constexpr bool empty() const { return 0 != m_size; }

        FLARE_INLINE_FUNCTION constexpr size_type max_size() const { return m_size; }

        template<typename iType>
        FLARE_INLINE_FUNCTION reference operator[](const iType &i) {
            static_assert(
                    (std::is_integral<iType>::value || std::is_enum<iType>::value),
                    "Must be integral argument");
            FLARE_ARRAY_BOUNDS_CHECK(i, m_size);
            return m_elem[i * m_stride];
        }

        template<typename iType>
        FLARE_INLINE_FUNCTION const_reference operator[](const iType &i) const {
            static_assert(
                    (std::is_integral<iType>::value || std::is_enum<iType>::value),
                    "Must be integral argument");
            FLARE_ARRAY_BOUNDS_CHECK(i, m_size);
            return m_elem[i * m_stride];
        }

        FLARE_INLINE_FUNCTION pointer data() { return m_elem; }

        FLARE_INLINE_FUNCTION const_pointer data() const { return m_elem; }

        FLARE_DEFAULTED_FUNCTION ~Array() = default;

        FLARE_INLINE_FUNCTION_DELETED Array() = delete;

        FLARE_INLINE_FUNCTION_DELETED Array(const Array &) = delete;

        // Some supported compilers are not sufficiently C++11 compliant
        // for default move constructor and move assignment operator.
        // Array( Array && rhs ) = default ;
        // Array & operator = ( Array && rhs ) = delete ;

        FLARE_INLINE_FUNCTION
        Array &operator=(const Array &rhs) {
            const size_t n = std::min(m_size, rhs.size());
            for (size_t i = 0; i < n; ++i) m_elem[i] = rhs[i];
            return *this;
        }

        template<size_t N, class P>
        FLARE_INLINE_FUNCTION Array &operator=(const Array<T, N, P> &rhs) {
            const size_t n = std::min(m_size, rhs.size());
            for (size_t i = 0; i < n; ++i) m_elem[i] = rhs[i];
            return *this;
        }

        FLARE_INLINE_FUNCTION constexpr Array(pointer arg_ptr, size_type arg_size,
                                              size_type arg_stride)
                : m_elem(arg_ptr), m_size(arg_size), m_stride(arg_stride) {}
    };

}  // namespace flare

template<class T, std::size_t N>
struct std::tuple_size<flare::Array<T, N>>
        : std::integral_constant<std::size_t, N> {
};

template<std::size_t I, class T, std::size_t N>
struct std::tuple_element<I, flare::Array<T, N>> {
    using type = T;
};

namespace flare {

    template<std::size_t I, class T, std::size_t N>
    FLARE_FUNCTION constexpr T &get(Array<T, N> &a) noexcept {
        return a[I];
    }

    template<std::size_t I, class T, std::size_t N>
    FLARE_FUNCTION constexpr T const &get(Array<T, N> const &a) noexcept {
        return a[I];
    }

    template<std::size_t I, class T, std::size_t N>
    FLARE_FUNCTION constexpr T &&get(Array<T, N> &&a) noexcept {
        return std::move(a[I]);
    }

    template<std::size_t I, class T, std::size_t N>
    FLARE_FUNCTION constexpr T const &&get(Array<T, N> const &&a) noexcept {
        return std::move(a[I]);
    }

}  // namespace flare

#endif  // FLARE_CORE_ARRAY_H_
