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

#ifndef FLARE_CORE_COMMON_HALF_FLOATING_POINT_WRAPPER_H_
#define FLARE_CORE_COMMON_HALF_FLOATING_POINT_WRAPPER_H_

#include <flare/core/defines.h>

#include <type_traits>
#include <iosfwd>  // istream & ostream for extraction and insertion ops
#include <string>

namespace flare::experimental::detail {
    /// @brief templated struct for determining if half_t is an alias to float.
    /// @tparam T The type to specialize on.
    template<class T>
    struct is_float16 : std::false_type {
    };

    /// @brief templated struct for determining if bhalf_t is an alias to float.
    /// @tparam T The type to specialize on.
    template<class T>
    struct is_bfloat16 : std::false_type {
    };
}  // namespace flare::experimental::detail

#ifdef FLARE_IMPL_HALF_TYPE_DEFINED

#if defined(__CUDA_ARCH__)
#define FLARE_HALF_IS_FULL_TYPE_ON_ARCH
#endif

/************************* BEGIN forward declarations *************************/
namespace flare::experimental::detail {
    template<class FloatType>
    class floating_point_wrapper;
}  // namespace flare::experimental::detail

namespace flare::experimental {

    // Declare half_t (binary16)
    using half_t = flare::experimental::detail::floating_point_wrapper<
            flare::detail::half_impl_t::type>;
    namespace detail {
        template<>
        struct is_float16<half_t> : std::true_type {
        };
    }  // namespace detail
    FLARE_INLINE_FUNCTION
    half_t cast_to_half(float val);

    FLARE_INLINE_FUNCTION
    half_t cast_to_half(bool val);

    FLARE_INLINE_FUNCTION
    half_t cast_to_half(double val);

    FLARE_INLINE_FUNCTION
    half_t cast_to_half(short val);

    FLARE_INLINE_FUNCTION
    half_t cast_to_half(int val);

    FLARE_INLINE_FUNCTION
    half_t cast_to_half(long val);

    FLARE_INLINE_FUNCTION
    half_t cast_to_half(long long val);

    FLARE_INLINE_FUNCTION
    half_t cast_to_half(unsigned short val);

    FLARE_INLINE_FUNCTION
    half_t cast_to_half(unsigned int val);

    FLARE_INLINE_FUNCTION
    half_t cast_to_half(unsigned long val);

    FLARE_INLINE_FUNCTION
    half_t cast_to_half(unsigned long long val);

    FLARE_INLINE_FUNCTION
    half_t cast_to_half(half_t);

    template<class T>
    FLARE_INLINE_FUNCTION std::enable_if_t<std::is_same<T, float>::value, T>
    cast_from_half(half_t);

    template<class T>
    FLARE_INLINE_FUNCTION std::enable_if_t<std::is_same<T, bool>::value, T>
    cast_from_half(half_t);

    template<class T>
    FLARE_INLINE_FUNCTION std::enable_if_t<std::is_same<T, double>::value, T>
    cast_from_half(half_t);

    template<class T>
    FLARE_INLINE_FUNCTION std::enable_if_t<std::is_same<T, short>::value, T>
    cast_from_half(half_t);

    template<class T>
    FLARE_INLINE_FUNCTION std::enable_if_t<std::is_same<T, int>::value, T>
    cast_from_half(half_t);

    template<class T>
    FLARE_INLINE_FUNCTION std::enable_if_t<std::is_same<T, long>::value, T>
    cast_from_half(half_t);

    template<class T>
    FLARE_INLINE_FUNCTION std::enable_if_t<std::is_same<T, long long>::value, T>
    cast_from_half(half_t);

    template<class T>
    FLARE_INLINE_FUNCTION
    std::enable_if_t<std::is_same<T, unsigned short>::value, T>
    cast_from_half(half_t);

    template<class T>
    FLARE_INLINE_FUNCTION std::enable_if_t<std::is_same<T, unsigned int>::value, T>
    cast_from_half(half_t);

    template<class T>
    FLARE_INLINE_FUNCTION
    std::enable_if_t<std::is_same<T, unsigned long>::value, T>
    cast_from_half(half_t);

    template<class T>
    FLARE_INLINE_FUNCTION
    std::enable_if_t<std::is_same<T, unsigned long long>::value, T>
    cast_from_half(half_t);

// declare bhalf_t
#ifdef FLARE_IMPL_BHALF_TYPE_DEFINED
    using bhalf_t = flare::experimental::detail::floating_point_wrapper<
            flare::detail::bhalf_impl_t::type>;
    namespace detail {
        template<>
        struct is_bfloat16<bhalf_t> : std::true_type {
        };
    }  // namespace detail
    FLARE_INLINE_FUNCTION
    bhalf_t cast_to_bhalf(float val);

    FLARE_INLINE_FUNCTION
    bhalf_t cast_to_bhalf(bool val);

    FLARE_INLINE_FUNCTION
    bhalf_t cast_to_bhalf(double val);

    FLARE_INLINE_FUNCTION
    bhalf_t cast_to_bhalf(short val);

    FLARE_INLINE_FUNCTION
    bhalf_t cast_to_bhalf(int val);

    FLARE_INLINE_FUNCTION
    bhalf_t cast_to_bhalf(long val);

    FLARE_INLINE_FUNCTION
    bhalf_t cast_to_bhalf(long long val);

    FLARE_INLINE_FUNCTION
    bhalf_t cast_to_bhalf(unsigned short val);

    FLARE_INLINE_FUNCTION
    bhalf_t cast_to_bhalf(unsigned int val);

    FLARE_INLINE_FUNCTION
    bhalf_t cast_to_bhalf(unsigned long val);

    FLARE_INLINE_FUNCTION
    bhalf_t cast_to_bhalf(unsigned long long val);

    FLARE_INLINE_FUNCTION
    bhalf_t cast_to_bhalf(bhalf_t val);

    template<class T>
    FLARE_INLINE_FUNCTION std::enable_if_t<std::is_same<T, float>::value, T>
    cast_from_bhalf(bhalf_t);

    template<class T>
    FLARE_INLINE_FUNCTION std::enable_if_t<std::is_same<T, bool>::value, T>
    cast_from_bhalf(bhalf_t);

    template<class T>
    FLARE_INLINE_FUNCTION std::enable_if_t<std::is_same<T, double>::value, T>
    cast_from_bhalf(bhalf_t);

    template<class T>
    FLARE_INLINE_FUNCTION std::enable_if_t<std::is_same<T, short>::value, T>
    cast_from_bhalf(bhalf_t);

    template<class T>
    FLARE_INLINE_FUNCTION std::enable_if_t<std::is_same<T, int>::value, T>
    cast_from_bhalf(bhalf_t);

    template<class T>
    FLARE_INLINE_FUNCTION std::enable_if_t<std::is_same<T, long>::value, T>
    cast_from_bhalf(bhalf_t);

    template<class T>
    FLARE_INLINE_FUNCTION std::enable_if_t<std::is_same<T, long long>::value, T>
    cast_from_bhalf(bhalf_t);

    template<class T>
    FLARE_INLINE_FUNCTION
    std::enable_if_t<std::is_same<T, unsigned short>::value, T>
    cast_from_bhalf(bhalf_t);

    template<class T>
    FLARE_INLINE_FUNCTION std::enable_if_t<std::is_same<T, unsigned int>::value, T>
    cast_from_bhalf(bhalf_t);

    template<class T>
    FLARE_INLINE_FUNCTION
    std::enable_if_t<std::is_same<T, unsigned long>::value, T>
    cast_from_bhalf(bhalf_t);

    template<class T>
    FLARE_INLINE_FUNCTION
    std::enable_if_t<std::is_same<T, unsigned long long>::value, T>
    cast_from_bhalf(bhalf_t);

#endif  // FLARE_IMPL_BHALF_TYPE_DEFINED

    template<class T>
    static FLARE_INLINE_FUNCTION flare::experimental::half_t cast_to_wrapper(
            T x, const volatile flare::detail::half_impl_t::type &);

#ifdef FLARE_IMPL_BHALF_TYPE_DEFINED

    template<class T>
    static FLARE_INLINE_FUNCTION flare::experimental::bhalf_t cast_to_wrapper(
            T x, const volatile flare::detail::bhalf_impl_t::type &);

#endif  // FLARE_IMPL_BHALF_TYPE_DEFINED

    template<class T>
    static FLARE_INLINE_FUNCTION T
    cast_from_wrapper(const flare::experimental::half_t &x);

#ifdef FLARE_IMPL_BHALF_TYPE_DEFINED

    template<class T>
    static FLARE_INLINE_FUNCTION T
    cast_from_wrapper(const flare::experimental::bhalf_t &x);

#endif  // FLARE_IMPL_BHALF_TYPE_DEFINED
/************************** END forward declarations **************************/

    namespace detail {
        template<class FloatType>
        class alignas(FloatType) floating_point_wrapper {
        public:
            using impl_type = FloatType;

        private:
            impl_type val;
            using fixed_width_integer_type = std::conditional_t<
                    sizeof(impl_type) == 2, uint16_t,
                    std::conditional_t<
                            sizeof(impl_type) == 4, uint32_t,
                            std::conditional_t<sizeof(impl_type) == 8, uint64_t, void>>>;
            static_assert(!std::is_void<fixed_width_integer_type>::value,
                          "Invalid impl_type");

        public:
            // In-class initialization and defaulted default constructors not used
            // since Cuda supports half precision initialization via the below constructor
            FLARE_FUNCTION
            floating_point_wrapper() : val(0.0F) {}

// Copy constructors
// Getting "C2580: multiple versions of a defaulted special
// member function are not allowed" with VS 16.11.3 and CUDA 11.4.2
#if defined(_WIN32) && defined(FLARE_ON_CUDA_DEVICE)
            FLARE_FUNCTION
            floating_point_wrapper(const floating_point_wrapper& rhs) : val(rhs.val) {}

            FLARE_FUNCTION
            floating_point_wrapper& operator=(const floating_point_wrapper& rhs) {
              val = rhs.val;
              return *this;
            }
#else
            FLARE_DEFAULTED_FUNCTION
            floating_point_wrapper(const floating_point_wrapper &) noexcept = default;

            FLARE_DEFAULTED_FUNCTION
            floating_point_wrapper &operator=(const floating_point_wrapper &) noexcept =
            default;

#endif

            FLARE_INLINE_FUNCTION
            floating_point_wrapper(const volatile floating_point_wrapper &rhs) {
#if defined(FLARE_HALF_IS_FULL_TYPE_ON_ARCH)
                val = rhs.val;
#else
                const volatile fixed_width_integer_type *rv_ptr =
                        reinterpret_cast<const volatile fixed_width_integer_type *>(&rhs.val);
                const fixed_width_integer_type rv_val = *rv_ptr;
                val = reinterpret_cast<const impl_type &>(rv_val);
#endif  // FLARE_HALF_IS_FULL_TYPE_ON_ARCH
            }

            // Don't support implicit conversion back to impl_type.
            // impl_type is a storage only type on host.
            FLARE_FUNCTION
            explicit operator impl_type() const { return val; }

            FLARE_FUNCTION
            explicit operator float() const { return cast_from_wrapper<float>(*this); }

            FLARE_FUNCTION
            explicit operator bool() const { return cast_from_wrapper<bool>(*this); }

            FLARE_FUNCTION
            explicit operator double() const { return cast_from_wrapper<double>(*this); }

            FLARE_FUNCTION
            explicit operator short() const { return cast_from_wrapper<short>(*this); }

            FLARE_FUNCTION
            explicit operator int() const { return cast_from_wrapper<int>(*this); }

            FLARE_FUNCTION
            explicit operator long() const { return cast_from_wrapper<long>(*this); }

            FLARE_FUNCTION
            explicit operator long long() const {
                return cast_from_wrapper<long long>(*this);
            }

            FLARE_FUNCTION
            explicit operator unsigned short() const {
                return cast_from_wrapper<unsigned short>(*this);
            }

            FLARE_FUNCTION
            explicit operator unsigned int() const {
                return cast_from_wrapper<unsigned int>(*this);
            }

            FLARE_FUNCTION
            explicit operator unsigned long() const {
                return cast_from_wrapper<unsigned long>(*this);
            }

            FLARE_FUNCTION
            explicit operator unsigned long long() const {
                return cast_from_wrapper<unsigned long long>(*this);
            }

            /**
             * Conversion constructors.
             *
             * Support implicit conversions from impl_type, float, double ->
             * floating_point_wrapper. Mixed precision expressions require upcasting which
             * is done in the
             * "// Binary Arithmetic" operator overloads below.
             *
             * Support implicit conversions from integral types -> floating_point_wrapper.
             * Expressions involving floating_point_wrapper with integral types require
             * downcasting the integral types to floating_point_wrapper. Existing operator
             * overloads can handle this with the addition of the below implicit
             * conversion constructors.
             */
            FLARE_FUNCTION
            constexpr floating_point_wrapper(impl_type rhs) : val(rhs) {}

            FLARE_FUNCTION
            floating_point_wrapper(float rhs) : val(cast_to_wrapper(rhs, val).val) {}

            FLARE_FUNCTION
            floating_point_wrapper(double rhs) : val(cast_to_wrapper(rhs, val).val) {}

            FLARE_FUNCTION
            explicit floating_point_wrapper(bool rhs)
                    : val(cast_to_wrapper(rhs, val).val) {}

            FLARE_FUNCTION
            floating_point_wrapper(short rhs) : val(cast_to_wrapper(rhs, val).val) {}

            FLARE_FUNCTION
            floating_point_wrapper(int rhs) : val(cast_to_wrapper(rhs, val).val) {}

            FLARE_FUNCTION
            floating_point_wrapper(long rhs) : val(cast_to_wrapper(rhs, val).val) {}

            FLARE_FUNCTION
            floating_point_wrapper(long long rhs) : val(cast_to_wrapper(rhs, val).val) {}

            FLARE_FUNCTION
            floating_point_wrapper(unsigned short rhs)
                    : val(cast_to_wrapper(rhs, val).val) {}

            FLARE_FUNCTION
            floating_point_wrapper(unsigned int rhs)
                    : val(cast_to_wrapper(rhs, val).val) {}

            FLARE_FUNCTION
            floating_point_wrapper(unsigned long rhs)
                    : val(cast_to_wrapper(rhs, val).val) {}

            FLARE_FUNCTION
            floating_point_wrapper(unsigned long long rhs)
                    : val(cast_to_wrapper(rhs, val).val) {}

            // Unary operators
            FLARE_FUNCTION
            floating_point_wrapper operator+() const {
                floating_point_wrapper tmp = *this;
#ifdef FLARE_HALF_IS_FULL_TYPE_ON_ARCH
                tmp.val = +tmp.val;
#else
                tmp.val = cast_to_wrapper(+cast_from_wrapper<float>(tmp), val).val;
#endif
                return tmp;
            }

            FLARE_FUNCTION
            floating_point_wrapper operator-() const {
                floating_point_wrapper tmp = *this;
#ifdef FLARE_HALF_IS_FULL_TYPE_ON_ARCH
                tmp.val = -tmp.val;
#else
                tmp.val = cast_to_wrapper(-cast_from_wrapper<float>(tmp), val).val;
#endif
                return tmp;
            }

            // Prefix operators
            FLARE_FUNCTION
            floating_point_wrapper &operator++() {
#ifdef FLARE_HALF_IS_FULL_TYPE_ON_ARCH
                val = val + impl_type(1.0F);  // cuda has no operator++ for __nv_bfloat
#else
                float tmp = cast_from_wrapper<float>(*this);
                ++tmp;
                val = cast_to_wrapper(tmp, val).val;
#endif
                return *this;
            }

            FLARE_FUNCTION
            floating_point_wrapper &operator--() {
#ifdef FLARE_HALF_IS_FULL_TYPE_ON_ARCH
                val = val - impl_type(1.0F);  // cuda has no operator-- for __nv_bfloat
#else
                float tmp = cast_from_wrapper<float>(*this);
                --tmp;
                val = cast_to_wrapper(tmp, val).val;
#endif
                return *this;
            }

            // Postfix operators
            FLARE_FUNCTION
            floating_point_wrapper operator++(int) {
                floating_point_wrapper tmp = *this;
                operator++();
                return tmp;
            }

            FLARE_FUNCTION
            floating_point_wrapper operator--(int) {
                floating_point_wrapper tmp = *this;
                operator--();
                return tmp;
            }

            // Binary operators
            FLARE_FUNCTION
            floating_point_wrapper &operator=(impl_type rhs) {
                val = rhs;
                return *this;
            }

            template<class T>
            FLARE_FUNCTION floating_point_wrapper &operator=(T rhs) {
                val = cast_to_wrapper(rhs, val).val;
                return *this;
            }

            template<class T>
            FLARE_FUNCTION void operator=(T rhs) volatile {
                impl_type new_val = cast_to_wrapper(rhs, val).val;
                volatile fixed_width_integer_type *val_ptr =
                        reinterpret_cast<volatile fixed_width_integer_type *>(
                                const_cast<impl_type *>(&val));
                *val_ptr = reinterpret_cast<fixed_width_integer_type &>(new_val);
            }

            // Compound operators
            FLARE_FUNCTION
            floating_point_wrapper &operator+=(floating_point_wrapper rhs) {
#ifdef FLARE_HALF_IS_FULL_TYPE_ON_ARCH
                val = val + rhs.val;  // cuda has no operator+= for __nv_bfloat
#else
                val = cast_to_wrapper(
                        cast_from_wrapper<float>(*this) + cast_from_wrapper<float>(rhs),
                        val)
                        .val;
#endif
                return *this;
            }

            FLARE_FUNCTION
            void operator+=(const volatile floating_point_wrapper &rhs) volatile {
                floating_point_wrapper tmp_rhs = rhs;
                floating_point_wrapper tmp_lhs = *this;

                tmp_lhs += tmp_rhs;
                *this = tmp_lhs;
            }

            // Compound operators: upcast overloads for +=
            template<class T>
            FLARE_FUNCTION friend std::enable_if_t<
                    std::is_same<T, float>::value || std::is_same<T, double>::value, T>
            operator+=(T &lhs, floating_point_wrapper rhs) {
                lhs += static_cast<T>(rhs);
                return lhs;
            }

            FLARE_FUNCTION
            floating_point_wrapper &operator+=(float rhs) {
                float result = static_cast<float>(val) + rhs;
                val = static_cast<impl_type>(result);
                return *this;
            }

            FLARE_FUNCTION
            floating_point_wrapper &operator+=(double rhs) {
                double result = static_cast<double>(val) + rhs;
                val = static_cast<impl_type>(result);
                return *this;
            }

            FLARE_FUNCTION
            floating_point_wrapper &operator-=(floating_point_wrapper rhs) {
#ifdef FLARE_HALF_IS_FULL_TYPE_ON_ARCH
                val = val - rhs.val;  // cuda has no operator-= for __nv_bfloat
#else
                val = cast_to_wrapper(
                        cast_from_wrapper<float>(*this) - cast_from_wrapper<float>(rhs),
                        val)
                        .val;
#endif
                return *this;
            }

            FLARE_FUNCTION
            void operator-=(const volatile floating_point_wrapper &rhs) volatile {
                floating_point_wrapper tmp_rhs = rhs;
                floating_point_wrapper tmp_lhs = *this;

                tmp_lhs -= tmp_rhs;
                *this = tmp_lhs;
            }

            // Compund operators: upcast overloads for -=
            template<class T>
            FLARE_FUNCTION friend std::enable_if_t<
                    std::is_same<T, float>::value || std::is_same<T, double>::value, T>
            operator-=(T &lhs, floating_point_wrapper rhs) {
                lhs -= static_cast<T>(rhs);
                return lhs;
            }

            FLARE_FUNCTION
            floating_point_wrapper &operator-=(float rhs) {
                float result = static_cast<float>(val) - rhs;
                val = static_cast<impl_type>(result);
                return *this;
            }

            FLARE_FUNCTION
            floating_point_wrapper &operator-=(double rhs) {
                double result = static_cast<double>(val) - rhs;
                val = static_cast<impl_type>(result);
                return *this;
            }

            FLARE_FUNCTION
            floating_point_wrapper &operator*=(floating_point_wrapper rhs) {
#ifdef FLARE_HALF_IS_FULL_TYPE_ON_ARCH
                val = val * rhs.val;  // cuda has no operator*= for __nv_bfloat
#else
                val = cast_to_wrapper(
                        cast_from_wrapper<float>(*this) * cast_from_wrapper<float>(rhs),
                        val)
                        .val;
#endif
                return *this;
            }

            FLARE_FUNCTION
            void operator*=(const volatile floating_point_wrapper &rhs) volatile {
                floating_point_wrapper tmp_rhs = rhs;
                floating_point_wrapper tmp_lhs = *this;

                tmp_lhs *= tmp_rhs;
                *this = tmp_lhs;
            }

            // Compund operators: upcast overloads for *=
            template<class T>
            FLARE_FUNCTION friend std::enable_if_t<
                    std::is_same<T, float>::value || std::is_same<T, double>::value, T>
            operator*=(T &lhs, floating_point_wrapper rhs) {
                lhs *= static_cast<T>(rhs);
                return lhs;
            }

            FLARE_FUNCTION
            floating_point_wrapper &operator*=(float rhs) {
                float result = static_cast<float>(val) * rhs;
                val = static_cast<impl_type>(result);
                return *this;
            }

            FLARE_FUNCTION
            floating_point_wrapper &operator*=(double rhs) {
                double result = static_cast<double>(val) * rhs;
                val = static_cast<impl_type>(result);
                return *this;
            }

            FLARE_FUNCTION
            floating_point_wrapper &operator/=(floating_point_wrapper rhs) {
#ifdef FLARE_HALF_IS_FULL_TYPE_ON_ARCH
                val = val / rhs.val;  // cuda has no operator/= for __nv_bfloat
#else
                val = cast_to_wrapper(
                        cast_from_wrapper<float>(*this) / cast_from_wrapper<float>(rhs),
                        val)
                        .val;
#endif
                return *this;
            }

            FLARE_FUNCTION
            void operator/=(const volatile floating_point_wrapper &rhs) volatile {
                floating_point_wrapper tmp_rhs = rhs;
                floating_point_wrapper tmp_lhs = *this;

                tmp_lhs /= tmp_rhs;
                *this = tmp_lhs;
            }

            // Compund operators: upcast overloads for /=
            template<class T>
            FLARE_FUNCTION friend std::enable_if_t<
                    std::is_same<T, float>::value || std::is_same<T, double>::value, T>
            operator/=(T &lhs, floating_point_wrapper rhs) {
                lhs /= static_cast<T>(rhs);
                return lhs;
            }

            FLARE_FUNCTION
            floating_point_wrapper &operator/=(float rhs) {
                float result = static_cast<float>(val) / rhs;
                val = static_cast<impl_type>(result);
                return *this;
            }

            FLARE_FUNCTION
            floating_point_wrapper &operator/=(double rhs) {
                double result = static_cast<double>(val) / rhs;
                val = static_cast<impl_type>(result);
                return *this;
            }

            // Binary Arithmetic
            FLARE_FUNCTION
            friend floating_point_wrapper operator+(floating_point_wrapper lhs,
                                                    floating_point_wrapper rhs) {
#ifdef FLARE_HALF_IS_FULL_TYPE_ON_ARCH
                lhs += rhs;
#else
                lhs.val = cast_to_wrapper(
                        cast_from_wrapper<float>(lhs) + cast_from_wrapper<float>(rhs),
                        lhs.val)
                        .val;
#endif
                return lhs;
            }

            // Binary Arithmetic upcast operators for +
            template<class T>
            FLARE_FUNCTION friend std::enable_if_t<
                    std::is_same<T, float>::value || std::is_same<T, double>::value, T>
            operator+(floating_point_wrapper lhs, T rhs) {
                return T(lhs) + rhs;
            }

            template<class T>
            FLARE_FUNCTION friend std::enable_if_t<
                    std::is_same<T, float>::value || std::is_same<T, double>::value, T>
            operator+(T lhs, floating_point_wrapper rhs) {
                return lhs + T(rhs);
            }

            FLARE_FUNCTION
            friend floating_point_wrapper operator-(floating_point_wrapper lhs,
                                                    floating_point_wrapper rhs) {
#ifdef FLARE_HALF_IS_FULL_TYPE_ON_ARCH
                lhs -= rhs;
#else
                lhs.val = cast_to_wrapper(
                        cast_from_wrapper<float>(lhs) - cast_from_wrapper<float>(rhs),
                        lhs.val)
                        .val;
#endif
                return lhs;
            }

            // Binary Arithmetic upcast operators for -
            template<class T>
            FLARE_FUNCTION friend std::enable_if_t<
                    std::is_same<T, float>::value || std::is_same<T, double>::value, T>
            operator-(floating_point_wrapper lhs, T rhs) {
                return T(lhs) - rhs;
            }

            template<class T>
            FLARE_FUNCTION friend std::enable_if_t<
                    std::is_same<T, float>::value || std::is_same<T, double>::value, T>
            operator-(T lhs, floating_point_wrapper rhs) {
                return lhs - T(rhs);
            }

            FLARE_FUNCTION
            friend floating_point_wrapper operator*(floating_point_wrapper lhs,
                                                    floating_point_wrapper rhs) {
#ifdef FLARE_HALF_IS_FULL_TYPE_ON_ARCH
                lhs *= rhs;
#else
                lhs.val = cast_to_wrapper(
                        cast_from_wrapper<float>(lhs) * cast_from_wrapper<float>(rhs),
                        lhs.val)
                        .val;
#endif
                return lhs;
            }

            // Binary Arithmetic upcast operators for *
            template<class T>
            FLARE_FUNCTION friend std::enable_if_t<
                    std::is_same<T, float>::value || std::is_same<T, double>::value, T>
            operator*(floating_point_wrapper lhs, T rhs) {
                return T(lhs) * rhs;
            }

            template<class T>
            FLARE_FUNCTION friend std::enable_if_t<
                    std::is_same<T, float>::value || std::is_same<T, double>::value, T>
            operator*(T lhs, floating_point_wrapper rhs) {
                return lhs * T(rhs);
            }

            FLARE_FUNCTION
            friend floating_point_wrapper operator/(floating_point_wrapper lhs,
                                                    floating_point_wrapper rhs) {
#ifdef FLARE_HALF_IS_FULL_TYPE_ON_ARCH
                lhs /= rhs;
#else
                lhs.val = cast_to_wrapper(
                        cast_from_wrapper<float>(lhs) / cast_from_wrapper<float>(rhs),
                        lhs.val)
                        .val;
#endif
                return lhs;
            }

            // Binary Arithmetic upcast operators for /
            template<class T>
            FLARE_FUNCTION friend std::enable_if_t<
                    std::is_same<T, float>::value || std::is_same<T, double>::value, T>
            operator/(floating_point_wrapper lhs, T rhs) {
                return T(lhs) / rhs;
            }

            template<class T>
            FLARE_FUNCTION friend std::enable_if_t<
                    std::is_same<T, float>::value || std::is_same<T, double>::value, T>
            operator/(T lhs, floating_point_wrapper rhs) {
                return lhs / T(rhs);
            }

            // Logical operators
            FLARE_FUNCTION
            bool operator!() const {
#ifdef FLARE_HALF_IS_FULL_TYPE_ON_ARCH
                return static_cast<bool>(!val);
#else
                return !cast_from_wrapper<float>(*this);
#endif
            }

            // NOTE: Loses short-circuit evaluation
            FLARE_FUNCTION
            bool operator&&(floating_point_wrapper rhs) const {
#ifdef FLARE_HALF_IS_FULL_TYPE_ON_ARCH
                return static_cast<bool>(val && rhs.val);
#else
                return cast_from_wrapper<float>(*this) && cast_from_wrapper<float>(rhs);
#endif
            }

            // NOTE: Loses short-circuit evaluation
            FLARE_FUNCTION
            bool operator||(floating_point_wrapper rhs) const {
#ifdef FLARE_HALF_IS_FULL_TYPE_ON_ARCH
                return static_cast<bool>(val || rhs.val);
#else
                return cast_from_wrapper<float>(*this) || cast_from_wrapper<float>(rhs);
#endif
            }

            // Comparison operators
            FLARE_FUNCTION
            bool operator==(floating_point_wrapper rhs) const {
#ifdef FLARE_HALF_IS_FULL_TYPE_ON_ARCH
                return static_cast<bool>(val == rhs.val);
#else
                return cast_from_wrapper<float>(*this) == cast_from_wrapper<float>(rhs);
#endif
            }

            FLARE_FUNCTION
            bool operator!=(floating_point_wrapper rhs) const {
#ifdef FLARE_HALF_IS_FULL_TYPE_ON_ARCH
                return static_cast<bool>(val != rhs.val);
#else
                return cast_from_wrapper<float>(*this) != cast_from_wrapper<float>(rhs);
#endif
            }

            FLARE_FUNCTION
            bool operator<(floating_point_wrapper rhs) const {
#ifdef FLARE_HALF_IS_FULL_TYPE_ON_ARCH
                return static_cast<bool>(val < rhs.val);
#else
                return cast_from_wrapper<float>(*this) < cast_from_wrapper<float>(rhs);
#endif
            }

            FLARE_FUNCTION
            bool operator>(floating_point_wrapper rhs) const {
#ifdef FLARE_HALF_IS_FULL_TYPE_ON_ARCH
                return static_cast<bool>(val > rhs.val);
#else
                return cast_from_wrapper<float>(*this) > cast_from_wrapper<float>(rhs);
#endif
            }

            FLARE_FUNCTION
            bool operator<=(floating_point_wrapper rhs) const {
#ifdef FLARE_HALF_IS_FULL_TYPE_ON_ARCH
                return static_cast<bool>(val <= rhs.val);
#else
                return cast_from_wrapper<float>(*this) <= cast_from_wrapper<float>(rhs);
#endif
            }

            FLARE_FUNCTION
            bool operator>=(floating_point_wrapper rhs) const {
#ifdef FLARE_HALF_IS_FULL_TYPE_ON_ARCH
                return static_cast<bool>(val >= rhs.val);
#else
                return cast_from_wrapper<float>(*this) >= cast_from_wrapper<float>(rhs);
#endif
            }

            FLARE_FUNCTION
            friend bool operator==(const volatile floating_point_wrapper &lhs,
                                   const volatile floating_point_wrapper &rhs) {
                floating_point_wrapper tmp_lhs = lhs, tmp_rhs = rhs;
                return tmp_lhs == tmp_rhs;
            }

            FLARE_FUNCTION
            friend bool operator!=(const volatile floating_point_wrapper &lhs,
                                   const volatile floating_point_wrapper &rhs) {
                floating_point_wrapper tmp_lhs = lhs, tmp_rhs = rhs;
                return tmp_lhs != tmp_rhs;
            }

            FLARE_FUNCTION
            friend bool operator<(const volatile floating_point_wrapper &lhs,
                                  const volatile floating_point_wrapper &rhs) {
                floating_point_wrapper tmp_lhs = lhs, tmp_rhs = rhs;
                return tmp_lhs < tmp_rhs;
            }

            FLARE_FUNCTION
            friend bool operator>(const volatile floating_point_wrapper &lhs,
                                  const volatile floating_point_wrapper &rhs) {
                floating_point_wrapper tmp_lhs = lhs, tmp_rhs = rhs;
                return tmp_lhs > tmp_rhs;
            }

            FLARE_FUNCTION
            friend bool operator<=(const volatile floating_point_wrapper &lhs,
                                   const volatile floating_point_wrapper &rhs) {
                floating_point_wrapper tmp_lhs = lhs, tmp_rhs = rhs;
                return tmp_lhs <= tmp_rhs;
            }

            FLARE_FUNCTION
            friend bool operator>=(const volatile floating_point_wrapper &lhs,
                                   const volatile floating_point_wrapper &rhs) {
                floating_point_wrapper tmp_lhs = lhs, tmp_rhs = rhs;
                return tmp_lhs >= tmp_rhs;
            }

            // Insertion and extraction operators
            friend std::ostream &operator<<(std::ostream &os,
                                            const floating_point_wrapper &x) {
                const std::string out = std::to_string(static_cast<double>(x));
                os << out;
                return os;
            }

            friend std::istream &operator>>(std::istream &is, floating_point_wrapper &x) {
                std::string in;
                is >> in;
                x = std::stod(in);
                return is;
            }
        };
    }  // namespace detail

// Declare wrapper overloads now that floating_point_wrapper is declared
    template<class T>
    static FLARE_INLINE_FUNCTION flare::experimental::half_t cast_to_wrapper(
            T x, const volatile flare::detail::half_impl_t::type &) {
        return flare::experimental::cast_to_half(x);
    }

#ifdef FLARE_IMPL_BHALF_TYPE_DEFINED

    template<class T>
    static FLARE_INLINE_FUNCTION flare::experimental::bhalf_t cast_to_wrapper(
            T x, const volatile flare::detail::bhalf_impl_t::type &) {
        return flare::experimental::cast_to_bhalf(x);
    }

#endif  // FLARE_IMPL_BHALF_TYPE_DEFINED

    template<class T>
    static FLARE_INLINE_FUNCTION T
    cast_from_wrapper(const flare::experimental::half_t &x) {
        return flare::experimental::cast_from_half<T>(x);
    }

#ifdef FLARE_IMPL_BHALF_TYPE_DEFINED

    template<class T>
    static FLARE_INLINE_FUNCTION T
    cast_from_wrapper(const flare::experimental::bhalf_t &x) {
        return flare::experimental::cast_from_bhalf<T>(x);
    }

#endif  // FLARE_IMPL_BHALF_TYPE_DEFINED

}  // namespace flare::experimental

#endif  // FLARE_IMPL_HALF_TYPE_DEFINED

// If none of the above actually did anything and defined a half precision type
// define a fallback implementation here using float

#ifndef FLARE_IMPL_HALF_TYPE_DEFINED
#define FLARE_IMPL_HALF_TYPE_DEFINED
#define FLARE_HALF_T_IS_FLOAT true
namespace flare::detail {
    struct half_impl_t {
        using type = float;
    };
}  // namespace flare::detail
namespace flare::experimental {

    using half_t = flare::detail::half_impl_t::type;

// cast_to_half
    FLARE_INLINE_FUNCTION
    half_t cast_to_half(float val) { return half_t(val); }

    FLARE_INLINE_FUNCTION
    half_t cast_to_half(bool val) { return half_t(val); }

    FLARE_INLINE_FUNCTION
    half_t cast_to_half(double val) { return half_t(val); }

    FLARE_INLINE_FUNCTION
    half_t cast_to_half(short val) { return half_t(val); }

    FLARE_INLINE_FUNCTION
    half_t cast_to_half(unsigned short val) { return half_t(val); }

    FLARE_INLINE_FUNCTION
    half_t cast_to_half(int val) { return half_t(val); }

    FLARE_INLINE_FUNCTION
    half_t cast_to_half(unsigned int val) { return half_t(val); }

    FLARE_INLINE_FUNCTION
    half_t cast_to_half(long val) { return half_t(val); }

    FLARE_INLINE_FUNCTION
    half_t cast_to_half(unsigned long val) { return half_t(val); }

    FLARE_INLINE_FUNCTION
    half_t cast_to_half(long long val) { return half_t(val); }

    FLARE_INLINE_FUNCTION
    half_t cast_to_half(unsigned long long val) { return half_t(val); }

    // cast_from_half
    // Using an explicit list here too, since the other ones are explicit and for
    // example don't include char
    template<class T>
    FLARE_INLINE_FUNCTION std::enable_if_t<
            std::is_same<T, float>::value || std::is_same<T, bool>::value ||
            std::is_same<T, double>::value || std::is_same<T, short>::value ||
            std::is_same<T, unsigned short>::value || std::is_same<T, int>::value ||
            std::is_same<T, unsigned int>::value || std::is_same<T, long>::value ||
            std::is_same<T, unsigned long>::value ||
            std::is_same<T, long long>::value ||
            std::is_same<T, unsigned long long>::value,
            T>
    cast_from_half(half_t val) {
        return T(val);
    }

}  // namespace flare::experimental

#else
#define FLARE_HALF_T_IS_FLOAT false
#endif  // FLARE_IMPL_HALF_TYPE_DEFINED

#ifndef FLARE_IMPL_BHALF_TYPE_DEFINED
#define FLARE_IMPL_BHALF_TYPE_DEFINED
#define FLARE_BHALF_T_IS_FLOAT true
namespace flare {
    namespace detail {
        struct bhalf_impl_t {
            using type = float;
        };
    }  // namespace detail

    namespace experimental {

        using bhalf_t = flare::detail::bhalf_impl_t::type;

// cast_to_bhalf
        FLARE_INLINE_FUNCTION
        bhalf_t cast_to_bhalf(float val) { return bhalf_t(val); }

        FLARE_INLINE_FUNCTION
        bhalf_t cast_to_bhalf(bool val) { return bhalf_t(val); }

        FLARE_INLINE_FUNCTION
        bhalf_t cast_to_bhalf(double val) { return bhalf_t(val); }

        FLARE_INLINE_FUNCTION
        bhalf_t cast_to_bhalf(short val) { return bhalf_t(val); }

        FLARE_INLINE_FUNCTION
        bhalf_t cast_to_bhalf(unsigned short val) { return bhalf_t(val); }

        FLARE_INLINE_FUNCTION
        bhalf_t cast_to_bhalf(int val) { return bhalf_t(val); }

        FLARE_INLINE_FUNCTION
        bhalf_t cast_to_bhalf(unsigned int val) { return bhalf_t(val); }

        FLARE_INLINE_FUNCTION
        bhalf_t cast_to_bhalf(long val) { return bhalf_t(val); }

        FLARE_INLINE_FUNCTION
        bhalf_t cast_to_bhalf(unsigned long val) { return bhalf_t(val); }

        FLARE_INLINE_FUNCTION
        bhalf_t cast_to_bhalf(long long val) { return bhalf_t(val); }

        FLARE_INLINE_FUNCTION
        bhalf_t cast_to_bhalf(unsigned long long val) { return bhalf_t(val); }

        // cast_from_bhalf
        template<class T>
        FLARE_INLINE_FUNCTION std::enable_if_t<
                std::is_same<T, float>::value || std::is_same<T, bool>::value ||
                std::is_same<T, double>::value || std::is_same<T, short>::value ||
                std::is_same<T, unsigned short>::value || std::is_same<T, int>::value ||
                std::is_same<T, unsigned int>::value || std::is_same<T, long>::value ||
                std::is_same<T, unsigned long>::value ||
                std::is_same<T, long long>::value ||
                std::is_same<T, unsigned long long>::value,
                T>
        cast_from_bhalf(bhalf_t val) {
            return T(val);
        }
    }  // namespace experimental
}  // namespace flare
#else
#define FLARE_BHALF_T_IS_FLOAT false
#endif  // FLARE_IMPL_BHALF_TYPE_DEFINED

#endif  // FLARE_CORE_COMMON_HALF_FLOATING_POINT_WRAPPER_H_
