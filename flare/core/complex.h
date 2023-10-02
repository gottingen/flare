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

#ifndef FLARE_CORE_COMPLEX_H_
#define FLARE_CORE_COMPLEX_H_

#include <flare/core/atomic.h>
#include <flare/core/mathematical_functions.h>
#include <flare/core/numeric_traits.h>
#include <flare/core/reduction_identity.h>
#include <flare/core/common/error.h>
#include <complex>
#include <type_traits>
#include <iosfwd>

namespace flare {

    /// \class complex
    /// \brief Partial reimplementation of std::complex that works as the
    ///   result of a flare::parallel_reduce.
    /// \tparam RealType The type of the real and imaginary parts of the
    ///   complex number.  As with std::complex, this is only defined for
    ///   \c float, \c double, and <tt>long double</tt>.  The latter is
    ///   currently forbidden in CUDA device kernels.
    template<class RealType>
    class
#ifdef FLARE_ENABLE_COMPLEX_ALIGN
    alignas(2 * sizeof(RealType))
#endif
    complex {
        static_assert(std::is_floating_point_v<RealType> &&
                      std::is_same_v<RealType, std::remove_cv_t<RealType>>,
                      "flare::complex can only be instantiated for a cv-unqualified "
                      "floating point type");

    private:
        RealType re_{};
        RealType im_{};

    public:
        //! The type of the real or imaginary parts of this complex number.
        using value_type = RealType;

        //! Default constructor (initializes both real and imaginary parts to zero).
        FLARE_DEFAULTED_FUNCTION
        complex() = default;

        //! Copy constructor.
        FLARE_DEFAULTED_FUNCTION
        complex(const complex &) noexcept = default;

        FLARE_DEFAULTED_FUNCTION
        complex &operator=(const complex &) noexcept = default;

        /// \brief Conversion constructor from compatible RType
        template<
                class RType,
                std::enable_if_t<std::is_convertible<RType, RealType>::value, int> = 0>
        FLARE_INLINE_FUNCTION complex(const complex<RType> &other) noexcept
        // Intentionally do the conversions implicitly here so that users don't
        // get any warnings about narrowing, etc., that they would expect to get
        // otherwise.
                : re_(other.real()), im_(other.imag()) {}

        /// \brief Conversion constructor from std::complex.
        ///
        /// This constructor cannot be called in a CUDA device function,
        /// because std::complex's methods and nonmember functions are not
        /// marked as CUDA device functions.
        FLARE_INLINE_FUNCTION
        complex(const std::complex<RealType> &src) noexcept
        // We can use this aspect of the standard to avoid calling
        // non-device-marked functions `std::real` and `std::imag`: "For any
        // object z of type complex<T>, reinterpret_cast<T(&)[2]>(z)[0] is the
        // real part of z and reinterpret_cast<T(&)[2]>(z)[1] is the imaginary
        // part of z." Now we don't have to provide a whole bunch of the overloads
        // of things taking either flare::complex or std::complex
                : re_(reinterpret_cast<const RealType (&)[2]>(src)[0]),
                  im_(reinterpret_cast<const RealType (&)[2]>(src)[1]) {}

        /// \brief Conversion operator to std::complex.
        ///
        /// This operator cannot be called in a CUDA device function,
        /// because std::complex's methods and nonmember functions are not
        /// marked as CUDA device functions.
        // TODO: make explicit.  DJS 2019-08-28
        operator std::complex<RealType>() const noexcept {
            return std::complex<RealType>(re_, im_);
        }

        /// \brief Constructor that takes just the real part, and sets the
        ///   imaginary part to zero.
        FLARE_INLINE_FUNCTION complex(const RealType &val) noexcept
                : re_(val), im_(static_cast<RealType>(0)) {}

        //! Constructor that takes the real and imaginary parts.
        FLARE_INLINE_FUNCTION
        complex(const RealType &re, const RealType &im) noexcept: re_(re), im_(im) {}

        //! Assignment operator (from a real number).
        FLARE_INLINE_FUNCTION complex &operator=(const RealType &val) noexcept {
            re_ = val;
            im_ = RealType(0);
            return *this;
        }

        /// \brief Assignment operator from std::complex.
        ///
        /// This constructor cannot be called in a CUDA device function,
        /// because std::complex's methods and nonmember functions are not
        /// marked as CUDA device functions.
        complex &operator=(const std::complex<RealType> &src) noexcept {
            *this = complex(src);
            return *this;
        }

        //! The imaginary part of this complex number.
        FLARE_INLINE_FUNCTION
        constexpr RealType &imag() noexcept { return im_; }

        //! The real part of this complex number.
        FLARE_INLINE_FUNCTION
        constexpr RealType &real() noexcept { return re_; }

        //! The imaginary part of this complex number.
        FLARE_INLINE_FUNCTION
        constexpr RealType imag() const noexcept { return im_; }

        //! The real part of this complex number.
        FLARE_INLINE_FUNCTION
        constexpr RealType real() const noexcept { return re_; }

        //! Set the imaginary part of this complex number.
        FLARE_INLINE_FUNCTION
        constexpr void imag(RealType v) noexcept { im_ = v; }

        //! Set the real part of this complex number.
        FLARE_INLINE_FUNCTION
        constexpr void real(RealType v) noexcept { re_ = v; }

        constexpr FLARE_INLINE_FUNCTION complex &operator+=(
                const complex<RealType> &src) noexcept {
            re_ += src.re_;
            im_ += src.im_;
            return *this;
        }

        constexpr FLARE_INLINE_FUNCTION complex &operator+=(
                const RealType &src) noexcept {
            re_ += src;
            return *this;
        }

        constexpr FLARE_INLINE_FUNCTION complex &operator-=(
                const complex<RealType> &src) noexcept {
            re_ -= src.re_;
            im_ -= src.im_;
            return *this;
        }

        constexpr FLARE_INLINE_FUNCTION complex &operator-=(
                const RealType &src) noexcept {
            re_ -= src;
            return *this;
        }

        constexpr FLARE_INLINE_FUNCTION complex &operator*=(
                const complex<RealType> &src) noexcept {
            const RealType realPart = re_ * src.re_ - im_ * src.im_;
            const RealType imagPart = re_ * src.im_ + im_ * src.re_;
            re_ = realPart;
            im_ = imagPart;
            return *this;
        }

        constexpr FLARE_INLINE_FUNCTION complex &operator*=(
                const RealType &src) noexcept {
            re_ *= src;
            im_ *= src;
            return *this;
        }

        // Conditional noexcept, just in case RType throws on divide-by-zero
        constexpr FLARE_INLINE_FUNCTION complex &operator/=(
                const complex<RealType> &y) noexcept(noexcept(RealType{} / RealType{})) {
            // Scale (by the "1-norm" of y) to avoid unwarranted overflow.
            // If the real part is +/-Inf and the imaginary part is -/+Inf,
            // this won't change the result.
            const RealType s = fabs(y.real()) + fabs(y.imag());

            // If s is 0, then y is zero, so x/y == real(x)/0 + i*imag(x)/0.
            // In that case, the relation x/y == (x/s) / (y/s) doesn't hold,
            // because y/s is NaN.
            // TODO mark this branch unlikely
            if (s == RealType(0)) {
                this->re_ /= s;
                this->im_ /= s;
            } else {
                const complex x_scaled(this->re_ / s, this->im_ / s);
                const complex y_conj_scaled(y.re_ / s, -(y.im_) / s);
                const RealType y_scaled_abs =
                        y_conj_scaled.re_ * y_conj_scaled.re_ +
                        y_conj_scaled.im_ * y_conj_scaled.im_;  // abs(y) == abs(conj(y))
                *this = x_scaled * y_conj_scaled;
                *this /= y_scaled_abs;
            }
            return *this;
        }

        constexpr FLARE_INLINE_FUNCTION complex &operator/=(
                const std::complex<RealType> &y) noexcept(noexcept(RealType{} /
                                                                   RealType{})) {
            // Scale (by the "1-norm" of y) to avoid unwarranted overflow.
            // If the real part is +/-Inf and the imaginary part is -/+Inf,
            // this won't change the result.
            const RealType s = fabs(y.real()) + fabs(y.imag());

            // If s is 0, then y is zero, so x/y == real(x)/0 + i*imag(x)/0.
            // In that case, the relation x/y == (x/s) / (y/s) doesn't hold,
            // because y/s is NaN.
            if (s == RealType(0)) {
                this->re_ /= s;
                this->im_ /= s;
            } else {
                const complex x_scaled(this->re_ / s, this->im_ / s);
                const complex y_conj_scaled(y.re_ / s, -(y.im_) / s);
                const RealType y_scaled_abs =
                        y_conj_scaled.re_ * y_conj_scaled.re_ +
                        y_conj_scaled.im_ * y_conj_scaled.im_;  // abs(y) == abs(conj(y))
                *this = x_scaled * y_conj_scaled;
                *this /= y_scaled_abs;
            }
            return *this;
        }

        constexpr FLARE_INLINE_FUNCTION complex &operator/=(
                const RealType &src) noexcept(noexcept(RealType{} / RealType{})) {
            re_ /= src;
            im_ /= src;
            return *this;
        }

    };


    // Note that this is not the same behavior as std::complex, which doesn't allow
    // implicit conversions, but since this is the way we had it before, we have
    // to do it this way now.

    //! Binary == operator for complex complex.
    template<class RealType1, class RealType2>
    FLARE_INLINE_FUNCTION bool operator==(complex<RealType1> const &x,
                                          complex<RealType2> const &y) noexcept {
        using common_type = std::common_type_t<RealType1, RealType2>;
        return common_type(x.real()) == common_type(y.real()) &&
               common_type(x.imag()) == common_type(y.imag());
    }

    // TODO (here and elsewhere) decide if we should convert to a flare::complex
    //      and do the comparison in a device-marked function
    //! Binary == operator for std::complex complex.
    template<class RealType1, class RealType2>
    inline bool operator==(std::complex<RealType1> const &x,
                           complex<RealType2> const &y) noexcept {
        using common_type = std::common_type_t<RealType1, RealType2>;
        return common_type(x.real()) == common_type(y.real()) &&
               common_type(x.imag()) == common_type(y.imag());
    }

    //! Binary == operator for complex std::complex.
    template<class RealType1, class RealType2>
    inline bool operator==(complex<RealType1> const &x,
                           std::complex<RealType2> const &y) noexcept {
        using common_type = std::common_type_t<RealType1, RealType2>;
        return common_type(x.real()) == common_type(y.real()) &&
               common_type(x.imag()) == common_type(y.imag());
    }

    //! Binary == operator for complex real.
    template<
            class RealType1, class RealType2,
            // Constraints to avoid participation in oparator==() for every possible RHS
            std::enable_if_t<std::is_convertible<RealType2, RealType1>::value, int> = 0>
    FLARE_INLINE_FUNCTION bool operator==(complex<RealType1> const &x,
                                          RealType2 const &y) noexcept {
        using common_type = std::common_type_t<RealType1, RealType2>;
        return common_type(x.real()) == common_type(y) &&
               common_type(x.imag()) == common_type(0);
    }

    //! Binary == operator for real complex.
    template<
            class RealType1, class RealType2,
            // Constraints to avoid participation in oparator==() for every possible RHS
            std::enable_if_t<std::is_convertible<RealType1, RealType2>::value, int> = 0>
    FLARE_INLINE_FUNCTION bool operator==(RealType1 const &x,
                                          complex<RealType2> const &y) noexcept {
        using common_type = std::common_type_t<RealType1, RealType2>;
        return common_type(x) == common_type(y.real()) &&
               common_type(0) == common_type(y.imag());
    }

    //! Binary != operator for complex complex.
    template<class RealType1, class RealType2>
    FLARE_INLINE_FUNCTION bool operator!=(complex<RealType1> const &x,
                                          complex<RealType2> const &y) noexcept {
        using common_type = std::common_type_t<RealType1, RealType2>;
        return common_type(x.real()) != common_type(y.real()) ||
               common_type(x.imag()) != common_type(y.imag());
    }

    //! Binary != operator for std::complex complex.
    template<class RealType1, class RealType2>
    inline bool operator!=(std::complex<RealType1> const &x,
                           complex<RealType2> const &y) noexcept {
        using common_type = std::common_type_t<RealType1, RealType2>;
        return common_type(x.real()) != common_type(y.real()) ||
               common_type(x.imag()) != common_type(y.imag());
    }

    //! Binary != operator for complex std::complex.
    template<class RealType1, class RealType2>
    inline bool operator!=(complex<RealType1> const &x,
                           std::complex<RealType2> const &y) noexcept {
        using common_type = std::common_type_t<RealType1, RealType2>;
        return common_type(x.real()) != common_type(y.real()) ||
               common_type(x.imag()) != common_type(y.imag());
    }

    //! Binary != operator for complex real.
    template<
            class RealType1, class RealType2,
            // Constraints to avoid participation in oparator==() for every possible RHS
            std::enable_if_t<std::is_convertible<RealType2, RealType1>::value, int> = 0>
    FLARE_INLINE_FUNCTION bool operator!=(complex<RealType1> const &x,
                                          RealType2 const &y) noexcept {
        using common_type = std::common_type_t<RealType1, RealType2>;
        return common_type(x.real()) != common_type(y) ||
               common_type(x.imag()) != common_type(0);
    }

    //! Binary != operator for real complex.
    template<
            class RealType1, class RealType2,
            // Constraints to avoid participation in oparator==() for every possible RHS
            std::enable_if_t<std::is_convertible<RealType1, RealType2>::value, int> = 0>
    FLARE_INLINE_FUNCTION bool operator!=(RealType1 const &x,
                                          complex<RealType2> const &y) noexcept {
        using common_type = std::common_type_t<RealType1, RealType2>;
        return common_type(x) != common_type(y.real()) ||
               common_type(0) != common_type(y.imag());
    }


    //! Binary + operator for complex complex.
    template<class RealType1, class RealType2>
    FLARE_INLINE_FUNCTION complex<std::common_type_t<RealType1, RealType2>>
    operator+(const complex<RealType1> &x, const complex<RealType2> &y) noexcept {
        return complex<std::common_type_t<RealType1, RealType2>>(x.real() + y.real(),
                                                                 x.imag() + y.imag());
    }

    //! Binary + operator for complex scalar.
    template<class RealType1, class RealType2>
    FLARE_INLINE_FUNCTION complex<std::common_type_t<RealType1, RealType2>>
    operator+(const complex<RealType1> &x, const RealType2 &y) noexcept {
        return complex<std::common_type_t<RealType1, RealType2>>(x.real() + y,
                                                                 x.imag());
    }

    //! Binary + operator for scalar complex.
    template<class RealType1, class RealType2>
    FLARE_INLINE_FUNCTION complex<std::common_type_t<RealType1, RealType2>>
    operator+(const RealType1 &x, const complex<RealType2> &y) noexcept {
        return complex<std::common_type_t<RealType1, RealType2>>(x + y.real(),
                                                                 y.imag());
    }

    //! Unary + operator for complex.
    template<class RealType>
    FLARE_INLINE_FUNCTION complex<RealType> operator+(
            const complex<RealType> &x) noexcept {
        return complex<RealType>{+x.real(), +x.imag()};
    }

    //! Binary - operator for complex.
    template<class RealType1, class RealType2>
    FLARE_INLINE_FUNCTION complex<std::common_type_t<RealType1, RealType2>>
    operator-(const complex<RealType1> &x, const complex<RealType2> &y) noexcept {
        return complex<std::common_type_t<RealType1, RealType2>>(x.real() - y.real(),
                                                                 x.imag() - y.imag());
    }

    //! Binary - operator for complex scalar.
    template<class RealType1, class RealType2>
    FLARE_INLINE_FUNCTION complex<std::common_type_t<RealType1, RealType2>>
    operator-(const complex<RealType1> &x, const RealType2 &y) noexcept {
        return complex<std::common_type_t<RealType1, RealType2>>(x.real() - y,
                                                                 x.imag());
    }

    //! Binary - operator for scalar complex.
    template<class RealType1, class RealType2>
    FLARE_INLINE_FUNCTION complex<std::common_type_t<RealType1, RealType2>>
    operator-(const RealType1 &x, const complex<RealType2> &y) noexcept {
        return complex<std::common_type_t<RealType1, RealType2>>(x - y.real(),
                                                                 -y.imag());
    }

    //! Unary - operator for complex.
    template<class RealType>
    FLARE_INLINE_FUNCTION complex<RealType> operator-(
            const complex<RealType> &x) noexcept {
        return complex<RealType>(-x.real(), -x.imag());
    }

    //! Binary * operator for complex.
    template<class RealType1, class RealType2>
    FLARE_INLINE_FUNCTION complex<std::common_type_t<RealType1, RealType2>>
    operator*(const complex<RealType1> &x, const complex<RealType2> &y) noexcept {
        return complex<std::common_type_t<RealType1, RealType2>>(
                x.real() * y.real() - x.imag() * y.imag(),
                x.real() * y.imag() + x.imag() * y.real());
    }

    /// \brief Binary * operator for std::complex and complex.
    ///
    /// This needs to exist because template parameters can't be deduced when
    /// conversions occur.  We could probably fix this using hidden friends patterns
    ///
    /// This function cannot be called in a CUDA device function, because
    /// std::complex's methods and nonmember functions are not marked as
    /// CUDA device functions.
    template<class RealType1, class RealType2>
    inline complex<std::common_type_t<RealType1, RealType2>> operator*(
            const std::complex<RealType1> &x, const complex<RealType2> &y) {
        return complex<std::common_type_t<RealType1, RealType2>>(
                x.real() * y.real() - x.imag() * y.imag(),
                x.real() * y.imag() + x.imag() * y.real());
    }

    /// \brief Binary * operator for RealType times complex.
    ///
    /// This function exists because the compiler doesn't know that
    /// RealType and complex<RealType> commute with respect to operator*.
    template<class RealType1, class RealType2>
    FLARE_INLINE_FUNCTION complex<std::common_type_t<RealType1, RealType2>>
    operator*(const RealType1 &x, const complex<RealType2> &y) noexcept {
        return complex<std::common_type_t<RealType1, RealType2>>(x * y.real(),
                                                                 x * y.imag());
    }

    /// \brief Binary * operator for RealType times complex.
    ///
    /// This function exists because the compiler doesn't know that
    /// RealType and complex<RealType> commute with respect to operator*.
    template<class RealType1, class RealType2>
    FLARE_INLINE_FUNCTION complex<std::common_type_t<RealType1, RealType2>>
    operator*(const complex<RealType1> &y, const RealType2 &x) noexcept {
        return complex<std::common_type_t<RealType1, RealType2>>(x * y.real(),
                                                                 x * y.imag());
    }

    //! Imaginary part of a complex number.
    template<class RealType>
    FLARE_INLINE_FUNCTION RealType imag(const complex<RealType> &x) noexcept {
        return x.imag();
    }

    template<class ArithmeticType>
    FLARE_INLINE_FUNCTION constexpr detail::promote_t<ArithmeticType> imag(
            ArithmeticType) {
        return ArithmeticType();
    }

    //! Real part of a complex number.
    template<class RealType>
    FLARE_INLINE_FUNCTION RealType real(const complex<RealType> &x) noexcept {
        return x.real();
    }

    template<class ArithmeticType>
    FLARE_INLINE_FUNCTION constexpr detail::promote_t<ArithmeticType> real(
            ArithmeticType x) {
        return x;
    }

    //! Constructs a complex number from magnitude and phase angle
    template<class T>
    FLARE_INLINE_FUNCTION complex<T> polar(const T &r, const T &theta = T()) {
        FLARE_EXPECTS(r >= 0);
        return complex<T>(r * cos(theta), r * sin(theta));
    }

    //! Absolute value (magnitude) of a complex number.
    template<class RealType>
    FLARE_INLINE_FUNCTION RealType abs(const complex<RealType> &x) {
        return hypot(x.real(), x.imag());
    }

    //! Power of a complex number
    template<class T>
    FLARE_INLINE_FUNCTION complex<T> pow(const complex<T> &x, const T &y) {
        T r = abs(x);
        T theta = atan2(x.imag(), x.real());
        return polar(pow(r, y), y * theta);
    }

    template<class T>
    FLARE_INLINE_FUNCTION complex<T> pow(const T &x, const complex<T> &y) {
        return pow(complex<T>(x), y);
    }

    template<class T>
    FLARE_INLINE_FUNCTION complex<T> pow(const complex<T> &x,
                                         const complex<T> &y) {
        return x == T() ? T() : exp(y * log(x));
    }

    template<class T, class U,
            class = std::enable_if_t<std::is_arithmetic<T>::value>>
    FLARE_INLINE_FUNCTION complex<detail::promote_2_t<T, U>> pow(
            const T &x, const complex<U> &y) {
        using type = detail::promote_2_t<T, U>;
        return pow(type(x), complex<type>(y));
    }

    template<class T, class U,
            class = std::enable_if_t<std::is_arithmetic<U>::value>>
    FLARE_INLINE_FUNCTION complex<detail::promote_2_t<T, U>> pow(const complex<T> &x,
                                                               const U &y) {
        using type = detail::promote_2_t<T, U>;
        return pow(complex<type>(x), type(y));
    }

    template<class T, class U>
    FLARE_INLINE_FUNCTION complex<detail::promote_2_t<T, U>> pow(
            const complex<T> &x, const complex<U> &y) {
        using type = detail::promote_2_t<T, U>;
        return pow(complex<type>(x), complex<type>(y));
    }

    //! Square root of a complex number. This is intended to match the stdc++
    //! implementation, which returns sqrt(z*z) = z; where z is complex number.
    template<class RealType>
    FLARE_INLINE_FUNCTION flare::complex<RealType> sqrt(
            const complex<RealType> &x) {
        RealType r = x.real();
        RealType i = x.imag();

        if (r == RealType()) {
            RealType t = sqrt(fabs(i) / 2);
            return flare::complex<RealType>(t, i < RealType() ? -t : t);
        } else {
            RealType t = sqrt(2 * (abs(x) + fabs(r)));
            RealType u = t / 2;
            return r > RealType() ? flare::complex<RealType>(u, i / t)
                                  : flare::complex<RealType>(fabs(i) / t,
                                                             i < RealType() ? -u : u);
        }
    }

    //! Conjugate of a complex number.
    template<class RealType>
    FLARE_INLINE_FUNCTION complex<RealType> conj(
            const complex<RealType> &x) noexcept {
        return complex<RealType>(real(x), -imag(x));
    }

    template<class ArithmeticType>
    FLARE_INLINE_FUNCTION constexpr complex<detail::promote_t<ArithmeticType>> conj(
            ArithmeticType x) {
        using type = detail::promote_t<ArithmeticType>;
        return complex<type>(x, -type());
    }

    //! Exponential of a complex number.
    template<class RealType>
    FLARE_INLINE_FUNCTION complex<RealType> exp(const complex<RealType> &x) {
        return exp(x.real()) * complex<RealType>(cos(x.imag()), sin(x.imag()));
    }

    //! natural log of a complex number.
    template<class RealType>
    FLARE_INLINE_FUNCTION flare::complex<RealType> log(
            const complex<RealType> &x) {
        RealType phi = atan2(x.imag(), x.real());
        return flare::complex<RealType>(log(abs(x)), phi);
    }

    //! base 10 log of a complex number.
    template<class RealType>
    FLARE_INLINE_FUNCTION flare::complex<RealType> log10(
            const complex<RealType> &x) {
        return log(x) / log(RealType(10));
    }

    //! sine of a complex number.
    template<class RealType>
    FLARE_INLINE_FUNCTION flare::complex<RealType> sin(
            const complex<RealType> &x) {
        return flare::complex<RealType>(sin(x.real()) * cosh(x.imag()),
                                        cos(x.real()) * sinh(x.imag()));
    }

    //! cosine of a complex number.
    template<class RealType>
    FLARE_INLINE_FUNCTION flare::complex<RealType> cos(
            const complex<RealType> &x) {
        return flare::complex<RealType>(cos(x.real()) * cosh(x.imag()),
                                        -sin(x.real()) * sinh(x.imag()));
    }

    //! tangent of a complex number.
    template<class RealType>
    FLARE_INLINE_FUNCTION flare::complex<RealType> tan(
            const complex<RealType> &x) {
        return sin(x) / cos(x);
    }

    //! hyperbolic sine of a complex number.
    template<class RealType>
    FLARE_INLINE_FUNCTION flare::complex<RealType> sinh(
            const complex<RealType> &x) {
        return flare::complex<RealType>(sinh(x.real()) * cos(x.imag()),
                                        cosh(x.real()) * sin(x.imag()));
    }

    //! hyperbolic cosine of a complex number.
    template<class RealType>
    FLARE_INLINE_FUNCTION flare::complex<RealType> cosh(
            const complex<RealType> &x) {
        return flare::complex<RealType>(cosh(x.real()) * cos(x.imag()),
                                        sinh(x.real()) * sin(x.imag()));
    }

    //! hyperbolic tangent of a complex number.
    template<class RealType>
    FLARE_INLINE_FUNCTION flare::complex<RealType> tanh(
            const complex<RealType> &x) {
        return sinh(x) / cosh(x);
    }

    //! inverse hyperbolic sine of a complex number.
    template<class RealType>
    FLARE_INLINE_FUNCTION flare::complex<RealType> asinh(
            const complex<RealType> &x) {
        return log(x + sqrt(x * x + RealType(1.0)));
    }

    //! inverse hyperbolic cosine of a complex number.
    template<class RealType>
    FLARE_INLINE_FUNCTION flare::complex<RealType> acosh(
            const complex<RealType> &x) {
        return RealType(2.0) * log(sqrt(RealType(0.5) * (x + RealType(1.0))) +
                                   sqrt(RealType(0.5) * (x - RealType(1.0))));
    }

    //! inverse hyperbolic tangent of a complex number.
    template<class RealType>
    FLARE_INLINE_FUNCTION flare::complex<RealType> atanh(
            const complex<RealType> &x) {
        const RealType i2 = x.imag() * x.imag();
        const RealType r = RealType(1.0) - i2 - x.real() * x.real();

        RealType p = RealType(1.0) + x.real();
        RealType m = RealType(1.0) - x.real();

        p = i2 + p * p;
        m = i2 + m * m;

        RealType phi = atan2(RealType(2.0) * x.imag(), r);
        return flare::complex<RealType>(RealType(0.25) * (log(p) - log(m)),
                                        RealType(0.5) * phi);
    }

    //! inverse sine of a complex number.
    template<class RealType>
    FLARE_INLINE_FUNCTION flare::complex<RealType> asin(
            const complex<RealType> &x) {
        flare::complex<RealType> t =
                asinh(flare::complex<RealType>(-x.imag(), x.real()));
        return flare::complex<RealType>(t.imag(), -t.real());
    }

    //! inverse cosine of a complex number.
    template<class RealType>
    FLARE_INLINE_FUNCTION flare::complex<RealType> acos(
            const complex<RealType> &x) {
        flare::complex<RealType> t = asin(x);
        RealType pi_2 = acos(RealType(0.0));
        return flare::complex<RealType>(pi_2 - t.real(), -t.imag());
    }

    //! inverse tangent of a complex number.
    template<class RealType>
    FLARE_INLINE_FUNCTION flare::complex<RealType> atan(
            const complex<RealType> &x) {
        const RealType r2 = x.real() * x.real();
        const RealType i = RealType(1.0) - r2 - x.imag() * x.imag();

        RealType p = x.imag() + RealType(1.0);
        RealType m = x.imag() - RealType(1.0);

        p = r2 + p * p;
        m = r2 + m * m;

        return flare::complex<RealType>(
                RealType(0.5) * atan2(RealType(2.0) * x.real(), i),
                RealType(0.25) * log(p / m));
    }

    /// This function cannot be called in a CUDA device function,
    /// because std::complex's methods and nonmember functions are not
    /// marked as CUDA device functions.
    template<class RealType>
    inline complex<RealType> exp(const std::complex<RealType> &c) {
        return complex<RealType>(std::exp(c.real()) * std::cos(c.imag()),
                                 std::exp(c.real()) * std::sin(c.imag()));
    }

    //! Binary operator / for complex and real numbers
    template<class RealType1, class RealType2>
    FLARE_INLINE_FUNCTION complex<std::common_type_t<RealType1, RealType2>>
    operator/(const complex<RealType1> &x,
              const RealType2 &y) noexcept(noexcept(RealType1{} / RealType2{})) {
        return complex<std::common_type_t<RealType1, RealType2>>(real(x) / y,
                                                                 imag(x) / y);
    }

    //! Binary operator / for complex.
    template<class RealType1, class RealType2>
    FLARE_INLINE_FUNCTION complex<std::common_type_t<RealType1, RealType2>>
    operator/(const complex<RealType1> &x,
              const complex<RealType2> &y) noexcept(noexcept(RealType1{} /
                                                             RealType2{})) {
        // Scale (by the "1-norm" of y) to avoid unwarranted overflow.
        // If the real part is +/-Inf and the imaginary part is -/+Inf,
        // this won't change the result.
        using common_real_type = std::common_type_t<RealType1, RealType2>;
        const common_real_type s = fabs(real(y)) + fabs(imag(y));

        // If s is 0, then y is zero, so x/y == real(x)/0 + i*imag(x)/0.
        // In that case, the relation x/y == (x/s) / (y/s) doesn't hold,
        // because y/s is NaN.
        if (s == 0.0) {
            return complex<common_real_type>(real(x) / s, imag(x) / s);
        } else {
            const complex<common_real_type> x_scaled(real(x) / s, imag(x) / s);
            const complex<common_real_type> y_conj_scaled(real(y) / s, -imag(y) / s);
            const RealType1 y_scaled_abs =
                    real(y_conj_scaled) * real(y_conj_scaled) +
                    imag(y_conj_scaled) * imag(y_conj_scaled);  // abs(y) == abs(conj(y))
            complex<common_real_type> result = x_scaled * y_conj_scaled;
            result /= y_scaled_abs;
            return result;
        }
    }

//! Binary operator / for complex and real numbers
    template<class RealType1, class RealType2>
    FLARE_INLINE_FUNCTION complex<std::common_type_t<RealType1, RealType2>>
    operator/(const RealType1 &x,
              const complex<RealType2> &y) noexcept(noexcept(RealType1{} /
                                                             RealType2{})) {
        return complex<std::common_type_t<RealType1, RealType2>>(x) / y;
    }

    template<class RealType>
    std::ostream &operator<<(std::ostream &os, const complex<RealType> &x) {
        const std::complex<RealType> x_std(flare::real(x), flare::imag(x));
        os << x_std;
        return os;
    }

    template<class RealType>
    std::istream &operator>>(std::istream &is, complex<RealType> &x) {
        std::complex<RealType> x_std;
        is >> x_std;
        x = x_std;  // only assigns on success of above
        return is;
    }

    template<class T>
    struct reduction_identity<flare::complex<T>> {
        using t_red_ident = reduction_identity<T>;

        FLARE_FORCEINLINE_FUNCTION constexpr static flare::complex<T>
        sum() noexcept {
            return flare::complex<T>(t_red_ident::sum(), t_red_ident::sum());
        }

        FLARE_FORCEINLINE_FUNCTION constexpr static flare::complex<T>
        prod() noexcept {
            return flare::complex<T>(t_red_ident::prod(), t_red_ident::sum());
        }
    };

}  // namespace flare

#endif  // FLARE_CORE_COMPLEX_H_
