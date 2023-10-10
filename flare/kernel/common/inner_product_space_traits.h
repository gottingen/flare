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

#ifndef FLARE_KERNEL_COMMON_INNER_PRODUCT_SPACE_TRAITS_H_
#define FLARE_KERNEL_COMMON_INNER_PRODUCT_SPACE_TRAITS_H_

#include <flare/core/arith_traits.h>

namespace flare::detail {

        template <class T>
        class InnerProductSpaceTraits {
        public:
            //! The type T itself.
            typedef T val_type;

            //! The type returned by norm(x) for a value x of type val_type.
            typedef typename flare::ArithTraits<val_type>::mag_type mag_type;

            //! The type returned by dot(x,y) for values x and y of type val_type.
            typedef val_type dot_type;

            //! The "norm" (absolute value or magnitude) of a value x of type val_type.
            static FLARE_FORCEINLINE_FUNCTION mag_type norm(const val_type& x) {
                return flare::ArithTraits<val_type>::abs(x);
            }
            /// \brief The "dot product" of two values x and y of type val_type.
            ///
            /// This default implementation should suffice unless val_type is
            /// complex.  In that case, see the partial specialization for
            /// flare::complex below to see our convention for which input gets
            /// conjugated.
            static FLARE_FORCEINLINE_FUNCTION dot_type dot(const val_type& x,
                                                            const val_type& y) {
                return x * y;
            }
        };

        /// \brief Partial specialization for long double.
        ///
        /// \warning CUDA does not support long double in device functions.
        template <>
        struct InnerProductSpaceTraits<long double> {
            typedef long double val_type;
            typedef flare::ArithTraits<val_type>::mag_type mag_type;
            typedef val_type dot_type;

            static mag_type norm(const val_type& x) {
                return flare::ArithTraits<val_type>::abs(x);
            }
            static dot_type dot(const val_type& x, const val_type& y) { return x * y; }
        };

        //! Partial specialization for flare::complex<T>.
        template <class T>
        class InnerProductSpaceTraits<flare::complex<T>> {
        public:
            typedef flare::complex<T> val_type;
            typedef typename flare::ArithTraits<val_type>::mag_type mag_type;
            typedef val_type dot_type;

            static FLARE_FORCEINLINE_FUNCTION mag_type norm(const val_type& x) {
                return flare::ArithTraits<val_type>::abs(x);
            }
            static FLARE_FORCEINLINE_FUNCTION dot_type dot(const val_type& x,
                                                            const val_type& y) {
                return flare::conj(x) * y;
            }
        };

        /// \brief Partial specialization for std::complex<T>.
        ///
        /// \warning CUDA does not support std::complex<T> in device
        ///   functions.
        template <class T>
        struct InnerProductSpaceTraits<std::complex<T>> {
        typedef std::complex<T> val_type;
        typedef typename flare::ArithTraits<val_type>::mag_type mag_type;
        typedef val_type dot_type;

        static mag_type norm(const val_type& x) {
            return flare::ArithTraits<val_type>::abs(x);
        }
        static dot_type dot(const val_type& x, const val_type& y) {
            return std::conj(x) * y;
        }
    };

#ifdef HAVE_FLARE_QUADMATH

    /// \brief Partial specialization for __float128.
    ///
    /// CUDA does not support __float128 in device functions, so none of
    /// the class methods in this specialization are marked as device
    /// functions.
template <>
struct InnerProductSpaceTraits<__float128> {
  typedef __float128 val_type;
  typedef typename flare::ArithTraits<val_type>::mag_type mag_type;
  typedef val_type dot_type;

  static mag_type norm(const val_type& x) {
    return flare::ArithTraits<val_type>::abs(x);
  }
  static dot_type dot(const val_type& x, const val_type& y) { return x * y; }
};

#endif  // HAVE_FLARE_QUADMATH

// dd_real and qd_real are floating-point types provided by the QD
// library of David Bailey (LBNL):
//
// http://crd-legacy.lbl.gov/~dhbailey/mpdist/
//
// dd_real uses two doubles (128 bits), and qd_real uses four doubles
// (256 bits).
//
// flare does <i>not</i> currently support these types in device
// functions.  It should be possible to use flare' support for
// aggregate types to implement device function support for dd_real
// and qd_real, but we have not done this yet (as of 07 Jan 2014).
// Hence, the class methods of the flare::ArithTraits specializations for
// dd_real and qd_real are not marked as device functions.
#ifdef HAVE_FLARE_QD
    template <>
struct InnerProductSpaceTraits<dd_real> {
  typedef dd_real val_type;
  typedef flare::ArithTraits<val_type>::mag_type mag_type;
  typedef val_type dot_type;

  static mag_type norm(const val_type& x) {
    return flare::ArithTraits<val_type>::abs(x);
  }
  static dot_type dot(const val_type& x, const val_type& y) { return x * y; }
};

template <>
struct InnerProductSpaceTraits<qd_real> {
  typedef qd_real val_type;
  typedef flare::ArithTraits<val_type>::mag_type mag_type;
  typedef val_type dot_type;

  static mag_type norm(const val_type& x) {
    return flare::ArithTraits<val_type>::abs(x);
  }
  static dot_type dot(const val_type& x, const val_type& y) { return x * y; }
};
#endif  // HAVE_FLARE_QD

    template <class ResultType, class InputType1, class InputType2>
    FLARE_INLINE_FUNCTION void updateDot(ResultType& sum, const InputType1& x,
                                          const InputType2& y) {
        // FIXME (mfh 22 Jan 2020) We should actually pick the type with the
        // greater precision.
        sum += InnerProductSpaceTraits<InputType1>::dot(x, y);
    }

    FLARE_INLINE_FUNCTION void updateDot(double& sum, const double x,
                                          const double y) {
        sum += x * y;
    }

    FLARE_INLINE_FUNCTION void updateDot(double& sum, const float x,
                                          const float y) {
        sum += x * y;
    }

    // This exists because complex<float> += complex<double> is not defined.
    FLARE_INLINE_FUNCTION void updateDot(flare::complex<double>& sum,
                                          const flare::complex<float> x,
                                          const flare::complex<float> y) {
        const auto tmp = flare::conj(x) * y;
        sum += flare::complex<double>(tmp.real(), tmp.imag());
    }

    // This exists in case people call the overload of flare::blas::dot
    // that takes an output View, and the output View has element type
    // flare::complex<float>.
    FLARE_INLINE_FUNCTION void updateDot(flare::complex<float>& sum,
                                          const flare::complex<float> x,
                                          const flare::complex<float> y) {
        sum += flare::conj(x) * y;
    }

    // This exists because flare::complex<double> =
    // flare::complex<float> is not defined.
    template <class Out, class In>
    struct CastPossiblyComplex {
        static Out cast(const In& x) { return x; }
    };

    template <class OutReal, class InReal>
    struct CastPossiblyComplex<flare::complex<OutReal>, flare::complex<InReal>> {
        static flare::complex<OutReal> cast(const flare::complex<InReal>& x) {
            return {static_cast<OutReal>(x.real()), static_cast<OutReal>(x.imag())};
        }
    };

}  // namespace flare::detail


#endif  // FLARE_KERNEL_COMMON_INNER_PRODUCT_SPACE_TRAITS_H_
