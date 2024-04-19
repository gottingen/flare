// Copyright 2023 The EA Authors.
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

#include <fly/defines.h>

#ifdef __cplusplus
namespace fly
{
    class array;

    /// C++ Interface to find the elementwise minimum between two arrays.
    ///
    /// \param[in] lhs input array
    /// \param[in] rhs input array
    /// \return        minimum
    ///
    /// \ingroup arith_func_min
    FLY_API array min    (const array &lhs, const array &rhs);

    /// C++ Interface to find the elementwise minimum between an array and a
    /// scalar value.
    ///
    /// \param[in] lhs input array
    /// \param[in] rhs scalar value
    /// \return        minimum
    ///
    /// \ingroup arith_func_min
    FLY_API array min    (const array &lhs, const double rhs);

    /// C++ Interface to find the elementwise minimum between an array and a
    /// scalar value.
    ///
    /// \param[in] lhs scalar value
    /// \param[in] rhs input array
    /// \return        minimum
    ///
    /// \ingroup arith_func_min
    FLY_API array min    (const double lhs, const array &rhs);

    /// C++ Interface to find the elementwise maximum between two arrays.
    ///
    /// \param[in] lhs input array
    /// \param[in] rhs input array
    /// \return        maximum
    ///
    /// \ingroup arith_func_max
    FLY_API array max    (const array &lhs, const array &rhs);

    /// C++ Interface to find the elementwise maximum between an array and a
    /// scalar value.
    ///
    /// \param[in] lhs input array
    /// \param[in] rhs scalar value
    /// \return        maximum
    ///
    /// \ingroup arith_func_max
    FLY_API array max    (const array &lhs, const double rhs);

    /// C++ Interface to find the elementwise maximum between an array and a
    /// scalar value.
    ///
    /// \param[in] lhs input array
    /// \param[in] rhs scalar value
    /// \return        maximum
    ///
    /// \ingroup arith_func_max
    FLY_API array max    (const double lhs, const array &rhs);

    /// @{
    /// C++ Interface to clamp an array between an upper and a lower limit.
    ///
    /// \param[in] in input array
    /// \param[in] lo lower limit; can be an array or a scalar
    /// \param[in] hi upper limit; can be an array or a scalar
    /// \return       clamped array
    /// 
    /// \ingroup arith_func_clamp
    FLY_API array clamp(const array &in, const array &lo, const array &hi);

    /// \copydoc clamp(const array&, const array&, const array&)
    FLY_API array clamp(const array &in, const array &lo, const double hi);

    /// \copydoc clamp(const array&, const array&, const array&)
    FLY_API array clamp(const array &in, const double lo, const array &hi);

    /// \copydoc clamp(const array&, const array&, const array&)
    FLY_API array clamp(const array &in, const double lo, const double hi);
    /// @}

    /// @{
    /// C++ Interface to calculate the remainder.
    ///
    /// \param[in] lhs numerator; can be an array or a scalar
    /// \param[in] rhs denominator; can be an array or a scalar
    /// \return        remainder
    /// 
    /// \ingroup arith_func_rem
    FLY_API array rem    (const array &lhs, const array &rhs);

    /// \copydoc rem(const array&, const array&)
    FLY_API array rem    (const array &lhs, const double rhs);

    /// \copydoc rem(const array&, const array&)
    FLY_API array rem    (const double lhs, const array &rhs);
    /// @}

    /// @{
    /// C++ Interface to calculate the modulus.
    ///
    /// \param[in] lhs dividend; can be an array or a scalar
    /// \param[in] rhs divisor; can be an array or a scalar
    /// \return        modulus
    /// 
    /// \ingroup arith_func_mod
    FLY_API array mod    (const array &lhs, const array &rhs);

    /// \copydoc mod(const array&, const array&)
    FLY_API array mod    (const array &lhs, const double rhs);

    /// \copydoc mod(const array&, const array&)
    FLY_API array mod    (const double lhs, const array &rhs);
    /// @}

    /// C++ Interface to calculate the absolute value.
    ///
    /// \param[in] in input array
    /// \return       absolute value
    ///
    /// \ingroup arith_func_abs
    FLY_API array abs    (const array &in);

    /// C++ Interface to calculate the phase angle (in radians) of a complex
    /// array.
    ///
    /// \param[in] in input array, typically complex
    /// \return       phase angle (in radians)
    /// 
    /// \ingroup arith_func_arg
    FLY_API array arg    (const array &in);

    /// C++ Interface to return the sign of elements in an array.
    ///
    /// \param[in] in input array
    /// \return       array containing 1's for negative values; 0's otherwise
    /// 
    /// \ingroup arith_func_sign
    FLY_API array sign  (const array &in);

    /// C++ Interface to round numbers.
    ///
    /// \param[in] in input array
    /// \return       nearest integer
    ///
    /// \ingroup arith_func_round
    FLY_API array round  (const array &in);

    /// C++ Interface to truncate numbers.
    ///
    /// \param[in] in input array
    /// \return       nearest integer not greater in magnitude than `in`
    /// 
    /// \ingroup arith_func_trunc
    FLY_API array trunc  (const array &in);

    /// C++ Interface to floor numbers.
    ///
    /// \param[in] in input array
    /// \return       nearest integer less than or equal to `in`
    ///
    /// \ingroup arith_func_floor
    FLY_API array floor  (const array &in);

    /// C++ Interface to ceil numbers.
    ///
    /// \param[in] in input array
    /// \return       nearest integer greater than or equal to `in`
    ///
    /// \ingroup arith_func_ceil
    FLY_API array ceil   (const array &in);

    /// \ingroup arith_func_hypot
    /// @{
    /// C++ Interface to calculate the length of the hypotenuse of two inputs.
    ///
    /// Calculates the hypotenuse of two inputs. The inputs can be both arrays
    /// or can be an array and a scalar.
    ///
    /// \param[in] lhs length of first side
    /// \param[in] rhs length of second side
    /// \return        length of the hypotenuse
    FLY_API array hypot  (const array &lhs, const array &rhs);

    /// \copydoc hypot(const array&, const array&)
    FLY_API array hypot  (const array &lhs, const double rhs);

    /// \copydoc hypot(const array&, const array&)
    FLY_API array hypot  (const double lhs, const array &rhs);
    /// @}

    /// C++ Interface to evaluate the sine function.
    ///
    /// \param[in] in input array
    /// \return       sine
    ///
    /// \ingroup arith_func_sin
    FLY_API array sin    (const array &in);

    /// C++ Interface to evaluate the cosine function.
    ///
    /// \param[in] in input array
    /// \return       cosine
    ///
    /// \ingroup arith_func_cos
    FLY_API array cos    (const array &in);

    /// C++ Interface to evaluate the tangent function.
    ///
    /// \param[in] in input array
    /// \return       tangent
    ///
    /// \ingroup arith_func_tan
    FLY_API array tan    (const array &in);

    /// C++ Interface to evaluate the inverse sine function.
    ///
    /// \param[in] in input array
    /// \return       inverse sine
    ///
    /// \ingroup arith_func_asin
    FLY_API array asin   (const array &in);

    /// C++ Interface to evaluate the inverse cosine function.
    ///
    /// \param[in] in input array
    /// \return       inverse cosine
    ///
    /// \ingroup arith_func_acos
    FLY_API array acos   (const array &in);

    /// C++ Interface to evaluate the inverse tangent function.
    ///
    /// \param[in] in input array
    /// \return       inverse tangent
    ///
    /// \ingroup arith_func_atan
    FLY_API array atan   (const array &in);

    /// \ingroup arith_func_atan
    /// @{
    /// C++ Interface to evaluate the inverse tangent of two arrays.
    ///
    /// \param[in] lhs value of numerator
    /// \param[in] rhs value of denominator
    /// \return        inverse tangent of the inputs
    FLY_API array atan2  (const array &lhs, const array &rhs);

    /// \copydoc atan2(const array&, const array&)
    FLY_API array atan2  (const array &lhs, const double rhs);

    /// \copydoc atan2(const array&, const array&)
    FLY_API array atan2  (const double lhs, const array &rhs);
    /// @}

    /// C++ Interface to evaluate the hyperbolic sine function.
    ///
    /// \param[in] in input array
    /// \return       hyperbolic sine
    ///
    /// \ingroup arith_func_sinh
    FLY_API array sinh(const array& in);

    /// C++ Interface to evaluate the hyperbolic cosine function.
    ///
    /// \param[in] in input array
    /// \return       hyperbolic cosine
    ///
    /// \ingroup arith_func_cosh
    FLY_API array cosh(const array& in);

    /// C++ Interface to evaluate the hyperbolic tangent function.
    ///
    /// \param[in] in input array
    /// \return       hyperbolic tangent
    ///
    /// \ingroup arith_func_tanh
    FLY_API array tanh(const array& in);

    /// C++ Interface to evaluate the inverse hyperbolic sine function.
    ///
    /// \param[in] in input array
    /// \return       inverse hyperbolic sine
    ///
    /// \ingroup arith_func_asinh
    FLY_API array asinh(const array& in);

    /// C++ Interface to evaluate the inverse hyperbolic cosine function.
    ///
    /// \param[in] in input array
    /// \return       inverse hyperbolic cosine
    ///
    /// \ingroup arith_func_acosh
    FLY_API array acosh(const array& in);

    /// C++ Interface to evaluate the inverse hyperbolic tangent function.
    ///
    /// \param[in] in input array
    /// \return       inverse hyperbolic tangent
    ///
    /// \ingroup arith_func_atanh
    FLY_API array atanh(const array& in);

    /// \ingroup arith_func_cplx
    /// @{
    /// C++ Interface to create a complex array from a single real array.
    ///
    /// \param[in] in input array
    /// \return       complex array
    FLY_API array complex(const array& in);
 
    /// C++ Interface to create a complex array from two real arrays.
    ///
    /// \param[in] real_ input array to be assigned as the real component of
    ///                  the returned complex array
    /// \param[in] imag_ input array to be assigned as the imaginary component
    ///                  of the returned complex array
    /// \return          complex array
    FLY_API array complex(const array &real_, const array &imag_);

    /// C++ Interface to create a complex array from a single real array for
    /// the real component and a single scalar for each imaginary component.
    ///
    /// \param[in] real_ input array to be assigned as the real component of
    ///                  the returned complex array
    /// \param[in] imag_ single scalar to be assigned as the imaginary
    ///                  component of each value of the returned complex array
    /// \return          complex array
    FLY_API array complex(const array &real_, const double imag_);

    /// C++ Interface to create a complex array from a single scalar for each
    /// real component and a single real array for the imaginary component.
    ///
    /// \param[in] real_ single scalar to be assigned as the real component of
    ///                  each value of the returned complex array
    /// \param[in] imag_ input array to be assigned as the imaginary component
    ///                  of the returned complex array
    /// \return          complex array
    FLY_API array complex(const double real_, const array &imag_);
    /// @}

    /// C++ Interface to return the real part of a complex array.
    ///
    /// \param[in] in input complex array
    /// \return       real part
    ///
    /// \ingroup arith_func_real
    FLY_API array real   (const array &in);

    /// C++ Interface to return the imaginary part of a complex array.
    ///
    /// \param[in] in input complex array
    /// \return       imaginary part
    ///
    /// \ingroup arith_func_imag
    FLY_API array imag   (const array &in);

    /// C++ Interface to calculate the complex conjugate of an input array.
    ///
    /// \param[in] in input complex array
    /// \return       complex conjugate
    ///
    /// \ingroup arith_func_conjg
    FLY_API array conjg  (const array &in);

    /// C++ Interface to evaluate the nth root.
    ///
    /// \param[in] nth_root nth root
    /// \param[in] value    value
    /// \return             `nth_root` th root of `value`
    ///
    /// \ingroup arith_func_root
    FLY_API array root    (const array &nth_root, const array &value);

    /// C++ Interface to evaluate the nth root.
    ///
    /// \param[in] nth_root nth root
    /// \param[in] value    value
    /// \return             `nth_root` th root of `value`
    ///
    /// \ingroup arith_func_root
    FLY_API array root    (const array &nth_root, const double value);

    /// C++ Interface to evaluate the nth root.
    ///
    /// \param[in] nth_root nth root
    /// \param[in] value    value
    /// \return             `nth_root` th root of `value`
    ///
    /// \ingroup arith_func_root
    FLY_API array root    (const double nth_root, const array &value);


    /// \ingroup arith_func_pow
    /// @{
    /// C++ Interface to raise a base to a power (or exponent).
    ///
    /// Computes the value of `base` raised to the power of `exponent`. The
    /// inputs can be two arrays or an array and a scalar.
    ///
    /// \param[in] base     base
    /// \param[in] exponent exponent
    /// \return             `base` raised to the power of `exponent`
    FLY_API array pow    (const array &base, const array &exponent);

    /// \copydoc pow(const array&, const array&)
    FLY_API array pow    (const array &base, const double exponent);

    /// \copydoc pow(const array&, const array&)
    FLY_API array pow    (const double base, const array &exponent);

    /// C++ Interface to raise 2 to a power (or exponent).
    ///
    /// \param[in] in power
    /// \return       2 raised to the power
    ///
    FLY_API array pow2    (const array &in);
    /// @}

    /// C++ Interface to evaluate the logistical sigmoid function.
    ///
    /// Computes \f$\frac{1}{1+e^{-x}}\f$.
    /// 
    /// \param[in] in input
    /// \return       sigmoid
    ///
    /// \ingroup arith_func_sigmoid
    FLY_API array sigmoid (const array &in);

    /// C++ Interface to evaluate the exponential.
    ///
    /// \param[in] in exponent
    /// \return       exponential
    ///
    /// \ingroup arith_func_exp
    FLY_API array exp    (const array &in);

    /// C++ Interface to evaluate the exponential of an array minus 1,
    /// `exp(in) - 1`.
    ///
    /// This function is useful when `in` is small.
    /// 
    /// \param[in] in exponent
    /// \return       exponential minus 1
    ///
    /// \ingroup arith_func_expm1
    FLY_API array expm1  (const array &in);

    /// C++ Interface to evaluate the error function.
    ///
    /// \param[in] in input array
    /// \return       error function
    ///
    /// \ingroup arith_func_erf
    FLY_API array erf    (const array &in);

    /// C++ Interface to evaluate the complementary error function.
    ///
    /// \param[in] in input array
    /// \return       complementary error function
    ///
    /// \ingroup arith_func_erfc
    FLY_API array erfc   (const array &in);

    /// C++ Interface to evaluate the natural logarithm.
    ///
    /// \param[in] in input array
    /// \return       natural logarithm
    ///
    /// \ingroup arith_func_log
    FLY_API array log    (const array &in);

    /// C++ Interface to evaluate the natural logarithm of 1 + input,
    /// `ln(1+in)`.
    /// 
    /// This function is useful when `in` is small.
    /// 
    /// \param[in] in input
    /// \return natural logarithm of `1 + input`
    ///
    /// \ingroup arith_func_log1p
    FLY_API array log1p  (const array &in);

    /// C++ Interface to evaluate the base 10 logarithm.
    ///
    /// \param[in] in input
    /// \return       base 10 logarithm
    ///
    /// \ingroup arith_func_log10
    FLY_API array log10  (const array &in);

    /// C++ Interface to evaluate the base 2 logarithm.
    ///
    /// \param[in] in input
    /// \return       base 2 logarithm
    ///
    /// \ingroup explog_func_log2
    FLY_API array log2   (const array &in);

    /// C++ Interface to evaluate the square root.
    ///
    /// \param[in] in input
    /// \return       square root
    ///
    /// \ingroup arith_func_sqrt
    FLY_API array sqrt   (const array &in);

    /// C++ Interface to evaluate the reciprocal square root.
    ///
    /// \param[in] in input
    /// \return       reciprocal square root
    ///
    /// \ingroup arith_func_rsqrt
    FLY_API array rsqrt   (const array &in);

    /// C++ Interface to evaluate the cube root.
    ///
    /// \param[in] in input
    /// \return       cube root
    ///
    /// \ingroup arith_func_cbrt
    FLY_API array cbrt   (const array &in);

    /// C++ Interface to calculate the factorial.
    ///
    /// \param[in] in input
    /// \return       factorial
    ///
    /// \ingroup arith_func_factorial
    FLY_API array factorial (const array &in);

    /// C++ Interface to evaluate the gamma function.
    ///
    /// \param[in] in input
    /// \return       gamma function
    ///
    /// \ingroup arith_func_tgamma
    FLY_API array tgamma (const array &in);

    /// C++ Interface to evaluate the logarithm of the absolute value of the
    /// gamma function.
    ///
    /// \param[in] in input
    /// \return       logarithm of the absolute value of the gamma function
    ///
    /// \ingroup arith_func_lgamma
    FLY_API array lgamma (const array &in);

    /// C++ Interface to check which values are zero.
    ///
    /// \param[in] in input
    /// \return       array containing 1's where input is 0; 0's otherwise
    ///
    /// \ingroup arith_func_iszero
    FLY_API array iszero (const array &in);

    /// C++ Interface to check if values are infinite.
    ///
    /// \param[in] in input
    /// \return       array containing 1's where input is Inf or -Inf; 0's
    ///               otherwise
    ///
    /// \ingroup arith_func_isinf
    FLY_API array isInf  (const array &in);

    /// C++ Interface to check if values are NaN.
    ///
    /// \param[in] in input
    /// \return       array containing 1's where input is NaN; 0's otherwise
    ///
    /// \ingroup arith_func_isnan
    FLY_API array isNaN  (const array &in);
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    /**
       C Interface to add two arrays.

       \param[out] out   +
       \param[in]  lhs   first input
       \param[in]  rhs   second input
       \param[in]  batch batch mode
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_add
    */
    FLY_API fly_err fly_add   (fly_array *out, const fly_array lhs, const fly_array rhs, const bool batch);

    /**
       C Interface to subtract one array from another array.

       \param[out] out   -
       \param[in]  lhs   first input
       \param[in]  rhs   second input
       \param[in]  batch batch mode
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_sub
    */
    FLY_API fly_err fly_sub   (fly_array *out, const fly_array lhs, const fly_array rhs, const bool batch);

    /**
       C Interface to multiply two arrays.

       \param[out] out   *
       \param[in]  lhs   first input
       \param[in]  rhs   second input
       \param[in]  batch batch mode
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_mul
    */
    FLY_API fly_err fly_mul   (fly_array *out, const fly_array lhs, const fly_array rhs, const bool batch);

    /**
       C Interface to divide one array by another array.

       \param[out] out   \
       \param[in]  lhs   first input
       \param[in]  rhs   second input
       \param[in]  batch batch mode
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_div
    */
    FLY_API fly_err fly_div   (fly_array *out, const fly_array lhs, const fly_array rhs, const bool batch);

    /**
       C Interface to perform a less-than comparison between corresponding
       elements of two arrays.

       Output type is b8.

       \param[out] out   1's where `lhs < rhs`, else 0's
       \param[in]  lhs   first input
       \param[in]  rhs   second input
       \param[in]  batch batch mode
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup logic_func_lt
    */
    FLY_API fly_err fly_lt    (fly_array *out, const fly_array lhs, const fly_array rhs, const bool batch);

    /**
       C Interface to perform a greater-than comparison between corresponding
       elements of two arrays.

       Output type is b8.

       \param[out] out   1's where `lhs > rhs`, else 0's
       \param[in]  lhs   first input
       \param[in]  rhs   second input
       \param[in]  batch batch mode
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_gt
    */
    FLY_API fly_err fly_gt    (fly_array *out, const fly_array lhs, const fly_array rhs, const bool batch);

    /**
       C Interface to perform a less-than-or-equal comparison between
       corresponding elements of two arrays.

       Output type is b8.

       \param[out] out   1's where `lhs <= rhs`, else 0's
       \param[in]  lhs   first input
       \param[in]  rhs   second input
       \param[in]  batch batch mode
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_le
    */
    FLY_API fly_err fly_le    (fly_array *out, const fly_array lhs, const fly_array rhs, const bool batch);

    /**
       C Interface to perform a greater-than-or-equal comparison between
       corresponding elements of two arrays.

       Output type is b8.

       \param[out] out   1's where `lhs >= rhs`, else 0's
       \param[in]  lhs   first input
       \param[in]  rhs   second input
       \param[in]  batch batch mode
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_ge
    */
    FLY_API fly_err fly_ge    (fly_array *out, const fly_array lhs, const fly_array rhs, const bool batch);

    /**
       C Interface to check if corresponding elements of two arrays are equal.

       Output type is b8.

       \param[out] out   1's where `lhs == rhs`, else 0's
       \param[in]  lhs   first input
       \param[in]  rhs   second input
       \param[in]  batch batch mode
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_eq
    */
    FLY_API fly_err fly_eq    (fly_array *out, const fly_array lhs, const fly_array rhs, const bool batch);

    /**
       C Interface to check if corresponding elements of two arrays are not
       equal.

       Output type is b8.

       \param[out] out   1's where `lhs != rhs`, else 0's
       \param[in]  lhs   first input
       \param[in]  rhs   second input
       \param[in]  batch batch mode
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_neq
    */
    FLY_API fly_err fly_neq   (fly_array *out, const fly_array lhs, const fly_array rhs, const bool batch);

    /**
       C Interface to evaluate the logical AND of two arrays.

       Output type is b8.

       \param[out] out   1's where `lhs && rhs`, else 0's
       \param[in]  lhs   first input
       \param[in]  rhs   second input
       \param[in]  batch batch mode
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_and
    */
    FLY_API fly_err fly_and   (fly_array *out, const fly_array lhs, const fly_array rhs, const bool batch);

    /**
       C Interface the evaluate the logical OR of two arrays.

       Output type is b8.

       \param[out] out   1's where `lhs || rhs`, else 0's
       \param[in]  lhs   first input
       \param[in]  rhs   second input
       \param[in]  batch batch mode
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_or
    */
    FLY_API fly_err fly_or    (fly_array *out, const fly_array lhs, const fly_array rhs, const bool batch);

    /**
       C Interface to evaluate the logical NOT of an array.

       Output type is b8.

       \param[out] out !, logical NOT
       \param[in]  in  input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_not
    */
    FLY_API fly_err fly_not   (fly_array *out, const fly_array in);

    /**
       C Interface to evaluate the bitwise NOT of an array.

       \param[out] out ~, bitwise NOT
       \param[in]  in  input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_bitnot
    */
    FLY_API fly_err fly_bitnot   (fly_array *out, const fly_array in);

    /**
       C Interface to evaluate the bitwise AND of two arrays.

       \param[out] out   &, bitwise AND
       \param[in]  lhs   first input
       \param[in]  rhs   second input
       \param[in]  batch batch mode
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_bitand
    */
    FLY_API fly_err fly_bitand   (fly_array *out, const fly_array lhs, const fly_array rhs, const bool batch);

    /**
       C Interface to evaluate the bitwise OR of two arrays.

       \param[out] out   |, bitwise OR
       \param[in]  lhs   first input
       \param[in]  rhs   second input
       \param[in]  batch batch mode
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_bitor
    */
    FLY_API fly_err fly_bitor    (fly_array *out, const fly_array lhs, const fly_array rhs, const bool batch);

    /**
       C Interface to evaluate the bitwise XOR of two arrays.

       \param[out] out   ^, bitwise XOR
       \param[in]  lhs   first input
       \param[in]  rhs   second input
       \param[in]  batch batch mode
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_bitxor
    */
    FLY_API fly_err fly_bitxor   (fly_array *out, const fly_array lhs, const fly_array rhs, const bool batch);

    /**
       C Interface to shift the bits of integer arrays left.

       \param[out] out   left shift
       \param[in]  lhs   values to shift
       \param[in]  rhs   n bits to shift
       \param[in]  batch batch mode
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_shiftl
    */
    FLY_API fly_err fly_bitshiftl(fly_array *out, const fly_array lhs, const fly_array rhs, const bool batch);

    /**
       C Interface to shift the bits of integer arrays right.

       \param[out] out   right shift
       \param[in]  lhs   values to shift
       \param[in]  rhs   n bits to shift
       \param[in]  batch batch mode
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_shiftr
    */
    FLY_API fly_err fly_bitshiftr(fly_array *out, const fly_array lhs, const fly_array rhs, const bool batch);

    /**
       C Interface to cast an array from one type to another.

       This function casts an fly_array object from one type to another. If the
       type of the original array is the same as `type` then the same array is
       returned.

       Consecutive casting operations may be may be optimized out if the
       original type of the fly_array is the same as the final type. For example
       if the original type is f64, which is cast to f32 and then back to
       f64, then the cast to f32 is skipped and that operation will *NOT*
       be performed by Flare. The following table shows which casts will
       be optimized out. outer -> inner -> outer

       | inner-> | f32 | f64 | c32 | c64 | s32 | u32 | u8 | b8 | s64 | u64 | s16 | u16 | f16 |
       |---------|-----|-----|-----|-----|-----|-----|----|----|-----|-----|-----|-----|-----|
       | f32     | x   | x   | x   | x   |     |     |    |    |     |     |     |     | x   |
       | f64     | x   | x   | x   | x   |     |     |    |    |     |     |     |     | x   |
       | c32     | x   | x   | x   | x   |     |     |    |    |     |     |     |     | x   |
       | c64     | x   | x   | x   | x   |     |     |    |    |     |     |     |     | x   |
       | s32     | x   | x   | x   | x   | x   | x   |    |    | x   | x   |     |     | x   |
       | u32     | x   | x   | x   | x   | x   | x   |    |    | x   | x   |     |     | x   |
       | u8      | x   | x   | x   | x   | x   | x   | x  | x  | x   | x   | x   | x   | x   |
       | b8      | x   | x   | x   | x   | x   | x   | x  | x  | x   | x   | x   | x   | x   |
       | s64     | x   | x   | x   | x   |     |     |    |    | x   | x   |     |     | x   |
       | u64     | x   | x   | x   | x   |     |     |    |    | x   | x   |     |     | x   |
       | s16     | x   | x   | x   | x   | x   | x   |    |    | x   | x   | x   | x   | x   |
       | u16     | x   | x   | x   | x   | x   | x   |    |    | x   | x   | x   | x   | x   |
       | f16     | x   | x   | x   | x   |     |     |    |    |     |     |     |     | x   |

       If you want to avoid this behavior use, fly_eval after the first cast
       operation. This will ensure that the cast operation is performed on the
       fly_array.

       \param[out] out  values in the specified type
       \param[in]  in   input
       \param[in]  type target data type \ref fly_dtype
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_cast
    */
    FLY_API fly_err fly_cast    (fly_array *out, const fly_array in, const fly_dtype type);

    /**
       C Interface to find the elementwise minimum between two arrays.

       \param[out] out   minimum
       \param[in]  lhs   input array
       \param[in]  rhs   input array
       \param[in]  batch batch mode
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_min
    */
    FLY_API fly_err fly_minof (fly_array *out, const fly_array lhs, const fly_array rhs, const bool batch);

    /**
       C Interface to find the elementwise minimum between an array and a
       scalar value.

       \param[out] out   maximum
       \param[in]  lhs   input array
       \param[in]  rhs   input array
       \param[in]  batch batch mode
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_max
    */
    FLY_API fly_err fly_maxof (fly_array *out, const fly_array lhs, const fly_array rhs, const bool batch);

    /**
       C Interface to clamp an array between an upper and a lower limit.

       \param[out] out   clamped array
       \param[in]  in    input array
       \param[in]  lo    lower limit array
       \param[in]  hi    upper limit array
       \param[in]  batch batch mode
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_clamp
    */
    FLY_API fly_err fly_clamp(fly_array *out, const fly_array in,
                          const fly_array lo, const fly_array hi, const bool batch);

    /**
       C Interface to calculate the remainder.

       \param[out] out   remainder
       \param[in]  lhs   numerator
       \param[in]  rhs   denominator
       \param[in]  batch batch mode
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_rem
    */
    FLY_API fly_err fly_rem   (fly_array *out, const fly_array lhs, const fly_array rhs, const bool batch);

    /**
       C Interface to calculate the modulus.

       \param[out] out   modulus
       \param[in]  lhs   dividend
       \param[in]  rhs   divisor
       \param[in]  batch batch mode
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_mod
    */
    FLY_API fly_err fly_mod   (fly_array *out, const fly_array lhs, const fly_array rhs, const bool batch);

    /**
       C Interface to calculate the absolute value.

       \param[out] out absolute value
       \param[in]  in  input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_abs
    */
    FLY_API fly_err fly_abs     (fly_array *out, const fly_array in);

    /**
       C Interface to calculate the phase angle (in radians) of a complex
       array.

       \param[out] out phase angle (in radians)
       \param[in]  in  input array, typically complex
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_arg
    */
    FLY_API fly_err fly_arg     (fly_array *out, const fly_array in);

    /**
       C Interface to calculate the sign of elements in an array.

       \param[out] out array containing 1's for negative values; 0's otherwise
       \param[in]  in  input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_sign
    */
    FLY_API fly_err fly_sign   (fly_array *out, const fly_array in);

    /**
       C Interface to round numbers.

       \param[out] out nearest integer
       \param[in]  in  input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_round
    */
    FLY_API fly_err fly_round   (fly_array *out, const fly_array in);

    /**
       C Interface to truncate numbers.

       \param[out] out nearest integer not greater in magnitude than `in`
       \param[in]  in  input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_trunc
    */
    FLY_API fly_err fly_trunc   (fly_array *out, const fly_array in);

    /**
       C Interface to floor numbers.

       \param[out] out nearest integer less than or equal to `in`
       \param[in]  in  input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_floor
    */
    FLY_API fly_err fly_floor   (fly_array *out, const fly_array in);

    /**
       C Interface to ceil numbers.

       \param[out] out nearest integer greater than or equal to `in`
       \param[in]  in  input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_ceil
    */
    FLY_API fly_err fly_ceil    (fly_array *out, const fly_array in);

    /**
       C Interface to calculate the length of the hypotenuse of two inputs.

       \param[out] out   length of the hypotenuse
       \param[in]  lhs   length of first side
       \param[in]  rhs   length of second side
       \param[in]  batch batch mode
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_floor
    */
    FLY_API fly_err fly_hypot (fly_array *out, const fly_array lhs, const fly_array rhs, const bool batch);

    /**
       C Interface to evaluate the sine function.

       \param[out] out sine
       \param[in]  in  input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_sin
    */
    FLY_API fly_err fly_sin     (fly_array *out, const fly_array in);

    /**
       C Interface to evaluate the cosine function.

       \param[out] out cosine
       \param[in]  in  input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_cos
    */
    FLY_API fly_err fly_cos     (fly_array *out, const fly_array in);

    /**
       C Interface to evaluate the tangent function.

       \param[out] out tangent
       \param[in]  in  input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_tan
    */
    FLY_API fly_err fly_tan     (fly_array *out, const fly_array in);

    /**
       C Interface to evaluate the inverse sine function.

       \param[out] out inverse sine
       \param[in]  in  input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_asin
    */
    FLY_API fly_err fly_asin    (fly_array *out, const fly_array in);

    /**
       C Interface to evaluate the inverse cosine function.

       \param[out] out inverse cos
       \param[in]  in  input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_acos
    */
    FLY_API fly_err fly_acos    (fly_array *out, const fly_array in);

    /**
       C Interface to evaluate the inverse tangent function.

       \param[out] out inverse tangent
       \param[in]  in  input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_atan
    */
    FLY_API fly_err fly_atan    (fly_array *out, const fly_array in);

    /**
       C Interface to evaluate the inverse tangent of two arrays.

       \param[out] out   inverse tangent of two arrays
       \param[in]  lhs   numerator
       \param[in]  rhs   denominator
       \param[in]  batch batch mode
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_atan
    */
    FLY_API fly_err fly_atan2 (fly_array *out, const fly_array lhs, const fly_array rhs, const bool batch);

    /**
       C Interface to evaluate the hyperbolic sine function.

       \param[out] out hyperbolic sine
       \param[in]  in  input
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_sinh
    */
    FLY_API fly_err fly_sinh    (fly_array *out, const fly_array in);

    /**
       C Interface to evaluate the hyperbolic cosine function.

       \param[out] out hyperbolic cosine
       \param[in]  in  input
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_cosh
    */
    FLY_API fly_err fly_cosh    (fly_array *out, const fly_array in);

    /**
       C Interface to evaluate the hyperbolic tangent function.

       \param[out] out hyperbolic tangent
       \param[in]  in  input
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_tanh
    */
    FLY_API fly_err fly_tanh    (fly_array *out, const fly_array in);

    /**
       C Interface to evaluate the inverse hyperbolic sine function.

       \param[out] out inverse hyperbolic sine
       \param[in]  in  input
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_asinh
    */
    FLY_API fly_err fly_asinh   (fly_array *out, const fly_array in);

    /**
       C Interface to evaluate the inverse hyperbolic cosine function.

       \param[out] out inverse hyperbolic cosine
       \param[in]  in  input
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_acosh
    */
    FLY_API fly_err fly_acosh   (fly_array *out, const fly_array in);

    /**
       C Interface to evaluate the inverse hyperbolic tangent function.

       \param[out] out inverse hyperbolic tangent
       \param[in]  in  input
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_atanh
    */
    FLY_API fly_err fly_atanh   (fly_array *out, const fly_array in);

    /**
       C Interface to create a complex array from a single real array.

       \param[out] out complex array
       \param[in]  in  real array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_cplx
    */
    FLY_API fly_err fly_cplx(fly_array* out, const fly_array in);

    /**
       C Interface to create a complex array from two real arrays.

       \param[out] out   complex array
       \param[in]  real  real array to be assigned as the real component of the
                         returned complex array
       \param[in]  imag  real array to be assigned as the imaginary component
                         of the returned complex array
       \param[in]  batch batch mode
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_cplx
    */
    FLY_API fly_err fly_cplx2(fly_array* out, const fly_array real, const fly_array imag, const bool batch);

    /**
       C Interface to return the real part of a complex array.

       \param[out] out real part
       \param[in]  in  complex array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_real
    */
    FLY_API fly_err fly_real(fly_array* out, const fly_array in);

    /**
       C Interface to return the imaginary part of a complex array.

       \param[out] out imaginary part
       \param[in]  in  complex array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_imag
    */
    FLY_API fly_err fly_imag(fly_array* out, const fly_array in);

    /**
       C Interface to evaluate the complex conjugate of an input array.

       \param[out] out complex conjugate
       \param[in]  in  complex array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_conjg
    */
    FLY_API fly_err fly_conjg(fly_array* out, const fly_array in);

    /**
       C Interface to evaluate the nth root.

       \param[out] out   `lhs` th root of `rhs`
       \param[in]  lhs   nth root
       \param[in]  rhs   value
       \param[in]  batch batch mode
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_root
    */
    FLY_API fly_err fly_root   (fly_array *out, const fly_array lhs, const fly_array rhs, const bool batch);


    /**
       C Interface to raise a base to a power (or exponent).

       \param[out] out   `lhs` raised to the power of `rhs`
       \param[in]  lhs   base
       \param[in]  rhs   exponent
       \param[in]  batch batch mode
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_pow
    */
    FLY_API fly_err fly_pow   (fly_array *out, const fly_array lhs, const fly_array rhs, const bool batch);

    /**
       C Interface to raise 2 to a power (or exponent).

       \param[out] out 2 raised to the power of `in`
       \param[in]  in  exponent
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_pow2
    */
    FLY_API fly_err fly_pow2     (fly_array *out, const fly_array in);

    /**
       C Interface to evaluate the logistical sigmoid function.

       Computes \f$\frac{1}{1+e^{-x}}\f$.

       \param[out] out output of the logistic sigmoid function
       \param[in]  in  input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_sigmoid
    */
    FLY_API fly_err fly_sigmoid(fly_array* out, const fly_array in);

    /**
       C Interface to evaluate the exponential.

       \param[out] out e raised to the power of `in`
       \param[in]  in  exponent
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_exp
    */
    FLY_API fly_err fly_exp     (fly_array *out, const fly_array in);

    /**
       C Interface to evaluate the exponential of an array minus 1,
       `exp(in) - 1`.

       \param[out] out exponential of `in - 1`
       \param[in]  in  input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_expm1
    */
    FLY_API fly_err fly_expm1   (fly_array *out, const fly_array in);

    /**
       C Interface to evaluate the error function.

       \param[out] out error function value
       \param[in]  in  input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_erf
    */
    FLY_API fly_err fly_erf     (fly_array *out, const fly_array in);

    /**
       C Interface to evaluate the complementary error function.

       \param[out] out complementary error function
       \param[in]  in  input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_erfc
    */
    FLY_API fly_err fly_erfc    (fly_array *out, const fly_array in);

    /**
       C Interface to evaluate the natural logarithm.

       \param[out] out natural logarithm
       \param[in]  in  input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_log
    */
    FLY_API fly_err fly_log     (fly_array *out, const fly_array in);

    /**
       C Interface to evaluate the natural logarithm of 1 + input, `ln(1+in)`.

       \param[out] out logarithm of `in + 1`
       \param[in]  in  input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_log1p
    */
    FLY_API fly_err fly_log1p   (fly_array *out, const fly_array in);

    /**
       C Interface to evaluate the base 10 logarithm.

       \param[out] out base 10 logarithm
       \param[in]  in  input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_log10
    */
    FLY_API fly_err fly_log10   (fly_array *out, const fly_array in);

    /**
       C Interface to evaluate the base 2 logarithm.

       \param[out] out base 2 logarithm
       \param[in]  in  input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup explog_func_log2
    */
    FLY_API fly_err fly_log2   (fly_array *out, const fly_array in);

    /**
       C Interface to evaluate the square root.

       \param[out] out square root
       \param[in]  in  input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_sqrt
    */
    FLY_API fly_err fly_sqrt    (fly_array *out, const fly_array in);

    /**
      C Interface to evaluate the reciprocal square root.

      \param[out] out reciprocal square root
      \param[in]  in  input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

      \ingroup arith_func_rsqrt
    */
    FLY_API fly_err fly_rsqrt    (fly_array *out, const fly_array in);
    /**
       C Interface to evaluate the cube root.

       \param[out] out cube root
       \param[in]  in  input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_cbrt
    */
    FLY_API fly_err fly_cbrt    (fly_array *out, const fly_array in);

    /**
       C Interface to calculate the factorial.

       \param[out] out factorial
       \param[in]  in  input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_factorial
    */
    FLY_API fly_err fly_factorial   (fly_array *out, const fly_array in);

    /**
       C Interface to evaluate the gamma function.

       \param[out] out gamma function
       \param[in]  in  input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_tgamma
    */
    FLY_API fly_err fly_tgamma   (fly_array *out, const fly_array in);

    /**
       C Interface to evaluate the logarithm of the absolute value of the
       gamma function.

       \param[out] out logarithm of the absolute value of the gamma function
       \param[in]  in  input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_lgamma
    */
    FLY_API fly_err fly_lgamma   (fly_array *out, const fly_array in);

    /**
       C Interface to check if values are zero.

       \param[out] out array containing 1's where input is 0; 0's otherwise
       \param[in]  in  input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_iszero
    */
    FLY_API fly_err fly_iszero  (fly_array *out, const fly_array in);

    /**
       C Interface to check if values are infinite.

       \param[out] out array containing 1's where input is Inf or -Inf; 0's
                       otherwise
       \param[in]  in  input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_isinf
    */
    FLY_API fly_err fly_isinf   (fly_array *out, const fly_array in);

    /**
       C Interface to check if values are NaN.

       \param[out] out array containing 1's where input is NaN; 0's otherwise
       \param[in]  in  input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup arith_func_isnan
    */
    FLY_API fly_err fly_isnan   (fly_array *out, const fly_array in);

#ifdef __cplusplus
}
#endif
