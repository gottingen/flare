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

#pragma once
#include <fly/defines.h>

#ifdef __cplusplus
namespace fly
{
class array;

/**
   C++ Interface for mean

   \param[in] in is the input array
   \param[in] dim the dimension along which the mean is extracted
   \return    the mean of the input array along dimension \p dim

   \ingroup stat_func_mean

   \note \p dim is -1 by default. -1 denotes the first non-singleton dimension.
*/
FLY_API array mean(const array& in, const dim_t dim=-1);

/**
   C++ Interface for mean of weighted inputs

   \param[in] in is the input array
   \param[in] weights is used to scale input \p in before getting mean
   \param[in] dim the dimension along which the mean is extracted
   \return    the mean of the weighted input array along dimension \p dim

   \ingroup stat_func_mean

   \note \p dim is -1 by default. -1 denotes the first non-singleton dimension.
*/
FLY_API array mean(const array& in, const array& weights, const dim_t dim=-1);

/**
   C++ Interface for variance

   \param[in] in is the input array
   \param[in] isbiased is boolean denoting Population variance (false) or Sample
              Variance (true)
   \param[in] dim the dimension along which the variance is extracted
   \return the variance of the input array along dimension \p dim

   \ingroup stat_func_var

   \note \p dim is -1 by default. -1 denotes the first non-singleton dimension.

   \deprecated Use \ref fly::var that takes \ref fly_var_bias instead
*/
FLY_DEPRECATED("Use \ref fly::var(const array&, const fly_var_bias, const dim_t)")
FLY_API array var(const array& in, const bool isbiased=false, const dim_t dim=-1);

/**
   C++ Interface for variance

   \param[in] in is the input array
   \param[in] bias The type of bias used for variance calculation. Takes o
              value of type \ref fly_var_bias.
   \param[in] dim the dimension along which the variance is extracted
   \return the variance of the input array along dimension \p dim

   \ingroup stat_func_var

   \note \p dim is -1 by default. -1 denotes the first non-singleton dimension.
*/
FLY_API array var(const array &in, const fly_var_bias bias, const dim_t dim = -1);

/**
   C++ Interface for variance of weighted inputs

   \param[in] in is the input array
   \param[in] weights is used to scale input \p in before getting variance
   \param[in] dim the dimension along which the variance is extracted
   \return    the variance of the weighted input array along dimension \p dim

   \ingroup stat_func_var

   \note \p dim is -1 by default. -1 denotes the first non-singleton dimension.
*/
FLY_API array var(const array& in, const array &weights, const dim_t dim=-1);

/**
   C++ Interface for mean and variance

   \param[out] mean     The mean of the input array along \p dim dimension
   \param[out] var      The variance of the input array along the \p dim dimension
   \param[in]  in       The input array
   \param[in]  weights  The weights to scale the input array before calculating
                        the mean and varience. If empty, the input is not scaled
   \param[in] bias      The type of bias used for variance calculation
   \param[in] dim       The dimension along which the variance and mean are
                        calculated. Default is -1 meaning the first non-zero dim
  */
FLY_API void meanvar(array& mean, array& var, const array& in, const array& weights,
                   const fly_var_bias bias = FLY_VARIANCE_POPULATION, const dim_t dim=-1);

/**
   C++ Interface for standard deviation

   \param[in] in is the input array
   \param[in] dim the dimension along which the standard deviation is extracted
   \return    the standard deviation of the input array along dimension \p dim

   \ingroup stat_func_stdev

   \note \p dim is -1 by default. -1 denotes the first non-singleton dimension.

   \deprecated Use \ref fly::stdev that takes \ref fly_var_bias instead
*/
FLY_DEPRECATED("Use fly::stdev(const array&, const fly_var_bias, const dim_t)")
FLY_API array stdev(const array& in, const dim_t dim=-1);

/**
   C++ Interface for standard deviation

   \param[in] in is the input array
   \param[in] bias The type of bias used for variance calculation. Takes of
              value of type \ref fly_var_bias.
   \param[in] dim the dimension along which the standard deviation is extracted
   \return    the standard deviation of the input array along dimension \p dim

   \ingroup stat_func_stdev

   \note \p dim is -1 by default. -1 denotes the first non-singleton dimension.
*/
FLY_API array stdev(const array &in, const fly_var_bias bias,
                  const dim_t dim = -1);

/**
   C++ Interface for covariance

   \param[in] X is the first input array
   \param[in] Y is the second input array
   \param[in] isbiased is boolean specifying if biased estimate should be
              taken (default: false)
   \return    the covariance of the input arrays

   \ingroup stat_func_cov

   \deprecated Use fly::cov(const array&, const array& const fly_var_bias)
*/
FLY_DEPRECATED("Use fly::cov(const fly::array&, const array&, conv fly_var_bias)")
FLY_API array cov(const array& X, const array& Y, const bool isbiased=false);

/**
   C++ Interface for covariance

   \param[in] X is the first input array
   \param[in] Y is the second input array
   \param[in] bias The type of bias used for variance calculation. Takes of
              value of type \ref fly_var_bias.
   \return the covariance of the input arrays

   \ingroup stat_func_cov
*/
FLY_API array cov(const array &X, const array &Y, const fly_var_bias bias);

/**
   C++ Interface for median

   \param[in] in is the input array
   \param[in] dim the dimension along which the median is extracted
   \return    the median of the input array along dimension \p dim

   \ingroup stat_func_median

   \note \p dim is -1 by default. -1 denotes the first non-singleton dimension.
*/
FLY_API array median(const array& in, const dim_t dim=-1);

/**
   C++ Interface for mean of all elements

   \param[in] in is the input array
   \return    mean of the entire input array

   \ingroup stat_func_mean
*/
template<typename T>
FLY_API T mean(const array& in);

/**
   C++ Interface for mean of all elements in weighted input

   \param[in] in is the input array
   \param[in] weights  is used to scale input \p in before getting mean
   \return    mean of the entire weighted input array

   \ingroup stat_func_mean
*/
template<typename T>
FLY_API T mean(const array& in, const array& weights);

/**
   C++ Interface for variance of all elements

   \param[in] in is the input array
   \param[in] isbiased is boolean denoting Population variance (false) or Sample
              Variance (true)
   \return variance of the entire input array

   \ingroup stat_func_var

   \deprecated Use \ref fly::var that takes \ref fly_var_bias instead
*/
template <typename T>
FLY_DEPRECATED("Use fly::var(const fly::array&, const fly_var_bias)")
FLY_API T var(const array &in, const bool isbiased = false);

/**
   C++ Interface for variance of all elements

   \param[in] in is the input array
   \param[in] bias The type of bias used for variance calculation. Takes of
              value of type \ref fly_var_bias.
   \return variance of the \p in array

   \ingroup stat_func_var
*/
template <typename T> FLY_API T var(const array &in, const fly_var_bias bias);

/**
   C++ Interface for variance of all elements in weighted input

   \param[in] in is the input array
   \param[in] weights  is used to scale input \p in before getting variance
   \return    variance of the entire input array

   \ingroup stat_func_var
*/
template<typename T>
FLY_API T var(const array& in, const array& weights);

/**
   C++ Interface for standard deviation of all elements

   \param[in] in is the input array
   \return    standard deviation of the entire input array

   \ingroup stat_func_stdev

   \deprecated Use \ref fly::stdev that takes \ref fly_var_bias instead
*/
template <typename T>
FLY_DEPRECATED("Use fly::stdev(const array&, const fly_var_bias)")
FLY_API T stdev(const array &in);

/**
   C++ Interface for standard deviation of all elements

   \param[in] in is the input array
   \param[in] bias The type of bias used for variance calculation. Takes of
              value of type \ref fly_var_bias.
   \return    standard deviation of the entire input array

   \ingroup stat_func_stdev
*/
template <typename T> FLY_API T stdev(const array &in, const fly_var_bias bias);

/**
   C++ Interface for median of all elements

   \param[in] in is the input array
   \return    median of the entire input array

   \ingroup stat_func_median
*/
template<typename T>
FLY_API T median(const array& in);

/**
   C++ Interface for correlation coefficient

   \param[in] X is the first input array
   \param[in] Y is the second input array
   \return    correlation coefficient of the input arrays

   \note There are many ways correlation coefficient is calculated. This algorithm returns Pearson product-moment correlation coefficient.

   \ingroup stat_func_corrcoef
*/
template<typename T>
FLY_API T corrcoef(const array& X, const array& Y);

/**
   C++ Interface for finding top k elements along a given dimension

   \param[out] values  The values of the top k elements along the \p dim dimension
   \param[out] indices The indices of the top k elements along the \p dim dimension
   \param[in]  in      Input \ref fly::array with at least \p k elements along
                       \p dim
   \param[in]  k       The number of elements to be retriefed along the \p dim dimension
   \param[in]  dim     The dimension along which top k elements are extracted.
                       (Must be 0)
   \param[in]  order   If Descending the highest values are returned. Otherwise
                       the lowest values are returned

   \note{This function is optimized for small values of k.}
   \note{The order of the returned keys may not be in the same order as the
   appear in the input array, for a stable topk, set the FLY_TOPK_STABLE flag
   in the order param. These are equivalent to FLY_TOPK_STABLE_MAX and FLY_TOPK_STABLE_MIN}
   \ingroup stat_func_topk
*/
FLY_API void topk(array &values, array &indices, const array& in, const int k,
                const int dim = -1, const topkFunction order = FLY_TOPK_MAX);
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
   C Interface for mean

   \param[out] out will contain the mean of the input array along dimension \p dim
   \param[in] in is the input array
   \param[in] dim the dimension along which the mean is extracted
   \return     \ref FLY_SUCCESS if the operation is successful,
   otherwise an appropriate error code is returned.

   \ingroup stat_func_mean
*/
FLY_API fly_err fly_mean(fly_array *out, const fly_array in, const dim_t dim);

/**
   C Interface for mean of weighted input array

   \param[out] out will contain the mean of the input array along dimension \p dim
   \param[in] in is the input array
   \param[in] weights is used to scale input \p in before getting mean
   \param[in] dim the dimension along which the mean is extracted
   \return     \ref FLY_SUCCESS if the operation is successful,
   otherwise an appropriate error code is returned.

   \ingroup stat_func_mean
*/
FLY_API fly_err fly_mean_weighted(fly_array *out, const fly_array in, const fly_array weights, const dim_t dim);

/**
   C Interface for variance

   \param[out] out will contain the variance of the input array along dimension \p dim
   \param[in] in is the input array
   \param[in] isbiased is boolean denoting Population variance (false) or Sample Variance (true)
   \param[in] dim the dimension along which the variance is extracted
   \return     \ref FLY_SUCCESS if the operation is successful,
   otherwise an appropriate error code is returned.

   \ingroup stat_func_var

   \deprecated Use \ref fly_var_v2 instead
*/
FLY_DEPRECATED("Use fly_var_v2")
FLY_API fly_err fly_var(fly_array *out, const fly_array in, const bool isbiased, const dim_t dim);

/**
   C Interface for variance

   \param[out] out will contain the variance of the input array along dimension
               \p dim
   \param[in] in is the input array
   \param[in] bias The type of bias used for variance calculation. Takes of
              value of type \ref fly_var_bias
   \param[in] dim the dimension along which the variance is extracted
   \return \ref FLY_SUCCESS if the operation is successful, otherwise an
           appropriate error code is returned.

   \ingroup stat_func_var

*/
FLY_API fly_err fly_var_v2(fly_array *out, const fly_array in, const fly_var_bias bias,
                       const dim_t dim);

/**
   C Interface for variance of weighted input array

   \param[out] out will contain the variance of the input array along dimension \p dim
   \param[in] in is the input array
   \param[in] weights is used to scale input \p in before getting variance
   \param[in] dim the dimension along which the variance is extracted
   \return     \ref FLY_SUCCESS if the operation is successful,
   otherwise an appropriate error code is returned.

   \ingroup stat_func_var

*/
FLY_API fly_err fly_var_weighted(fly_array *out, const fly_array in, const fly_array weights, const dim_t dim);

/**
   C Interface for mean and variance

   \param[out] mean     The mean of the input array along \p dim dimension
   \param[out] var      The variance of the input array along the \p dim dimension
   \param[in]  in       The input array
   \param[in]  weights  The weights to scale the input array before calculating
                        the mean and varience. If empty, the input is not scaled
   \param[in]  bias     The type of bias used for variance calculation
   \param[in]  dim      The dimension along which the variance and mean are
                        calculated. Default is -1 meaning the first non-zero dim
  */
FLY_API fly_err fly_meanvar(fly_array *mean, fly_array *var, const fly_array in,
                        const fly_array weights, const fly_var_bias bias, const dim_t dim);

/**
   C Interface for standard deviation

   \param[out] out will contain the standard deviation of the input array along dimension \p dim
   \param[in] in is the input array
   \param[in] dim the dimension along which the standard deviation is extracted
   \return     \ref FLY_SUCCESS if the operation is successful,
   otherwise an appropriate error code is returned.

   \ingroup stat_func_stdev

   \deprecated Use \ref fly_stdev_v2 instead
*/
FLY_DEPRECATED("Use fly_stdev_v2")
FLY_API fly_err fly_stdev(fly_array *out, const fly_array in, const dim_t dim);

/**
   C Interface for standard deviation

   \param[out] out will contain the standard deviation of the input array along
               dimension \p dim
   \param[in] in is the input array
   \param[in] bias The type of bias used for variance calculation. Takes of
              value of type \ref fly_var_bias
   \param[in] dim the dimension along which the standard deviation is extracted
   \return \ref FLY_SUCCESS if the operation is successful, otherwise an
           appropriate error code is returned.

   \ingroup stat_func_stdev

*/
FLY_API fly_err fly_stdev_v2(fly_array *out, const fly_array in,
                         const fly_var_bias bias, const dim_t dim);

/**
   C Interface for covariance

   \param[out] out will the covariance of the input arrays
   \param[in] X is the first input array
   \param[in] Y is the second input array
   \param[in] isbiased is boolean specifying if biased estimate should be taken (default: false)
   \return     \ref FLY_SUCCESS if the operation is successful,
   otherwise an appropriate error code is returned.

   \ingroup stat_func_cov

   \deprecated Use \ref fly_cov_v2 instead
*/
FLY_DEPRECATED("Use fly_cov_v2")
FLY_API fly_err fly_cov(fly_array* out, const fly_array X, const fly_array Y, const bool isbiased);

/**
   C Interface for covariance

   \param[out] out will the covariance of the input arrays
   \param[in] X is the first input array
   \param[in] Y is the second input array
   \param[in] bias The type of bias used for variance calculation. Takes of
              value of type \ref fly_var_bias
   \return \ref FLY_SUCCESS if the operation is successful, otherwise an
           appropriate error code is returned.

   \ingroup stat_func_cov
*/
FLY_API fly_err fly_cov_v2(fly_array *out, const fly_array X, const fly_array Y,
                       const fly_var_bias bias);

/**
   C Interface for median

   \param[out] out will contain the median of the input array along dimension \p dim
   \param[in] in is the input array
   \param[in] dim the dimension along which the median is extracted
   \return     \ref FLY_SUCCESS if the operation is successful,
   otherwise an appropriate error code is returned.

   \ingroup stat_func_median
*/
FLY_API fly_err fly_median(fly_array* out, const fly_array in, const dim_t dim);

/**
   C Interface for mean of all elements

   \param[out] real will contain the real part of mean of the entire input array
   \param[out] imag will contain the imaginary part of mean of the entire input array
   \param[in] in is the input array
   \return     \ref FLY_SUCCESS if the operation is successful,
   otherwise an appropriate error code is returned.

   \ingroup stat_func_mean
*/
FLY_API fly_err fly_mean_all(double *real, double *imag, const fly_array in);

/**
   C Interface for mean of all elements in weighted input

   \param[out] real will contain the real part of mean of the entire weighted input array
   \param[out] imag will contain the imaginary part of mean of the entire weighted input array
   \param[in] in is the input array
   \param[in] weights  is used to scale input \p in before getting mean
   \return     \ref FLY_SUCCESS if the operation is successful,
   otherwise an appropriate error code is returned.

   \ingroup stat_func_mean
*/
FLY_API fly_err fly_mean_all_weighted(double *real, double *imag, const fly_array in, const fly_array weights);


/**
   C Interface for variance of all elements

   \param[out] realVal will contain the real part of variance of the entire input array
   \param[out] imagVal will contain the imaginary part of variance of the entire input array
   \param[in] in is the input array
   \param[in] isbiased is boolean denoting Population variance (false) or Sample Variance (true)
   \return     \ref FLY_SUCCESS if the operation is successful,
   otherwise an appropriate error code is returned.

   \ingroup stat_func_var

   \deprecated Use \ref fly_var_all_v2 instead
*/
FLY_DEPRECATED("Use fly_var_all_v2")
FLY_API fly_err fly_var_all(double *realVal, double *imagVal, const fly_array in, const bool isbiased);

/**
   C Interface for variance of all elements

   \param[out] realVal will contain the real part of variance of the entire
               input array
   \param[out] imagVal will contain the imaginary part of variance
               of the entire input array
   \param[in] in is the input array
   \param[in] bias The type of bias used for variance calculation. Takes of
              value of type \ref fly_var_bias
   \return \ref FLY_SUCCESS if the operation is successful, otherwise an
           appropriate error code is returned.

   \ingroup stat_func_var
*/
FLY_API fly_err fly_var_all_v2(double *realVal, double *imagVal, const fly_array in,
                           const fly_var_bias bias);

/**
   C Interface for variance of all elements in weighted input

   \param[out] realVal will contain the real part of variance of the entire weighted input array
   \param[out] imagVal will contain the imaginary part of variance of the entire weighted input array
   \param[in] in is the input array
   \param[in] weights  is used to scale input \p in before getting variance
   \return     \ref FLY_SUCCESS if the operation is successful,
   otherwise an appropriate error code is returned.

   \ingroup stat_func_var
*/
FLY_API fly_err fly_var_all_weighted(double *realVal, double *imagVal, const fly_array in, const fly_array weights);

/**
   C Interface for standard deviation of all elements

   \param[out] real will contain the real part of standard deviation of the entire input array
   \param[out] imag will contain the imaginary part of standard deviation of the entire input array
   \param[in] in is the input array
   \return     \ref FLY_SUCCESS if the operation is successful,
   otherwise an appropriate error code is returned.

   \ingroup stat_func_stdev

   \deprecated Use \ref fly_stdev_all_v2 instead
*/
FLY_DEPRECATED("Use fly_stdev_all_v2")
FLY_API fly_err fly_stdev_all(double *real, double *imag, const fly_array in);

/**
   C Interface for standard deviation of all elements

   \param[out] real will contain the real part of standard deviation of the
               entire input array
   \param[out] imag will contain the imaginary part of standard deviation
               of the entire input array
   \param[in] in is the input array
   \param[in] bias The type of bias used for variance calculation. Takes of
              value of type \ref fly_var_bias
   \return     \ref FLY_SUCCESS if the operation is successful,
   otherwise an appropriate error code is returned.

   \ingroup stat_func_stdev
*/
FLY_API fly_err fly_stdev_all_v2(double *real, double *imag, const fly_array in,
                             const fly_var_bias bias);

/**
   C Interface for median

   \param[out] realVal will contain the real part of median of the entire input array
   \param[out] imagVal will contain the imaginary part of median of the entire input array
   \param[in] in is the input array
   \return     \ref FLY_SUCCESS if the operation is successful,
   otherwise an appropriate error code is returned.

   \ingroup stat_func_median
*/
FLY_API fly_err fly_median_all(double *realVal, double *imagVal, const fly_array in);

/**
   C Interface for correlation coefficient

   \param[out] realVal will contain the real part of correlation coefficient of the inputs
   \param[out] imagVal will contain the imaginary part of correlation coefficient of the inputs
   \param[in] X is the first input array
   \param[in] Y is the second input array
   \return     \ref FLY_SUCCESS if the operation is successful,
   otherwise an appropriate error code is returned.

   \note There are many ways correlation coefficient is calculated. This algorithm returns Pearson product-moment correlation coefficient.

   \ingroup stat_func_corrcoef
*/
FLY_API fly_err fly_corrcoef(double *realVal, double *imagVal, const fly_array X, const fly_array Y);

/**
   C Interface for finding top k elements along a given dimension

   \param[out] values  The values of the top k elements along the \p dim dimension
   \param[out] indices The indices of the top k elements along the \p dim dimension
   \param[in]  in      Input \ref fly::array with at least \p k elements along
                       \p dim
   \param[in]  k       The number of elements to be retriefed along the \p dim dimension
   \param[in]  dim     The dimension along which top k elements are extracted.
                       (Must be 0)
   \param[in]  order   If Descending the highest values are returned. Otherwise
                       the lowest values are returned

   \note{This function is optimized for small values of k.}
   \note{The order of the returned keys may not be in the same order as the
   appear in the input array, for a stable topk, set the FLY_TOPK_STABLE flag
   in the order param. These are equivalent to FLY_TOPK_STABLE_MAX and FLY_TOPK_STABLE_MIN}
   \ingroup stat_func_topk
*/
FLY_API fly_err fly_topk(fly_array *values, fly_array *indices, const fly_array in,
                     const int k, const int dim, const fly_topk_function order);

#ifdef __cplusplus
}
#endif
