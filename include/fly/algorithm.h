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
       C++ Interface to sum array elements over a given dimension.

       \param[in] in  input array
       \param[in] dim dimension along which the summation occurs, -1 denotes
                      the first non-singleton dimension
       \return        sum

       \ingroup reduce_func_sum
    */
    FLY_API array sum(const array &in, const int dim = -1);

    /**
       C++ Interface to sum array elements over a given dimension, replacing
       any NaNs with a specified value.

       \param[in] in     input array
       \param[in] dim    dimension along which the summation occurs
       \param[in] nanval value that replaces NaNs
       \return           sum

       \ingroup reduce_func_sum
    */
    FLY_API array sum(const array &in, const int dim, const double nanval);

    /**
       C++ Interface to sum array elements over a given dimension, according to
       an array of keys.

       \param[out] keys_out reduced keys
       \param[out] vals_out sum
       \param[in]  keys     keys array
       \param[in]  vals     input array
       \param[in]  dim      dimension along which the summation occurs, -1
                            denotes the first non-singleton dimension

       \ingroup reduce_func_sum_by_key
    */
    FLY_API void sumByKey(array &keys_out, array &vals_out,
                        const array &keys, const array &vals,
                        const int dim = -1);

    /**
       C++ Interface to sum array elements over a given dimension, replacing
       any NaNs with a specified value, according to an array of keys.

       \param[out] keys_out reduced keys
       \param[out] vals_out sum
       \param[in]  keys     keys array
       \param[in]  vals     input array
       \param[in]  dim      dimension along which the summation occurs
       \param[in]  nanval   value that replaces NaNs

       \ingroup reduce_func_sum_by_key
    */
    FLY_API void sumByKey(array &keys_out, array &vals_out,
                        const array &keys, const array &vals,
                        const int dim, const double nanval);

    /**
       C++ Interface to multiply array elements over a given dimension.

       \param[in] in  input array
       \param[in] dim dimension along which the product occurs, -1 denotes the
                      first non-singleton dimension
       \return        product

       \ingroup reduce_func_product
    */
    FLY_API array product(const array &in, const int dim = -1);

    /**
       C++ Interface to multiply array elements over a given dimension,
       replacing any NaNs with a specified value.

       \param[in] in     input array
       \param[in] dim    dimension along which the product occurs
       \param[in] nanval value that replaces NaNs
       \return           product

       \ingroup reduce_func_product
    */
    FLY_API array product(const array &in, const int dim, const double nanval);

    /**
       C++ Interface to multiply array elements over a given dimension,
       according to an array of keys.

       \param[out] keys_out reduced keys
       \param[out] vals_out product
       \param[in]  keys     keys array
       \param[in]  vals     input array
       \param[in]  dim      dimension along which the product occurs, -1
                            denotes the first non-singleton dimension

       \ingroup reduce_func_product_by_key
    */
    FLY_API void productByKey(array &keys_out, array &vals_out,
                            const array &keys, const array &vals,
                            const int dim = -1);

    /**
       C++ Interface to multiply array elements over a given dimension,
       replacing any NaNs with a specified value, according to an array of
       keys.

       \param[out] keys_out reduced keys
       \param[out] vals_out product
       \param[in]  keys     keys array
       \param[in]  vals     input array
       \param[in]  dim      dimension along which the product occurs
       \param[in]  nanval   value that replaces NaNs

       \ingroup reduce_func_product_by_key

    */
    FLY_API void productByKey(array &keys_out, array &vals_out,
                            const array &keys, const array &vals,
                            const int dim, const double nanval);

    /**
       C++ Interface to return the minimum along a given dimension.

       NaN values are ignored.

       \param[in] in  input array
       \param[in] dim dimension along which the minimum is found, -1 denotes
                      the first non-singleton dimension
       \return        minimum

       \ingroup reduce_func_min
    */
    FLY_API array min(const array &in, const int dim = -1);

    /**
       C++ Interface to return the minimum along a given dimension, according
       to an array of keys.

       NaN values are ignored.

       \param[out] keys_out reduced keys
       \param[out] vals_out minimum
       \param[in]  keys     keys array
       \param[in]  vals     input array
       \param[in]  dim      dimension along which the minimum is found, -1
                            denotes the first non-singleton dimension

       \ingroup reduce_func_min_by_key
    */
    FLY_API void minByKey(array &keys_out, array &vals_out,
                        const array &keys, const array &vals,
                        const int dim = -1);

    /**
       C++ Interface to return the maximum along a given dimension.

       NaN values are ignored.

       \param[in] in  input array
       \param[in] dim dimension along which the maximum is found, -1 denotes
                      the first non-singleton dimension
       \return        maximum

       \ingroup reduce_func_max
    */
    FLY_API array max(const array &in, const int dim = -1);

    /**
       C++ Interface to return the maximum along a given dimension, according
       to an array of keys.

       NaN values are ignored.

       \param[out] keys_out reduced keys
       \param[out] vals_out maximum
       \param[in]  keys     keys array
       \param[in]  vals     input array
       \param[in]  dim      dimension along which the maximum is found, -1
                            denotes the first non-singleton dimension

       \ingroup reduce_func_max_by_key
    */
    FLY_API void maxByKey(array &keys_out, array &vals_out,
                        const array &keys, const array &vals,
                        const int dim = -1);

    /**
       C++ Interface to return the ragged maximum along a given dimension.

       Input parameter `ragged_len` sets the number of elements to consider.

       NaN values are ignored.

       \param[out] val        ragged maximum
       \param[out] idx        locations of the maximum ragged values
       \param[in]  in         input array
       \param[in]  ragged_len array containing the number of elements to use
       \param[in]  dim        dimension along which the maximum is found

       \ingroup reduce_func_max
    */
    FLY_API void max(array &val, array &idx, const array &in, const array &ragged_len, const int dim);

    /**
       C++ Interface to check if all values along a given dimension are true.

       NaN values are ignored.

       \param[in] in  input array
       \param[in] dim dimension along which the check occurs, -1 denotes the
                      first non-singleton dimension
       \return        array containing 1's if all true; 0's otherwise

       \ingroup reduce_func_all_true
    */
    FLY_API array allTrue(const array &in, const int dim = -1);

    /**
       C++ Interface to check if all values along a given dimension are true,
       according to an array of keys.

       NaN values are ignored.

       \param[out] keys_out reduced keys
       \param[out] vals_out array containing 1's if all true; 0's otherwise
       \param[in]  keys     keys array
       \param[in]  vals     input array
       \param[in]  dim      dimension along which the check occurs

       \ingroup reduce_func_alltrue_by_key
    */
    FLY_API void allTrueByKey(array &keys_out, array &vals_out,
                            const array &keys, const array &vals,
                            const int dim = -1);

    /**
       C++ Interface to check if any values along a given dimension are true.

       NaN values are ignored.

       \param[in] in  input array
       \param[in] dim dimension along which the check occurs, -1 denotes the
                      first non-singleton dimension
       \return        array containing 1's if any true; 0's otherwise

       \ingroup reduce_func_any_true
    */
    FLY_API array anyTrue(const array &in, const int dim = -1);

    /**
       C++ Interface to check if any values along a given dimension are true,
       according to an array of keys.

       NaN values are ignored.

       \param[out] keys_out reduced keys
       \param[out] vals_out array containing 1's if any true; 0's otherwise
       \param[in]  keys     keys array
       \param[in]  vals     input array
       \param[in]  dim      dimension along which the check occurs

       \ingroup reduce_func_anytrue_by_key
    */
    FLY_API void anyTrueByKey(array &keys_out, array &vals_out,
                            const array &keys, const array &vals,
                            const int dim = -1);

    /**
       C++ Interface to count non-zero values in an array along a given
       dimension.

       NaN values are treated as non-zero.

       \param[in] in  input array
       \param[in] dim dimension along which the count occurs, -1 denotes the
                      first non-singleton dimension
       \return        count

       \ingroup reduce_func_count
    */
    FLY_API array count(const array &in, const int dim = -1);

    /**
       C++ Interface to count non-zero values in an array, according to an
       array of keys.

       NaN values are treated as non-zero.

       \param[out] keys_out reduced keys
       \param[out] vals_out count
       \param[in]  keys     keys array
       \param[in]  vals     input array
       \param[in]  dim      dimension along which the count occurs, -1 denotes
                            the first non-singleton dimension

       \ingroup reduce_func_count_by_key
    */
    FLY_API void countByKey(array &keys_out, array &vals_out,
                          const array &keys, const array &vals,
                          const int dim = -1);

    /**
       C++ Interface to sum array elements over all dimensions.

       Results in a single value as an output, which may be a single element
       `fly::array`.

       \param[in] in  input array
       \return        sum

       \ingroup reduce_func_sum
    */
    template<typename T> T sum(const array &in);

    /**
       C++ Interface to sum array elements over all dimensions, replacing any
       NaNs with a specified value.

       Results in a single value as an output, which may be a single element
       `fly::array`.

       \param[in] in     input array
       \param[in] nanval value that replaces NaNs
       \return           sum

       \ingroup reduce_func_sum
    */
    template<typename T> T sum(const array &in, double nanval);

    /**
       C++ Interface to multiply array elements over the first non-singleton
       dimension.

       \param[in] in input array
       \return       product

       \ingroup reduce_func_product
    */
    template<typename T> T product(const array &in);

    /**
       C++ Interface to multiply array elements over the first non-singleton
       dimension, replacing any NaNs with a specified value.

       \param[in] in     input array
       \param[in] nanval value that replaces NaNs
       \return           product

       \ingroup reduce_func_product
    */
    template<typename T> T product(const array &in, double nanval);

    /**
       C++ Interface to return the minimum along the first non-singleton
       dimension.

       NaN values are ignored.

       \param[in] in input array
       \return       minimum

       \ingroup reduce_func_min
    */
    template<typename T> T min(const array &in);

    /**
       C++ Interface to return the maximum along the first non-singleton
       dimension.

       NaN values are ignored.

       \param[in] in input array
       \return       maximum

       \ingroup reduce_func_max
    */
    template<typename T> T max(const array &in);

    /**
       C++ Interface to check if all values along the first non-singleton
       dimension are true.

       NaN values are ignored.

       \param[in] in input array
       \return       array containing 1's if all true; 0's otherwise

       \ingroup reduce_func_all_true
    */
    template<typename T> T allTrue(const array &in);

    /**
       C++ Interface to check if any values along the first non-singleton
       dimension are true.

       NaN values are ignored.

       \param[in] in input array
       \return       array containing 1's if any true; 0's otherwise

       \ingroup reduce_func_any_true
    */
    template<typename T> T anyTrue(const array &in);

    /**
       C++ Interface to count non-zero values along the first non-singleton
       dimension.

       NaN values are treated as non-zero.

       \param[in] in input array
       \return       count

       \ingroup reduce_func_count
    */
    template<typename T> T count(const array &in);

    /**
       C++ Interface to return the minimum and its location along a given
       dimension.

       NaN values are ignored.

       \param[out] val minimum
       \param[out] idx location
       \param[in]  in  input array
       \param[in]  dim dimension along which the minimum is found, -1 denotes
                       the first non-singleton dimension

       \ingroup reduce_func_min
    */
    FLY_API void min(array &val, array &idx, const array &in, const int dim = -1);

    /**
       C++ Interface to return the maximum and its location along a given
       dimension.

       NaN values are ignored.

       \param[out] val maximum
       \param[out] idx location
       \param[in]  in  input array
       \param[in]  dim dimension along which the maximum is found, -1 denotes
                       the first non-singleton dimension

       \ingroup reduce_func_max
    */
    FLY_API void max(array &val, array &idx, const array &in, const int dim = -1);

    /**
       C++ Interface to return the minimum and its location over all
       dimensions.

       NaN values are ignored.

       Often used to return values directly to the host.

       \param[out] val minimum
       \param[out] idx location
       \param[in]  in  input array

       \ingroup reduce_func_min
    */
    template<typename T> void min(T *val, unsigned *idx, const array &in);

    /**
       C++ Interface to return the maximum and its location over all
       dimensions.

       NaN values are ignored.

       Often used to return values directly to the host.

       \param[out] val maximum
       \param[out] idx location
       \param[in]  in  input array

       \ingroup reduce_func_max
    */
    template<typename T> void max(T *val, unsigned *idx, const array &in);

    /**
       C++ Interface to evaluate the cumulative sum (inclusive) along a given
       dimension.

       \param[in] in  input array
       \param[in] dim dimension along which the sum is accumulated, 0 denotes
                      the first non-singleton dimension
       \return        cumulative sum

       \ingroup scan_func_accum
    */
    FLY_API array accum(const array &in, const int dim = 0);

    /**
       C++ Interface to scan an array (generalized) over a given dimension.

       \param[in] in             input array
       \param[in] dim            dimension along which the scan occurs, 0
                                 denotes the first non-singleton dimension
       \param[in] op             type of binary operation used
       \param[in] inclusive_scan flag specifying whether the scan is inclusive
       \return                   scan

       \ingroup scan_func_scan
    */
    FLY_API array scan(const array &in, const int dim = 0,
                     binaryOp op = FLY_BINARY_ADD, bool inclusive_scan = true);

    /**
       C++ Interface to scan an array (generalized) over a given dimension,
       according to an array of keys.

       \param[in] key            keys array
       \param[in] in             input array
       \param[in] dim            dimension along which the scan occurs, 0
                                 denotes the first non-singleton dimension
       \param[in] op             type of binary operation used
       \param[in] inclusive_scan flag specifying whether the scan is inclusive
       \return                   scan

       \ingroup scan_func_scanbykey
    */
    FLY_API array scanByKey(const array &key, const array& in, const int dim = 0,
                          binaryOp op = FLY_BINARY_ADD, bool inclusive_scan = true);

    /**
       C++ Interface to locate the indices of the non-zero values in an array.

       \param[in] in input array
       \return       linear indices where `in` is non-zero

       \ingroup scan_func_where
    */
    FLY_API array where(const array &in);

    /**
       C++ Interface to calculate the first order difference in an array over a
       given dimension.

       \param[in] in  input array
       \param[in] dim dimension along which the difference occurs, 0
                      denotes the first non-singleton dimension
       \return        first order numerical difference

       \ingroup calc_func_diff1
    */
    FLY_API array diff1(const array &in, const int dim = 0);

    /**
       C++ Interface to calculate the second order difference in an array over
       a given dimension.

       \param[in] in  input array
       \param[in] dim dimension along which the difference occurs, 0
                      denotes the first non-singleton dimension
       \return        second order numerical difference

       \ingroup calc_func_diff2
    */
    FLY_API array diff2(const array &in, const int dim = 0);

    /**
       C++ Interface to sort an array over a given dimension.

       \param[in] in          input array
       \param[in] dim         dimension along which the sort occurs, 0 denotes
                              the first non-singleton dimension
       \param[in] isAscending specifies the sorting order
       \return                sorted output

       \ingroup sort_func_sort
    */
    FLY_API array sort(const array &in, const unsigned dim = 0,
                     const bool isAscending = true);

    /**
       C++ Interface to sort an array over a given dimension and to return the
       original indices.

       \param[out] out         sorted output
       \param[out] indices     indices from the input
       \param[in]  in          input array
       \param[in]  dim         dimension along which the sort occurs, 0 denotes
                               the first non-singleton dimension
       \param[in]  isAscending specifies the sorting order

       \ingroup sort_func_sort_index
    */
    FLY_API void  sort(array &out, array &indices, const array &in, const unsigned dim = 0,
                     const bool isAscending = true);

    /**
       C++ Interface to sort an array over a given dimension, according to an
       array of keys.

       \param[out] out_keys    sorted keys
       \param[out] out_values  sorted output
       \param[in]  keys        keys array
       \param[in]  values      input array
       \param[in]  dim         dimension along which the sort occurs, 0 denotes
                               the first non-singleton dimension
       \param[in]  isAscending specifies the sorting order

       \ingroup sort_func_sort_keys
    */
    FLY_API void  sort(array &out_keys, array &out_values, const array &keys,
                     const array &values, const unsigned dim = 0,
                     const bool isAscending = true);

    /**
       C++ Interface to return the unique values in an array.

       \param[in] in        input array
       \param[in] is_sorted if true, skip the sorting steps internally
       \return              unique values

       \ingroup set_func_unique
    */
    FLY_API array setUnique(const array &in, const bool is_sorted=false);

    /**
       C++ Interface to evaluate the union of two arrays.

       \param[in] first     input array
       \param[in] second    input array
       \param[in] is_unique if true, skip calling setUnique internally
       \return              union, values in increasing order

       \ingroup set_func_union
    */
    FLY_API array setUnion(const array &first, const array &second,
                         const bool is_unique=false);

    /**
       C++ Interface to evaluate the intersection of two arrays.

       \param[in] first     input array
       \param[in] second    input array
       \param[in] is_unique if true, skip calling setUnique internally
       \return              intersection, values in increasing order

       \ingroup set_func_intersect
    */
    FLY_API array setIntersect(const array &first, const array &second,
                             const bool is_unique=false);
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    /**
       C Interface to sum array elements over a given dimension.

       \param[out] out sum
       \param[in]  in  input array
       \param[in]  dim dimension along which the summation occurs
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup reduce_func_sum
    */
    FLY_API fly_err fly_sum(fly_array *out, const fly_array in, const int dim);

    /**
       C Interface to sum array elements over all dimensions.

       Results in a single element `fly::array`.

       \param[out] out sum
       \param[in]  in  input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup reduce_func_sum
    */
    FLY_API fly_err fly_sum_all_array(fly_array *out, const fly_array in);

    /**
       C Interface to sum array elements over a given dimension, replacing any
       NaNs with a specified value.

       \param[out] out    sum
       \param[in]  in     input array
       \param[in]  dim    dimension along which the summation occurs
       \param[in]  nanval value that replaces NaNs
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup reduce_func_sum
    */
    FLY_API fly_err fly_sum_nan(fly_array *out, const fly_array in,
                            const int dim, const double nanval);

    /**
       C Interface to sum array elements over all dimensions, replacing any
       NaNs with a specified value.

       Results in a single element `fly::array`.

       \param[out] out    sum
       \param[in]  in     input array
       \param[in]  nanval value that replaces NaNs
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup reduce_func_sum
    */
    FLY_API fly_err fly_sum_nan_all_array(fly_array *out, const fly_array in, const double nanval);

    /**
       C Interface to sum array elements over a given dimension, according to
       an array of keys.

       \param[out] keys_out reduced keys
       \param[out] vals_out sum
       \param[in]  keys     keys array
       \param[in]  vals     input array
       \param[in]  dim      dimension along which the summation occurs
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup reduce_func_sum_by_key
    */
    FLY_API fly_err fly_sum_by_key(fly_array *keys_out, fly_array *vals_out,
                               const fly_array keys, const fly_array vals, const int dim);

    /**
       C Interface to sum array elements over a given dimension, replacing any
       NaNs with a specified value, according to an array of keys.

       \param[out] keys_out reduced keys
       \param[out] vals_out sum
       \param[in]  keys     keys array
       \param[in]  vals     input array
       \param[in]  dim      dimension along which the summation occurs
       \param[in]  nanval   value that replaces NaNs
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup reduce_func_sum_by_key
    */
    FLY_API fly_err fly_sum_by_key_nan(fly_array *keys_out, fly_array *vals_out,
                                   const fly_array keys, const fly_array vals,
                                   const int dim, const double nanval);

    /**
       C Interface to multiply array elements over a given dimension.

       \param[out] out product
       \param[in]  in  input array
       \param[in]  dim dimension along which the product occurs
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup reduce_func_product
    */
    FLY_API fly_err fly_product(fly_array *out, const fly_array in, const int dim);

    /**
       C Interface to multiply array elements over all dimensions.

       Results in a single element `fly::array`.

       \param[out] out product
       \param[in]  in  input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup reduce_func_product
    */
    FLY_API fly_err fly_product_all_array(fly_array *out, const fly_array in);

    /**
       C Interface to multiply array elements over a given dimension, replacing
       any NaNs with a specified value.

       \param[out] out    product
       \param[in]  in     input array
       \param[in]  dim    dimension along with the product occurs
       \param[in]  nanval value that replaces NaNs
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup reduce_func_product
    */
    FLY_API fly_err fly_product_nan(fly_array *out, const fly_array in, const int dim, const double nanval);

    /**
       C Interface to multiply array elements over all dimensions, replacing
       any NaNs with a specified value.

       \param[out] out    product
       \param[in]  in     input array
       \param[in]  nanval value that replaces NaNs
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup reduce_func_product
    */
    FLY_API fly_err fly_product_nan_all_array(fly_array *out, const fly_array in, const double nanval);

    /**
       C Interface to multiply array elements over a given dimension, according
       to an array of keys.

       \param[out] keys_out reduced keys
       \param[out] vals_out product
       \param[in]  keys     keys array
       \param[in]  vals     input array
       \param[in]  dim      dimension along which the product occurs
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup reduce_func_product_by_key
    */
    FLY_API fly_err fly_product_by_key(fly_array *keys_out, fly_array *vals_out,
                                   const fly_array keys, const fly_array vals, const int dim);

    /**
       C Interface to multiply array elements over a given dimension, replacing
       any NaNs with a specified value, according to an array of keys.

       \param[out] keys_out reduced keys
       \param[out] vals_out product
       \param[in]  keys     keys array
       \param[in]  vals     input array
       \param[in]  dim      dimension along which the product occurs
       \param[in]  nanval   value that replaces NaNs
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup reduce_func_product_by_key
    */
    FLY_API fly_err fly_product_by_key_nan(fly_array *keys_out, fly_array *vals_out,
                                       const fly_array keys, const fly_array vals,
                                       const int dim, const double nanval);

    /**
       C Interface to return the minimum along a given dimension.

       \param[out] out minimum
       \param[in]  in  input array
       \param[in]  dim dimension along which the minimum is found
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup reduce_func_min
    */
    FLY_API fly_err fly_min(fly_array *out, const fly_array in, const int dim);

    /**
       C Interface to return the minimum along a given dimension, according to
       an array of keys.

       \param[out] keys_out reduced keys
       \param[out] vals_out minimum
       \param[in]  keys     keys array
       \param[in]  vals     input array
       \param[in]  dim      dimension along which the minimum is found
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup reduce_func_min_by_key
    */
    FLY_API fly_err fly_min_by_key(fly_array *keys_out, fly_array *vals_out,
                               const fly_array keys, const fly_array vals,
                               const int dim);

    /**
       C Interface to return the maximum along a given dimension.

       \param[out] out  maximum
       \param[in]  in   input array
       \param[in]  dim dimension along which the maximum is found
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup reduce_func_max
    */
    FLY_API fly_err fly_max(fly_array *out, const fly_array in, const int dim);

    /**
       C Interface to return the maximum along a given dimension, according to
       an array of keys.

       \param[out] keys_out reduced keys
       \param[out] vals_out maximum
       \param[in]  keys     keys array
       \param[in]  vals     input array
       \param[in]  dim      dimension along which the maximum is found
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup reduce_func_max_by_key
    */
    FLY_API fly_err fly_max_by_key(fly_array *keys_out, fly_array *vals_out,
                               const fly_array keys, const fly_array vals,
                               const int dim);

    /**
       C Interface to return the ragged maximum over a given dimension.

       Input parameter `ragged_len` sets the number of elements to consider.

       NaN values are ignored.

       \param[out] val        ragged maximum
       \param[out] idx        locations of the maximum ragged values
       \param[in]  in         input array
       \param[in]  ragged_len array containing the number of elements to use
       \param[in]  dim        dimension along which the maximum is found
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup reduce_func_max
    */
    FLY_API fly_err fly_max_ragged(fly_array *val, fly_array *idx, const fly_array in, const fly_array ragged_len, const int dim);

    /**
       C Interface  to check if all values along a given dimension are true.

       NaN values are ignored.

       \param[out] out array containing 1's if all true; 0's otherwise
       \param[in]  in  input array
       \param[in]  dim dimention along which the check occurs
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup reduce_func_all_true
    */
    FLY_API fly_err fly_all_true(fly_array *out, const fly_array in, const int dim);

    /**
       C Interface to check if all values along a given dimension are true,
       according to an array of keys.

       NaN values are ignored.

       \param[out] keys_out reduced keys
       \param[out] vals_out array containing 1's if all true; 0's otherwise
       \param[in]  keys     keys array
       \param[in]  vals     input array
       \param[in]  dim      dimension along which the check occurs
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup reduce_func_alltrue_by_key
    */
    FLY_API fly_err fly_all_true_by_key(fly_array *keys_out, fly_array *vals_out,
                                    const fly_array keys, const fly_array vals,
                                    const int dim);

    /**
       C Interface to check if any values along a given dimension are true.

       NaN values are ignored.

       \param[out] out array containing 1's if any true; 0's otherwise
       \param[in]  in  input array
       \param[in]  dim dimension along which the check occurs
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup reduce_func_any_true
    */
    FLY_API fly_err fly_any_true(fly_array *out, const fly_array in, const int dim);

    /**
       C Interface to check if any values along a given dimension are true.

       NaN values are ignored.

       \param[out] keys_out reduced keys
       \param[out] vals_out array containing 1's if any true; 0's otherwise
       \param[in]  keys     keys array
       \param[in]  vals     input array
       \param[in]  dim      dimensions along which the check occurs
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup reduce_func_anytrue_by_key
    */
    FLY_API fly_err fly_any_true_by_key(fly_array *keys_out, fly_array *vals_out,
                                    const fly_array keys, const fly_array vals,
                                    const int dim);

    /**
       C Interface to count non-zero values in an array along a given
       dimension.

       NaN values are treated as non-zero.

       \param[out] out count
       \param[in]  in  input array
       \param[in]  dim dimension along which the count occurs
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup reduce_func_count
    */
    FLY_API fly_err fly_count(fly_array *out, const fly_array in, const int dim);

    /**
       C Interface to count non-zero values in an array, according to an array
       of keys.

       NaN values are treated as non-zero.

       \param[out] keys_out reduced keys
       \param[out] vals_out count
       \param[in]  keys     keys array
       \param[in]  vals     input array
       \param[in]  dim      dimension along which the count occurs
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup reduce_func_count_by_key
    */
    FLY_API fly_err fly_count_by_key(fly_array *keys_out, fly_array *vals_out,
                                 const fly_array keys, const fly_array vals,
                                 const int dim);

    /**
       C Interface to sum array elements over all dimensions.

       If `in` is real, `imag` will be set to zeros.

       \param[out] real sum of all real components
       \param[out] imag sum of all imaginary components
       \param[in]  in   input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup reduce_func_sum
    */
    FLY_API fly_err fly_sum_all(double *real, double *imag, const fly_array in);

    /**
       C Interface to sum array elements over all dimensions, replacing any
       NaNs with a specified value.

       If `in` is real, `imag` will be set to zeros.

       \param[out] real   sum of all real components
       \param[out] imag   sum of all imaginary components
       \param[in]  in     input array
       \param[in]  nanval value that replaces NaNs
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup reduce_func_sum
    */
    FLY_API fly_err fly_sum_nan_all(double *real, double *imag,
                                const fly_array in, const double nanval);

    /**
       C Interface to multiply array elements over all dimensions.

       If `in` is real, `imag` will be set to zeros.

       \param[out] real product of all real components
       \param[out] imag product of all imaginary components
       \param[in]  in   input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup reduce_func_product
    */
    FLY_API fly_err fly_product_all(double *real, double *imag, const fly_array in);

    /**
       C Interface to multiply array elements over all dimensions, replacing
       any NaNs with a specified value.

       If `in` is real, `imag` will be set to zeros.

       \param[out] real   product of all real components
       \param[out] imag   product of all imaginary components
       \param[in]  in     input array
       \param[in]  nanval value that replaces NaNs
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup reduce_func_product
    */
    FLY_API fly_err fly_product_nan_all(double *real, double *imag,
                                    const fly_array in, const double nanval);

    /**
       C Interface to return the minimum over all dimensions.

       If `in` is real, `imag` will be set to zeros.

       \param[out] real real component of the minimum
       \param[out] imag imaginary component of the minimum
       \param[in]  in   input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup reduce_func_min
    */
    FLY_API fly_err fly_min_all(double *real, double *imag, const fly_array in);

    /**
       C Interface to return the minimum over all dimensions.

       \param[out] out minimum
       \param[in]  in  input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup reduce_func_min
    */
    FLY_API fly_err fly_min_all_array(fly_array *out, const fly_array in);

    /**
       C Interface to return the maximum over all dimensions.

       If `in` is real, `imag` will be set to zeros.

       \param[out] real real component of the maximum
       \param[out] imag imaginary component of the maximum
       \param[in]  in   input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup reduce_func_max
    */
    FLY_API fly_err fly_max_all(double *real, double *imag, const fly_array in);

    /**
       C Interface to return the maximum over all dimensions.

       \param[out] out maximum
       \param[in]  in  input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup reduce_func_max
    */
    FLY_API fly_err fly_max_all_array(fly_array *out, const fly_array in);

    /**
       C Interface to check if all values over all dimensions are true.
 
       \param[out] real 1 if all true; 0 otherwise
       \param[out] imag 0
       \param[in]  in   input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup reduce_func_all_true
    */
    FLY_API fly_err fly_all_true_all(double *real, double *imag, const fly_array in);

    /**
       C Interface to check if all values over all dimensions are true.
 
       \param[out] out 1 if all true; 0 otherwise
       \param[in]  in  input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup reduce_func_all_true
    */
    FLY_API fly_err fly_all_true_all_array(fly_array *out, const fly_array in);

    /**
       C Interface to check if any values over all dimensions are true.

       \param[out] real 1 if any true; 0 otherwise
       \param[out] imag 0
       \param[in]  in   input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup reduce_func_any_true
    */
    FLY_API fly_err fly_any_true_all(double *real, double *imag, const fly_array in);

    /**
       C Interface to check if any values over all dimensions are true.

       \param[out] out 1 if any true; 0 otherwise
       \param[in]  in  input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup reduce_func_any_true
    */
    FLY_API fly_err fly_any_true_all_array(fly_array *out, const fly_array in);

    /**
       C Interface to count non-zero values over all dimensions.

       \param[out] real count
       \param[out] imag 0
       \param[in]  in   input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup reduce_func_count
    */
    FLY_API fly_err fly_count_all(double *real, double *imag, const fly_array in);

    /**
       C Interface to count non-zero values over all dimensions.

       \param[out] out count
       \param[in]  in  input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup reduce_func_count
    */
    FLY_API fly_err fly_count_all_array(fly_array *out, const fly_array in);

    /**
       C Interface to return the minimum and its location along a given
       dimension.

       \param[out] out minimum
       \param[out] idx location
       \param[in]  in  input array
       \param[in]  dim dimension along which the minimum is found
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup reduce_func_min
    */
    FLY_API fly_err fly_imin(fly_array *out, fly_array *idx, const fly_array in,
                         const int dim);

    /**
       C Interface to return the maximum and its location along a given
       dimension.

       \param[out] out maximum
       \param[out] idx location
       \param[in]  in  input array
       \param[in]  dim dimension along which the maximum is found
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup reduce_func_max
    */
    FLY_API fly_err fly_imax(fly_array *out, fly_array *idx, const fly_array in,
                         const int dim);

    /**
       C Interface to return the minimum and its location over all dimensions.

       NaN values are ignored.

       \param[out] real real component of the minimum
       \param[out] imag imaginary component of the minimum; 0 if `idx` is real
       \param[out] idx  location
       \param[in]  in   input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup reduce_func_min
    */
    FLY_API fly_err fly_imin_all(double *real, double *imag, unsigned *idx,
                             const fly_array in);

    /**
       C Interface to return the maximum and its location over all dimensions.

       NaN values are ignored.

       \param[out] real real component of the maximum
       \param[out] imag imaginary component of the maximum; 0 if `idx` is real
       \param[out] idx  location
       \param[in]  in   input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup reduce_func_max
    */
    FLY_API fly_err fly_imax_all(double *real, double *imag, unsigned *idx, const fly_array in);

    /**
       C Interface to evaluate the cumulative sum (inclusive) along a given
       dimension.

       \param[out] out cumulative sum
       \param[in]  in  input array
       \param[in]  dim dimension along which the sum is accumulated
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup scan_func_accum
    */
    FLY_API fly_err fly_accum(fly_array *out, const fly_array in, const int dim);

    /**
       C Interface to scan an array (generalized) over a given dimension.

       \param[out] out            scan
       \param[in]  in             input array
       \param[in]  dim            dimension along which the scan occurs
       \param[in]  op             type of binary operation used
       \param[in]  inclusive_scan flag specifying whether the scan is inclusive
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup scan_func_scan
    */
    FLY_API fly_err fly_scan(fly_array *out, const fly_array in, const int dim,
                         fly_binary_op op, bool inclusive_scan);

    /**
       C Interface to scan an array (generalized) over a given dimension,
       according to an array of keys.

       \param[out] out            scan
       \param[in]  key            keys array
       \param[in]  in             input array
       \param[in]  dim            dimension along which the scan occurs
       \param[in]  op             type of binary operation used
       \param[in]  inclusive_scan flag specifying whether the scan is inclusive
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup scan_func_scanbykey
    */
    FLY_API fly_err fly_scan_by_key(fly_array *out, const fly_array key,
                                const fly_array in, const int dim,
                                fly_binary_op op, bool inclusive_scan);


    /**
       C Interface to locate the indices of the non-zero values in an array.

       \param[out] idx linear indices where `in` is non-zero
       \param[in]  in  input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup scan_func_where
    */
    FLY_API fly_err fly_where(fly_array *idx, const fly_array in);

    /**
       C Interface to calculate the first order difference in an array over a
       given dimension.

       \param[out] out first order numerical difference
       \param[in]  in  input array
       \param[in]  dim dimension along which the difference occurs
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup calc_func_diff1
    */
    FLY_API fly_err fly_diff1(fly_array *out, const fly_array in, const int dim);

    /**
       C Interface to calculate the second order difference in an array over a
       given dimension.

       \param[out] out second order numerical difference
       \param[in]  in  input array
       \param[in]  dim dimension along which the difference occurs
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup calc_func_diff2
    */
    FLY_API fly_err fly_diff2(fly_array *out, const fly_array in, const int dim);

    /**
       C Interface to sort an array over a given dimension.

       \param[out] out         sorted output
       \param[in]  in          input array
       \param[in]  dim         dimension along which the sort occurs
       \param[in]  isAscending specifies the sorting order
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup sort_func_sort
    */
    FLY_API fly_err fly_sort(fly_array *out, const fly_array in, const unsigned dim,
                         const bool isAscending);

    /**
       C Interface to sort an array over a given dimension and to return the
       original indices.

       \param[out] out         sorted output
       \param[out] indices     indices from the input
       \param[in]  in          input array
       \param[in]  dim         dimension along which the sort occurs
       \param[in]  isAscending specifies the sorting order
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup sort_func_sort_index
    */
    FLY_API fly_err fly_sort_index(fly_array *out, fly_array *indices, const fly_array in,
                               const unsigned dim, const bool isAscending);
    /**
       C Interface to sort an array over a given dimension, according to an
       array of keys.

       \param[out] out_keys    sorted keys
       \param[out] out_values  sorted output
       \param[in]  keys        keys array
       \param[in]  values      input array
       \param[in]  dim         dimension along which the sort occurs
       \param[in]  isAscending specifies the sorting order
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup sort_func_sort_keys
    */
    FLY_API fly_err fly_sort_by_key(fly_array *out_keys, fly_array *out_values,
                                const fly_array keys, const fly_array values,
                                const unsigned dim, const bool isAscending);

    /**
       C Interface to return the unique values in an array.

       \param[out] out       unique values
       \param[in]  in        input array
       \param[in]  is_sorted if true, skip the sorting steps internally
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup set_func_unique
    */
    FLY_API fly_err fly_set_unique(fly_array *out, const fly_array in, const bool is_sorted);

    /**
       C Interface to evaluate the union of two arrays.

       \param[out] out       union, values in increasing order
       \param[in]  first     input array
       \param[in]  second    input array
       \param[in]  is_unique if true, skip calling unique internally
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup set_func_union
    */
    FLY_API fly_err fly_set_union(fly_array *out, const fly_array first,
                              const fly_array second, const bool is_unique);

    /**
       C Interface to evaluate the intersection of two arrays.

       \param[out] out       intersection, values in increasing order
       \param[in]  first     input array
       \param[in]  second    input array
       \param[in]  is_unique if true, skip calling unique internally
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup set_func_intersect
    */
    FLY_API fly_err fly_set_intersect(fly_array *out, const fly_array first,
                                  const fly_array second, const bool is_unique);

#ifdef __cplusplus
}
#endif
