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
#include <fly/dim4.hpp>
#include <fly/traits.hpp>
namespace fly
{
    class array;

    /// C++ Interface to generate an array with elements set to a specified
    /// value.
    ///
    /// \param[in] val  constant value
    /// \param[in] dims dimensions of the array to be generated
    /// \param[in] ty   type
    /// \return         constant array
    ///
    /// \ingroup data_func_constant
    template<typename T>
    array constant(T val, const dim4 &dims, const dtype ty=(fly_dtype)dtype_traits<T>::ctype);

    /// C++ Interface to generate a 1-D array with elements set to a specified
    /// value.
    ///
    /// \param[in] val constant value
    /// \param[in] d0  size of the first dimension
    /// \param[in] ty  type
    /// \return        constant 1-D array
    ///
    /// \ingroup data_func_constant
    template<typename T>
    array constant(T val, const dim_t d0, const fly_dtype ty=(fly_dtype)dtype_traits<T>::ctype);

    /// C++ Interface to generate a 2-D array with elements set to a specified
    /// value.
    ///
    /// \param[in] val constant value
    /// \param[in] d0  size of the first dimension
    /// \param[in] d1  size of the second dimension
    /// \param[in] ty  type
    /// \return        constant 2-D array
    ///
    /// \ingroup data_func_constant
    template<typename T>
    array constant(T val, const dim_t d0, const dim_t d1, const fly_dtype ty=(fly_dtype)dtype_traits<T>::ctype);

    /// C++ Interface to generate a 3-D array with elements set to a specified
    /// value.
    ///
    /// \param[in] val constant value
    /// \param[in] d0  size of the first dimension
    /// \param[in] d1  size of the second dimension
    /// \param[in] d2  size of the third dimension
    /// \param[in] ty  type
    /// \return        constant 3-D array
    ///
    /// \ingroup data_func_constant
    template<typename T>
    array constant(T val, const dim_t d0, const dim_t d1, const dim_t d2, const fly_dtype ty=(fly_dtype)dtype_traits<T>::ctype);

    /// C++ Interface to generate a 4-D array with elements set to a specified
    /// value.
    ///
    /// \param[in] val constant value
    /// \param[in] d0  size of the first dimension
    /// \param[in] d1  size of the second dimension
    /// \param[in] d2  size of the third dimension
    /// \param[in] d3  size of the fourth dimension
    /// \param[in] ty  type
    /// \return        constant 4-D array
    ///
    /// \ingroup data_func_constant
    template<typename T>
    array constant(T val, const dim_t d0, const dim_t d1, const dim_t d2, const dim_t d3, const fly_dtype ty=(fly_dtype)dtype_traits<T>::ctype);

    /// C++ Interface to generate an identity array.
    ///
    /// \param[in] dims size
    /// \param[in] ty   type
    /// \return         identity array
    ///
    /// \ingroup data_func_identity
    FLY_API array identity(const dim4 &dims, const dtype ty=f32);

    /// C++ Interface to generate a 1-D identity array.
    ///
    /// \param[in] d0 size of the first dimension
    /// \param[in] ty type
    /// \return       identity array
    ///
    /// \ingroup data_func_identity
    FLY_API array identity(const dim_t d0, const dtype ty=f32);

    /// C++ Interface to generate a 2-D identity array.
    ///
    /// \param[in] d0 size of the first dimension
    /// \param[in] d1 size of the second dimension
    /// \param[in] ty type
    /// \return       identity array
    ///
    /// \ingroup data_func_identity
    FLY_API array identity(const dim_t d0, const dim_t d1, const dtype ty=f32);

    /// C++ Interface to generate a 3-D identity array.
    ///
    /// \param[in] d0 size of the first dimension
    /// \param[in] d1 size of the second dimension
    /// \param[in] d2 size of the third dimension
    /// \param[in] ty type
    /// \return       identity array
    ///
    /// \ingroup data_func_identity
    FLY_API array identity(const dim_t d0, const dim_t d1,
                         const dim_t d2, const dtype ty=f32);

    /// C++ Interface to generate a 4-D identity array.
    ///
    /// \param[in] d0 size of the first dimension
    /// \param[in] d1 size of the second dimension
    /// \param[in] d2 size of the third dimension
    /// \param[in] d3 size of the fourth dimension
    /// \param[in] ty type
    /// \return       identity array
    ///
    /// \ingroup data_func_identity
    FLY_API array identity(const dim_t d0, const dim_t d1,
                         const dim_t d2, const dim_t d3, const dtype ty=f32);

    /// C++ Interface to generate an array with `[0, n-1]` values along the
    /// `seq_dim` dimension and tiled across other dimensions of shape `dim4`.
    ///
    /// \param[in] dims    size
    /// \param[in] seq_dim dimesion along which the range is created
    /// \param[in] ty      type
    /// \return            range array
    ///
    /// \ingroup data_func_range
    FLY_API array range(const dim4 &dims, const int seq_dim = -1, const dtype ty=f32);

    /// C++ Interface to generate an array with `[0, n-1]` values along the
    /// `seq_dim` dimension and tiled across other dimensions described by
    /// dimension parameters.
    ///
    /// \param[in] d0      size of the first dimension
    /// \param[in] d1      size of the second dimension
    /// \param[in] d2      size of the third dimension
    /// \param[in] d3      size of the fourth dimension
    /// \param[in] seq_dim dimesion along which the range is created
    /// \param[in] ty      type
    /// \return            range array
    ///
    /// \ingroup data_func_range
    FLY_API array range(const dim_t d0, const dim_t d1 = 1, const dim_t d2 = 1,
                      const dim_t d3 = 1, const int seq_dim = -1, const dtype ty=f32);

    /// C++ Interface to generate an array with `[0, n-1]` values modified to
    /// specified dimensions and tiling.
    ///
    /// \param[in] dims      size
    /// \param[in] tile_dims number of tiled repetitions in each dimension
    /// \param[in] ty        type
    /// \return              iota array
    ///
    /// \ingroup data_func_iota
    FLY_API array iota(const dim4 &dims, const dim4 &tile_dims = dim4(1), const dtype ty=f32);

    /// C++ Interface to extract the diagonal from an array.
    ///
    /// \param[in] in      input array
    /// \param[in] num     diagonal index
    /// \param[in] extract if true, returns an array containing diagonal of the
    ///                    matrix; if false, returns a diagonal matrix
    /// \return            diagonal array (or matrix)
    ///
    /// \ingroup data_func_diag
    FLY_API array diag(const array &in, const int num = 0, const bool extract = true);

    /// C++ Interface to join 2 arrays along a dimension.
    ///
    /// Empty arrays are ignored.
    ///
    /// \param[in] dim    dimension along which the join occurs
    /// \param[in] first  input array
    /// \param[in] second input array
    /// \return           joined array
    ///
    /// \ingroup manip_func_join
    FLY_API array join(const int dim, const array &first, const array &second);

    /// C++ Interface to join 3 arrays along a dimension.
    ///
    /// Empty arrays are ignored.
    ///
    /// \param[in] dim    dimension along which the join occurs
    /// \param[in] first  input array
    /// \param[in] second input array
    /// \param[in] third  input array
    /// \return           joined array
    ///
    /// \ingroup manip_func_join
    FLY_API array join(const int dim, const array &first, const array &second, const array &third);

    /// C++ Interface to join 4 arrays along a dimension.
    ///
    /// Empty arrays are ignored.
    ///
    /// \param[in] dim    dimension along which the join occurs
    /// \param[in] first  input array
    /// \param[in] second input array
    /// \param[in] third  input array
    /// \param[in] fourth input array
    /// \return           joined array
    ///
    /// \ingroup manip_func_join
    FLY_API array join(const int dim, const array &first, const array &second,
                     const array &third, const array &fourth);

    /// C++ Interface to generate a tiled array.
    ///
    /// Note, `x`, `y`, `z`, and `w` include the original in the count.
    ///
    /// \param[in] in input array
    /// \param[in] x  number tiles along the first dimension
    /// \param[in] y  number tiles along the second dimension
    /// \param[in] z  number tiles along the third dimension
    /// \param[in] w  number tiles along the fourth dimension
    /// \return       tiled array
    ///
    /// \ingroup manip_func_tile
    FLY_API array tile(const array &in, const unsigned x, const unsigned y=1,
                     const unsigned z=1, const unsigned w=1);

    /// C++ Interface to generate a tiled array.
    ///
    /// Each component of `dims` includes the original in the count. Thus, if
    /// no duplicates are needed in a certain dimension, it is left as 1, the
    /// default value for just one copy.
    ///
    /// \param[in] in   input array
    /// \param[in] dims number of times `in` is copied along each dimension
    /// \return         tiled array
    ///
    /// \ingroup manip_func_tile
    FLY_API array tile(const array &in, const dim4 &dims);

    /// C++ Interface to reorder an array. 
    ///
    /// \param[in] in input array
    /// \param[in] x  specifies which dimension should be first
    /// \param[in] y  specifies which dimension should be second
    /// \param[in] z  specifies which dimension should be third
    /// \param[in] w  specifies which dimension should be fourth
    /// \return       reordered array
    ///
    /// \ingroup manip_func_reorder
    FLY_API array reorder(const array& in, const unsigned x,
                        const unsigned y=1, const unsigned z=2, const unsigned w=3);

    /// C++ Interface to shift an array.
    ///
    /// \param[in] in input array
    /// \param[in] x  specifies the shift along the first dimension
    /// \param[in] y  specifies the shift along the second dimension
    /// \param[in] z  specifies the shift along the third dimension
    /// \param[in] w  specifies the shift along the fourth dimension
    /// \return       shifted array
    ///
    /// \ingroup manip_func_shift
    FLY_API array shift(const array& in, const int x, const int y=0, const int z=0, const int w=0);

    /// C++ Interface to modify the dimensions of an input array to a specified
    /// shape.
    ///
    /// \param[in] in   input array
    /// \param[in] dims new dimension sizes
    /// \return         modded output
    ///
    /// \ingroup manip_func_moddims
    FLY_API array moddims(const array& in, const dim4& dims);

    /// C++ Interface to modify the dimensions of an input array to a specified
    /// shape.
    ///
    /// \param[in] in input array
    /// \param[in] d0 new size of the first dimension
    /// \param[in] d1 new size of the second dimension (optional)
    /// \param[in] d2 new size of the third dimension (optional)
    /// \param[in] d3 new size of the fourth dimension (optional)
    /// \return       modded output
    ///
    /// \ingroup manip_func_moddims
    FLY_API array moddims(const array& in, const dim_t d0, const dim_t d1=1, const dim_t d2=1, const dim_t d3=1);

    /// C++ Interface to modify the dimensions of an input array to a specified
    /// shape.
    ///
    /// \param[in] in    input array
    /// \param[in] ndims number of dimensions
    /// \param[in] dims  new dimension sizes
    /// \return          modded output
    ///
    /// \ingroup manip_func_moddims
    FLY_API array moddims(const array& in, const unsigned ndims, const dim_t* const dims);

    /// C++ Interface to flatten an array.
    ///
    /// \param[in] in input array
    /// \return       flat array
    ///
    /// \ingroup manip_func_flat
    FLY_API array flat(const array &in);

    /// C++ Interface to flip an array.
    ///
    /// \param[in] in  input array
    /// \param[in] dim dimension to flip
    /// \return        flipped array
    ///
    /// \ingroup manip_func_flip
    FLY_API array flip(const array &in, const unsigned dim);

    /// C++ Interface to return the lower triangle array.
    ///
    /// \param[in] in           input array
    /// \param[in] is_unit_diag boolean specifying if diagonal elements are 1's
    /// \return                 lower triangle array
    ///
    /// \ingroup data_func_lower
    FLY_API array lower(const array &in, bool is_unit_diag=false);

    /// C++ Interface to return the upper triangle array.
    ///
    /// \param[in] in           input array
    /// \param[in] is_unit_diag boolean specifying if diagonal elements are 1's
    /// \return                 upper triangle matrix
    ///
    /// \ingroup data_func_upper
    FLY_API array upper(const array &in, bool is_unit_diag=false);

    /// C++ Interface to select elements based on a conditional array.
    ///
    /// \param[in] cond conditional array
    /// \param[in] a    when true, select array element
    /// \param[in] b    when false, select array element
    /// \return         `a` when `cond` is true, else `b`
    ///
    /// \ingroup data_func_select
    FLY_API array select(const array &cond, const array  &a, const array  &b);

    /// C++ Interface to select elements based on a conditional array.
    ///
    /// \param[in] cond conditional array
    /// \param[in] a    when true, select array element
    /// \param[in] b    when false, select scalar value
    /// \return         `a` when `cond` is true, else `b`
    ///
    /// \ingroup data_func_select
    FLY_API array select(const array &cond, const array  &a, const double &b);

    /// C++ Interface to select elements based on a conditional array.
    ///
    /// \param[in] cond conditional array
    /// \param[in] a    when true, select scalar value
    /// \param[in] b    when false, select array element
    /// \return         `a` when `cond` is true, else `b`
    ///
    /// \ingroup data_func_select
    FLY_API array select(const array &cond, const double &a, const array  &b);

    /// C++ Interface to replace elements of an array with elements of another
    /// array.
    ///
    /// Elements of `a` are replaced with corresponding elements of `b` when
    /// `cond` is false.
    ///
    /// \param[inout] a    input array
    /// \param[in]    cond conditional array
    /// \param[in]    b    replacement array
    ///
    /// \ingroup data_func_replace
    FLY_API void replace(array &a, const array  &cond, const array  &b);

    /// C++ Interface to replace elements of an array with a scalar value.
    ///
    /// Elements of `a` are replaced with a scalar value when `cond` is false.
    ///
    /// \param[inout] a    input array
    /// \param[in]    cond conditional array
    /// \param[in]    b    replacement scalar value
    ///
    /// \ingroup data_func_replace
    FLY_API void replace(array &a, const array  &cond, const double &b);

    /// C++ Interface to pad an array.
    ///
    /// \param[in] in           input array
    /// \param[in] beginPadding number of elements to be padded at the start of
    ///                         each dimension
    /// \param[in] endPadding   number of elements to be padded at the end of
    ///                         each dimension
    /// \param[in] padFillType  values to fill into the padded region
    /// \return                 padded array
    ///
    /// \ingroup data_func_pad
    FLY_API array pad(const array &in, const dim4 &beginPadding,
                    const dim4 &endPadding, const borderType padFillType);

    /// C++ Interface to replace elements of an array with a scalar value.
    ///
    /// Elements of `a` are replaced with a scalar value when `cond` is false.
    ///
    /// \param[inout] a    input array
    /// \param[in]    cond conditional array
    /// \param[in]    b    replacement scalar value
    ///
    /// \ingroup data_func_replace
    FLY_API void replace(array &a, const array &cond, const long long b);

    /// C++ Interface to replace elements of an array with a scalar value.
    ///
    /// Elements of `a` are replaced with a scalar value when `cond` is false.
    ///
    /// \param[inout] a    input array
    /// \param[in]    cond conditional array
    /// \param[in]    b    replacement scalar value
    ///
    /// \ingroup data_func_replace
    FLY_API void replace(array &a, const array &cond,
                       const unsigned long long b);

    /// C++ Interface to select elements based on a conditional array.
    ///
    /// \param[in] cond conditional array
    /// \param[in] a    when true, select array element
    /// \param[in] b    when false, select scalar value
    /// \return         `a` when `cond` is true, else `b`
    ///
    /// \ingroup data_func_select
    FLY_API array select(const array &cond, const array &a, const long long b);

    /// C++ Interface to select elements based on a conditional array.
    ///
    /// \param[in] cond conditional array
    /// \param[in] a    when true, select array element
    /// \param[in] b    when false, select scalar value
    /// \return         `a` when `cond` is true, else `b`
    ///
    /// \ingroup data_func_select
    FLY_API array select(const array &cond, const array &a,
                       const unsigned long long b);

    /// C++ Interface to select elements based on a conditional array.
    ///
    /// \param[in] cond conditional array
    /// \param[in] a    when true, select scalar value
    /// \param[in] b    when false, select array element
    /// \return         `a` when `cond` is true, else `b`
    ///
    /// \ingroup data_func_select
    FLY_API array select(const array &cond, const long long a, const array &b);

    /// C++ Interface to select elements based on a conditional array.
    ///
    /// \param[in] cond conditional array
    /// \param[in] a    when true, select scalar value
    /// \param[in] b    when false, select array element
    /// \return         `a` when `cond` is true, else `b`
    ///
    /// \ingroup data_func_select
    FLY_API array select(const array &cond, const unsigned long long a,
                       const array &b);
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
    /**
       C Interface to generate an array with elements set to a specified value.

       \param[out] arr   constant array
       \param[in]  val   constant value
       \param[in]  ndims size of the dimension array
       \param[in]  dims  dimensions of the array to be generated
       \param[in]  type  type
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup data_func_constant
    */
    FLY_API fly_err fly_constant(fly_array *arr, const double val, const unsigned ndims, const dim_t * const dims, const fly_dtype type);

    /**
       C Interface to generate a complex array with elements set to a specified
       value.

       \param[out] arr   constant complex array
       \param[in]  real  real constant value
       \param[in]  imag  imaginary constant value
       \param[in]  ndims size of the dimension array
       \param[in]  dims  dimensions of the array to be generated
       \param[in]  type  type, \ref c32 or \ref c64
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup data_func_constant
    */
    FLY_API fly_err fly_constant_complex(fly_array *arr, const double real, const double imag,
                                     const unsigned ndims, const dim_t * const dims, const fly_dtype type);

    /**
       C Interface to generate an array with elements set to a specified value.

       Output type is \ref s64.

       \param[out] arr   constant array
       \param[in]  val   constant value
       \param[in]  ndims size of the dimension array
       \param[in]  dims  dimensions of the array to be generated
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup data_func_constant
    */
    FLY_API fly_err fly_constant_long (fly_array *arr, const long long val, const unsigned ndims, const dim_t * const dims);

    /**
       C Interface to generate an array with elements set to a specified value.

       Output type is \ref u64.

       \param[out] arr   constant array
       \param[in]  val   constant value
       \param[in]  ndims size of the dimension array
       \param[in]  dims  dimensions of the array to be generated
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup data_func_constant
    */

    FLY_API fly_err fly_constant_ulong(fly_array *arr, const unsigned long long val, const unsigned ndims, const dim_t * const dims);

    /**
       C Interface to generate an identity array.

       \param[out] out   identity array
       \param[in]  ndims number of dimensions
       \param[in]  dims  size
       \param[in]  type  type
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup data_func_identity
    */
    FLY_API fly_err fly_identity(fly_array* out, const unsigned ndims, const dim_t* const dims, const fly_dtype type);

    /**
       C Interface to generate an array with `[0, n-1]` values along the
       `seq_dim` dimension and tiled across other dimensions of shape `dim4`.

       \param[out] out     range array
       \param[in]  ndims   number of dimensions, specified by the size of `dims`
       \param[in]  dims    size
       \param[in]  seq_dim dimension along which the range is created
       \param[in]  type    type
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup data_func_range
    */
    FLY_API fly_err fly_range(fly_array *out, const unsigned ndims, const dim_t * const dims,
                          const int seq_dim, const fly_dtype type);

    /**
       C Interface to generate an array with `[0, n-1]` values modified to
       specified dimensions and tiling.

       \param[out] out     iota array
       \param[in]  ndims   number of dimensions
       \param[in]  dims    size
       \param[in]  t_ndims number of dimensions of tiled array
       \param[in]  tdims   number of tiled repetitions in each dimension
       \param[in]  type    type
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup data_func_iota
    */
    FLY_API fly_err fly_iota(fly_array *out, const unsigned ndims, const dim_t * const dims,
                         const unsigned t_ndims, const dim_t * const tdims, const fly_dtype type);

    /**
       C Interface to create a diagonal matrix from an extracted diagonal
       array.

       See also, \ref fly_diag_extract.

       \param[out] out diagonal matrix
       \param[in]  in  diagonal array
       \param[in]  num diagonal index
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup data_func_diag
    */
    FLY_API fly_err fly_diag_create(fly_array *out, const fly_array in, const int num);

    /**
       C Interface to extract the diagonal from an array.

       See also, \ref fly_diag_create.

       \param[out] out `num`-th diagonal array
       \param[in]  in  input array
       \param[in]  num diagonal index
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup data_func_diag
    */
    FLY_API fly_err fly_diag_extract(fly_array *out, const fly_array in, const int num);

    /**
       C Interface to join 2 arrays along a dimension.

       Empty arrays are ignored.

       \param[out] out    joined array
       \param[in]  dim    dimension along which the join occurs
       \param[in]  first  input array
       \param[in]  second input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup manip_func_join
    */
    FLY_API fly_err fly_join(fly_array *out, const int dim, const fly_array first, const fly_array second);

    /**
       C Interface to join many arrays along a dimension.

       Limited to 10 arrays. Empty arrays are ignored.

       \param[out] out      joined array
       \param[in]  dim      dimension along which the join occurs
       \param[in]  n_arrays number of arrays to join
       \param[in]  inputs   array of fly_arrays containing handles to the
                             arrays to be joined
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup manip_func_join
    */
    FLY_API fly_err fly_join_many(fly_array *out, const int dim, const unsigned n_arrays, const fly_array *inputs);

    /**
       C Interface to generate a tiled array.

       Note, `x`, `y`, `z`, and `w` include the original in the count.

       \param[out] out tiled array
       \param[in]  in  input array
       \param[in]  x   number of tiles along the first dimension
       \param[in]  y   number of tiles along the second dimension
       \param[in]  z   number of tiles along the third dimension
       \param[in]  w   number of tiles along the fourth dimension
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup manip_func_tile
    */
    FLY_API fly_err fly_tile(fly_array *out, const fly_array in,
                         const unsigned x, const unsigned y, const unsigned z, const unsigned w);

    /**
       C Interface to reorder an array.

       \param[out] out reordered array
       \param[in]  in  input array
       \param[in]  x   specifies which dimension should be first
       \param[in]  y   specifies which dimension should be second
       \param[in]  z   specifies which dimension should be third
       \param[in]  w   specifies which dimension should be fourth
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup manip_func_reorder
    */
    FLY_API fly_err fly_reorder(fly_array *out, const fly_array in,
                            const unsigned x, const unsigned y, const unsigned z, const unsigned w);

    /**
       C Interface to shift an array.

       \param[out] out shifted array
       \param[in]  in  input array
       \param[in]  x   specifies the shift along first dimension
       \param[in]  y   specifies the shift along second dimension
       \param[in]  z   specifies the shift along third dimension
       \param[in]  w   specifies the shift along fourth dimension
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup manip_func_shift
    */
    FLY_API fly_err fly_shift(fly_array *out, const fly_array in, const int x, const int y, const int z, const int w);

    /**
       C Interface to modify the dimensions of an input array to a specified
       shape.

       \param[out] out   modded output
       \param[in]  in    input array
       \param[in]  ndims number of dimensions
       \param[in]  dims  new dimension sizes
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup manip_func_moddims
    */
    FLY_API fly_err fly_moddims(fly_array *out, const fly_array in, const unsigned ndims, const dim_t * const dims);

    /**
       C Interface to flatten an array.

       \param[out] out flat array
       \param[in]  in  input array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup manip_func_flat
    */
    FLY_API fly_err fly_flat(fly_array *out, const fly_array in);

    /**
       C Interface to flip an array.

       \param[out] out flipped array
       \param[in]  in  input array
       \param[in]  dim dimension to flip
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup manip_func_flip
    */
    FLY_API fly_err fly_flip(fly_array *out, const fly_array in, const unsigned dim);

    /**
       C Interface to return the lower triangle array.

       \param[out] out          lower traingle array
       \param[in]  in           input array
       \param[in]  is_unit_diag boolean specifying if diagonal elements are 1's
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup data_func_lower
    */
    FLY_API fly_err fly_lower(fly_array *out, const fly_array in, bool is_unit_diag);

    /**
       C Interface to return the upper triangle array.

       \param[out] out          upper triangle array
       \param[in]  in           input array
       \param[in]  is_unit_diag boolean specifying if diagonal elements are 1's
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup data_func_upper
    */
    FLY_API fly_err fly_upper(fly_array *out, const fly_array in, bool is_unit_diag);

    /**
       C Interface to select elements based on a conditional array.

       \param[out] out  `a` when `cond` is true, else `b`
       \param[in]  cond conditional array
       \param[in]  a    when true, select array element
       \param[in]  b    when false, select array element
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup data_func_select
    */
    FLY_API fly_err fly_select(fly_array *out, const fly_array cond, const fly_array a, const fly_array b);

    /**
       C Interface to select elements based on a conditional array.

       \param[out] out  `a` when `cond` is true, else `b`
       \param[in]  cond conditional array
       \param[in]  a    when true, select array element
       \param[in]  b    when false, select scalar value
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup data_func_select
    */
    FLY_API fly_err fly_select_scalar_r(fly_array *out, const fly_array cond, const fly_array a, const double b);

    /**
       C Interface to select elements based on a conditional array.

       \param[out] out  `a` when `cond` is true, else `b`
       \param[in]  cond conditional array
       \param[in]  a    when true, select scalar value
       \param[in]  b    when false, select array element
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup data_func_select
    */
    FLY_API fly_err fly_select_scalar_l(fly_array *out, const fly_array cond, const double a, const fly_array b);

    /**
       C Interface to replace elements of an array with elements of another
       array.

       Elements of `a` are replaced with corresponding elements of `b` when
       `cond` is false.

       \param[inout]  a    input array
       \param[in]     cond conditional array
       \param[in]     b    replacement array
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup data_func_replace
    */
    FLY_API fly_err fly_replace(fly_array a, const fly_array cond, const fly_array b);

    /**
       C Interface to replace elements of an array with a scalar value.

       Elements of `a` are replaced with a scalar value when `cond` is false.

       \param[inout] a    input array
       \param[in]    cond conditional array
       \param[in]    b    replacement scalar value
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup data_func_replace
    */
    FLY_API fly_err fly_replace_scalar(fly_array a, const fly_array cond, const double b);

    /**
       C Interface to pad an array.

       \param[out] out           padded array
       \param[in]  in            input array
       \param[in]  begin_ndims   number of dimensions for start padding
       \param[in]  begin_dims    number of elements to be padded at the start
                                 of each dimension
       \param[in]  end_ndims     number of dimensions for end padding
       \param[in]  end_dims      number of elements to be padded at the end of
                                 each dimension
       \param[in]  pad_fill_type values to fill into the padded region
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup data_func_pad
    */
    FLY_API fly_err fly_pad(fly_array *out, const fly_array in,
                        const unsigned begin_ndims,
                        const dim_t *const begin_dims, const unsigned end_ndims,
                        const dim_t *const end_dims,
                        const fly_border_type pad_fill_type);

    /**
       C Interface to replace elements of an array with a scalar value.

       Elements of `a` are replaced with a scalar value when `cond` is false.

       \param[inout] a    input array
       \param[in]    cond conditional array
       \param[in]    b    replacement scalar value
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup data_func_replace
    */
    FLY_API fly_err fly_replace_scalar_long(fly_array a, const fly_array cond,
                                        const long long b);

    /**
       C Interface to replace elements of an array with a scalar value.

       Elements of `a` are replaced with a scalar value when `cond` is false.

       \param[inout] a    input array
       \param[in]    cond conditional array
       \param[in]    b    replacement scalar value
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup data_func_replace
    */
    FLY_API fly_err fly_replace_scalar_ulong(fly_array a, const fly_array cond,
                                         const unsigned long long b);

    /**
       C Interface to select elements based on a conditional array.

       \param[out] out  `a` when `cond` is true, else `b`
       \param[in]  cond conditional array
       \param[in]  a    when true, select array element
       \param[in]  b    when false, select scalar value
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup data_func_select
    */
    FLY_API fly_err fly_select_scalar_r_long(fly_array *out, const fly_array cond,
                                         const fly_array a, const long long b);

    /**
       C Interface to select elements based on a conditional array.

       \param[out] out  `a` when `cond` is true, else `b`
       \param[in]  cond conditional array
       \param[in]  a    when true, select array element
       \param[in]  b    when false, select scalar value
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup data_func_select
    */
    FLY_API fly_err fly_select_scalar_r_ulong(fly_array *out, const fly_array cond,
                                          const fly_array a,
                                          const unsigned long long b);

    /**
       C Interface to select elements based on a conditional array.

       \param[out] out  `a` when `cond` is true, else `b`
       \param[in]  cond conditional array
       \param[in]  a    when true, select scalar value
       \param[in]  b    when false, select array element
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup data_func_select
    */
    FLY_API fly_err fly_select_scalar_l_long(fly_array *out, const fly_array cond,
                                         const long long a, const fly_array b);

    /**
       C Interface to select elements based on a conditional array.

       \param[out] out  `a` when `cond` is true, else `b`
       \param[in]  cond conditional array
       \param[in]  a    when true, select scalar value
       \param[in]  b    when false, select array element
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup data_func_select
    */
    FLY_API fly_err fly_select_scalar_l_ulong(fly_array *out, const fly_array cond,
                                          const unsigned long long a,
                                          const fly_array b);

#ifdef __cplusplus
}
#endif
