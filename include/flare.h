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

/**

\defgroup flare_func Flare Functions
@{
@}

\defgroup flare_class Flare Classes
@{
@}

\defgroup graphics_func Graphics
@{
@}

@defgroup func_categories Flare Functions by Category
@{

   @defgroup array_mat Functions to create and modify Arrays
   @{

      Array constructors, random number generation, transpose, indexing, etc.

      @defgroup device_mat Managing devices in Flare
      getting device pointer, allocating and freeing memory

      @defgroup data_mat Functions to create arrays.
      constant, random, range, etc.

      @defgroup c_api_mat C API to manage arrays
      Create, release, copy, fetch-properties of \ref fly_array

      @defgroup index_mat Assignment & Indexing operation on arrays
      Access sub regions of an array object

      @defgroup manip_mat Move and Reorder array content
      reorder, transpose, flip, join, tile, etc.

      @defgroup helper_mat Helper functions for arrays
      iszero, isInf, isNan, etc.
   @}

   @defgroup mathfunc_mat Mathematical functions
   @{

      Functions from standard math library

      @defgroup arith_mat Arithmetic operations
      +, -, *, /, >>, <<

      @defgroup logic_mat  Logical operations
      &&, ||, |, &, <, >, <=, >=, ==, !

      @defgroup numeric_mat Numeric functions
      floor, round, min, max, etc.

      @defgroup trig_mat Trigonometric functions
      sin, cos, tan, etc.

      @defgroup explog_mat Exponential and logarithmic functions
      exp, log, expm1, log1p, etc.

      @defgroup hyper_mat Hyperbolic functions
      sinh, cosh, tanh, etc.

      @defgroup complex_mat Complex operations
      real, imag, conjugate etc.
   @}

   @defgroup vector_mat Vector Algorithms
   @{

      sum, min, max, sort, set operations, etc.

      @defgroup reduce_mat Reduction operations
      sum, min, max, etc.

      @defgroup sort_mat Sort operations
      sort, sort by key, etc.

      @defgroup scan_mat Inclusive scan operations
      inclusive / cumulative sum, etc.

      @defgroup set_mat Set operations
      unique, union, intersect

      @defgroup calc_mat Numerical differentiation
      diff, gradient, etc.
   @}

   @defgroup memory_manager Memory Management
   @{
      Interfaces for writing custom memory managers.

      Create and set a custom memory manager by first defining the relevant
      closures for each required function, for example:

      \code{.cpp}
          fly_err my_initialize(fly_memory_manager manager) {
              void* myPayload = malloc(sizeof(MyPayload_t));
              fly_memory_manager_set_payload(manager, myPayload);
              // ...
          }

          fly_err my_allocated(fly_memory_manager handle, size_t* size, void* ptr) {
              void* myPayload;
              fly_memory_manager_get_payload(manager, &myPayload);
              // ...
          }
      \endcode

      Create an \ref fly_memory_manager and attach relevant closures:

      \code{.cpp}
          fly_memory_manager manager;
          fly_create_memory_manager(&manager);

          fly_memory_manager_set_initialize_fn(manager, my_initialize);
          fly_memory_manager_set_allocated_fn(manager, my_allocated);

          // ...
      \endcode

      Set the memory manager to be active, which shuts down the existing memory
      manager:

      \code{.cpp}
          fly_set_memory_manager(manager);
      \endcode

      Unset to re-create and reset an instance of the default memory manager:

      \code{.cpp}
          fly_unset_memory_manager();
      \endcode

      @defgroup native_memory_interface Native Memory Interface
      \brief Native alloc, native free, get device id, etc.

      @defgroup memory_manager_utils Memory Manager Utils
      \brief Set and unset memory managers, set and get manager payloads,
              function setters

      @defgroup memory_manager_api Memory Manager API
      \brief Functions for defining custom memory managers
   @}

   @defgroup event Events
   @{

      \brief Managing Flare Events which allows manipulation of operations
              on computation queues.

      \defgroup event_api Event API
      \brief fly_create_event, fly_mark_event, etc.
   @}

   @defgroup linalg_mat Linear Algebra
   @{

     Matrix multiply, solve, decompositions, sparse matrix

     @defgroup blas_mat BLAS operations
     Matrix multiply, dot product, etc.

     @defgroup lapack_factor_mat Matrix factorizations and decompositions
     LU, QR, Cholesky etc.

     @defgroup lapack_solve_mat Linear solve and least squares
     solve, solveLU, etc.

     @defgroup lapack_ops_mat Matrix operations
     inverse, det, rank, norm etc.

     @defgroup lapack_helper LAPACK Helper functions

     @defgroup sparse_func Sparse functions
        \brief Functions to create and handle sparse arrays and matrix operations

        Sparse array in Flare use the same \ref fly::array (or \ref fly_array)
        handle as normal. Internally, this handle is used to maintain a structure
        of the sparse array (components listed below).

        Description     | Data Type
        ----------------|-------------------
        Values          | T (one of \ref f32, \ref f64, \ref c32, \ref c64)
        Row Indices     | Int (\ref s32)
        Column Indices  | Int (\ref s32)
        Storage         | \ref fly::storage

        The value array contains the non-zero elements of the matrix. The
        \ref fly::dtype of the value array is the same as that of the matrix.
        The size of this array is the same as the number of non-zero elements
        of the matrix.

        The row indices and column indices contain the indices based on
        \ref fly::storage type. These \ref fly::array are always of type \ref s32.

        The \ref fly::storage is used to determin the type of storage to use.
        Currently \ref FLY_STORAGE_CSR and \ref FLY_STORAGE_COO are available.

        A sparse array can be identied using the \ref fly::array::issparse()
        function.  This function will return true for a sparse array and false
        for a regular \ref fly::array.

        The valid operations on sparse arrays are \ref fly::matmul (sparse-dense).
        When calling matmul for sparse matrices, the sparse array is required to
        be the left hand side matrix and can be used with transposing options.
        The dense matrix on the right hand side cannot be used with any transpose
        options.

        Most functions cannot use sparse arrays and will throw an error with
        \ref FLY_ERR_ARG if a sparse array is given as input.

        \note Sparse functionality support was added to Flare in v3.4.0.

   @}

   @defgroup image_mat Image Processing
   @{

     Image filtering, morphing and transformations

     @defgroup colorconv_mat Colorspace conversions
     RGB to gray, gray to RGB, RGB to HSV, etc.

     @defgroup hist_mat Histograms
     Image and data histograms

     @defgroup moments_mat Image moments
     Centroids, areas, etc.

     @defgroup transform_mat Image transformations
     rotate, skew, etc.

     @defgroup morph_mat Morphological Operations
     erode, dilate, etc.

     @defgroup imageflt_mat Filters
     bilateral, sobel, mean shift, median / min / max filters etc.

     @defgroup connected_comps_mat Connected Components & Labeling
     regions

     @defgroup image_mod_mat Wrapping and unwrapping image windows
     wrap, unwrap, etc.

     @defgroup utility_mat Utility Functions
     loadImage, saveImage, gaussianKernel
   @}

   @defgroup cv_mat Computer Vision
   @{

     A list of computer vision algorithms

     @defgroup featdetect_mat Feature detectors
     FAST feature detector

     @defgroup featdescriptor_mat Feature descriptors
     ORB feature descriptor

     @defgroup featmatcher_mat Feature matchers
     Feature matchers

     @defgroup match_mat Template matching
   @}

   @defgroup signal_mat Signal Processing
   @{

     Convolutions, FFTs, filters

     @defgroup convolve_mat Convolutions
     1D, 2D and 3D convolutions

     @defgroup sigfilt_mat Filter
     fir, iir, etc.

     @defgroup fft_mat      Fast Fourier Transforms
     1D, 2D and 3D forward, inverse FFTs

     @defgroup approx_mat   Interpolation and approximation
     1D and 2D interpolation
   @}

   @defgroup stats_mat Statistics
   @{

     A list of Statistics functions
     @defgroup basicstats_mat Basic statistics functions
     mean, median, variance, etc.
   @}

   @defgroup io_mat Input and Output functions
   @{

     Functions to read and write data

     @defgroup dataio_mat Reading and writing arrays
     printing data to screen / files

     @defgroup imageio_mat Reading and writing images
     Reading and writing images
   @}

   @defgroup unified_func Unified API Functions
   @{

     Functions to set current backend and utilities

   @}

   @defgroup internal_func Functions to work with internal array layout
   @{

     Functions to work with flare's internal data structure.

     Note: The behavior of these functions is not promised to be consistent across versions.

   @}

   @defgroup external Interface Functions
   @{

     Backend specific functions

     @defgroup cuda_mat CUDA specific functions

        \brief Accessing Flare's stream, and native device id with other CUDA code.

        If your software is using Flare's CUDA backend, you can also write custom
        kernels and do custom memory operations using native CUDA commands. The functions
        contained in the \p flycu namespace provide methods to get the stream and native
        device id that Flare is using.
   @}

   @defgroup ml Machine Learning
   @{

     Machine learning functions

     @defgroup ml_convolution Convolutions
     Forward and backward convolution passes
   @}
@}


*/

#include "fly/compatible.h"
#include "fly/algorithm.h"
#include "fly/arith.h"
#include "fly/array.h"
#include "fly/backend.h"
#include "fly/blas.h"
#include "fly/complex.h"
#include "fly/constants.h"
#include "fly/data.h"
#include "fly/device.h"
#include "fly/event.h"
#include "fly/exception.h"
#include "fly/features.h"
#include "fly/gfor.h"
#include "fly/graphics.h"
#include "fly/half.h"
#include "fly/image.h"
#include "fly/index.h"
#include "fly/lapack.h"
#include "fly/memory.h"
#include "fly/ml.h"
#include "fly/random.h"
#include "fly/seq.h"
#include "fly/signal.h"
#include "fly/sparse.h"
#include "fly/statistics.h"
#include "fly/timing.h"
#include "fly/util.h"
#include "fly/version.h"
#include "fly/vision.h"
