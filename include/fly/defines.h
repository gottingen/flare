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

#ifndef __CUDACC_RTC__
#include <fly/compilers.h>
#endif

#if defined(_WIN32) || defined(_MSC_VER)
    // http://msdn.microsoft.com/en-us/library/b0084kay(v=VS.80).aspx
    // http://msdn.microsoft.com/en-us/library/3y1sfaz2%28v=VS.80%29.aspx
    #ifdef FLY_DLL // libaf
        #define FLY_API  __declspec(dllexport)
    #else
        #define FLY_API  __declspec(dllimport)
    #endif

    // bool
    #ifndef __cplusplus
        #define bool unsigned char
        #define false 0
        #define true  1
    #endif
    #define __PRETTY_FUNCTION__ __FUNCSIG__
    #define SIZE_T_FRMT_SPECIFIER "%Iu"
    #define FLY_DEPRECATED(msg) __declspec(deprecated( msg ))
    #if _MSC_VER >= 1800
        #define FLY_HAS_VARIADIC_TEMPLATES
    #endif
#else
    #define FLY_API   __attribute__((visibility("default")))
    #include <stdbool.h>
    #define SIZE_T_FRMT_SPECIFIER "%zu"
#if __GNUC__ >= 4 && __GNUC_MINOR > 4
    #define FLY_DEPRECATED(msg) __attribute__((deprecated( msg )))
#else
    #define FLY_DEPRECATED(msg) __attribute__((deprecated))
#endif
#endif

// Known 64-bit x86 and ARM architectures use long long
#if defined(__x86_64__) || defined(_M_X64) || defined(_WIN64) || defined(__aarch64__) || defined(__LP64__)   // 64-bit Architectures
    typedef long long   dim_t;
// Known 32-bit x86 and ARM architectures use int
#elif defined(__i386__) || defined(_M_IX86) || defined(__arm__) || defined(_M_ARM)     // 32-bit x86 Architecture
    typedef int         dim_t;
// All other platforms use long long
#else
    typedef long long   dim_t;
#endif

#include <stdlib.h>

#ifndef FLY_DLL  // prevents the use of these types internally
typedef FLY_DEPRECATED("intl is deprecated. Use long long instead.") long long intl;
typedef FLY_DEPRECATED("uintl is deprecated. Use unsigned long long instead.") unsigned long long uintl;
#endif

#include <fly/version.h>
#ifndef FLY_API_VERSION
#define FLY_API_VERSION FLY_API_VERSION_CURRENT
#endif

typedef enum {
    ///
    /// The function returned successfully
    ///
    FLY_SUCCESS            =   0

    // 100-199 Errors in environment

    ///
    /// The system or device ran out of memory
    ///
    , FLY_ERR_NO_MEM         = 101

    ///
    /// There was an error in the device driver
    ///
    , FLY_ERR_DRIVER         = 102

    ///
    /// There was an error with the runtime environment
    ///
    , FLY_ERR_RUNTIME        = 103

    // 200-299 Errors in input parameters

    ///
    /// The input array is not a valid fly_array object
    ///
    , FLY_ERR_INVALID_ARRAY  = 201

    ///
    /// One of the function arguments is incorrect
    ///
    , FLY_ERR_ARG            = 202

    ///
    /// The size is incorrect
    ///
    , FLY_ERR_SIZE           = 203

    ///
    /// The type is not suppported by this function
    ///
    , FLY_ERR_TYPE           = 204

    ///
    /// The type of the input arrays are not compatible
    ///
    , FLY_ERR_DIFF_TYPE      = 205

    ///
    /// Function does not support GFOR / batch mode
    ///
    , FLY_ERR_BATCH          = 207


    ///
    /// Input does not belong to the current device.
    ///
    , FLY_ERR_DEVICE         = 208

    // 300-399 Errors for missing software features

    ///
    /// The option is not supported
    ///
    , FLY_ERR_NOT_SUPPORTED  = 301

    ///
    /// This build of Flare does not support this feature
    ///
    , FLY_ERR_NOT_CONFIGURED = 302

    ///
    /// This build of Flare is not compiled with "nonfree" algorithms
    ///
    , FLY_ERR_NONFREE        = 303

    // 400-499 Errors for missing hardware features

    ///
    /// This device does not support double
    ///
    , FLY_ERR_NO_DBL         = 401

    ///
    /// This build of Flare was not built with graphics or this device does
    /// not support graphics
    ///
    , FLY_ERR_NO_GFX         = 402

    ///
    /// This device does not support half
    ///
    , FLY_ERR_NO_HALF        = 403

    // 500-599 Errors specific to heterogenous API

    ///
    /// There was an error when loading the libraries
    ///
    , FLY_ERR_LOAD_LIB       = 501

    ///
    /// There was an error when loading the symbols
    ///
    , FLY_ERR_LOAD_SYM       = 502

    ///
    /// There was a mismatch between the input array and the active backend
    ///
    , FLY_ERR_ARR_BKND_MISMATCH    = 503

    // 900-999 Errors from upstream libraries and runtimes

    ///
    /// There was an internal error either in Flare or in a project
    /// upstream
    ///
    , FLY_ERR_INTERNAL       = 998

    ///
    /// Unknown Error
    ///
    , FLY_ERR_UNKNOWN        = 999
} fly_err;

typedef enum {
    f32,    ///< 32-bit floating point values
    c32,    ///< 32-bit complex floating point values
    f64,    ///< 64-bit floating point values
    c64,    ///< 64-bit complex floating point values
    b8 ,    ///< 8-bit boolean values
    s32,    ///< 32-bit signed integral values
    u32,    ///< 32-bit unsigned integral values
    u8 ,    ///< 8-bit unsigned integral values
    s64,    ///< 64-bit signed integral values
    u64    ///< 64-bit unsigned integral values
    , s16    ///< 16-bit signed integral values
    , u16    ///< 16-bit unsigned integral values
    , f16    ///< 16-bit floating point value
} fly_dtype;

typedef enum {
    flyDevice,   ///< Device pointer
    flyHost     ///< Host pointer
} fly_source;

#define FLY_MAX_DIMS 4

// A handle for an internal array object
typedef void * fly_array;

typedef enum {
    FLY_INTERP_NEAREST,         ///< Nearest Interpolation
    FLY_INTERP_LINEAR,          ///< Linear Interpolation
    FLY_INTERP_BILINEAR,        ///< Bilinear Interpolation
    FLY_INTERP_CUBIC,           ///< Cubic Interpolation
    FLY_INTERP_LOWER           ///< Floor Indexed
    , FLY_INTERP_LINEAR_COSINE   ///< Linear Interpolation with cosine smoothing
    , FLY_INTERP_BILINEAR_COSINE ///< Bilinear Interpolation with cosine smoothing
    , FLY_INTERP_BICUBIC         ///< Bicubic Interpolation
    , FLY_INTERP_CUBIC_SPLINE    ///< Cubic Interpolation with Catmull-Rom splines
    , FLY_INTERP_BICUBIC_SPLINE  ///< Bicubic Interpolation with Catmull-Rom splines

} fly_interp_type;

typedef enum {
    ///
    /// Out of bound values are 0
    ///
    FLY_PAD_ZERO = 0,

    ///
    /// Out of bound values are symmetric over the edge
    ///
    FLY_PAD_SYM,

    ///
    /// Out of bound values are clamped to the edge
    ///
    FLY_PAD_CLAMP_TO_EDGE,

    ///
    /// Out of bound values are mapped to range of the dimension in cyclic fashion
    ///
    FLY_PAD_PERIODIC
} fly_border_type;

typedef enum {
    ///
    /// Connectivity includes neighbors, North, East, South and West of current pixel
    ///
    FLY_CONNECTIVITY_4 = 4,

    ///
    /// Connectivity includes 4-connectivity neigbors and also those on Northeast, Northwest, Southeast and Southwest
    ///
    FLY_CONNECTIVITY_8_4 = 8
} fly_connectivity;

typedef enum {

    ///
    /// Output of the convolution is the same size as input
    ///
    FLY_CONV_DEFAULT,

    ///
    /// Output of the convolution is signal_len + filter_len - 1
    ///
    FLY_CONV_EXPAND
} fly_conv_mode;

typedef enum {
    FLY_CONV_AUTO,    ///< Flare automatically picks the right convolution algorithm
    FLY_CONV_SPATIAL, ///< Perform convolution in spatial domain
    FLY_CONV_FREQ     ///< Perform convolution in frequency domain
} fly_conv_domain;

typedef enum {
    FLY_SAD = 0,   ///< Match based on Sum of Absolute Differences (SAD)
    FLY_ZSAD,      ///< Match based on Zero mean SAD
    FLY_LSAD,      ///< Match based on Locally scaled SAD
    FLY_SSD,       ///< Match based on Sum of Squared Differences (SSD)
    FLY_ZSSD,      ///< Match based on Zero mean SSD
    FLY_LSSD,      ///< Match based on Locally scaled SSD
    FLY_NCC,       ///< Match based on Normalized Cross Correlation (NCC)
    FLY_ZNCC,      ///< Match based on Zero mean NCC
    FLY_SHD        ///< Match based on Sum of Hamming Distances (SHD)
} fly_match_type;

typedef enum {
    FLY_YCC_601 = 601,  ///< ITU-R BT.601 (formerly CCIR 601) standard
    FLY_YCC_709 = 709,  ///< ITU-R BT.709 standard
    FLY_YCC_2020 = 2020 ///< ITU-R BT.2020 standard
} fly_ycc_std;

typedef enum {
    FLY_GRAY = 0, ///< Grayscale
    FLY_RGB,      ///< 3-channel RGB
    FLY_HSV       ///< 3-channel HSV
    , FLY_YCbCr     ///< 3-channel YCbCr
} fly_cspace_t;

typedef enum {
    FLY_MAT_NONE       = 0,    ///< Default
    FLY_MAT_TRANS      = 1,    ///< Data needs to be transposed
    FLY_MAT_CTRANS     = 2,    ///< Data needs to be conjugate tansposed
    FLY_MAT_CONJ       = 4,    ///< Data needs to be conjugate
    FLY_MAT_UPPER      = 32,   ///< Matrix is upper triangular
    FLY_MAT_LOWER      = 64,   ///< Matrix is lower triangular
    FLY_MAT_DIAG_UNIT  = 128,  ///< Matrix diagonal contains unitary values
    FLY_MAT_SYM        = 512,  ///< Matrix is symmetric
    FLY_MAT_POSDEF     = 1024, ///< Matrix is positive definite
    FLY_MAT_ORTHOG     = 2048, ///< Matrix is orthogonal
    FLY_MAT_TRI_DIAG   = 4096, ///< Matrix is tri diagonal
    FLY_MAT_BLOCK_DIAG = 8192  ///< Matrix is block diagonal
} fly_mat_prop;

typedef enum {
    FLY_NORM_VECTOR_1,      ///< treats the input as a vector and returns the sum of absolute values
    FLY_NORM_VECTOR_INF,    ///< treats the input as a vector and returns the max of absolute values
    FLY_NORM_VECTOR_2,      ///< treats the input as a vector and returns euclidean norm
    FLY_NORM_VECTOR_P,      ///< treats the input as a vector and returns the p-norm
    FLY_NORM_MATRIX_1,      ///< return the max of column sums
    FLY_NORM_MATRIX_INF,    ///< return the max of row sums
    FLY_NORM_MATRIX_2,      ///< returns the max singular value). Currently NOT SUPPORTED
    FLY_NORM_MATRIX_L_PQ,   ///< returns Lpq-norm

    FLY_NORM_EUCLID = FLY_NORM_VECTOR_2 ///< The default. Same as FLY_NORM_VECTOR_2
} fly_norm_type;

typedef enum {
    FLY_FIF_BMP          = 0,    ///< FreeImage Enum for Bitmap File
    FLY_FIF_ICO          = 1,    ///< FreeImage Enum for Windows Icon File
    FLY_FIF_JPEG         = 2,    ///< FreeImage Enum for JPEG File
    FLY_FIF_JNG          = 3,    ///< FreeImage Enum for JPEG Network Graphics File
    FLY_FIF_PNG          = 13,   ///< FreeImage Enum for Portable Network Graphics File
    FLY_FIF_PPM          = 14,   ///< FreeImage Enum for Portable Pixelmap (ASCII) File
    FLY_FIF_PPMRAW       = 15,   ///< FreeImage Enum for Portable Pixelmap (Binary) File
    FLY_FIF_TIFF         = 18,   ///< FreeImage Enum for Tagged Image File Format File
    FLY_FIF_PSD          = 20,   ///< FreeImage Enum for Adobe Photoshop File
    FLY_FIF_HDR          = 26,   ///< FreeImage Enum for High Dynamic Range File
    FLY_FIF_EXR          = 29,   ///< FreeImage Enum for ILM OpenEXR File
    FLY_FIF_JP2          = 31,   ///< FreeImage Enum for JPEG-2000 File
    FLY_FIF_RAW          = 34    ///< FreeImage Enum for RAW Camera Image File
} fly_image_format;

typedef enum {
    FLY_MOMENT_M00 = 1,
    FLY_MOMENT_M01 = 2,
    FLY_MOMENT_M10 = 4,
    FLY_MOMENT_M11 = 8,
    FLY_MOMENT_FIRST_ORDER = FLY_MOMENT_M00 | FLY_MOMENT_M01 | FLY_MOMENT_M10 | FLY_MOMENT_M11
} fly_moment_type;

typedef enum {
    FLY_HOMOGRAPHY_RANSAC = 0,   ///< Computes homography using RANSAC
    FLY_HOMOGRAPHY_LMEDS  = 1    ///< Computes homography using Least Median of Squares
} fly_homography_type;

// These enums should be 2^x
typedef enum {
    FLY_BACKEND_DEFAULT = 0,  ///< Default backend order: OpenCL -> CUDA -> CPU
    FLY_BACKEND_CPU     = 1,  ///< CPU a.k.a sequential algorithms
    FLY_BACKEND_CUDA    = 2,  ///< CUDA Compute Backend
} fly_backend;

// Below enum is purely added for example purposes
// it doesn't and shoudn't be used anywhere in the
// code. No Guarantee's provided if it is used.
typedef enum {
    FLY_ID = 0
} fly_someenum_t;

typedef enum {
    FLY_BINARY_ADD  = 0,
    FLY_BINARY_MUL  = 1,
    FLY_BINARY_MIN  = 2,
    FLY_BINARY_MAX  = 3
} fly_binary_op;

typedef enum {
    FLY_RANDOM_ENGINE_PHILOX_4X32_10     = 100,                                  //Philox variant with N = 4, W = 32 and Rounds = 10
    FLY_RANDOM_ENGINE_THREEFRY_2X32_16   = 200,                                  //Threefry variant with N = 2, W = 32 and Rounds = 16
    FLY_RANDOM_ENGINE_MERSENNE_GP11213   = 300,                                  //Mersenne variant with MEXP = 11213
    FLY_RANDOM_ENGINE_PHILOX             = FLY_RANDOM_ENGINE_PHILOX_4X32_10,      //Resolves to Philox 4x32_10
    FLY_RANDOM_ENGINE_THREEFRY           = FLY_RANDOM_ENGINE_THREEFRY_2X32_16,    //Resolves to Threefry 2X32_16
    FLY_RANDOM_ENGINE_MERSENNE           = FLY_RANDOM_ENGINE_MERSENNE_GP11213,    //Resolves to Mersenne GP 11213
    FLY_RANDOM_ENGINE_DEFAULT            = FLY_RANDOM_ENGINE_PHILOX               //Resolves to Philox
} fly_random_engine_type;

////////////////////////////////////////////////////////////////////////////////
// theia / Graphics Related Enums
// These enums have values corresponsding to Theia enums in theia defines.h
////////////////////////////////////////////////////////////////////////////////
typedef enum {
    FLY_COLORMAP_DEFAULT = 0,    ///< Default grayscale map
    FLY_COLORMAP_SPECTRUM= 1,    ///< Spectrum map (390nm-830nm, in sRGB colorspace)
    FLY_COLORMAP_COLORS  = 2,    ///< Colors, aka. Rainbow
    FLY_COLORMAP_RED     = 3,    ///< Red hue map
    FLY_COLORMAP_MOOD    = 4,    ///< Mood map
    FLY_COLORMAP_HEAT    = 5,    ///< Heat map
    FLY_COLORMAP_BLUE    = 6,    ///< Blue hue map
    FLY_COLORMAP_INFERNO = 7,    ///< Perceptually uniform shades of black-red-yellow
    FLY_COLORMAP_MAGMA   = 8,    ///< Perceptually uniform shades of black-red-white
    FLY_COLORMAP_PLASMA  = 9,    ///< Perceptually uniform shades of blue-red-yellow
    FLY_COLORMAP_VIRIDIS = 10    ///< Perceptually uniform shades of blue-green-yellow
} fly_colormap;

typedef enum {
    FLY_MARKER_NONE         = 0,
    FLY_MARKER_POINT        = 1,
    FLY_MARKER_CIRCLE       = 2,
    FLY_MARKER_SQUARE       = 3,
    FLY_MARKER_TRIANGLE     = 4,
    FLY_MARKER_CROSS        = 5,
    FLY_MARKER_PLUS         = 6,
    FLY_MARKER_STAR         = 7
} fly_marker_type;

////////////////////////////////////////////////////////////////////////////////


typedef enum {
    FLY_CANNY_THRESHOLD_MANUAL    = 0, ///< User has to define canny thresholds manually
    FLY_CANNY_THRESHOLD_AUTO_OTSU = 1  ///< Determine canny algorithm thresholds using Otsu algorithm
} fly_canny_threshold;

typedef enum {
    FLY_STORAGE_DENSE     = 0,   ///< Storage type is dense
    FLY_STORAGE_CSR       = 1,   ///< Storage type is CSR
    FLY_STORAGE_CSC       = 2,   ///< Storage type is CSC
    FLY_STORAGE_COO       = 3    ///< Storage type is COO
} fly_storage;

typedef enum {
    FLY_FLUX_QUADRATIC   = 1,    ///< Quadratic flux function
    FLY_FLUX_EXPONENTIAL = 2,    ///< Exponential flux function
    FLY_FLUX_DEFAULT     = 0     ///< Default flux function is exponential
} fly_flux_function;

typedef enum {
    FLY_DIFFUSION_GRAD = 1,      ///< Gradient diffusion equation
    FLY_DIFFUSION_MCDE = 2,      ///< Modified curvature diffusion equation
    FLY_DIFFUSION_DEFAULT = 0    ///< Default option is same as FLY_DIFFUSION_GRAD
} fly_diffusion_eq;

typedef enum {
    FLY_TOPK_MIN         = 1,  ///< Top k min values
    FLY_TOPK_MAX         = 2,  ///< Top k max values
    FLY_TOPK_STABLE      = 4,  ///< Preserve order of indices for equal values
    FLY_TOPK_STABLE_MIN  = FLY_TOPK_STABLE | FLY_TOPK_MIN, ///< Top k min with stable indices
    FLY_TOPK_STABLE_MAX  = FLY_TOPK_STABLE | FLY_TOPK_MAX, ///< Top k max with stable indices
    FLY_TOPK_DEFAULT = 0   ///< Default option (max)
} fly_topk_function;

typedef enum {
    FLY_VARIANCE_DEFAULT    = 0, ///< Default (Population) variance
    FLY_VARIANCE_SAMPLE     = 1, ///< Sample variance
    FLY_VARIANCE_POPULATION = 2  ///< Population variance
} fly_var_bias;

typedef enum {
    FLY_ITERATIVE_DECONV_LANDWEBER       = 1,        ///< Landweber Deconvolution
    FLY_ITERATIVE_DECONV_RICHARDSONLUCY  = 2,        ///< Richardson-Lucy Deconvolution
    FLY_ITERATIVE_DECONV_DEFAULT         = 0         ///< Default is Landweber deconvolution
} fly_iterative_deconv_algo;

typedef enum {
    FLY_INVERSE_DECONV_TIKHONOV       = 1,        ///< Tikhonov Inverse deconvolution
    FLY_INVERSE_DECONV_DEFAULT        = 0         ///< Default is Tikhonov deconvolution
} fly_inverse_deconv_algo;


typedef enum {
    FLY_CONV_GRADIENT_DEFAULT = 0,
    FLY_CONV_GRADIENT_FILTER  = 1,
    FLY_CONV_GRADIENT_DATA    = 2,
    FLY_CONV_GRADIENT_BIAS    = 3
} fly_conv_gradient_type;

#ifdef __cplusplus
namespace fly
{
    typedef fly_dtype dtype;
    typedef fly_source source;
    typedef fly_interp_type interpType;
    typedef fly_border_type borderType;
    typedef fly_connectivity connectivity;
    typedef fly_match_type matchType;
    typedef fly_cspace_t CSpace;
    typedef fly_someenum_t SomeEnum; // Purpose of Addition: How to add Function example
    typedef fly_mat_prop trans;
    typedef fly_conv_mode convMode;
    typedef fly_conv_domain convDomain;
    typedef fly_mat_prop matProp;
    typedef fly_colormap ColorMap;
    typedef fly_norm_type normType;
    typedef fly_ycc_std YCCStd;
    typedef fly_image_format imageFormat;
    typedef fly_backend Backend;
    typedef fly_marker_type markerType;
    typedef fly_moment_type momentType;
    typedef fly_storage storage;
    typedef fly_binary_op binaryOp;
    typedef fly_random_engine_type randomEngineType;
    typedef fly_canny_threshold cannyThreshold;
    typedef fly_flux_function fluxFunction;
    typedef fly_diffusion_eq diffusionEq;
    typedef fly_topk_function topkFunction;
    typedef fly_var_bias varBias;
    typedef fly_iterative_deconv_algo iterativeDeconvAlgo;
    typedef fly_inverse_deconv_algo inverseDeconvAlgo;
    typedef fly_conv_gradient_type convGradientType;
}

#endif
