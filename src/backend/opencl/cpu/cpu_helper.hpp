/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#ifndef FLY_OPENCL_CPU
#define FLY_OPENCL_CPU

#include <Array.hpp>
#include <memory.hpp>
#include <platform.hpp>
#include <types.hpp>

//********************************************************/
// LAPACK
//********************************************************/
#if defined(WITH_LINEAR_ALGEBRA)

#define lapack_complex_float flare::opencl::cfloat
#define lapack_complex_double flare::opencl::cdouble
#define LAPACK_PREFIX LAPACKE_
#define ORDER_TYPE int
#define FLY_LAPACK_COL_MAJOR LAPACK_COL_MAJOR
#define LAPACK_NAME(fn) LAPACKE_##fn

#ifdef USE_MKL
#include <mkl_lapack.h>
#include <mkl_lapacke.h>
#else
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#include <common/lapacke.hpp>
#undef FLY_LAPACK_COL_MAJOR
#define FLY_LAPACK_COL_MAJOR 0
#else  // NETLIB LAPACKE
#include <lapacke.h>
#endif
#endif

#endif  // WITH_LINEAR_ALGEBRA

#endif  // FLY_OPENCL_CPU
