/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#ifndef FLYCPU_LAPACK
#define FLYCPU_LAPACK

#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
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

#endif
