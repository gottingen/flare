/*******************************************************
 * Copyright (c) 2022, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#ifdef __DH__
#undef __DH__
#endif

#ifdef __CUDACC__
#define __DH__ __device__ __host__
#else
#define __DH__
#endif

namespace flare {
namespace oneapi {}
}  // namespace flare

namespace detail = flare::oneapi;
