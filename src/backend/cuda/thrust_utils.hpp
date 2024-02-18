/*******************************************************
 * Copyright (c) 2019, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <ThrustFlarePolicy.hpp>
#include <thrust/system/cuda/detail/par.h>
#include <thrust/version.h>
#include <ThrustAllocator.cuh>

namespace flare {
namespace cuda {
template<typename T>
using ThrustVector = thrust::device_vector<T, ThrustAllocator<T>>;
}  // namespace cuda
}  // namespace flare

#define THRUST_SELECT(fn, ...) \
    fn(flare::cuda::ThrustFlarePolicy(), __VA_ARGS__)
#define THRUST_SELECT_OUT(res, fn, ...) \
    res = fn(flare::cuda::ThrustFlarePolicy(), __VA_ARGS__)
