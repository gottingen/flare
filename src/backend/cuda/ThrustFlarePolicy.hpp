/*******************************************************
 * Copyright (c) 2020, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <backend.hpp>
#include <memory.hpp>
#include <platform.hpp>
#include <thrust/memory.h>
#include <thrust/system/cuda/execution_policy.h>

namespace flare {
namespace cuda {
struct ThrustFlarePolicy
    : thrust::cuda::execution_policy<ThrustFlarePolicy> {};

template<typename T>
thrust::pair<thrust::pointer<T, ThrustFlarePolicy>, std::ptrdiff_t>
get_temporary_buffer(ThrustFlarePolicy, std::ptrdiff_t n) {
    thrust::pointer<T, ThrustFlarePolicy> result(
        flare::cuda::memAlloc<T>(n / sizeof(T)).release());

    return thrust::make_pair(result, n);
}

template<typename Pointer>
inline void return_temporary_buffer(ThrustFlarePolicy, Pointer p) {
    memFree(thrust::raw_pointer_cast(p));
}

}  // namespace cuda
}  // namespace flare

namespace thrust {
namespace cuda_cub {
template<>
__DH__ inline cudaStream_t get_stream<flare::cuda::ThrustFlarePolicy>(
    execution_policy<flare::cuda::ThrustFlarePolicy> &) {
#if defined(__CUDA_ARCH__)
    return 0;
#else
    return flare::cuda::getActiveStream();
#endif
}

__DH__
inline cudaError_t synchronize_stream(
    const flare::cuda::ThrustFlarePolicy &) {
#if defined(__CUDA_ARCH__)
    return cudaSuccess;
#else
    return cudaStreamSynchronize(flare::cuda::getActiveStream());
#endif
}

}  // namespace cuda_cub
}  // namespace thrust
