/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fly/dim4.hpp>
#include <fly/half.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wparentheses"
#include "half.hpp"
#pragma GCC diagnostic pop

#ifdef FLY_CUDA
#include <cuda_fp16.h>
#endif

#include <cstring>

namespace fly {

/// Get the first non-zero dimension
static inline dim_t getFNSD(const int dim, fly::dim4 dims) {
    if (dim >= 0) return dim;

    dim_t fNSD = 0;
    for (dim_t i = 0; i < 4; ++i) {
        if (dims[i] > 1) {
            fNSD = i;
            break;
        }
    }
    return fNSD;
}

namespace {
// casts from one type to another. Needed for fly_half conversions specialization
template<typename To, typename T>
inline To cast(T in) {
    return static_cast<To>(in);
}

#if defined(FLY_CUDA) && CUDA_VERSION < 10000
template<>
inline __half cast<__half, double>(double in) {
    __half_raw out;
    half_float::half h(in);
    memcpy(&out, &h, sizeof(__half_raw));
    return out;
}
#endif

template<>
[[gnu::unused]] fly_half cast<fly_half, double>(double in) {
    half_float::half tmp = static_cast<half_float::half>(in);
    fly_half out;
    memcpy(&out, &tmp, sizeof(fly_half));
    return out;
}

}  // namespace
}  // namespace fly
