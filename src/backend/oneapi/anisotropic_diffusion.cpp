/*******************************************************
 * Copyright (c) 2022, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <anisotropic_diffusion.hpp>
#include <copy.hpp>
#include <err_oneapi.hpp>
#include <fly/dim4.hpp>

namespace flare {
namespace oneapi {
template<typename T>
void anisotropicDiffusion(Array<T>& inout, const float dt, const float mct,
                          const fly::fluxFunction fftype,
                          const fly::diffusionEq eq) {
    ONEAPI_NOT_SUPPORTED("");
}

#define INSTANTIATE(T)                                     \
    template void anisotropicDiffusion<T>(                 \
        Array<T> & inout, const float dt, const float mct, \
        const fly::fluxFunction fftype, const fly::diffusionEq eq);

INSTANTIATE(double)
INSTANTIATE(float)
}  // namespace oneapi
}  // namespace flare
