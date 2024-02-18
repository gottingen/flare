/*******************************************************
 * Copyright (c) 2017, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <anisotropic_diffusion.hpp>
#include <copy.hpp>
#include <kernel/anisotropic_diffusion.hpp>
#include <fly/dim4.hpp>

namespace flare {
namespace opencl {
template<typename T>
void anisotropicDiffusion(Array<T>& inout, const float dt, const float mct,
                          const fly::fluxFunction fftype,
                          const fly::diffusionEq eq) {
    if (eq == FLY_DIFFUSION_MCDE) {
        kernel::anisotropicDiffusion<T, true>(inout, dt, mct, fftype);
    } else {
        kernel::anisotropicDiffusion<T, false>(inout, dt, mct, fftype);
    }
}

#define INSTANTIATE(T)                                     \
    template void anisotropicDiffusion<T>(                 \
        Array<T> & inout, const float dt, const float mct, \
        const fly::fluxFunction fftype, const fly::diffusionEq eq);

INSTANTIATE(double)
INSTANTIATE(float)
}  // namespace opencl
}  // namespace flare
