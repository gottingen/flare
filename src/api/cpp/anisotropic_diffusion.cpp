/*******************************************************
 * Copyright (c) 2017, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fly/array.h>
#include <fly/image.h>
#include "error.hpp"

namespace fly {
array anisotropicDiffusion(const array& in, const float timestep,
                           const float conductance, const unsigned iterations,
                           const fluxFunction fftype, const diffusionEq eq) {
    fly_array out = 0;
    FLY_THROW(fly_anisotropic_diffusion(&out, in.get(), timestep, conductance,
                                      iterations, fftype, eq));
    return array(out);
}
}  // namespace fly
