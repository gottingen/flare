/*******************************************************
 * Copyright (c) 2017, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>

namespace flare {
namespace opencl {
template<typename T>
void anisotropicDiffusion(Array<T>& inout, const float dt, const float mct,
                          const fly::fluxFunction fftype,
                          const fly::diffusionEq eq);
}  // namespace opencl
}  // namespace flare
