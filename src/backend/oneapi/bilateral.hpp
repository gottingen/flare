/*******************************************************
 * Copyright (c) 2022, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>

namespace flare {
namespace oneapi {
template<typename inType, typename outType>
Array<outType> bilateral(const Array<inType> &in, const float &spatialSigma,
                         const float &chromaticSigma);
}  // namespace oneapi
}  // namespace flare
