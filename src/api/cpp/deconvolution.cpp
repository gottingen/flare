/*******************************************************
 * Copyright (c) 2018, Flare
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
array iterativeDeconv(const array& in, const array& ker,
                      const unsigned iterations, const float relaxFactor,
                      const iterativeDeconvAlgo algo) {
    fly_array temp = 0;
    FLY_THROW(fly_iterative_deconv(&temp, in.get(), ker.get(), iterations,
                                 relaxFactor, algo));
    return array(temp);
}

array inverseDeconv(const array& in, const array& psf, const float gamma,
                    const inverseDeconvAlgo algo) {
    fly_array temp = 0;
    FLY_THROW(fly_inverse_deconv(&temp, in.get(), psf.get(), gamma, algo));
    return array(temp);
}
}  // namespace fly
