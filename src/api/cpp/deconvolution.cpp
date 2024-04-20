// Copyright 2023 The EA Authors.
// part of Elastic AI Search
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

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
