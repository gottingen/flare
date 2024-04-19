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

#include <anisotropic_diffusion.hpp>

#include <arith.hpp>
#include <backend.hpp>
#include <common/cast.hpp>
#include <common/err_common.hpp>
#include <copy.hpp>
#include <gradient.hpp>
#include <handle.hpp>
#include <reduce.hpp>

#include <fly/dim4.hpp>
#include <fly/image.h>

#include <type_traits>

using fly::dim4;
using flare::common::cast;
using detail::arithOp;
using detail::Array;
using detail::createEmptyArray;
using detail::getScalar;
using detail::gradient;
using detail::reduce_all;

template<typename T>
fly_array diffusion(const Array<float>& in, const float dt, const float K,
                   const unsigned iterations, const fly_flux_function fftype,
                   const fly::diffusionEq eq) {
    auto out  = copyArray(in);
    auto dims = out.dims();
    auto g0   = createEmptyArray<float>(dims);
    auto g1   = createEmptyArray<float>(dims);
    float cnst =
        -2.0f * K * K / dims.elements();  // NOLINT(readability-magic-numbers)

    for (unsigned i = 0; i < iterations; ++i) {
        gradient<float>(g0, g1, out);

        auto g0Sqr = arithOp<float, fly_mul_t>(g0, g0, dims);
        auto g1Sqr = arithOp<float, fly_mul_t>(g1, g1, dims);
        auto sumd  = arithOp<float, fly_add_t>(g0Sqr, g1Sqr, dims);
        float avg =
            getScalar<float>(reduce_all<fly_add_t, float, float>(sumd, true, 0));

        anisotropicDiffusion(out, dt, 1.0f / (cnst * avg), fftype, eq);
    }

    return getHandle(cast<T, float>(out));
}

fly_err fly_anisotropic_diffusion(fly_array* out, const fly_array in,
                                const float dt, const float K,
                                const unsigned iterations,
                                const fly_flux_function fftype,
                                const fly_diffusion_eq eq) {
    try {
        const ArrayInfo& info = getInfo(in);

        const fly::dim4& inputDimensions = info.dims();
        const fly_dtype inputType        = info.getType();
        const unsigned inputNumDims     = inputDimensions.ndims();

        DIM_ASSERT(1, (inputNumDims >= 2));

        ARG_ASSERT(3, (K > 0 || K < 0));
        ARG_ASSERT(4, (iterations > 0));

        const fly_flux_function F =
            (fftype == FLY_FLUX_DEFAULT ? FLY_FLUX_EXPONENTIAL : fftype);

        auto input = castArray<float>(in);

        fly_array output = nullptr;
        switch (inputType) {
            case f64:
                output = diffusion<double>(input, dt, K, iterations, F, eq);
                break;
            case f32:
            case s32:
            case u32:
            case s16:
            case u16:
            case u8:
                output = diffusion<float>(input, dt, K, iterations, F, eq);
                break;
            default: TYPE_ERROR(1, inputType);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return FLY_SUCCESS;
}
