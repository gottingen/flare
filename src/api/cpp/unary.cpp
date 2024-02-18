/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fly/arith.h>
#include <fly/array.h>
#include <fly/data.h>
#include "error.hpp"

namespace fly {

#define fly_complex(...) fly_cplx(__VA_ARGS__)

#define INSTANTIATE(func)                    \
    array func(const array &in) {            \
        fly_array out = 0;                    \
        FLY_THROW(fly_##func(&out, in.get())); \
        return array(out);                   \
    }

INSTANTIATE(complex)
INSTANTIATE(real)
INSTANTIATE(imag)
INSTANTIATE(arg)
INSTANTIATE(abs)
INSTANTIATE(conjg)

INSTANTIATE(sign)
INSTANTIATE(round)
INSTANTIATE(trunc)
INSTANTIATE(floor)
INSTANTIATE(ceil)

INSTANTIATE(sin)
INSTANTIATE(cos)
INSTANTIATE(tan)

INSTANTIATE(asin)
INSTANTIATE(acos)
INSTANTIATE(atan)

INSTANTIATE(sinh)
INSTANTIATE(cosh)
INSTANTIATE(tanh)

INSTANTIATE(asinh)
INSTANTIATE(acosh)
INSTANTIATE(atanh)

INSTANTIATE(pow2)
INSTANTIATE(exp)
INSTANTIATE(expm1)
INSTANTIATE(erf)
INSTANTIATE(erfc)
INSTANTIATE(sigmoid)

INSTANTIATE(log)
INSTANTIATE(log1p)
INSTANTIATE(log10)
INSTANTIATE(log2)

INSTANTIATE(sqrt)
INSTANTIATE(rsqrt)
INSTANTIATE(cbrt)

INSTANTIATE(iszero)

INSTANTIATE(factorial)
INSTANTIATE(tgamma)
INSTANTIATE(lgamma)

// isinf and isnan are defined by C++.
// Thus we need a difference nomenclature.
array isInf(const array &in) {
    fly_array out = 0;
    FLY_THROW(fly_isinf(&out, in.get()));
    return array(out);
}

array isNaN(const array &in) {
    fly_array out = 0;
    FLY_THROW(fly_isnan(&out, in.get()));
    return array(out);
}

}  // namespace fly
