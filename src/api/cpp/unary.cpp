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
