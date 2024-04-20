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
#include <fly/blas.h>
#include "error.hpp"

namespace fly {
array matmul(const array &lhs, const array &rhs, const matProp optLhs,
             const matProp optRhs) {
    fly_array out = 0;
    FLY_THROW(fly_matmul(&out, lhs.get(), rhs.get(), optLhs, optRhs));
    return array(out);
}

array matmulNT(const array &lhs, const array &rhs) {
    fly_array out = 0;
    FLY_THROW(fly_matmul(&out, lhs.get(), rhs.get(), FLY_MAT_NONE, FLY_MAT_TRANS));
    return array(out);
}

array matmulTN(const array &lhs, const array &rhs) {
    fly_array out = 0;
    FLY_THROW(fly_matmul(&out, lhs.get(), rhs.get(), FLY_MAT_TRANS, FLY_MAT_NONE));
    return array(out);
}

array matmulTT(const array &lhs, const array &rhs) {
    fly_array out = 0;
    FLY_THROW(fly_matmul(&out, lhs.get(), rhs.get(), FLY_MAT_TRANS, FLY_MAT_TRANS));
    return array(out);
}

array matmul(const array &a, const array &b, const array &c) {
    dim_t tmp1 = a.dims(0) * b.dims(1);
    dim_t tmp2 = b.dims(0) * c.dims(1);

    if (tmp1 < tmp2) {
        return matmul(matmul(a, b), c);
    } else {
        return matmul(a, matmul(b, c));
    }
}

array matmul(const array &a, const array &b, const array &c, const array &d) {
    dim_t tmp1 = a.dims(0) * c.dims(1);
    dim_t tmp2 = b.dims(0) * d.dims(1);

    if (tmp1 < tmp2) {
        return matmul(matmul(a, b, c), d);
    } else {
        return matmul(a, matmul(b, c, d));
    }
}

array dot(const array &lhs, const array &rhs, const matProp optLhs,
          const matProp optRhs) {
    fly_array out = 0;
    FLY_THROW(fly_dot(&out, lhs.get(), rhs.get(), optLhs, optRhs));
    return array(out);
}

#define INSTANTIATE_REAL(TYPE)                                               \
    template<>                                                               \
    FLY_API TYPE dot(const array &lhs, const array &rhs, const matProp optLhs, \
                   const matProp optRhs) {                                   \
        double rval = 0, ival = 0;                                           \
        FLY_THROW(                                                            \
            fly_dot_all(&rval, &ival, lhs.get(), rhs.get(), optLhs, optRhs)); \
        return (TYPE)(rval);                                                 \
    }

#define INSTANTIATE_CPLX(TYPE, REAL)                                         \
    template<>                                                               \
    FLY_API TYPE dot(const array &lhs, const array &rhs, const matProp optLhs, \
                   const matProp optRhs) {                                   \
        double rval = 0, ival = 0;                                           \
        FLY_THROW(                                                            \
            fly_dot_all(&rval, &ival, lhs.get(), rhs.get(), optLhs, optRhs)); \
        TYPE out((REAL)rval, (REAL)ival);                                    \
        return out;                                                          \
    }

INSTANTIATE_REAL(float)
INSTANTIATE_REAL(double)
INSTANTIATE_CPLX(cfloat, float)
INSTANTIATE_CPLX(cdouble, double)

}  // namespace fly
