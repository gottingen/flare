/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

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
