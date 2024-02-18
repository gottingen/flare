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
#include <fly/gfor.h>
#include "error.hpp"

namespace fly {

#define INSTANTIATE(cppfunc, cfunc)                             \
    array cppfunc(const array &lhs, const array &rhs) {         \
        fly_array out = 0;                                       \
        FLY_THROW(cfunc(&out, lhs.get(), rhs.get(), gforGet())); \
        return array(out);                                      \
    }

INSTANTIATE(min, fly_minof)
INSTANTIATE(max, fly_maxof)
INSTANTIATE(pow, fly_pow)
INSTANTIATE(root, fly_root)
INSTANTIATE(rem, fly_rem)
INSTANTIATE(mod, fly_mod)

INSTANTIATE(complex, fly_cplx2)
INSTANTIATE(atan2, fly_atan2)
INSTANTIATE(hypot, fly_hypot)

#define WRAPPER(func)                                             \
    array func(const array &lhs, const double rhs) {              \
        fly::dtype ty = lhs.type();                                \
        if (lhs.iscomplex()) { ty = lhs.issingle() ? f32 : f64; } \
        return func(lhs, constant(rhs, lhs.dims(), ty));          \
    }                                                             \
    array func(const double lhs, const array &rhs) {              \
        fly::dtype ty = rhs.type();                                \
        if (rhs.iscomplex()) { ty = rhs.issingle() ? f32 : f64; } \
        return func(constant(lhs, rhs.dims(), ty), rhs);          \
    }

WRAPPER(min)
WRAPPER(max)
WRAPPER(pow)
WRAPPER(root)
WRAPPER(rem)
WRAPPER(mod)
WRAPPER(complex)
WRAPPER(atan2)
WRAPPER(hypot)
}  // namespace fly
