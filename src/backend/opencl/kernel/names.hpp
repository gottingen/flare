/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <common/defines.hpp>
#include <optypes.hpp>

template<fly_op_t T>
static const char *binOpName() {
    return "ADD_OP";
}

template<>
inline const char *binOpName<fly_add_t>() {
    return "ADD_OP";
}
template<>
inline const char *binOpName<fly_mul_t>() {
    return "MUL_OP";
}
template<>
inline const char *binOpName<fly_and_t>() {
    return "AND_OP";
}
template<>
inline const char *binOpName<fly_or_t>() {
    return "OR_OP";
}
template<>
inline const char *binOpName<fly_min_t>() {
    return "MIN_OP";
}
template<>
inline const char *binOpName<fly_max_t>() {
    return "MAX_OP";
}
template<>
inline const char *binOpName<fly_notzero_t>() {
    return "NOTZERO_OP";
}
