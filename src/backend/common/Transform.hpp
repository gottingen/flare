/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <backend.hpp>
#include <common/Binary.hpp>
#include <math.hpp>
#include <types.hpp>

#ifndef __DH__
#define __DH__
#endif

#include "optypes.hpp"

namespace flare {
namespace common {

using namespace detail;  // NOLINT

// Because isnan(cfloat) and isnan(cdouble) is not defined
#define IS_NAN(val) !((val) == (val))

template<typename Ti, typename To, fly_op_t op>
struct Transform {
    __DH__ To operator()(Ti in) { return static_cast<To>(in); }
};

template<typename Ti, typename To>
struct Transform<Ti, To, fly_min_t> {
    __DH__ To operator()(Ti in) {
        return IS_NAN(in) ? Binary<To, fly_min_t>::init() : To(in);
    }
};

template<typename Ti, typename To>
struct Transform<Ti, To, fly_max_t> {
    __DH__ To operator()(Ti in) {
        return IS_NAN(in) ? Binary<To, fly_max_t>::init() : To(in);
    }
};

template<typename Ti, typename To>
struct Transform<Ti, To, fly_or_t> {
    __DH__ To operator()(Ti in) { return (in != scalar<Ti>(0.)); }
};

template<typename Ti, typename To>
struct Transform<Ti, To, fly_and_t> {
    __DH__ To operator()(Ti in) { return (in != scalar<Ti>(0.)); }
};

template<typename Ti, typename To>
struct Transform<Ti, To, fly_notzero_t> {
    __DH__ To operator()(Ti in) { return (in != scalar<Ti>(0.)); }
};

}  // namespace common
}  // namespace flare
