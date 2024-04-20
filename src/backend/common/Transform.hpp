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
