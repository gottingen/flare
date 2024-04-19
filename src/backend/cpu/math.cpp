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
#include <common/defines.hpp>
#include <math.hpp>
#include <complex>

namespace flare {
namespace cpu {

uint abs(uint val) { return val; }
uchar abs(uchar val) { return val; }
uintl abs(uintl val) { return val; }

cfloat scalar(float val) {
    cfloat cval = {val, 0};
    return cval;
}

cdouble scalar(double val) {
    cdouble cval = {val, 0};
    return cval;
}

cfloat min(cfloat lhs, cfloat rhs) { return abs(lhs) < abs(rhs) ? lhs : rhs; }

cdouble min(cdouble lhs, cdouble rhs) {
    return abs(lhs) < abs(rhs) ? lhs : rhs;
}

cfloat max(cfloat lhs, cfloat rhs) { return abs(lhs) > abs(rhs) ? lhs : rhs; }

cdouble max(cdouble lhs, cdouble rhs) {
    return abs(lhs) > abs(rhs) ? lhs : rhs;
}

}  // namespace cpu
}  // namespace flare
