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

#include <implicit.hpp>

/*
Implicit type mimics C/C++ behavior.

Order of precedence:
- complex > real
- double > float > uintl > intl > uint > int > uchar > char
*/

fly_dtype implicit(const fly_dtype lty, const fly_dtype rty) {
    if (lty == rty) { return lty; }

    if (lty == c64 || rty == c64) { return c64; }

    if (lty == c32 || rty == c32) {
        if (lty == f64 || rty == f64) { return c64; }
        return c32;
    }

    if (lty == f64 || rty == f64) { return f64; }
    if (lty == f32 || rty == f32) { return f32; }
    if ((lty == f16) || (rty == f16)) { return f16; }

    if ((lty == u64) || (rty == u64)) { return u64; }
    if ((lty == s64) || (rty == s64)) { return s64; }
    if ((lty == u32) || (rty == u32)) { return u32; }
    if ((lty == s32) || (rty == s32)) { return s32; }
    if ((lty == u16) || (rty == u16)) { return u16; }
    if ((lty == s16) || (rty == s16)) { return s16; }
    if ((lty == u8) || (rty == u8)) { return u8; }
    if ((lty == b8) && (rty == b8)) { return b8; }

    return f32;
}

fly_dtype implicit(const fly_array lhs, const fly_array rhs) {
    const ArrayInfo& lInfo = getInfo(lhs);
    const ArrayInfo& rInfo = getInfo(rhs);

    return implicit(lInfo.getType(), rInfo.getType());
}
