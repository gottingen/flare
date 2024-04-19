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

#if defined(WITH_LINEAR_ALGEBRA)
#ifndef CPU_LAPACK_TRIANGLE
#define CPU_LAPACK_TRIANGLE

#include <math.hpp>

namespace flare {
namespace opencl {
namespace cpu {

template<typename T, bool is_upper, bool is_unit_diag>
void triangle(T *o, const T *i, const dim4 odm, const dim4 ost,
              const dim4 ist) {
    for (dim_t ow = 0; ow < odm[3]; ow++) {
        const dim_t oW = ow * ost[3];
        const dim_t iW = ow * ist[3];

        for (dim_t oz = 0; oz < odm[2]; oz++) {
            const dim_t oZW = oW + oz * ost[2];
            const dim_t iZW = iW + oz * ist[2];

            for (dim_t oy = 0; oy < odm[1]; oy++) {
                const dim_t oYZW = oZW + oy * ost[1];
                const dim_t iYZW = iZW + oy * ist[1];

                for (dim_t ox = 0; ox < odm[0]; ox++) {
                    const dim_t oMem = oYZW + ox;
                    const dim_t iMem = iYZW + ox;

                    bool cond         = is_upper ? (oy >= ox) : (oy <= ox);
                    bool do_unit_diag = (is_unit_diag && ox == oy);
                    if (cond) {
                        o[oMem] = do_unit_diag ? scalar<T>(1) : i[iMem];
                    } else {
                        o[oMem] = scalar<T>(0);
                    }
                }
            }
        }
    }
}

}  // namespace cpu
}  // namespace opencl
}  // namespace flare

#endif
#endif  // WITH_LINEAR_ALGEBRA
