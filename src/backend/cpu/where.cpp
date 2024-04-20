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

#include <Array.hpp>
#include <common/Binary.hpp>
#include <common/Transform.hpp>
#include <math.hpp>
#include <memory.hpp>
#include <platform.hpp>
#include <where.hpp>
#include <fly/dim4.hpp>

#include <complex>
#include <vector>

using fly::dim4;

namespace flare {
namespace cpu {

template<typename T>
Array<uint> where(const Array<T> &in) {
    const dim_t *dims    = in.dims().get();
    const dim_t *strides = in.strides().get();
    static const T zero  = scalar<T>(0);

    const T *iptr = in.get();
    auto out_vec  = memAlloc<uint>(in.elements());
    getQueue().sync();

    dim_t count = 0;
    dim_t idx   = 0;
    for (dim_t w = 0; w < dims[3]; w++) {
        uint offw = w * strides[3];

        for (dim_t z = 0; z < dims[2]; z++) {
            uint offz = offw + z * strides[2];

            for (dim_t y = 0; y < dims[1]; y++) {
                uint offy = y * strides[1] + offz;

                for (dim_t x = 0; x < dims[0]; x++) {
                    T val = iptr[offy + x];
                    if (val != zero) {
                        out_vec[count] = idx;
                        count++;
                    }
                    idx++;
                }
            }
        }
    }

    Array<uint> out = createDeviceDataArray<uint>(dim4(count), out_vec.get());
    out_vec.release();
    return out;
}

#define INSTANTIATE(T) template Array<uint> where<T>(const Array<T> &in);

INSTANTIATE(float)
INSTANTIATE(cfloat)
INSTANTIATE(double)
INSTANTIATE(cdouble)
INSTANTIATE(char)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)

}  // namespace cpu
}  // namespace flare
