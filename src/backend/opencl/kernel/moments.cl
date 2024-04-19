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

#define FLY_MOMENT_M00 1
#define FLY_MOMENT_M01 2
#define FLY_MOMENT_M10 4
#define FLY_MOMENT_M11 8

inline void fatomic_add_l(volatile local float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal, prevVal, expVal;

    prevVal.floatVal = *source;
    do {
        expVal.floatVal = prevVal.floatVal;
        newVal.floatVal = expVal.floatVal + operand;
        prevVal.intVal  = atomic_cmpxchg((volatile local unsigned int *)source,
                                        expVal.intVal, newVal.intVal);
    } while (expVal.intVal != prevVal.intVal);
}

inline void fatomic_add_g(volatile global float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal, prevVal, expVal;

    prevVal.floatVal = *source;
    do {
        expVal.floatVal = prevVal.floatVal;
        newVal.floatVal = expVal.floatVal + operand;
        prevVal.intVal  = atomic_cmpxchg((volatile global unsigned int *)source,
                                        expVal.intVal, newVal.intVal);
    } while (expVal.intVal != prevVal.intVal);
}

kernel void moments(global float *d_out, const KParam out, global const T *d_in,
                    const KParam in, const int moment, const int pBatch) {
    const dim_t idw = get_group_id(1) / in.dims[2];
    const dim_t idz = get_group_id(1) - idw * in.dims[2];

    const dim_t idy = get_group_id(0);
    dim_t idx       = get_local_id(0);

    if (idy >= in.dims[1] || idz >= in.dims[2] || idw >= in.dims[3]) return;

    local float wkg_moment_sum[MOMENTS_SZ];
    if (get_local_id(0) < MOMENTS_SZ) { wkg_moment_sum[get_local_id(0)] = 0.f; }
    barrier(CLK_LOCAL_MEM_FENCE);

    int mId = idy * in.strides[1] + idx;
    if (pBatch) { mId += idw * in.strides[3] + idz * in.strides[2]; }

    for (; idx < in.dims[0]; idx += get_local_size(0)) {
        dim_t m_off = 0;
        float val   = d_in[mId];
        mId += get_local_size(0);

        if ((moment & FLY_MOMENT_M00) > 0) {
            fatomic_add_l(wkg_moment_sum + m_off++, val);
        }
        if ((moment & FLY_MOMENT_M01) > 0) {
            fatomic_add_l(wkg_moment_sum + m_off++, idx * val);
        }
        if ((moment & FLY_MOMENT_M10) > 0) {
            fatomic_add_l(wkg_moment_sum + m_off++, idy * val);
        }
        if ((moment & FLY_MOMENT_M11) > 0) {
            fatomic_add_l(wkg_moment_sum + m_off, idx * idy * val);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_local_id(0) < out.dims[0])
        fatomic_add_g(d_out + (idw * out.strides[3] + idz * out.strides[2]) +
                          get_local_id(0),
                      wkg_moment_sum[get_local_id(0)]);
}
