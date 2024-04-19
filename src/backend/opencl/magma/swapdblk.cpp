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

#include "kernel/swapdblk.hpp"
#include "magma_data.h"

template<typename T>
void magmablas_swapdblk(magma_int_t n, magma_int_t nb, cl_mem dA,
                        magma_int_t dA_offset, magma_int_t ldda,
                        magma_int_t inca, cl_mem dB, magma_int_t dB_offset,
                        magma_int_t lddb, magma_int_t incb,
                        magma_queue_t queue) {
    flare::opencl::kernel::swapdblk<T>(n, nb, dA, dA_offset, ldda, inca, dB,
                                           dB_offset, lddb, incb, queue);
}

#define INSTANTIATE(T)                                                        \
    template void magmablas_swapdblk<T>(                                      \
        magma_int_t n, magma_int_t nb, cl_mem dA, magma_int_t dA_offset,      \
        magma_int_t ldda, magma_int_t inca, cl_mem dB, magma_int_t dB_offset, \
        magma_int_t lddb, magma_int_t incb, magma_queue_t queue);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(magmaFloatComplex)
INSTANTIATE(magmaDoubleComplex)
