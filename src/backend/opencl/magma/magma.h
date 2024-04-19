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

#ifndef __MAGMA_H
#define __MAGMA_H

#include "magma_common.h"

template<typename Ty>
magma_int_t magma_getrf_gpu(magma_int_t m, magma_int_t n, cl_mem dA,
                            size_t dA_offset, magma_int_t ldda,
                            magma_int_t *ipiv, magma_queue_t queue,
                            magma_int_t *info);

template<typename Ty>
magma_int_t magma_potrf_gpu(magma_uplo_t uplo, magma_int_t n, cl_mem dA,
                            size_t dA_offset, magma_int_t ldda,
                            magma_queue_t queue, magma_int_t *info);

template<typename Ty>
magma_int_t magma_larfb_gpu(magma_side_t side, magma_trans_t trans,
                            magma_direct_t direct, magma_storev_t storev,
                            magma_int_t m, magma_int_t n, magma_int_t k,
                            cl_mem dV, size_t dV_offset, magma_int_t lddv,
                            cl_mem dT, size_t dT_offset, magma_int_t lddt,
                            cl_mem dC, size_t dC_offset, magma_int_t lddc,
                            cl_mem dwork, size_t dwork_offset,
                            magma_int_t ldwork, magma_queue_t queue);

template<typename Ty>
magma_int_t magma_geqrf2_gpu(magma_int_t m, magma_int_t n, cl_mem dA,
                             size_t dA_offset, magma_int_t ldda, Ty *tau,
                             magma_queue_t *queue, magma_int_t *info);

template<typename Ty>
magma_int_t magma_geqrf3_gpu(magma_int_t m, magma_int_t n, cl_mem dA,
                             size_t dA_offset, magma_int_t ldda, Ty *tau,
                             cl_mem dT, size_t dT_offset, magma_queue_t queue,
                             magma_int_t *info);

template<typename Ty>
magma_int_t magma_unmqr_gpu(magma_side_t side, magma_trans_t trans,
                            magma_int_t m, magma_int_t n, magma_int_t k,
                            cl_mem dA, size_t dA_offset, magma_int_t ldda,
                            Ty *tau, cl_mem dC, size_t dC_offset,
                            magma_int_t lddc, Ty *hwork, magma_int_t lwork,
                            cl_mem dT, size_t dT_offset, magma_int_t nb,
                            magma_queue_t queue, magma_int_t *info);

#if 0  // Needs to be enabled when unmqr2 is enabled
template<typename Ty> magma_int_t
magma_unmqr2_gpu(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    cl_mem dA, size_t dA_offset, magma_int_t ldda,
    Ty    *tau,
    cl_mem dC, size_t dC_offset, magma_int_t lddc,
    Ty    *wA, magma_int_t ldwa,
    magma_queue_t queue,
    magma_int_t *info);
#endif

template<typename Ty>
magma_int_t magma_ungqr_gpu(magma_int_t m, magma_int_t n, magma_int_t k,
                            cl_mem dA, size_t dA_offset, magma_int_t ldda,
                            Ty *tau, cl_mem dT, size_t dT_offset,
                            magma_int_t nb, magma_queue_t queue,
                            magma_int_t *info);

template<typename Ty>
magma_int_t magma_getrs_gpu(magma_trans_t trans, magma_int_t n,
                            magma_int_t nrhs, cl_mem dA, size_t dA_offset,
                            magma_int_t ldda, magma_int_t *ipiv, cl_mem dB,
                            size_t dB_offset, magma_int_t lddb,
                            magma_queue_t queue, magma_int_t *info);

template<typename Ty>
magma_int_t magma_labrd_gpu(magma_int_t m, magma_int_t n, magma_int_t nb, Ty *a,
                            magma_int_t lda, cl_mem da, size_t da_offset,
                            magma_int_t ldda, void *_d, void *_e, Ty *tauq,
                            Ty *taup, Ty *x, magma_int_t ldx, cl_mem dx,
                            size_t dx_offset, magma_int_t lddx, Ty *y,
                            magma_int_t ldy, cl_mem dy, size_t dy_offset,
                            magma_int_t lddy, magma_queue_t queue);

template<typename Ty>
magma_int_t magma_gebrd_hybrid(magma_int_t m, magma_int_t n, Ty *a,
                               magma_int_t lda, cl_mem da, size_t da_offset,
                               magma_int_t ldda, void *_d, void *_e, Ty *tauq,
                               Ty *taup, Ty *work, magma_int_t lwork,
                               magma_queue_t queue, magma_int_t *info,
                               bool copy);

#endif
