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

#include <common/SparseArray.hpp>
#include <common/defines.hpp>
#include <common/unique_handle.hpp>
#include <cudaDataType.hpp>
#include <cusparseModule.hpp>
#include <cusparse_v2.h>
#include <err_cuda.hpp>

#if defined(FLY_USE_NEW_CUSPARSE_API)
namespace flare {
namespace cuda {

template<typename T>
cusparseStatus_t createSpMatDescr(
    cusparseSpMatDescr_t *out, const flare::common::SparseArray<T> &arr) {
    auto &_ = flare::cuda::getCusparsePlugin();
    switch (arr.getStorage()) {
        case FLY_STORAGE_CSR: {
            return _.cusparseCreateCsr(
                out, arr.dims()[0], arr.dims()[1], arr.getNNZ(),
                (void *)arr.getRowIdx().get(), (void *)arr.getColIdx().get(),
                (void *)arr.getValues().get(), CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, getType<T>());
        }
#if CUSPARSE_VERSION >= 11300
        case FLY_STORAGE_CSC: {
            return _.cusparseCreateCsc(
                out, arr.dims()[0], arr.dims()[1], arr.getNNZ(),
                (void *)arr.getColIdx().get(), (void *)arr.getRowIdx().get(),
                (void *)arr.getValues().get(), CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, getType<T>());
        }
#else
        case FLY_STORAGE_CSC:
            CUDA_NOT_SUPPORTED(
                "Sparse not supported for CSC on this version of the CUDA "
                "Toolkit");
#endif
        case FLY_STORAGE_COO: {
            return _.cusparseCreateCoo(
                out, arr.dims()[0], arr.dims()[1], arr.getNNZ(),
                (void *)arr.getColIdx().get(), (void *)arr.getRowIdx().get(),
                (void *)arr.getValues().get(), CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO, getType<T>());
        }
    }
    return CUSPARSE_STATUS_SUCCESS;
}

}  // namespace cuda
}  // namespace flare
#endif

// clang-format off
DEFINE_HANDLER(cusparseHandle_t, flare::cuda::getCusparsePlugin().cusparseCreate, flare::cuda::getCusparsePlugin().cusparseDestroy);
DEFINE_HANDLER(cusparseMatDescr_t, flare::cuda::getCusparsePlugin().cusparseCreateMatDescr, flare::cuda::getCusparsePlugin().cusparseDestroyMatDescr);
#if defined(FLY_USE_NEW_CUSPARSE_API)
DEFINE_HANDLER(cusparseSpMatDescr_t, flare::cuda::createSpMatDescr, flare::cuda::getCusparsePlugin().cusparseDestroySpMat);
DEFINE_HANDLER(cusparseDnVecDescr_t, flare::cuda::getCusparsePlugin().cusparseCreateDnVec, flare::cuda::getCusparsePlugin().cusparseDestroyDnVec);
DEFINE_HANDLER(cusparseDnMatDescr_t, flare::cuda::getCusparsePlugin().cusparseCreateDnMat, flare::cuda::getCusparsePlugin().cusparseDestroyDnMat);
#endif
// clang-format on

namespace flare {
namespace cuda {

const char *errorString(cusparseStatus_t err);

#define CUSPARSE_CHECK(fn)                                                    \
    do {                                                                      \
        cusparseStatus_t _error = fn;                                         \
        if (_error != CUSPARSE_STATUS_SUCCESS) {                              \
            char _err_msg[1024];                                              \
            snprintf(_err_msg, sizeof(_err_msg), "CUSPARSE Error (%d): %s\n", \
                     (int)(_error), flare::cuda::errorString(_error));    \
                                                                              \
            FLY_ERROR(_err_msg, FLY_ERR_INTERNAL);                              \
        }                                                                     \
    } while (0)

}  // namespace cuda
}  // namespace flare
