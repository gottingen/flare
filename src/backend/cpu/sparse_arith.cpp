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

#include <arith.hpp>
#include <common/SparseArray.hpp>
#include <common/err_common.hpp>
#include <complex.hpp>
#include <copy.hpp>
#include <math.hpp>
#include <optypes.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <sparse.hpp>
#include <sparse_arith.hpp>
#include <fly/dim4.hpp>

#include <kernel/sparse_arith.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

using flare::common::createArrayDataSparseArray;
using flare::common::createEmptySparseArray;
using flare::common::SparseArray;
using std::numeric_limits;

namespace flare {
namespace cpu {

template<typename T>
T getInf() {
    return scalar<T>(numeric_limits<T>::infinity());
}

template<>
cfloat getInf() {
    return scalar<cfloat, float>(numeric_limits<float>::infinity(),
                                 numeric_limits<float>::infinity());
}

template<>
cdouble getInf() {
    return scalar<cdouble, double>(numeric_limits<double>::infinity(),
                                   numeric_limits<double>::infinity());
}

template<typename T, fly_op_t op>
Array<T> arithOpD(const SparseArray<T> &lhs, const Array<T> &rhs,
                  const bool reverse) {
    Array<T> out  = createEmptyArray<T>(dim4(0));
    Array<T> zero = createValueArray<T>(rhs.dims(), scalar<T>(0));
    switch (op) {
        case fly_add_t: out = copyArray<T>(rhs); break;
        case fly_sub_t:
            out = reverse ? copyArray<T>(rhs)
                          : arithOp<T, fly_sub_t>(zero, rhs, rhs.dims());
            break;
        default: out = copyArray<T>(rhs);
    }
    switch (lhs.getStorage()) {
        case FLY_STORAGE_CSR:
            getQueue().enqueue(kernel::sparseArithOpD<T, op, FLY_STORAGE_CSR>,
                               out, lhs.getValues(), lhs.getRowIdx(),
                               lhs.getColIdx(), rhs, reverse);
            break;
        case FLY_STORAGE_COO:
            getQueue().enqueue(kernel::sparseArithOpD<T, op, FLY_STORAGE_COO>,
                               out, lhs.getValues(), lhs.getRowIdx(),
                               lhs.getColIdx(), rhs, reverse);
            break;
        default:
            FLY_ERROR("Sparse Arithmetic only supported for CSR or COO",
                     FLY_ERR_NOT_SUPPORTED);
    }

    return out;
}

template<typename T, fly_op_t op>
SparseArray<T> arithOp(const SparseArray<T> &lhs, const Array<T> &rhs,
                       const bool reverse) {
    SparseArray<T> out = createArrayDataSparseArray<T>(
        lhs.dims(), lhs.getValues(), lhs.getRowIdx(), lhs.getColIdx(),
        lhs.getStorage(), true);
    switch (out.getStorage()) {
        case FLY_STORAGE_CSR:
            getQueue().enqueue(kernel::sparseArithOpS<T, op, FLY_STORAGE_CSR>,
                               out.getValues(), out.getRowIdx(),
                               out.getColIdx(), rhs, reverse);
            break;
        case FLY_STORAGE_COO:
            getQueue().enqueue(kernel::sparseArithOpS<T, op, FLY_STORAGE_COO>,
                               out.getValues(), out.getRowIdx(),
                               out.getColIdx(), rhs, reverse);
            break;
        default:
            FLY_ERROR("Sparse Arithmetic only supported for CSR or COO",
                     FLY_ERR_NOT_SUPPORTED);
    }

    return out;
}

template<typename T, fly_op_t op>
SparseArray<T> arithOp(const SparseArray<T> &lhs, const SparseArray<T> &rhs) {
    fly::storage sfmt = lhs.getStorage();

    const dim4 &dims = lhs.dims();
    const uint M     = dims[0];
    const uint N     = dims[1];

    auto rowArr = createEmptyArray<int>(dim4(M + 1));

    getQueue().enqueue(kernel::calcOutNNZ, rowArr, M, N, lhs.getRowIdx(),
                       lhs.getColIdx(), rhs.getRowIdx(), rhs.getColIdx());
    getQueue().sync();

    uint nnz = rowArr.get()[M];
    auto out = createEmptySparseArray<T>(dims, nnz, sfmt);

    copyArray(out.getRowIdx(), rowArr);

    getQueue().enqueue(kernel::sparseArithOp<T, op>, out.getValues(),
                       out.getColIdx(), out.getRowIdx(), M, lhs.getValues(),
                       lhs.getRowIdx(), lhs.getColIdx(), rhs.getValues(),
                       rhs.getRowIdx(), rhs.getColIdx());
    return out;
}

#define INSTANTIATE(T)                                                         \
    template Array<T> arithOpD<T, fly_add_t>(                                   \
        const SparseArray<T> &lhs, const Array<T> &rhs, const bool reverse);   \
    template Array<T> arithOpD<T, fly_sub_t>(                                   \
        const SparseArray<T> &lhs, const Array<T> &rhs, const bool reverse);   \
    template Array<T> arithOpD<T, fly_mul_t>(                                   \
        const SparseArray<T> &lhs, const Array<T> &rhs, const bool reverse);   \
    template Array<T> arithOpD<T, fly_div_t>(                                   \
        const SparseArray<T> &lhs, const Array<T> &rhs, const bool reverse);   \
    template SparseArray<T> arithOp<T, fly_add_t>(                              \
        const SparseArray<T> &lhs, const Array<T> &rhs, const bool reverse);   \
    template SparseArray<T> arithOp<T, fly_sub_t>(                              \
        const SparseArray<T> &lhs, const Array<T> &rhs, const bool reverse);   \
    template SparseArray<T> arithOp<T, fly_mul_t>(                              \
        const SparseArray<T> &lhs, const Array<T> &rhs, const bool reverse);   \
    template SparseArray<T> arithOp<T, fly_div_t>(                              \
        const SparseArray<T> &lhs, const Array<T> &rhs, const bool reverse);   \
    template SparseArray<T> arithOp<T, fly_add_t>(                              \
        const common::SparseArray<T> &lhs, const common::SparseArray<T> &rhs); \
    template SparseArray<T> arithOp<T, fly_sub_t>(                              \
        const common::SparseArray<T> &lhs, const common::SparseArray<T> &rhs); \
    template SparseArray<T> arithOp<T, fly_mul_t>(                              \
        const common::SparseArray<T> &lhs, const common::SparseArray<T> &rhs); \
    template SparseArray<T> arithOp<T, fly_div_t>(                              \
        const common::SparseArray<T> &lhs, const common::SparseArray<T> &rhs);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)

}  // namespace cpu
}  // namespace flare
