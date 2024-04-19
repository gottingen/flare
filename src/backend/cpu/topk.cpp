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
#include <common/half.hpp>
#include <index.hpp>
#include <sort.hpp>
#include <sort_index.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

using flare::common::half;
using std::iota;
using std::min;
using std::partial_sort_copy;
using std::vector;

namespace flare {
namespace cpu {
template<typename T>
void topk(Array<T>& vals, Array<unsigned>& idxs, const Array<T>& in,
          const int k, const int dim, const fly::topkFunction order) {
    // The out_dims is of size k along the dimension of the topk operation
    // and the same as the input dimension otherwise.
    dim4 out_dims(1);
    int ndims = in.dims().ndims();
    for (int i = 0; i < ndims; i++) {
        if (i == dim) {
            out_dims[i] = min(k, static_cast<int>(in.dims()[i]));
        } else {
            out_dims[i] = in.dims()[i];
        }
    }

    auto values  = createEmptyArray<T>(out_dims);
    auto indices = createEmptyArray<unsigned>(out_dims);

    auto func = [=](Param<T> values, Param<unsigned> indices, CParam<T> in) {
        const T* ptr   = in.get();
        unsigned* iptr = indices.get();
        T* vptr        = values.get();

        // Create a linear index
        vector<uint> idx(in.dims().elements());
        iota(begin(idx), end(idx), 0);

        int iter = in.dims()[1] * in.dims()[2] * in.dims()[3];
        for (int i = 0; i < iter; i++) {
            auto idx_itr = begin(idx) + i * in.strides()[1];
            auto* kiptr  = iptr + k * i;

            if (order & FLY_TOPK_MIN) {
                if (order & FLY_TOPK_STABLE) {
                    partial_sort_copy(
                        idx_itr, idx_itr + in.strides()[1], kiptr, kiptr + k,
                        [ptr](const uint lhs, const uint rhs) -> bool {
                            return compute_t<T>(ptr[lhs]) <
                                           compute_t<T>(ptr[rhs])
                                       ? true
                                   : compute_t<T>(ptr[lhs]) ==
                                           compute_t<T>(ptr[rhs])
                                       ? (lhs < rhs)
                                       : false;
                        });
                } else {
                    partial_sort_copy(
                        idx_itr, idx_itr + in.strides()[1], kiptr, kiptr + k,
                        [ptr](const uint lhs, const uint rhs) -> bool {
                            return compute_t<T>(ptr[lhs]) <
                                   compute_t<T>(ptr[rhs]);
                        });
                    // Sort the top k values in each column
                }
            } else {
                if (order & FLY_TOPK_STABLE) {
                    partial_sort_copy(
                        idx_itr, idx_itr + in.strides()[1], kiptr, kiptr + k,
                        [ptr](const uint lhs, const uint rhs) -> bool {
                            return compute_t<T>(ptr[lhs]) >
                                           compute_t<T>(ptr[rhs])
                                       ? true
                                   : compute_t<T>(ptr[lhs]) ==
                                           compute_t<T>(ptr[rhs])
                                       ? (lhs < rhs)
                                       : false;
                        });
                } else {
                    partial_sort_copy(
                        idx_itr, idx_itr + in.strides()[1], kiptr, kiptr + k,
                        [ptr](const uint lhs, const uint rhs) -> bool {
                            return compute_t<T>(ptr[lhs]) >
                                   compute_t<T>(ptr[rhs]);
                        });
                }
            }

            auto* kvptr = vptr + k * i;
            for (int j = 0; j < k; j++) {
                // Update the value arrays with the original values
                kvptr[j] = ptr[kiptr[j]];
                // Convert linear indices back to column indices
                kiptr[j] -= i * in.strides()[1];
            }
        }
    };

    getQueue().enqueue(func, values, indices, in);

    vals = values;
    idxs = indices;
}

#define INSTANTIATE(T)                                                  \
    template void topk<T>(Array<T>&, Array<unsigned>&, const Array<T>&, \
                          const int, const int, const fly::topkFunction);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(long long)
INSTANTIATE(unsigned long long)
INSTANTIATE(half)
}  // namespace cpu
}  // namespace flare
