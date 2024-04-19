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

#include <Param.hpp>
#include <common/dispatch.hpp>
#include <debug_cuda.hpp>
#include <err_cuda.hpp>
#include <iota.hpp>
#include <kernel/thrust_sort_by_key.hpp>
#include <math.hpp>
#include <memory.hpp>
#include <fly/dim4.hpp>

namespace flare {
namespace cuda {
namespace kernel {
// Wrapper functions
template<typename Tk, typename Tv>
void sort0ByKeyIterative(Param<Tk> okey, Param<Tv> oval, bool isAscending) {
    for (int w = 0; w < okey.dims[3]; w++) {
        int okeyW = w * okey.strides[3];
        int ovalW = w * oval.strides[3];
        for (int z = 0; z < okey.dims[2]; z++) {
            int okeyWZ = okeyW + z * okey.strides[2];
            int ovalWZ = ovalW + z * oval.strides[2];
            for (int y = 0; y < okey.dims[1]; y++) {
                int okeyOffset = okeyWZ + y * okey.strides[1];
                int ovalOffset = ovalWZ + y * oval.strides[1];

                thrustSortByKey<Tk, Tv>(okey.ptr + okeyOffset,
                                        oval.ptr + ovalOffset, okey.dims[0],
                                        isAscending);
            }
        }
    }
    POST_LAUNCH_CHECK();
}

template<typename Tk, typename Tv>
void sortByKeyBatched(Param<Tk> pKey, Param<Tv> pVal, const int dim,
                      bool isAscending) {
    fly::dim4 inDims;
    for (int i = 0; i < 4; i++) inDims[i] = pKey.dims[i];

    const dim_t elements = inDims.elements();

    // Sort dimension
    // tileDims * seqDims = inDims
    fly::dim4 tileDims(1);
    fly::dim4 seqDims = inDims;
    tileDims[dim]    = inDims[dim];
    seqDims[dim]     = 1;

    // Create/call iota
    Array<uint> Seq = iota<uint>(seqDims, tileDims);

    Tk *Key   = pKey.ptr;
    auto cKey = memAlloc<Tk>(elements);
    CUDA_CHECK(cudaMemcpyAsync(cKey.get(), Key, elements * sizeof(Tk),
                               cudaMemcpyDeviceToDevice, getActiveStream()));

    Tv *Val = pVal.ptr;
    thrustSortByKey(Key, Val, elements, isAscending);
    thrustSortByKey(cKey.get(), Seq.get(), elements, isAscending);

    auto cSeq = memAlloc<uint>(elements);
    CUDA_CHECK(cudaMemcpyAsync(cSeq.get(), Seq.get(), elements * sizeof(uint),
                               cudaMemcpyDeviceToDevice, getActiveStream()));

    // This always needs to be ascending
    thrustSortByKey(Seq.get(), Val, elements, true);
    thrustSortByKey(cSeq.get(), Key, elements, true);

    // No need of doing moddims here because the original Array<T>
    // dimensions have not been changed
    // val.modDims(inDims);
}

template<typename Tk, typename Tv>
void sort0ByKey(Param<Tk> okey, Param<Tv> oval, bool isAscending) {
    int higherDims = okey.dims[1] * okey.dims[2] * okey.dims[3];

    // Batced sort performs 4x sort by keys But this is only useful
    // before GPU is saturated The GPU is saturated at around 100,000
    // integers Call batched sort only if both conditions are met
    if (higherDims > 4 && okey.dims[0] < 100000)
        kernel::sortByKeyBatched<Tk, Tv>(okey, oval, 0, isAscending);
    else
        kernel::sort0ByKeyIterative<Tk, Tv>(okey, oval, isAscending);
}
}  // namespace kernel
}  // namespace cuda
}  // namespace flare
