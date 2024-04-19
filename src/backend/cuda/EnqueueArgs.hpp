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

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace flare {
namespace cuda {

///
/// EnqueueArgs is a kernel launch configuration composition object
///
/// This structure is an composition of various parameters that are
/// required to successfully launch a CUDA kernel.
///
struct EnqueueArgs {
    // TODO(pradeep): this can be easily templated
    // template<typename Queue, typename Event>
    dim3 mBlocks;                  ///< Number of blocks per grid/kernel-launch
    dim3 mThreads;                 ///< Number of threads per block
    CUstream mStream;              ///< CUDA stream to enqueue the kernel on
    unsigned int mSharedMemSize;   ///< Size(in bytes) of shared memory used
    std::vector<CUevent> mEvents;  ///< Events to wait for kernel execution

    ///
    /// \brief EnqueueArgs constructor
    ///
    /// \param[in] blks is number of blocks per grid
    /// \param[in] thrds is number of threads per block
    /// \param[in] stream is CUDA steam on which kernel has to be enqueued
    /// \param[in] sharedMemSize is number of bytes of shared memory allocation
    /// \param[in] events is list of events to wait for kernel execution
    ///
    EnqueueArgs(dim3 blks, dim3 thrds, CUstream stream = 0,
                const unsigned int sharedMemSize   = 0,
                const std::vector<CUevent> &events = {})
        : mBlocks(blks)
        , mThreads(thrds)
        , mStream(stream)
        , mSharedMemSize(sharedMemSize)
        , mEvents(events) {}
};

}  // namespace cuda
}  // namespace flare
