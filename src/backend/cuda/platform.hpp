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

#include <memory>
#include <string>
#include <utility>

/* Forward declarations of Opaque structure holding
 * the following library contexts
 *  * cuBLAS
 *  * cuSparse
 *  * cuSolver
 */
struct cublasContext;
typedef struct cublasContext* BlasHandle;
struct cusparseContext;
typedef struct cusparseContext* SparseHandle;
struct cusolverDnContext;
typedef struct cusolverDnContext* SolveHandle;

#ifdef WITH_CUDNN
struct cudnnContext;
typedef struct cudnnContext* cudnnHandle_t;
#endif

namespace clog {
class logger;
}

namespace flare {
namespace common {
class TheiaManager;
class MemoryManagerBase;
}  // namespace common
}  // namespace flare

using flare::common::MemoryManagerBase;

namespace flare {
namespace cuda {

class GraphicsResourceManager;
class PlanCache;

int getBackend();

std::string getDeviceInfo() noexcept;
std::string getDeviceInfo(int device) noexcept;

std::string getPlatformInfo() noexcept;

std::string getDriverVersion() noexcept;

// Returns the cuda runtime version as a string for the current build. If no
// runtime is found or an error occured, the string "N/A" is returned
std::string getCUDARuntimeVersion() noexcept;

// Returns true if double is supported by the device
bool isDoubleSupported(int device) noexcept;

// Returns true if half is supported by the device
bool isHalfSupported(int device);

void devprop(char* d_name, char* d_platform, char* d_toolkit, char* d_compute);

int& getMaxJitSize();

int getDeviceCount();

void init();

int getActiveDeviceId();

int getDeviceNativeId(int device);

cudaStream_t getStream(int device);

cudaStream_t getActiveStream();

/// Returns true if the buffer on device buf_device_id can be accessed by
/// kernels on device execution_id
///
/// \param[in] buf_device_id The device id of the buffer
/// \param[in] execution_id The device where the buffer will be accessed.
bool isDeviceBufferAccessible(int buf_device_id, int execution_id);

/// Return a handle to the stream for the device.
///
/// \param[in] device The device of the returned stream
/// \returns The handle to the queue/stream
cudaStream_t getQueueHandle(int device);

size_t getDeviceMemorySize(int device);

size_t getHostMemorySize();

size_t getL2CacheSize(const int device);

// Returns int[3] of maxGridSize
const int* getMaxGridSize(const int device);

unsigned getMemoryBusWidth(const int device);

// maximum nr of threads the device really can run in parallel, without
// scheduling
unsigned getMaxParallelThreads(const int device);

unsigned getMultiProcessorCount(const int device);

int setDevice(int device);

void sync(int device);

// Returns true if the FLY_SYNCHRONIZE_CALLS environment variable is set to 1
bool synchronize_calls();

const cudaDeviceProp& getDeviceProp(const int device);

std::pair<int, int> getComputeCapability(const int device);

bool& evalFlag();

MemoryManagerBase& memoryManager();

MemoryManagerBase& pinnedMemoryManager();

void setMemoryManager(std::unique_ptr<MemoryManagerBase> mgr);

void resetMemoryManager();

void setMemoryManagerPinned(std::unique_ptr<MemoryManagerBase> mgr);

void resetMemoryManagerPinned();

flare::common::TheiaManager& theiaManager();

GraphicsResourceManager& interopManager();

PlanCache& fftManager();

BlasHandle blasHandle();

#ifdef WITH_CUDNN
cudnnHandle_t nnHandle();
#endif

SolveHandle solverDnHandle();

SparseHandle sparseHandle();

}  // namespace cuda
}  // namespace flare
