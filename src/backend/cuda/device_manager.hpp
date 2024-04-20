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

#include <platform.hpp>

#include <array>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

using flare::common::MemoryManagerBase;

#ifndef FLY_CUDA_MEM_DEBUG
#define FLY_CUDA_MEM_DEBUG 0
#endif

namespace flare {
namespace cuda {

struct cudaDevice_t {
    cudaDeviceProp prop;
    size_t flops;
    int nativeId;
};

int& tlocalActiveDeviceId();

bool checkDeviceWithRuntime(int runtime, std::pair<int, int> compute);

class DeviceManager {
   public:
    static const int MAX_DEVICES = 16;

    static bool checkGraphicsInteropCapability();

    static DeviceManager& getInstance();
    ~DeviceManager();

    clog::logger* getLogger();

    friend MemoryManagerBase& memoryManager();

    friend void setMemoryManager(std::unique_ptr<MemoryManagerBase> mgr);

    void setMemoryManager(std::unique_ptr<MemoryManagerBase> mgr);

    friend void resetMemoryManager();

    void resetMemoryManager();

    friend MemoryManagerBase& pinnedMemoryManager();

    friend void setMemoryManagerPinned(std::unique_ptr<MemoryManagerBase> mgr);

    void setMemoryManagerPinned(std::unique_ptr<MemoryManagerBase> mgr);

    friend void resetMemoryManagerPinned();

    void resetMemoryManagerPinned();

    friend flare::common::TheiaManager& theiaManager();

    friend GraphicsResourceManager& interopManager();

    friend std::string getDeviceInfo(int device) noexcept;

    friend std::string getPlatformInfo() noexcept;

    friend std::string getDriverVersion() noexcept;

    friend std::string getCUDARuntimeVersion() noexcept;

    friend std::string getDeviceInfo() noexcept;

    friend int getDeviceCount();

    friend int getDeviceNativeId(int device);

    friend int getDeviceIdFromNativeId(int nativeId);

    friend cudaStream_t getStream(int device);

    friend int setDevice(int device);

    friend const cudaDeviceProp& getDeviceProp(int device);

    friend std::pair<int, int> getComputeCapability(const int device);

    friend bool isDeviceBufferAccessible(int buf_device_id, int execution_id);

   private:
    DeviceManager();

    // Following two declarations are required to
    // avoid copying accidental copy/assignment
    // of instance returned by getInstance to other
    // variables
    DeviceManager(DeviceManager const&);
    void operator=(DeviceManager const&);

    // Attributes
    enum sort_mode { flops = 0, memory = 1, compute = 2, none = 3 };

    // Checks if the Graphics driver is capable of running the CUDA toolkit
    // version that Flare was compiled against
    void checkCudaVsDriverVersion();
    void sortDevices(sort_mode mode = flops);

    int setActiveDevice(int device, int nId = -1);

    std::shared_ptr<clog::logger> logger;

    /// A matrix of booleans where true indicates that the corresponding
    /// corrdinate devices can access each other buffers. False indicates
    /// buffers need to be copied over to the other device
    std::array<std::array<bool, MAX_DEVICES>, MAX_DEVICES>
        device_peer_access_map;

    std::vector<cudaDevice_t> cuDevices;
    std::vector<std::pair<int, int>> devJitComputes;

    int nDevices;
    cudaStream_t streams[MAX_DEVICES]{};

    std::unique_ptr<flare::common::TheiaManager> fgMngr;

    std::unique_ptr<MemoryManagerBase> memManager;

    std::unique_ptr<MemoryManagerBase> pinnedMemManager;

    std::unique_ptr<GraphicsResourceManager> gfxManagers[MAX_DEVICES];

    std::mutex mutex;
};

}  // namespace cuda
}  // namespace flare
