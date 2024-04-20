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
#include <queue.hpp>
#include <memory>
#include <mutex>
#include <string>

using flare::common::MemoryManagerBase;

#ifndef FLY_CPU_MEM_DEBUG
#define FLY_CPU_MEM_DEBUG 0
#endif

#if defined(FLY_WITH_CPUID) &&                                       \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || \
     defined(_M_IX86) || defined(_WIN64))
#define CPUID_CAPABLE
#endif

#ifdef _WIN32
#include <intrin.h>
#include <limits.h>
typedef unsigned __int32 uint32_t;
#else
#include <stdint.h>
#endif

#ifdef CPUID_CAPABLE

#define MAX_INTEL_TOP_LVL 4

class CPUID {
    uint32_t regs[4];

   public:
    explicit CPUID(unsigned funcId, unsigned subFuncId) {
#ifdef _WIN32
        __cpuidex((int*)regs, (int)funcId, (int)subFuncId);

#else
        asm volatile("cpuid"
                     : "=a"(regs[0]), "=b"(regs[1]), "=c"(regs[2]),
                       "=d"(regs[3])
                     : "a"(funcId), "c"(subFuncId));
#endif
    }

    inline const uint32_t& EAX() const { return regs[0]; }
    inline const uint32_t& EBX() const { return regs[1]; }
    inline const uint32_t& ECX() const { return regs[2]; }
    inline const uint32_t& EDX() const { return regs[3]; }
};

#endif

class CPUInfo {
   public:
    CPUInfo();
    std::string vendor() const { return mVendorId; }
    std::string model() const { return mModelName; }
    int threads() const { return mNumLogCpus; }

   private:
    // Bit positions for data extractions
    static const uint32_t LVL_NUM   = 0x000000FF;
    static const uint32_t LVL_TYPE  = 0x0000FF00;
    static const uint32_t LVL_CORES = 0x0000FFFF;
    static const uint32_t HTT_POS   = 0x10000000;

    // Attributes
    std::string mVendorId;
    std::string mModelName;
    unsigned mNumSMT;
    unsigned mNumCores;
    unsigned mNumLogCpus;
    bool mIsHTT;
};

namespace flare {
namespace cpu {

class DeviceManager {
   public:
    static const int MAX_QUEUES            = 1;
    static const int NUM_DEVICES           = 1;
    static const unsigned ACTIVE_DEVICE_ID = 0;
    static const bool IS_DOUBLE_SUPPORTED  = true;

    // TODO(umar): Half is not supported for BLAS and FFT on x86_64
    static const bool IS_HALF_SUPPORTED = true;

    static DeviceManager& getInstance();

    friend queue& getQueue(int device);

    friend MemoryManagerBase& memoryManager();

    friend void setMemoryManager(std::unique_ptr<MemoryManagerBase> mgr);

    friend void resetMemoryManager();

    // Pinned memory not supported in CPU
    friend void setMemoryManagerPinned(std::unique_ptr<MemoryManagerBase> mgr);

    void setMemoryManagerPinned(std::unique_ptr<MemoryManagerBase> mgr);

    friend void resetMemoryManagerPinned();

    void resetMemoryManagerPinned();

    friend flare::common::TheiaManager& theiaManager();

    void setMemoryManager(std::unique_ptr<MemoryManagerBase> mgr);

    void resetMemoryManager();

    CPUInfo getCPUInfo() const;

   private:
    DeviceManager();
    // Following two declarations are required to
    // avoid copying accidental copy/assignment
    // of instance returned by getInstance to other
    // variables
    DeviceManager(DeviceManager const&)  = delete;
    void operator=(DeviceManager const&) = delete;

    // Attributes
    std::vector<queue> queues;
    std::unique_ptr<flare::common::TheiaManager> fgMngr;
    const CPUInfo cinfo;
    std::unique_ptr<MemoryManagerBase> memManager;
    std::mutex mutex;
};

}  // namespace cpu
}  // namespace flare
