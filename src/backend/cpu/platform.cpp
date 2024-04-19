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

#include <build_version.hpp>
#include <common/MemoryManagerBase.hpp>
#include <common/defines.hpp>
#include <common/host_memory.hpp>
#include <device_manager.hpp>
#include <platform.hpp>
#include <fly/version.h>

#include <cctype>
#include <cstdio>
#include <memory>
#include <sstream>
#include <string>

using flare::common::TheiaManager;
using flare::common::getEnvVar;
using flare::common::ltrim;
using flare::common::MemoryManagerBase;
using std::endl;
using std::ostringstream;
using std::stoi;
using std::string;
using std::unique_ptr;

namespace flare {
namespace cpu {

static string get_system() {
    string arch = (sizeof(void*) == 4) ? "32-bit " : "64-bit ";

    return arch +
#if defined(OS_LNX)
           "Linux";
#elif defined(OS_WIN)
           "Windows";
#elif defined(OS_MAC)
           "Mac OSX";
#endif
}

int getBackend() { return FLY_BACKEND_CPU; }

string getDeviceInfo() noexcept {
    const CPUInfo cinfo = DeviceManager::getInstance().getCPUInfo();

    ostringstream info;

    info << "Flare v" << FLY_VERSION << " (CPU, " << get_system()
         << ", build " << FLY_REVISION << ")" << endl;

    string model = cinfo.model();

    size_t memMB =
        getDeviceMemorySize(static_cast<int>(getActiveDeviceId())) / 1048576;

    info << string("[0] ") << cinfo.vendor() << ": " << ltrim(model);

    if (memMB) {
        info << ", " << memMB << " MB, ";
    } else {
        info << ", Unknown MB, ";
    }

    info << "Max threads(" << cinfo.threads() << ") ";
#ifndef NDEBUG
    info << FLY_COMPILER_STR;
#endif
    info << endl;

    return info.str();
}

bool isDoubleSupported(int device) {
    UNUSED(device);
    return DeviceManager::IS_DOUBLE_SUPPORTED;
}

bool isHalfSupported(int device) {
    UNUSED(device);
    return DeviceManager::IS_HALF_SUPPORTED;
}

void devprop(char* d_name, char* d_platform, char* d_toolkit, char* d_compute) {
    const CPUInfo cinfo = DeviceManager::getInstance().getCPUInfo();

    snprintf(d_name, 64, "%s", cinfo.vendor().c_str());
    snprintf(d_platform, 10, "CPU");
    // report the compiler for toolkit
    snprintf(d_toolkit, 64, "%s", FLY_COMPILER_STR);
    snprintf(d_compute, 10, "%s", "0.0");
}

int& getMaxJitSize() {
    constexpr int MAX_JIT_LEN = 100;
    thread_local int length   = 0;
    if (length <= 0) {
        string env_var = getEnvVar("FLY_CPU_MAX_JIT_LEN");
        if (!env_var.empty()) {
            int input_len = stoi(env_var);
            length        = input_len > 0 ? input_len : MAX_JIT_LEN;
        } else {
            length = MAX_JIT_LEN;
        }
    }
    return length;
}

int getDeviceCount() { return DeviceManager::NUM_DEVICES; }

void init() {
    thread_local const auto& instance = DeviceManager::getInstance();
    UNUSED(instance);
}

// Get the currently active device id
unsigned getActiveDeviceId() { return DeviceManager::ACTIVE_DEVICE_ID; }

size_t getDeviceMemorySize(int device) {
    UNUSED(device);
    return common::getHostMemorySize();
}

size_t getHostMemorySize() { return common::getHostMemorySize(); }

int setDevice(int device) {
    thread_local bool flag = false;
    if (!flag && device != 0) {
#ifndef NDEBUG
        fprintf(
            stderr,
            "WARNING fly_set_device(device): device can only be 0 for CPU\n");
#endif
        flag = true;
    }
    return 0;
}

queue& getQueue(int device) {
    return DeviceManager::getInstance().queues[device];
}

queue* getQueueHandle(int device) { return &getQueue(device); }

void sync(int device) { getQueue(device).sync(); }

bool& evalFlag() {
    thread_local bool flag = true;
    return flag;
}

MemoryManagerBase& memoryManager() {
    DeviceManager& inst = DeviceManager::getInstance();
    return *(inst.memManager);
}

void setMemoryManager(unique_ptr<MemoryManagerBase> mgr) {
    return DeviceManager::getInstance().setMemoryManager(move(mgr));
}

void resetMemoryManager() {
    return DeviceManager::getInstance().resetMemoryManager();
}

void setMemoryManagerPinned(unique_ptr<MemoryManagerBase> mgr) {
    return DeviceManager::getInstance().setMemoryManagerPinned(move(mgr));
}

void resetMemoryManagerPinned() {
    return DeviceManager::getInstance().resetMemoryManagerPinned();
}

TheiaManager& theiaManager() { return *(DeviceManager::getInstance().fgMngr); }

}  // namespace cpu
}  // namespace flare
