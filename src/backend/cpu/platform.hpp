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

#include <queue.hpp>
#include <string>

namespace flare {
namespace common {
class TheiaManager;
class MemoryManagerBase;
}  // namespace common
}  // namespace flare

using flare::common::MemoryManagerBase;

namespace flare {
namespace cpu {

int getBackend();

std::string getDeviceInfo() noexcept;

bool isDoubleSupported(int device);

bool isHalfSupported(int device);

void devprop(char* d_name, char* d_platform, char* d_toolkit, char* d_compute);

int& getMaxJitSize();

int getDeviceCount();

void init();

unsigned getActiveDeviceId();

size_t getDeviceMemorySize(int device);

size_t getHostMemorySize();

int setDevice(int device);

queue& getQueue(int device = 0);

/// Return a handle to the queue for the device.
///
/// \param[in] device The device of the returned queue
/// \returns The handle to the queue
queue* getQueueHandle(int device);

void sync(int device);

bool& evalFlag();

MemoryManagerBase& memoryManager();

void setMemoryManager(std::unique_ptr<MemoryManagerBase> mgr);

void resetMemoryManager();

// Pinned memory not supported
void setMemoryManagerPinned(std::unique_ptr<MemoryManagerBase> mgr);

void resetMemoryManagerPinned();

flare::common::TheiaManager& theiaManager();

}  // namespace cpu
}  // namespace flare
