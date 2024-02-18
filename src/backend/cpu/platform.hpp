/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <queue.hpp>
#include <string>

namespace flare {
namespace common {
class ForgeManager;
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

flare::common::ForgeManager& forgeManager();

}  // namespace cpu
}  // namespace flare
