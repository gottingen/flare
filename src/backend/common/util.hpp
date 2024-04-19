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

/// This file contains platform independent utility functions
#pragma once

#include <optypes.hpp>
#include <fly/defines.h>

#include <string>

namespace flare {
namespace common {
/// The environment variable that determines where the runtime kernels
/// will be stored on the file system
constexpr const char* JIT_KERNEL_CACHE_DIRECTORY_ENV_NAME =
    "FLY_JIT_KERNEL_CACHE_DIRECTORY";

std::string getEnvVar(const std::string& key);

std::string& ltrim(std::string& s);

// Dump the kernel sources only if the environment variable is defined
void saveKernel(const std::string& funcName, const std::string& jit_ker,
                const std::string& ext);

std::string& getCacheDirectory();

bool directoryExists(const std::string& path);

bool createDirectory(const std::string& path);

bool removeFile(const std::string& path);

bool renameFile(const std::string& sourcePath, const std::string& destPath);

bool isDirectoryWritable(const std::string& path);

/// Return a string suitable for naming a temporary file.
///
/// Every call to this function will generate a new string with a very low
/// probability of colliding with past or future outputs of this function,
/// including calls from other threads or processes. The string contains
/// no extension.
std::string makeTempFilename();

const char* getName(fly_dtype type);

std::string getOpEnumStr(fly_op_t val);

template<typename T>
std::string toString(T value);

}  // namespace common
}  // namespace flare
