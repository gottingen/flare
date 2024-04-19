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

#ifdef _WIN32
#include <windows.h>  // clog needs this
#endif

#include <common/Logger.hpp>
#include <common/util.hpp>

#include <collie/log/sinks/stdout_sinks.h>
#include <array>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>

using std::array;
using std::shared_ptr;
using std::string;

using clog::get;
using clog::logger;
using clog::stdout_logger_mt;

namespace flare {
namespace common {

shared_ptr<logger> loggerFactory(const string& name) {
    shared_ptr<logger> logger;
    if (!(logger = get(name))) {
        logger = stdout_logger_mt(name);
        logger->set_pattern("[%n][%E][%t] %v");

        // Log mode
        string env_var = getEnvVar("FLY_TRACE");
        if (env_var.find("all") != string::npos ||
            env_var.find(name) != string::npos) {
            logger->set_level(clog::level::trace);
        } else {
            logger->set_level(clog::level::off);
        }
    }
    return logger;
}

string bytesToString(size_t bytes) {
    constexpr array<const char*, 7> units{
        {"B", "KB", "MB", "GB", "TB", "PB", "EB"}};
    size_t count     = 0;
    auto fbytes      = static_cast<double>(bytes);
    size_t num_units = units.size();
    for (count = 0; count < num_units && fbytes > 1000.0f; count++) {
        fbytes *= (1.0f / 1024.0f);
    }
    if (count == units.size()) { count--; }
    return fmt::format("{:.3g} {}", fbytes, units[count]);
}
}  // namespace common
}  // namespace flare
