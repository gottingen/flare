// Copyright 2023 The Elastic-AI Authors.
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


#include <flare/core.h>
#include <flare/core/common/device.h>
#include <flare/core/common/error.h>
#include <flare/core/common/command_line_parsing.h>
#include <flare/core/common/parse_command_line.h>
#include <flare/core/common/device_management.h>
#include <flare/core/common/exec_space_manager.h>
#include <flare/core/common/cpu_discovery.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <stack>
#include <functional>
#include <list>
#include <cerrno>
#include <random>
#include <regex>

#ifndef _WIN32

#include <unistd.h>

#else
#include <windows.h>
#endif

//----------------------------------------------------------------------------
namespace {
    bool g_is_initialized = false;
    bool g_is_finalized = false;
    bool g_show_warnings = true;
    bool g_tune_internals = false;
// When compiling with clang/LLVM and using the GNU (GCC) C++ Standard Library
// (any recent version between GCC 7.3 and GCC 9.2), std::deque SEGV's during
// the unwinding of the atexit(3C) handlers at program termination.  However,
// this bug is not observable when building with GCC.
// As an added bonus, std::list<T> provides constant insertion and
// deletion time complexity, which translates to better run-time performance. As
// opposed to std::deque<T> which does not provide the same constant time
// complexity for inserts/removals, since std::deque<T> is implemented as a
// segmented array.
    using hook_function_type = std::function<void()>;
    std::stack<hook_function_type, std::list<hook_function_type>> finalize_hooks;

/**
 * The category is only used in printing, tools
 * get all metadata free of category
 */
    using metadata_category_type = std::string;
    using metadata_key_type = std::string;
    using metadata_value_type = std::string;

    std::map<metadata_category_type,
            std::map<metadata_key_type, metadata_value_type>>
            metadata_map;

    void declare_configuration_metadata(const std::string &category,
                                        const std::string &key,
                                        const std::string &value) {
        metadata_map[category][key] = value;
    }

    void combine(flare::InitializationSettings &out,
                 flare::InitializationSettings const &in) {
#define FLARE_IMPL_COMBINE_SETTING(NAME) \
  if (in.has_##NAME()) {                  \
    out.set_##NAME(in.get_##NAME());      \
  }                                       \
  static_assert(true, "no-op to require trailing semicolon")
        FLARE_IMPL_COMBINE_SETTING(num_threads);
        FLARE_IMPL_COMBINE_SETTING(map_device_id_by);
        FLARE_IMPL_COMBINE_SETTING(device_id);
        FLARE_IMPL_COMBINE_SETTING(num_devices);
        FLARE_IMPL_COMBINE_SETTING(skip_device);
        FLARE_IMPL_COMBINE_SETTING(disable_warnings);
        FLARE_IMPL_COMBINE_SETTING(tune_internals);
        FLARE_IMPL_COMBINE_SETTING(tools_help);
        FLARE_IMPL_COMBINE_SETTING(tools_libs);
        FLARE_IMPL_COMBINE_SETTING(tools_args);
#undef FLARE_IMPL_COMBINE_SETTING
    }

    void combine(flare::InitializationSettings &out,
                 flare::Tools::InitArguments const &in) {
        using flare::Tools::InitArguments;
        if (in.help != InitArguments::PossiblyUnsetOption::unset) {
            out.set_tools_help(in.help == InitArguments::PossiblyUnsetOption::on);
        }
        if (in.lib != InitArguments::unset_string_option) {
            out.set_tools_libs(in.lib);
        }
        if (in.args != InitArguments::unset_string_option) {
            out.set_tools_args(in.args);
        }
    }

    void combine(flare::Tools::InitArguments &out,
                 flare::InitializationSettings const &in) {
        using flare::Tools::InitArguments;
        if (in.has_tools_help()) {
            out.help = in.get_tools_help() ? InitArguments::PossiblyUnsetOption::on
                                           : InitArguments::PossiblyUnsetOption::off;
        }
        if (in.has_tools_libs()) {
            out.lib = in.get_tools_libs();
        }
        if (in.has_tools_args()) {
            out.args = in.get_tools_args();
        }
    }

#ifndef FLARE_ON_CUDA_DEVICE

    int get_device_count() {
        flare::abort("implementation bug");
        return -1;
    }

#endif

    unsigned get_process_id() {
#ifdef _WIN32
        return unsigned(GetCurrentProcessId());
#else
        return unsigned(getpid());
#endif
    }

    bool is_valid_num_threads(int x) { return x > 0; }

    bool is_valid_device_id(int x) { return x >= 0; }

    bool is_valid_map_device_id_by(std::string const &x) {
        return x == "mpi_rank" || x == "random";
    }

}  // namespace



[[nodiscard]] int flare::device_id() noexcept {
#ifndef FLARE_ON_CUDA_DEVICE
    return -1;
#else
    return Cuda().cuda_device();
#endif
}


[[nodiscard]] int flare::num_threads() noexcept {
    return DefaultHostExecutionSpace().concurrency();
}

flare::detail::ExecSpaceManager &flare::detail::ExecSpaceManager::get_instance() {
    static ExecSpaceManager space_initializer = {};
    return space_initializer;
}

void flare::detail::ExecSpaceManager::register_space_factory(
        const std::string name, std::unique_ptr<ExecSpaceBase> space) {
    exec_space_factory_list[name] = std::move(space);
}

void flare::detail::ExecSpaceManager::initialize_spaces(
        const InitializationSettings &settings) {
    // Note: the names of the execution spaces, used as keys in the map, encode
    // the ordering of the initialization code from the old initialization stuff.
    // Eventually, we may want to do something less brittle than this, but for now
    // we're just preserving compatibility with the old implementation.
    for (auto &to_init: exec_space_factory_list) {
        to_init.second->initialize(settings);
    }
}

void flare::detail::ExecSpaceManager::finalize_spaces() {
    for (auto &to_finalize: exec_space_factory_list) {
        to_finalize.second->finalize();
    }
}

void flare::detail::ExecSpaceManager::static_fence(const std::string &name) {
    for (auto &to_fence: exec_space_factory_list) {
        to_fence.second->static_fence(name);
    }
}

void flare::detail::ExecSpaceManager::print_configuration(std::ostream &os,
                                                          bool verbose) {
    for (auto const &to_print: exec_space_factory_list) {
        to_print.second->print_configuration(os, verbose);
    }
}

int flare::detail::get_ctest_gpu(int local_rank) {
    auto const *ctest_flare_profile_device_type =
            std::getenv("CTEST_FLARE_DEVICE_TYPE");
    if (!ctest_flare_profile_device_type) {
        return 0;
    }

    auto const *ctest_resource_group_count_str =
            std::getenv("CTEST_RESOURCE_GROUP_COUNT");
    if (!ctest_resource_group_count_str) {
        return 0;
    }

    // Make sure rank is within bounds of resource groups specified by CTest
    auto resource_group_count = std::stoi(ctest_resource_group_count_str);
    assert(local_rank >= 0);
    if (local_rank >= resource_group_count) {
        std::ostringstream ss;
        ss << "Error: local rank " << local_rank
           << " is outside the bounds of resource groups provided by CTest. Raised"
           << " by flare::detail::get_ctest_gpu().";
        throw_runtime_exception(ss.str());
    }

    // Get the resource types allocated to this resource group
    std::ostringstream ctest_resource_group;
    ctest_resource_group << "CTEST_RESOURCE_GROUP_" << local_rank;
    std::string ctest_resource_group_name = ctest_resource_group.str();
    auto const *ctest_resource_group_str =
            std::getenv(ctest_resource_group_name.c_str());
    if (!ctest_resource_group_str) {
        std::ostringstream ss;
        ss << "Error: " << ctest_resource_group_name << " is not specified. Raised"
           << " by flare::detail::get_ctest_gpu().";
        throw_runtime_exception(ss.str());
    }

    // Look for the device type specified in CTEST_FLARE_DEVICE_TYPE
    bool found_device = false;
    std::string ctest_resource_group_cxx_str = ctest_resource_group_str;
    std::istringstream instream(ctest_resource_group_cxx_str);
    while (true) {
        std::string devName;
        std::getline(instream, devName, ',');
        if (devName == ctest_flare_profile_device_type) {
            found_device = true;
            break;
        }
        if (instream.eof() || devName.length() == 0) {
            break;
        }
    }

    if (!found_device) {
        std::ostringstream ss;
        ss << "Error: device type '" << ctest_flare_profile_device_type
           << "' not included in " << ctest_resource_group_name
           << ". Raised by flare::detail::get_ctest_gpu().";
        throw_runtime_exception(ss.str());
    }

    // Get the device ID
    std::string ctest_device_type_upper = ctest_flare_profile_device_type;
    for (auto &c: ctest_device_type_upper) {
        c = std::toupper(c);
    }
    ctest_resource_group << "_" << ctest_device_type_upper;

    std::string ctest_resource_group_id_name = ctest_resource_group.str();
    auto resource_str = std::getenv(ctest_resource_group_id_name.c_str());
    if (!resource_str) {
        std::ostringstream ss;
        ss << "Error: " << ctest_resource_group_id_name
           << " is not specified. Raised by flare::detail::get_ctest_gpu().";
        throw_runtime_exception(ss.str());
    }

    auto const *comma = std::strchr(resource_str, ',');
    if (!comma || strncmp(resource_str, "id:", 3)) {
        std::ostringstream ss;
        ss << "Error: invalid value of " << ctest_resource_group_id_name << ": '"
           << resource_str << "'. Raised by flare::detail::get_ctest_gpu().";
        throw_runtime_exception(ss.str());
    }

    std::string id(resource_str + 3, comma - resource_str - 3);
    return std::stoi(id.c_str());
}

std::vector<int> flare::detail::get_visible_devices(
        flare::InitializationSettings const &settings, int device_count) {
    std::vector<int> visible_devices;
    char *env_visible_devices = std::getenv("FLARE_VISIBLE_DEVICES");
    if (env_visible_devices) {
        std::stringstream ss(env_visible_devices);
        for (int i; ss >> i;) {
            visible_devices.push_back(i);
            if (ss.peek() == ',') ss.ignore();
        }
        for (auto id: visible_devices) {
            if (id < 0) {
                ss << "Error: Invalid device id '" << id
                   << "' in environment variable 'FLARE_VISIBLE_DEVICES="
                   << env_visible_devices << "'."
                   << " Device id cannot be negative!"
                   << " Raised by flare::initialize().\n";
            }
            if (id >= device_count) {
                ss << "Error: Invalid device id '" << id
                   << "' in environment variable 'FLARE_VISIBLE_DEVICES="
                   << env_visible_devices << "'."
                   << " Device id must be smaller than the number of GPUs available"
                   << " for execution '" << device_count << "'!"
                   << " Raised by flare::initialize().\n";
            }
        }
    } else {
        int num_devices =
                settings.has_num_devices() ? settings.get_num_devices() : device_count;
        if (num_devices > device_count) {
            std::stringstream ss;
            ss << "Error: Specified number of devices '" << num_devices
               << "' exceeds the actual number of GPUs available for execution '"
               << device_count << "'."
               << " Raised by flare::initialize().\n";
            flare::abort(ss.str().c_str());
        }
        for (int i = 0; i < num_devices; ++i) {
            visible_devices.push_back(i);
        }
        if (settings.has_skip_device()) {
            if (visible_devices.size() == 1 && settings.get_skip_device() == 0) {
                flare::abort(
                        "Error: skipping the only GPU available for execution.\n"
                        " Raised by flare::initialize().\n");
            }
            visible_devices.erase(
                    std::remove(visible_devices.begin(), visible_devices.end(),
                                settings.get_skip_device()),
                    visible_devices.end());
        }
    }
    if (visible_devices.empty()) {
        flare::abort(
                "Error: no GPU available for execution.\n"
                " Raised by flare::initialize().\n");
    }
    return visible_devices;
}

int flare::detail::get_gpu(const InitializationSettings &settings) {
    std::vector<int> visible_devices =
            get_visible_devices(settings, get_device_count());
    int const num_devices = visible_devices.size();
    // device_id is provided
    if (settings.has_device_id()) {
        int const id = settings.get_device_id();
        if (id < 0) {
            std::stringstream ss;
            ss << "Error: Requested GPU with invalid id '" << id << "'."
               << " Device id cannot be negative!"
               << " Raised by flare::initialize().\n";
            flare::abort(ss.str().c_str());
        }
        if (id >= num_devices) {
            std::stringstream ss;
            ss << "Error: Requested GPU with id '" << id << "' but only "
               << num_devices << "GPU(s) available!"
               << " Raised by flare::initialize().\n";
            flare::abort(ss.str().c_str());
        }
        return visible_devices[settings.get_device_id()];
    }

    // either random or round-robin assignment based on local MPI rank
    if (settings.has_map_device_id_by() &&
        !is_valid_map_device_id_by(settings.get_map_device_id_by())) {
        std::stringstream ss;
        ss << "Error: map_device_id_by setting '" << settings.get_map_device_id_by()
           << "' is not recognized."
           << " Raised by flare::initialize().\n";
        flare::abort(ss.str().c_str());
    }

    if (settings.has_map_device_id_by() &&
        settings.get_map_device_id_by() == "random") {
        std::default_random_engine gen(get_process_id());
        std::uniform_int_distribution<int> distribution(0, num_devices - 1);
        return visible_devices[distribution(gen)];
    }

    // either map_device_id_by is not specified or it is mpi_rank
    if (settings.has_map_device_id_by() &&
        settings.get_map_device_id_by() != "mpi_rank") {
        flare::abort("implementation bug");
    }

    int const mpi_local_rank = mpi_local_rank_on_node();

    // use first GPU available for execution if unable to detect local MPI rank
    if (mpi_local_rank < 0) {
        if (settings.has_map_device_id_by()) {
            std::cerr << "Warning: unable to detect local MPI rank."
                      << " Falling back to the first GPU available for execution."
                      << " Raised by flare::initialize()." << std::endl;
        }
        return visible_devices[0];
    }

    // use device assigned by CTest when resource allocation is activated
    if (std::getenv("CTEST_FLARE_DEVICE_TYPE") &&
        std::getenv("CTEST_RESOURCE_GROUP_COUNT")) {
        return get_ctest_gpu(mpi_local_rank);
    }

    return visible_devices[mpi_local_rank % visible_devices.size()];
}

namespace {

    void initialize_backends(const flare::InitializationSettings &settings) {
// This is an experimental setting
// For KNL in Flat mode this variable should be set, so that
// memkind allocates high bandwidth memory correctly.
#ifdef FLARE_ENABLE_HBWSPACE
        setenv("MEMKIND_HBW_NODES", "1", 0);
#endif

        flare::detail::ExecSpaceManager::get_instance().initialize_spaces(settings);
    }

    void initialize_profiling(const flare::Tools::InitArguments &args) {
        auto initialization_status =
                flare::Tools::detail::initialize_tools_subsystem(args);
        if (initialization_status.result ==
            flare::Tools::detail::InitializationStatus::InitializationResult::
            help_request) {
            g_is_initialized = true;
            ::flare::finalize();
            std::exit(EXIT_SUCCESS);
        } else if (initialization_status.result ==
                   flare::Tools::detail::InitializationStatus::InitializationResult::
                   success) {
            flare::Tools::parseArgs(args.args);
            for (const auto &category_value: metadata_map) {
                for (const auto &key_value: category_value.second) {
                    flare::Tools::declareMetadata(key_value.first, key_value.second);
                }
            }
        } else {
            std::cerr << "Error initializing flare Tools subsystem" << std::endl;
            g_is_initialized = true;
            ::flare::finalize();
            std::exit(EXIT_FAILURE);
        }
    }

    std::string version_string_from_int(int version_number) {
        std::stringstream str_builder;
        str_builder << version_number / 10000 << "." << (version_number % 10000) / 100
                    << "." << version_number % 100;
        return str_builder.str();
    }

    void pre_initialize_internal(const flare::InitializationSettings &settings) {
        if (settings.has_disable_warnings() && settings.get_disable_warnings())
            g_show_warnings = false;
        if (settings.has_tune_internals() && settings.get_tune_internals())
            g_tune_internals = true;
        declare_configuration_metadata("version_info", "flare Version",
                                       version_string_from_int(FLARE_VERSION));
#ifdef FLARE_COMPILER_APPLECC
        declare_configuration_metadata("compiler_version", "FLARE_COMPILER_APPLECC",
                                       std::to_string(FLARE_COMPILER_APPLECC));
        declare_configuration_metadata("tools_only", "compiler_family", "apple");
#endif
#ifdef FLARE_COMPILER_CLANG
        declare_configuration_metadata("compiler_version", "FLARE_COMPILER_CLANG",
                                       std::to_string(FLARE_COMPILER_CLANG));
        declare_configuration_metadata("tools_only", "compiler_family", "clang");
#endif
#ifdef FLARE_COMPILER_CRAYC
        declare_configuration_metadata("compiler_version", "FLARE_COMPILER_CRAYC",
                                       std::to_string(FLARE_COMPILER_CRAYC));
        declare_configuration_metadata("tools_only", "compiler_family", "cray");
#endif
#ifdef FLARE_COMPILER_GNU
        declare_configuration_metadata("compiler_version", "FLARE_COMPILER_GNU",
                                       std::to_string(FLARE_COMPILER_GNU));
        declare_configuration_metadata("tools_only", "compiler_family", "gnu");
#endif
#ifdef FLARE_COMPILER_INTEL
        declare_configuration_metadata("compiler_version", "FLARE_COMPILER_INTEL",
                                       std::to_string(FLARE_COMPILER_INTEL));
        declare_configuration_metadata("tools_only", "compiler_family", "intel");
#endif
#ifdef FLARE_COMPILER_INTEL_LLVM
        declare_configuration_metadata("compiler_version",
                                       "FLARE_COMPILER_INTEL_LLVM",
                                       std::to_string(FLARE_COMPILER_INTEL_LLVM));
        declare_configuration_metadata("tools_only", "compiler_family", "intel_llvm");
#endif
#ifdef FLARE_COMPILER_NVCC
        declare_configuration_metadata("compiler_version", "FLARE_COMPILER_NVCC",
                                       std::to_string(FLARE_COMPILER_NVCC));
        declare_configuration_metadata("tools_only", "compiler_family", "nvcc");
#endif
#ifdef FLARE_COMPILER_NVHPC
        declare_configuration_metadata("compiler_version", "FLARE_COMPILER_NVHPC",
                                       std::to_string(FLARE_COMPILER_NVHPC));
        declare_configuration_metadata("tools_only", "compiler_family", "pgi");
#endif
#ifdef FLARE_COMPILER_MSVC
        declare_configuration_metadata("compiler_version", "FLARE_COMPILER_MSVC",
                                       std::to_string(FLARE_COMPILER_MSVC));
        declare_configuration_metadata("tools_only", "compiler_family", "msvc");
#endif

#ifdef FLARE_ENABLE_PRAGMA_IVDEP
        declare_configuration_metadata("vectorization", "FLARE_ENABLE_PRAGMA_IVDEP",
                                       "yes");
#else
        declare_configuration_metadata("vectorization", "FLARE_ENABLE_PRAGMA_IVDEP",
                                       "no");
#endif
#ifdef FLARE_ENABLE_PRAGMA_LOOPCOUNT
        declare_configuration_metadata("vectorization",
                                       "FLARE_ENABLE_PRAGMA_LOOPCOUNT", "yes");
#else
        declare_configuration_metadata("vectorization",
                                       "FLARE_ENABLE_PRAGMA_LOOPCOUNT", "no");
#endif
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
        declare_configuration_metadata("vectorization", "FLARE_ENABLE_PRAGMA_UNROLL",
                                       "yes");
#else
        declare_configuration_metadata("vectorization", "FLARE_ENABLE_PRAGMA_UNROLL",
                                       "no");
#endif
#ifdef FLARE_ENABLE_PRAGMA_VECTOR
        declare_configuration_metadata("vectorization", "FLARE_ENABLE_PRAGMA_VECTOR",
                                       "yes");
#else
        declare_configuration_metadata("vectorization", "FLARE_ENABLE_PRAGMA_VECTOR",
                                       "no");
#endif

#ifdef FLARE_ENABLE_HBWSPACE
        declare_configuration_metadata("memory", "FLARE_ENABLE_HBWSPACE", "yes");
#else
        declare_configuration_metadata("memory", "FLARE_ENABLE_HBWSPACE", "no");
#endif
#ifdef FLARE_ENABLE_INTEL_MM_ALLOC
        declare_configuration_metadata("memory", "FLARE_ENABLE_INTEL_MM_ALLOC",
                                       "yes");
#else
        declare_configuration_metadata("memory", "FLARE_ENABLE_INTEL_MM_ALLOC",
                                       "no");
#endif

#ifdef FLARE_ENABLE_ASM
        declare_configuration_metadata("options", "FLARE_ENABLE_ASM", "yes");
#else
        declare_configuration_metadata("options", "FLARE_ENABLE_ASM", "no");
#endif
#ifdef FLARE_ENABLE_CXX17
        declare_configuration_metadata("options", "FLARE_ENABLE_CXX17", "yes");
#else
        declare_configuration_metadata("options", "FLARE_ENABLE_CXX17", "no");
#endif
#ifdef FLARE_ENABLE_CXX20
        declare_configuration_metadata("options", "FLARE_ENABLE_CXX20", "yes");
#else
        declare_configuration_metadata("options", "FLARE_ENABLE_CXX20", "no");
#endif
#ifdef FLARE_ENABLE_CXX23
        declare_configuration_metadata("options", "FLARE_ENABLE_CXX23", "yes");
#else
        declare_configuration_metadata("options", "FLARE_ENABLE_CXX23", "no");
#endif
#ifdef FLARE_ENABLE_DEBUG_BOUNDS_CHECK
        declare_configuration_metadata("options", "FLARE_ENABLE_DEBUG_BOUNDS_CHECK",
                                       "yes");
#else
        declare_configuration_metadata("options", "FLARE_ENABLE_DEBUG_BOUNDS_CHECK",
                                       "no");
#endif
#ifdef FLARE_ENABLE_HWLOC
        declare_configuration_metadata("options", "FLARE_ENABLE_HWLOC", "yes");
#else
        declare_configuration_metadata("options", "FLARE_ENABLE_HWLOC", "no");
#endif
#ifdef FLARE_ENABLE_LIBRT
        declare_configuration_metadata("options", "FLARE_ENABLE_LIBRT", "yes");
#else
        declare_configuration_metadata("options", "FLARE_ENABLE_LIBRT", "no");
#endif
#ifdef FLARE_ENABLE_LIBDL
        declare_configuration_metadata("options", "FLARE_ENABLE_LIBDL", "yes");
#else
        declare_configuration_metadata("options", "FLARE_ENABLE_LIBDL", "no");
#endif
        declare_configuration_metadata("architecture", "Default Device",
                                       typeid(flare::DefaultExecutionSpace).name());

#if defined(FLARE_ARCH_A64FX)
        declare_configuration_metadata("architecture", "CPU architecture", "A64FX");
#elif defined(FLARE_ARCH_AMDAVX)
        declare_configuration_metadata("architecture", "CPU architecture", "AMDAVX");
#elif defined(FLARE_ARCH_ARMV80)
        declare_configuration_metadata("architecture", "CPU architecture", "ARMV80");
#elif defined(FLARE_ARCH_ARMV81)
        declare_configuration_metadata("architecture", "CPU architecture", "ARMV81");
#elif defined(FLARE_ARCH_ARMV8_THUNDERX)
        declare_configuration_metadata("architecture", "CPU architecture",
                                       "ARMV8_THUNDERX");
#elif defined(FLARE_ARCH_ARMV8_THUNDERX2)
        declare_configuration_metadata("architecture", "CPU architecture",
                                       "ARMV8_THUNDERX2");
#elif defined(FLARE_ARCH_BDW)
        declare_configuration_metadata("architecture", "CPU architecture", "BDW");
#elif defined(FLARE_ARCH_BGQ)
        declare_configuration_metadata("architecture", "CPU architecture", "BGQ");
#elif defined(FLARE_ARCH_HSW)
        declare_configuration_metadata("architecture", "CPU architecture", "HSW");
#elif defined(FLARE_ARCH_ICL)
        declare_configuration_metadata("architecture", "CPU architecture", "ICL");
#elif defined(FLARE_ARCH_ICX)
        declare_configuration_metadata("architecture", "CPU architecture", "ICX");
#elif defined(FLARE_ARCH_KNC)
        declare_configuration_metadata("architecture", "CPU architecture", "KNC");
#elif defined(FLARE_ARCH_KNL)
        declare_configuration_metadata("architecture", "CPU architecture", "KNL");
#elif defined(FLARE_ARCH_NATIVE)
        declare_configuration_metadata("architecture", "CPU architecture", "NATIVE");
#elif defined(FLARE_ARCH_POWER7)
        declare_configuration_metadata("architecture", "CPU architecture", "POWER7");
#elif defined(FLARE_ARCH_POWER8)
        declare_configuration_metadata("architecture", "CPU architecture", "POWER8");
#elif defined(FLARE_ARCH_POWER9)
        declare_configuration_metadata("architecture", "CPU architecture", "POWER9");
#elif defined(FLARE_ARCH_SKL)
        declare_configuration_metadata("architecture", "CPU architecture", "SKL");
#elif defined(FLARE_ARCH_SKX)
        declare_configuration_metadata("architecture", "CPU architecture", "SKX");
#elif defined(FLARE_ARCH_SNB)
        declare_configuration_metadata("architecture", "CPU architecture", "SNB");
#elif defined(FLARE_ARCH_SPR)
        declare_configuration_metadata("architecture", "CPU architecture", "SPR");
#elif defined(FLARE_ARCH_WSM)
        declare_configuration_metadata("architecture", "CPU architecture", "WSM");
#elif defined(FLARE_ARCH_AMD_ZEN)
        declare_configuration_metadata("architecture", "CPU architecture", "AMD_ZEN");
#elif defined(FLARE_ARCH_AMD_ZEN2)
        declare_configuration_metadata("architecture", "CPU architecture",
                                       "AMD_ZEN2");
#elif defined(FLARE_ARCH_AMD_ZEN3)
        declare_configuration_metadata("architecture", "CPU architecture",
                                       "AMD_ZEN3");
#else
        declare_configuration_metadata("architecture", "CPU architecture", "none");
#endif

#if defined(FLARE_ARCH_INTEL_GEN)
        declare_configuration_metadata("architecture", "GPU architecture",
                                       "INTEL_GEN");
#elif defined(FLARE_ARCH_INTEL_DG1)
        declare_configuration_metadata("architecture", "GPU architecture",
                                       "INTEL_DG1");
#elif defined(FLARE_ARCH_INTEL_GEN9)
        declare_configuration_metadata("architecture", "GPU architecture",
                                       "INTEL_GEN9");
#elif defined(FLARE_ARCH_INTEL_GEN11)
        declare_configuration_metadata("architecture", "GPU architecture",
                                       "INTEL_GEN11");
#elif defined(FLARE_ARCH_INTEL_GEN12LP)
        declare_configuration_metadata("architecture", "GPU architecture",
                                       "INTEL_GEN12LP");
#elif defined(FLARE_ARCH_INTEL_XEHP)
        declare_configuration_metadata("architecture", "GPU architecture",
                                       "INTEL_XEHP");
#elif defined(FLARE_ARCH_INTEL_PVC)
        declare_configuration_metadata("architecture", "GPU architecture",
                                       "INTEL_PVC");

#elif defined(FLARE_ARCH_KEPLER30)
        declare_configuration_metadata("architecture", "GPU architecture",
                                       "KEPLER30");
#elif defined(FLARE_ARCH_KEPLER32)
        declare_configuration_metadata("architecture", "GPU architecture",
                                       "KEPLER32");
#elif defined(FLARE_ARCH_KEPLER35)
        declare_configuration_metadata("architecture", "GPU architecture",
                                       "KEPLER35");
#elif defined(FLARE_ARCH_KEPLER37)
        declare_configuration_metadata("architecture", "GPU architecture",
                                       "KELPER37");
#elif defined(FLARE_ARCH_MAXWELL50)
        declare_configuration_metadata("architecture", "GPU architecture",
                                       "MAXWELL50");
#elif defined(FLARE_ARCH_MAXWELL52)
        declare_configuration_metadata("architecture", "GPU architecture",
                                       "MAXWELL52");
#elif defined(FLARE_ARCH_MAXWELL53)
        declare_configuration_metadata("architecture", "GPU architecture",
                                       "MAXWELL53");
#elif defined(FLARE_ARCH_PASCAL60)
        declare_configuration_metadata("architecture", "GPU architecture",
                                       "PASCAL60");
#elif defined(FLARE_ARCH_PASCAL61)
        declare_configuration_metadata("architecture", "GPU architecture",
                                       "PASCAL61");
#elif defined(FLARE_ARCH_VOLTA70)
        declare_configuration_metadata("architecture", "GPU architecture", "VOLTA70");
#elif defined(FLARE_ARCH_VOLTA72)
        declare_configuration_metadata("architecture", "GPU architecture", "VOLTA72");
#elif defined(FLARE_ARCH_TURING75)
        declare_configuration_metadata("architecture", "GPU architecture",
                                       "TURING75");
#elif defined(FLARE_ARCH_AMPERE80)
        declare_configuration_metadata("architecture", "GPU architecture",
                                       "AMPERE80");
#elif defined(FLARE_ARCH_AMPERE86)
        declare_configuration_metadata("architecture", "GPU architecture",
                                       "AMPERE86");
#elif defined(FLARE_ARCH_ADA89)
        declare_configuration_metadata("architecture", "GPU architecture", "ADA89");
#elif defined(FLARE_ARCH_HOPPER90)
        declare_configuration_metadata("architecture", "GPU architecture",
                                       "HOPPER90");
#elif defined(FLARE_ARCH_AMD_GFX906)
        declare_configuration_metadata("architecture", "GPU architecture",
                                       "AMD_GFX906");
#elif defined(FLARE_ARCH_AMD_GFX908)
        declare_configuration_metadata("architecture", "GPU architecture",
                                       "AMD_GFX908");
#elif defined(FLARE_ARCH_AMD_GFX90A)
        declare_configuration_metadata("architecture", "GPU architecture",
                                       "AMD_GFX90A");
#elif defined(FLARE_ARCH_AMD_GFX1030)
        declare_configuration_metadata("architecture", "GPU architecture",
                                       "AMD_GFX1030");
#elif defined(FLARE_ARCH_AMD_GFX1100)
        declare_configuration_metadata("architecture", "GPU architecture",
                                       "AMD_GFX1100");

#else
        declare_configuration_metadata("architecture", "GPU architecture", "none");
#endif

#ifdef FLARE_IMPL_32BIT
        declare_configuration_metadata("architecture", "platform", "32bit");
#else
        declare_configuration_metadata("architecture", "platform", "64bit");
#endif
    }

    void post_initialize_internal(const flare::InitializationSettings &settings) {
        flare::Tools::InitArguments tools_init_arguments;
        combine(tools_init_arguments, settings);
        initialize_profiling(tools_init_arguments);
        g_is_initialized = true;
        if (settings.has_print_configuration() &&
            settings.get_print_configuration()) {
            ::flare::print_configuration(std::cout);
        }
    }

    void initialize_internal(const flare::InitializationSettings &settings) {
        // The tool initialization is only called in post_initialize_internal.
        // Pausing tools here, so that if someone has set callbacks programmatically
        // these callbacks are not called inside the backend initialization, before
        // the tool initialization happened.
        flare::Tools::experimental::pause_tools();
        pre_initialize_internal(settings);
        initialize_backends(settings);
        flare::Tools::experimental::resume_tools();
        post_initialize_internal(settings);
    }

    void pre_finalize_internal() {
        typename decltype(finalize_hooks)::size_type numSuccessfulCalls = 0;
        while (!finalize_hooks.empty()) {
            auto f = finalize_hooks.top();
            try {
                f();
            } catch (...) {
                std::cerr << "flare::finalize: A finalize hook (set via "
                             "flare::push_finalize_hook) threw an exception that it did "
                             "not catch."
                             "  Per std::atexit rules, this results in std::terminate.  "
                             "This is "
                             "finalize hook number "
                          << numSuccessfulCalls
                          << " (1-based indexing) "
                             "out of "
                          << finalize_hooks.size()
                          << " to call.  Remember that "
                             "flare::finalize calls finalize hooks in reverse order "
                             "from how they "
                             "were pushed."
                          << std::endl;
                std::terminate();
            }
            finalize_hooks.pop();
            ++numSuccessfulCalls;
        }

        flare::Profiling::finalize();
    }

    void post_finalize_internal() {
        g_is_initialized = false;
        g_is_finalized = true;
        g_show_warnings = true;
        g_tune_internals = false;
    }

    void fence_internal(const std::string &name) {
        flare::detail::ExecSpaceManager::get_instance().static_fence(name);
    }

    void print_help_message() {
        auto const help_message = R"(
--------------------------------------------------------------------------------
-------------flare command line arguments--------------------------------------
--------------------------------------------------------------------------------
This program is using flare.  You can use the following command line flags to
control its behavior:

flare Core Options:
  --flare-help                  : print this message
  --flare-disable-warnings      : disable flare warning messages
  --flare-print-configuration   : print configuration
  --flare-tune-internals        : allow flare to autotune policies and declare
                                   tuning features through the tuning system. If
                                   left off, flare uses heuristics
  --flare-num-threads=INT       : specify total number of threads to use for
                                   parallel regions on the host.
  --flare-device-id=INT         : specify device id to be used by flare.
  --flare-map-device-id-by=(random|mpi_rank)
                                 : strategy to select device-id automatically from
                                   available devices.
                                   - random:   choose a random device from available.
                                   - mpi_rank: choose device-id based on a round robin
                                               assignment of local MPI ranks.
                                               Works with OpenMPI, MVAPICH, SLURM, and
                                               derived implementations.

flare Tools Options:
  --flare-tools-libs=STR        : Specify which of the tools to use. Must either
                                   be full path to library or name of library if the
                                   path is present in the runtime library search path
                                   (e.g. LD_LIBRARY_PATH)
  --flare-tools-help            : Query the (loaded) flare-tool for its command-line
                                   option support (which should then be passed via
                                   --flare-tools-args="...")
  --flare-tools-args=STR        : A single (quoted) string of options which will be
                                   whitespace delimited and passed to the loaded
                                   flare-tool as command-line arguments. E.g.
                                   `<EXE> --flare-tools-args="-c input.txt"` will
                                   pass `<EXE> -c input.txt` as argc/argv to tool

Except for --flare[-tools]-help, you can alternatively set the corresponding
environment variable of a flag (all letters in upper-case and underscores
instead of hyphens). For example, to disable warning messages, you can either
specify --flare-disable-warnings or set the FLARE_DISABLE_WARNINGS
environment variable to yes.
--------------------------------------------------------------------------------
)";
        std::cout << help_message << std::endl;
    }

}  // namespace

void flare::detail::parse_command_line_arguments(
        int &argc, char *argv[], InitializationSettings &settings) {
    Tools::InitArguments tools_init_arguments;
    combine(tools_init_arguments, settings);
    Tools::detail::parse_command_line_arguments(argc, argv, tools_init_arguments);
    combine(settings, tools_init_arguments);

    int num_threads;
    int device_id;
    int num_devices;  // deprecated
    int skip_device;  // deprecated
    std::string map_device_id_by;
    bool disable_warnings;
    bool print_configuration;
    bool tune_internals;

    auto get_flag = [](std::string s) -> std::string {
        return s.erase(s.find('='));
    };

    bool help_flag = false;

    int iarg = 0;
    while (iarg < argc) {
        bool remove_flag = false;

        if (check_arg(argv[iarg], "--flare-numa") ||
            check_arg(argv[iarg], "--numa")) {
            warn_deprecated_command_line_argument(get_flag(argv[iarg]));
            // remove flag if prefixed with '--flare-'
            remove_flag = std::string(argv[iarg]).find("--flare-") == 0;
        } else if (check_arg_int(argv[iarg], "--flare-num-threads", num_threads) ||
                   check_arg_int(argv[iarg], "--num-threads", num_threads) ||
                   check_arg_int(argv[iarg], "--flare-threads", num_threads) ||
                   check_arg_int(argv[iarg], "--threads", num_threads)) {
            if (get_flag(argv[iarg]) != "--flare-num-threads") {
                warn_deprecated_command_line_argument(get_flag(argv[iarg]),
                                                      "--flare-num-threads");
            }
            if (!is_valid_num_threads(num_threads)) {
                std::stringstream ss;
                ss << "Error: command line argument '" << argv[iarg] << "' is invalid."
                   << " The number of threads must be greater than or equal to one."
                   << " Raised by flare::initialize().\n";
                flare::abort(ss.str().c_str());
            }
            settings.set_num_threads(num_threads);
            remove_flag = std::string(argv[iarg]).find("--flare-") == 0;
        } else if (check_arg_int(argv[iarg], "--flare-device-id", device_id) ||
                   check_arg_int(argv[iarg], "--device-id", device_id) ||
                   check_arg_int(argv[iarg], "--flare-device", device_id) ||
                   check_arg_int(argv[iarg], "--device", device_id)) {
            if (get_flag(argv[iarg]) != "--flare-device-id") {
                warn_deprecated_command_line_argument(get_flag(argv[iarg]),
                                                      "--flare-device-id");
            }
            if (!is_valid_device_id(device_id)) {
                std::stringstream ss;
                ss << "Error: command line argument '" << argv[iarg] << "' is invalid."
                   << " The device id must be greater than or equal to zero."
                   << " Raised by flare::initialize().\n";
                flare::abort(ss.str().c_str());
            }
            settings.set_device_id(device_id);
            remove_flag = std::string(argv[iarg]).find("--flare-") == 0;
        } else if (check_arg(argv[iarg], "--flare-num-devices") ||
                   check_arg(argv[iarg], "--num-devices") ||
                   check_arg(argv[iarg], "--flare-ndevices") ||
                   check_arg(argv[iarg], "--ndevices")) {
            if (check_arg(argv[iarg], "--num-devices")) {
                warn_deprecated_command_line_argument("--num-devices",
                                                      "--flare-num-devices");
            }
            if (check_arg(argv[iarg], "--ndevices")) {
                warn_deprecated_command_line_argument("--ndevices",
                                                      "--flare-num-devices");
            }
            if (check_arg(argv[iarg], "--flare-ndevices")) {
                warn_deprecated_command_line_argument("--flare-ndevices",
                                                      "--flare-num-devices");
            }
            warn_deprecated_command_line_argument(
                    "--flare-num-devices", "--flare-map-device-id-by=mpi_rank");
            // Find the number of device (expecting --device=XX)
            if (!((strncmp(argv[iarg], "--flare-num-devices=", 20) == 0) ||
                  (strncmp(argv[iarg], "--num-devices=", 14) == 0) ||
                  (strncmp(argv[iarg], "--flare-ndevices=", 17) == 0) ||
                  (strncmp(argv[iarg], "--ndevices=", 11) == 0)))
                throw_runtime_exception(
                        "Error: expecting an '=INT[,INT]' after command line argument "
                        "'--flare-num-devices'."
                        " Raised by flare::initialize().");

            char *num1 = strchr(argv[iarg], '=') + 1;
            char *num2 = strpbrk(num1, ",");
            int num1_len = num2 == nullptr ? strlen(num1) : num2 - num1;
            char *num1_only = new char[num1_len + 1];
            strncpy(num1_only, num1, num1_len);
            num1_only[num1_len] = '\0';

            if (!is_unsigned_int(num1_only) || (strlen(num1_only) == 0)) {
                throw_runtime_exception(
                        "Error: expecting an integer number after command line argument "
                        "'--flare-num-devices'."
                        " Raised by flare::initialize().");
            }
            if (check_arg(argv[iarg], "--flare-num-devices") ||
                check_arg(argv[iarg], "--flare-ndevices")) {
                num_devices = std::stoi(num1_only);
                settings.set_num_devices(num_devices);
                settings.set_map_device_id_by("mpi_rank");
            }
            delete[] num1_only;

            if (num2 != nullptr) {
                if ((!is_unsigned_int(num2 + 1)) || (strlen(num2) == 1))
                    throw_runtime_exception(
                            "Error: expecting an integer number after command line argument "
                            "'--flare-num-devices=XX,'."
                            " Raised by flare::initialize().");

                if (check_arg(argv[iarg], "--flare-num-devices") ||
                    check_arg(argv[iarg], "--flare-ndevices")) {
                    skip_device = std::stoi(num2 + 1);
                    settings.set_skip_device(skip_device);
                }
            }
            remove_flag = std::string(argv[iarg]).find("--flare-") == 0;
        } else if (check_arg_bool(argv[iarg], "--flare-disable-warnings",
                                  disable_warnings)) {
            settings.set_disable_warnings(disable_warnings);
            remove_flag = true;
        } else if (check_arg_bool(argv[iarg], "--flare-print-configuration",
                                  print_configuration)) {
            settings.set_print_configuration(print_configuration);
            remove_flag = true;
        } else if (check_arg_bool(argv[iarg], "--flare-tune-internals",
                                  tune_internals)) {
            settings.set_tune_internals(tune_internals);
            remove_flag = true;
        } else if (check_arg(argv[iarg], "--flare-help") ||
                   check_arg(argv[iarg], "--help")) {
            help_flag = true;
            remove_flag = std::string(argv[iarg]).find("--flare-") == 0;
        } else if (check_arg_str(argv[iarg], "--flare-map-device-id-by",
                                 map_device_id_by)) {
            if (!is_valid_map_device_id_by(map_device_id_by)) {
                std::stringstream ss;
                ss << "Warning: command line argument '--flare-map-device-id-by="
                   << map_device_id_by << "' is not recognized."
                   << " Raised by flare::initialize().\n";
                flare::abort(ss.str().c_str());
            }
            settings.set_map_device_id_by(map_device_id_by);
            remove_flag = true;
        } else if (std::regex_match(argv[iarg],
                                    std::regex("-?-flare.*", std::regex::egrep))) {
            warn_not_recognized_command_line_argument(argv[iarg]);
        }

        if (remove_flag) {
            // Shift the remainder of the argv list by one.  Note that argv has
            // (argc + 1) arguments, the last one always being nullptr.  The following
            // loop moves the trailing nullptr element as well
            for (int k = iarg; k < argc; ++k) {
                argv[k] = argv[k + 1];
            }
            argc--;
        } else {
            iarg++;
        }
    }

    if (help_flag) {
        print_help_message();
    }

    if ((tools_init_arguments.args ==
         flare::Tools::InitArguments::unset_string_option) &&
        argc > 0) {
        settings.set_tools_args(argv[0]);
    }
}

void flare::detail::parse_environment_variables(
        InitializationSettings &settings) {
    Tools::InitArguments tools_init_arguments;
    combine(tools_init_arguments, settings);
    auto init_result =
            Tools::detail::parse_environment_variables(tools_init_arguments);
    if (init_result.result ==
        Tools::detail::InitializationStatus::environment_argument_mismatch) {
        detail::throw_runtime_exception(init_result.error_message);
    }
    combine(settings, tools_init_arguments);

    if (std::getenv("FLARE_NUMA")) {
        warn_deprecated_environment_variable("FLARE_NUMA");
    }
    int num_threads;
    if (check_env_int("FLARE_NUM_THREADS", num_threads)) {
        if (!is_valid_num_threads(num_threads)) {
            std::stringstream ss;
            ss << "Error: environment variable 'FLARE_NUM_THREADS=" << num_threads
               << "' is invalid."
               << " The number of threads must be greater than or equal to one."
               << " Raised by flare::initialize().\n";
            flare::abort(ss.str().c_str());
        }
        settings.set_num_threads(num_threads);
    }
    int device_id;
    if (check_env_int("FLARE_DEVICE_ID", device_id)) {
        if (!is_valid_device_id(device_id)) {
            std::stringstream ss;
            ss << "Error: environment variable 'FLARE_DEVICE_ID" << device_id
               << "' is invalid."
               << " The device id must be greater than or equal to zero."
               << " Raised by flare::initialize().\n";
            flare::abort(ss.str().c_str());
        }
        settings.set_device_id(device_id);
    }
    int num_devices;
    int rand_devices;
    bool has_num_devices = check_env_int("FLARE_NUM_DEVICES", num_devices);
    bool has_rand_devices = check_env_int("FLARE_RAND_DEVICES", rand_devices);
    if (has_rand_devices && has_num_devices) {
        detail::throw_runtime_exception(
                "Error: cannot specify both FLARE_NUM_DEVICES and "
                "FLARE_RAND_DEVICES."
                " Raised by flare::initialize().");
    }
    if (has_num_devices) {
        warn_deprecated_environment_variable("FLARE_NUM_DEVICES",
                                             "FLARE_MAP_DEVICE_ID_BY=mpi_rank");
        settings.set_map_device_id_by("mpi_rank");
        settings.set_num_devices(num_devices);
    }
    if (has_rand_devices) {
        warn_deprecated_environment_variable("FLARE_RAND_DEVICES",
                                             "FLARE_MAP_DEVICE_ID_BY=random");
        settings.set_map_device_id_by("random");
        settings.set_num_devices(rand_devices);
    }
    if (has_num_devices || has_rand_devices) {
        int skip_device;
        if (check_env_int("FLARE_SKIP_DEVICE", skip_device)) {
            settings.set_skip_device(skip_device);
        }
    }
    bool disable_warnings;
    if (check_env_bool("FLARE_DISABLE_WARNINGS", disable_warnings)) {
        settings.set_disable_warnings(disable_warnings);
    }
    bool print_configuration;
    if (check_env_bool("FLARE_PRINT_CONFIGURATION", print_configuration)) {
        settings.set_print_configuration(print_configuration);
    }
    bool tune_internals;
    if (check_env_bool("FLARE_TUNE_INTERNALS", tune_internals)) {
        settings.set_tune_internals(tune_internals);
    }
    char const *map_device_id_by = std::getenv("FLARE_MAP_DEVICE_ID_BY");
    if (map_device_id_by != nullptr) {
        if (std::getenv("FLARE_DEVICE_ID")) {
            std::cerr << "Warning: environment variable FLARE_MAP_DEVICE_ID_BY"
                      << "ignored since FLARE_DEVICE_ID is specified."
                      << " Raised by flare::initialize()." << std::endl;
        }
        if (!is_valid_map_device_id_by(map_device_id_by)) {
            std::stringstream ss;
            ss << "Warning: environment variable 'FLARE_MAP_DEVICE_ID_BY="
               << map_device_id_by << "' is not recognized."
               << " Raised by flare::initialize().\n";
            flare::abort(ss.str().c_str());
        }
        settings.set_map_device_id_by(map_device_id_by);
    }
}

//----------------------------------------------------------------------------
namespace {
    bool flare_initialize_was_called() {
        return flare::is_initialized() || flare::is_finalized();
    }

    bool flare_finalize_was_called() { return flare::is_finalized(); }
}  // namespace

void flare::initialize(int &argc, char *argv[]) {
    if (flare_initialize_was_called()) {
        flare::abort(
                "Error: flare::initialize() has already been called."
                " flare can be initialized at most once.\n");
    }
    InitializationSettings settings;
    detail::parse_environment_variables(settings);
    detail::parse_command_line_arguments(argc, argv, settings);
    initialize_internal(settings);
}

void flare::initialize(InitializationSettings const &settings) {
    if (flare_initialize_was_called()) {
        flare::abort(
                "Error: flare::initialize() has already been called."
                " flare can be initialized at most once.\n");
    }
    InitializationSettings tmp;
    detail::parse_environment_variables(tmp);
    combine(tmp, settings);
    initialize_internal(tmp);
}

void flare::detail::pre_initialize(const InitializationSettings &settings) {
    pre_initialize_internal(settings);
}

void flare::detail::post_initialize(const InitializationSettings &settings) {
    post_initialize_internal(settings);
}

void flare::detail::pre_finalize() { pre_finalize_internal(); }

void flare::detail::post_finalize() { post_finalize_internal(); }

void flare::push_finalize_hook(std::function<void()> f) {
    finalize_hooks.push(f);
}

void flare::finalize() {
    if (!flare_initialize_was_called()) {
        flare::abort(
                "Error: flare::finalize() may only be called after flare has been "
                "initialized.\n");
    }
    if (flare_finalize_was_called()) {
        flare::abort("Error: flare::finalize() has already been called.\n");
    }
    pre_finalize_internal();
    detail::ExecSpaceManager::get_instance().finalize_spaces();
    post_finalize_internal();
}

#ifdef FLARE_COMPILER_INTEL
void flare::fence() { fence("flare::fence: Unnamed Global Fence"); }
#endif

void flare::fence(const std::string &name) { fence_internal(name); }

namespace {
    void print_helper(std::ostream &os,
                      const std::map<std::string, std::string> &print_me) {
        for (const auto &kv: print_me) {
            os << "  " << kv.first << ": " << kv.second << '\n';
        }
    }
}  // namespace

void flare::print_configuration(std::ostream &os, bool verbose) {
    print_helper(os, metadata_map["version_info"]);

    os << "Compiler:\n";
    print_helper(os, metadata_map["compiler_version"]);

    os << "Architecture:\n";
    print_helper(os, metadata_map["architecture"]);

    os << "Atomics:\n";
    print_helper(os, metadata_map["atomics"]);

    os << "Vectorization:\n";
    print_helper(os, metadata_map["vectorization"]);

    os << "Memory:\n";
    print_helper(os, metadata_map["memory"]);

    os << "Options:\n";
    print_helper(os, metadata_map["options"]);

    detail::ExecSpaceManager::get_instance().print_configuration(os, verbose);
}

[[nodiscard]] bool flare::is_initialized() noexcept {
    return g_is_initialized;
}

[[nodiscard]] bool flare::is_finalized() noexcept { return g_is_finalized; }

bool flare::show_warnings() noexcept { return g_show_warnings; }

bool flare::tune_internals() noexcept { return g_tune_internals; }
