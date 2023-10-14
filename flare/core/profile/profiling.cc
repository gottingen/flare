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

#ifndef FLARE_TOOLS_INDEPENDENT_BUILD

#include <flare/core/defines.h>
#include <flare/core/policy/tuners.h>

#endif

#include <flare/core/profile/profiling.h>
#include <flare/core/profile/interface.h>
#include <flare/core/common/command_line_parsing.h>

#if defined(FLARE_ENABLE_LIBDL) || defined(FLARE_TOOLS_INDEPENDENT_BUILD)
#include <dlfcn.h>
#define FLARE_TOOLS_ENABLE_LIBDL
#endif

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <memory>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <sstream>
#include <iostream>

namespace {
    void warn_cmd_line_arg_ignored_when_flare_tools_disabled(char const *arg) {
#ifndef FLARE_TOOLS_ENABLE_LIBDL
        if (flare::show_warnings()) {
            std::cerr << "Warning: command line argument '" << arg
                      << "' ignored because flare-tools is disabled."
                      << " Raised by flare::initialize()." << std::endl;
        }
#else
        (void)arg;
#endif
    }

    void warn_env_var_ignored_when_flare_tools_disabled(char const *env_var,
                                                        char const *val) {
#ifndef FLARE_TOOLS_ENABLE_LIBDL
        if (flare::show_warnings()) {
            std::cerr << "Warning: environment variable '" << env_var << "=" << val
                      << "' ignored because flare-tools is disabled."
                      << " Raised by flare::initialize()." << std::endl;
        }
#else
        (void)env_var;
        (void)val;
#endif
    }
}  // namespace

namespace flare {

    namespace Tools {

        const std::string InitArguments::unset_string_option = {
                "flare_tools_impl_unset_option"};

        InitArguments tool_arguments;

        namespace detail {
            void parse_command_line_arguments(int &argc, char *argv[],
                                              InitArguments &arguments) {
                int iarg = 0;
                using flare::detail::check_arg;
                using flare::detail::check_arg_str;

                auto &libs = arguments.lib;
                auto &args = arguments.args;
                auto &help = arguments.help;
                while (iarg < argc) {
                    bool remove_flag = false;
                    if (check_arg_str(argv[iarg], "--flare-tools-libs", libs) ||
                        check_arg_str(argv[iarg], "--flare-tools-library", libs)) {
                        if (check_arg(argv[iarg], "--flare-tools-library")) {
                            using flare::detail::warn_deprecated_command_line_argument;
                            warn_deprecated_command_line_argument("--flare-tools-library",
                                                                  "--flare-tools-libs");
                        }
                        warn_cmd_line_arg_ignored_when_flare_tools_disabled(argv[iarg]);
                        remove_flag = true;
                    } else if (check_arg_str(argv[iarg], "--flare-tools-args", args)) {
                        warn_cmd_line_arg_ignored_when_flare_tools_disabled(argv[iarg]);
                        remove_flag = true;
                        // strip any leading and/or trailing quotes if they were retained in the
                        // string because this will very likely cause parsing issues for tools.
                        // If the quotes are retained (via bypassing the shell):
                        //    <EXE> --flare-tools-args="-c my example"
                        // would be tokenized as:
                        //    "<EXE>" "\"-c" "my" "example\""
                        // instead of:
                        //    "<EXE>" "-c" "my" "example"
                        if (!args.empty()) {
                            if (args.front() == '"') args = args.substr(1);
                            if (args.back() == '"') args = args.substr(0, args.length() - 1);
                        }
                        // add the name of the executable to the beginning
                        if (argc > 0) args = std::string(argv[0]) + " " + args;
                    } else if (check_arg(argv[iarg], "--flare-tools-help")) {
                        help = InitArguments::PossiblyUnsetOption::on;
                        warn_cmd_line_arg_ignored_when_flare_tools_disabled(argv[iarg]);
                        remove_flag = true;
                    } else if (std::regex_match(argv[iarg], std::regex("-?-flare-tool.*",
                                                                       std::regex::egrep))) {
                        std::cerr << "Warning: command line argument '" << argv[iarg]
                                  << "' is not recognized."
                                  << " Raised by flare::initialize()." << std::endl;
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
                    if ((args == flare::Tools::InitArguments::unset_string_option) && argc > 0)
                        args = argv[0];
                }
            }

            flare::Tools::detail::InitializationStatus parse_environment_variables(
                    InitArguments &arguments) {
                auto &libs = arguments.lib;
                auto &args = arguments.args;
                auto env_profile_library = std::getenv("FLARE_PROFILE_LIBRARY");
                if (env_profile_library != nullptr) {
                    using flare::detail::warn_deprecated_environment_variable;
                    warn_deprecated_environment_variable("FLARE_PROFILE_LIBRARY",
                                                         "FLARE_TOOLS_LIBS");
                    warn_env_var_ignored_when_flare_tools_disabled("FLARE_PROFILE_LIBRARY",
                                                                   env_profile_library);
                    libs = env_profile_library;
                }
                auto env_tools_libs = std::getenv("FLARE_TOOLS_LIBS");
                if (env_tools_libs != nullptr) {
                    warn_env_var_ignored_when_flare_tools_disabled("FLARE_TOOLS_LIBS",
                                                                   env_tools_libs);
                    if (env_profile_library != nullptr && libs != env_tools_libs) {
                        std::stringstream ss;
                        ss << "Error: environment variables 'FLARE_PROFILE_LIBRARY="
                           << env_profile_library << "' and 'FLARE_TOOLS_LIBS=" << env_tools_libs
                           << "' are both set and do not match."
                           << " Raised by flare::initialize().\n";
                        flare::abort(ss.str().c_str());
                    }
                    libs = env_tools_libs;
                }
                auto env_tools_args = std::getenv("FLARE_TOOLS_ARGS");
                if (env_tools_args != nullptr) {
                    warn_env_var_ignored_when_flare_tools_disabled("FLARE_TOOLS_ARGS",
                                                                   env_tools_args);
                    args = env_tools_args;
                }
                return {
                        flare::Tools::detail::InitializationStatus::InitializationResult::success,
                        ""};
            }

            InitializationStatus initialize_tools_subsystem(
                    const flare::Tools::InitArguments &args) {
#ifdef FLARE_TOOLS_ENABLE_LIBDL
                flare::Profiling::initialize(args.lib);
                auto final_args =
                    (args.args != flare::Tools::InitArguments::unset_string_option)
                        ? args.args
                        : "";

                if (args.help) {
                  if (!flare::Tools::printHelp(final_args)) {
                    std::cerr << "Tool has not provided a help message" << std::endl;
                  }
                  return {InitializationStatus::InitializationResult::help_request, ""};
                }
                flare::Tools::parseArgs(final_args);
#else
                (void) args;
#endif
                return {InitializationStatus::InitializationResult::success, ""};
            }

        }  // namespace detail
        void initialize(const InitArguments &arguments) {
            detail::initialize_tools_subsystem(arguments);
        }

        void initialize(int argc, char *argv[]) {
            InitArguments arguments;
            detail::parse_environment_variables(arguments);
            detail::parse_command_line_arguments(argc, argv, arguments);
            initialize(arguments);
        }

        namespace experimental {

            namespace detail {
                void tool_invoked_fence(const uint32_t /* devID */) {
                    /**
                     * Currently the function ignores the device ID,
                     * Eventually we want to support fencing only
                     * a given stream/resource
                     */
#ifndef FLARE_TOOLS_INDEPENDENT_BUILD
                    flare::fence(
                            "flare::Tools::experimental::detail::tool_invoked_fence: Tool Requested "
                            "Fence");
#endif
                }
            }  // namespace detail

#ifdef FLARE_ENABLE_TUNING
            static size_t kernel_name_context_variable_id;
            static size_t kernel_type_context_variable_id;
            static std::unordered_map<size_t, std::unordered_set<size_t>>
                features_per_context;
            static std::unordered_set<size_t> active_features;
            static std::unordered_map<size_t, VariableValue> feature_values;
            static std::unordered_map<size_t, VariableInfo> variable_metadata;
#endif
            static EventSet current_callbacks;
            static EventSet backup_callbacks;
            static EventSet no_profiling;
            static ToolSettings tool_requirements;

            bool eventSetsEqual(const EventSet &l, const EventSet &r) {
                return l.init == r.init && l.finalize == r.finalize &&
                       l.parse_args == r.parse_args && l.print_help == r.print_help &&
                       l.begin_parallel_for == r.begin_parallel_for &&
                       l.end_parallel_for == r.end_parallel_for &&
                       l.begin_parallel_reduce == r.begin_parallel_reduce &&
                       l.end_parallel_reduce == r.end_parallel_reduce &&
                       l.begin_parallel_scan == r.begin_parallel_scan &&
                       l.end_parallel_scan == r.end_parallel_scan &&
                       l.push_region == r.push_region && l.pop_region == r.pop_region &&
                       l.allocate_data == r.allocate_data &&
                       l.deallocate_data == r.deallocate_data &&
                       l.create_profile_section == r.create_profile_section &&
                       l.start_profile_section == r.start_profile_section &&
                       l.stop_profile_section == r.stop_profile_section &&
                       l.destroy_profile_section == r.destroy_profile_section &&
                       l.profile_event == r.profile_event &&
                       l.begin_deep_copy == r.begin_deep_copy &&
                       l.end_deep_copy == r.end_deep_copy && l.begin_fence == r.begin_fence &&
                       l.end_fence == r.end_fence && l.sync_dual_tensor == r.sync_dual_tensor &&
                       l.modify_dual_tensor == r.modify_dual_tensor &&
                       l.declare_metadata == r.declare_metadata &&
                       l.request_tool_settings == r.request_tool_settings &&
                       l.provide_tool_programming_interface ==
                       r.provide_tool_programming_interface &&
                       l.declare_input_type == r.declare_input_type &&
                       l.declare_output_type == r.declare_output_type &&
                       l.end_tuning_context == r.end_tuning_context &&
                       l.begin_tuning_context == r.begin_tuning_context &&
                       l.request_output_values == r.request_output_values &&
                       l.declare_optimization_goal == r.declare_optimization_goal;
            }

            enum class MayRequireGlobalFencing : bool {
                No, Yes
            };

            template<typename Callback, typename... Args>
            inline void invoke_flare_profile_callback(
                    MayRequireGlobalFencing may_require_global_fencing,
                    const Callback &callback, Args &&... args) {
                if (callback != nullptr) {
                    // two clause if statement
                    // may_require_global_fencing: "if this callback ever needs a fence", AND
                    // if the tool requires global fencing (default true, but tools can
                    // overwrite)
                    if (may_require_global_fencing == MayRequireGlobalFencing::Yes &&
                        (tool_requirements.requires_global_fencing)) {
#ifndef FLARE_TOOLS_INDEPENDENT_BUILD
                        flare::fence(
                                "flare::Tools::invoke_flare_profile_callback: flare Profile Tool Fence");
#endif
                    }
                    (*callback)(std::forward<Args>(args)...);
                }
            }
        }  // namespace experimental
        bool profileLibraryLoaded() {
            return !experimental::eventSetsEqual(experimental::current_callbacks,
                                                 experimental::no_profiling);
        }

        void beginParallelFor(const std::string &kernelPrefix, const uint32_t devID,
                              uint64_t *kernelID) {
            experimental::invoke_flare_profile_callback(
                    experimental::MayRequireGlobalFencing::Yes,
                    experimental::current_callbacks.begin_parallel_for, kernelPrefix.c_str(),
                    devID, kernelID);
#ifdef FLARE_ENABLE_TUNING
            if (flare::tune_internals()) {
              auto context_id = experimental::get_new_context_id();
              experimental::begin_context(context_id);
              experimental::VariableValue contextValues[] = {
                  experimental::make_variable_value(
                      experimental::kernel_name_context_variable_id, kernelPrefix),
                  experimental::make_variable_value(
                      experimental::kernel_type_context_variable_id, "parallel_for")};
              experimental::set_input_values(context_id, 2, contextValues);
            }
#endif
        }

        void endParallelFor(const uint64_t kernelID) {
            experimental::invoke_flare_profile_callback(
                    experimental::MayRequireGlobalFencing::Yes,
                    experimental::current_callbacks.end_parallel_for, kernelID);
#ifdef FLARE_ENABLE_TUNING
            if (flare::tune_internals()) {
              experimental::end_context(experimental::get_current_context_id());
            }
#endif
        }

        void beginParallelScan(const std::string &kernelPrefix, const uint32_t devID,
                               uint64_t *kernelID) {
            experimental::invoke_flare_profile_callback(
                    experimental::MayRequireGlobalFencing::Yes,
                    experimental::current_callbacks.begin_parallel_scan, kernelPrefix.c_str(),
                    devID, kernelID);
#ifdef FLARE_ENABLE_TUNING
            if (flare::tune_internals()) {
              auto context_id = experimental::get_new_context_id();
              experimental::begin_context(context_id);
              experimental::VariableValue contextValues[] = {
                  experimental::make_variable_value(
                      experimental::kernel_name_context_variable_id, kernelPrefix),
                  experimental::make_variable_value(
                      experimental::kernel_type_context_variable_id, "parallel_for")};
              experimental::set_input_values(context_id, 2, contextValues);
            }
#endif
        }

        void endParallelScan(const uint64_t kernelID) {
            experimental::invoke_flare_profile_callback(
                    experimental::MayRequireGlobalFencing::Yes,
                    experimental::current_callbacks.end_parallel_scan, kernelID);
#ifdef FLARE_ENABLE_TUNING
            if (flare::tune_internals()) {
              experimental::end_context(experimental::get_current_context_id());
            }
#endif
        }

        void beginParallelReduce(const std::string &kernelPrefix, const uint32_t devID,
                                 uint64_t *kernelID) {
            experimental::invoke_flare_profile_callback(
                    experimental::MayRequireGlobalFencing::Yes,
                    experimental::current_callbacks.begin_parallel_reduce,
                    kernelPrefix.c_str(), devID, kernelID);
#ifdef FLARE_ENABLE_TUNING
            if (flare::tune_internals()) {
              auto context_id = experimental::get_new_context_id();
              experimental::begin_context(context_id);
              experimental::VariableValue contextValues[] = {
                  experimental::make_variable_value(
                      experimental::kernel_name_context_variable_id, kernelPrefix),
                  experimental::make_variable_value(
                      experimental::kernel_type_context_variable_id, "parallel_for")};
              experimental::set_input_values(context_id, 2, contextValues);
            }
#endif
        }

        void endParallelReduce(const uint64_t kernelID) {
            experimental::invoke_flare_profile_callback(
                    experimental::MayRequireGlobalFencing::Yes,
                    experimental::current_callbacks.end_parallel_reduce, kernelID);
#ifdef FLARE_ENABLE_TUNING
            if (flare::tune_internals()) {
              experimental::end_context(experimental::get_current_context_id());
            }
#endif
        }

        void pushRegion(const std::string &kName) {
            experimental::invoke_flare_profile_callback(
                    experimental::MayRequireGlobalFencing::Yes,
                    experimental::current_callbacks.push_region, kName.c_str());
        }

        void popRegion() {
            experimental::invoke_flare_profile_callback(
                    experimental::MayRequireGlobalFencing::Yes,
                    experimental::current_callbacks.pop_region);
        }

        void allocateData(const SpaceHandle space, const std::string label,
                          const void *ptr, const uint64_t size) {
            experimental::invoke_flare_profile_callback(
                    experimental::MayRequireGlobalFencing::No,
                    experimental::current_callbacks.allocate_data, space, label.c_str(), ptr,
                    size);
        }

        void deallocateData(const SpaceHandle space, const std::string label,
                            const void *ptr, const uint64_t size) {
            experimental::invoke_flare_profile_callback(
                    experimental::MayRequireGlobalFencing::No,
                    experimental::current_callbacks.deallocate_data, space, label.c_str(),
                    ptr, size);
        }

        void beginDeepCopy(const SpaceHandle dst_space, const std::string dst_label,
                           const void *dst_ptr, const SpaceHandle src_space,
                           const std::string src_label, const void *src_ptr,
                           const uint64_t size) {
            experimental::invoke_flare_profile_callback(
                    experimental::MayRequireGlobalFencing::No,
                    experimental::current_callbacks.begin_deep_copy, dst_space,
                    dst_label.c_str(), dst_ptr, src_space, src_label.c_str(), src_ptr, size);
#ifdef FLARE_ENABLE_TUNING
            if (experimental::current_callbacks.begin_deep_copy != nullptr) {
              if (flare::tune_internals()) {
                auto context_id = experimental::get_new_context_id();
                experimental::begin_context(context_id);
                experimental::VariableValue contextValues[] = {
                    experimental::make_variable_value(
                        experimental::kernel_name_context_variable_id,
                        "deep_copy_kernel"),
                    experimental::make_variable_value(
                        experimental::kernel_type_context_variable_id, "deep_copy")};
                experimental::set_input_values(context_id, 2, contextValues);
              }
            }
#endif
        }

        void endDeepCopy() {
            experimental::invoke_flare_profile_callback(
                    experimental::MayRequireGlobalFencing::No,
                    experimental::current_callbacks.end_deep_copy);
#ifdef FLARE_ENABLE_TUNING
            if (experimental::current_callbacks.end_deep_copy != nullptr) {
              if (flare::tune_internals()) {
                experimental::end_context(experimental::get_current_context_id());
              }
            }
#endif
        }

        void beginFence(const std::string name, const uint32_t deviceId,
                        uint64_t *handle) {
            experimental::invoke_flare_profile_callback(
                    experimental::MayRequireGlobalFencing::No,
                    experimental::current_callbacks.begin_fence, name.c_str(), deviceId,
                    handle);
        }

        void endFence(const uint64_t handle) {
            experimental::invoke_flare_profile_callback(
                    experimental::MayRequireGlobalFencing::No,
                    experimental::current_callbacks.end_fence, handle);
        }

        void createProfileSection(const std::string &sectionName, uint32_t *secID) {
            experimental::invoke_flare_profile_callback(
                    experimental::MayRequireGlobalFencing::No,
                    experimental::current_callbacks.create_profile_section,
                    sectionName.c_str(), secID);
        }

        void startSection(const uint32_t secID) {
            experimental::invoke_flare_profile_callback(
                    experimental::MayRequireGlobalFencing::No,
                    experimental::current_callbacks.start_profile_section, secID);
        }

        void stopSection(const uint32_t secID) {
            experimental::invoke_flare_profile_callback(
                    experimental::MayRequireGlobalFencing::No,
                    experimental::current_callbacks.stop_profile_section, secID);
        }

        void destroyProfileSection(const uint32_t secID) {
            experimental::invoke_flare_profile_callback(
                    experimental::MayRequireGlobalFencing::No,
                    experimental::current_callbacks.destroy_profile_section, secID);
        }

        void markEvent(const std::string &eventName) {
            experimental::invoke_flare_profile_callback(
                    experimental::MayRequireGlobalFencing::No,
                    experimental::current_callbacks.profile_event, eventName.c_str());
        }

        bool printHelp(const std::string &args) {
            if (experimental::current_callbacks.print_help == nullptr) {
                return false;
            }
            std::string arg0 = args.substr(0, args.find_first_of(' '));
            const char *carg0 = arg0.c_str();
            experimental::invoke_flare_profile_callback(
                    experimental::MayRequireGlobalFencing::No,
                    experimental::current_callbacks.print_help, const_cast<char *>(carg0));
            return true;
        }

        void parseArgs(int _argc, char **_argv) {
            if (experimental::current_callbacks.parse_args != nullptr && _argc > 0) {
                experimental::invoke_flare_profile_callback(
                        experimental::MayRequireGlobalFencing::No,
                        experimental::current_callbacks.parse_args, _argc, _argv);
            }
        }

        void parseArgs(const std::string &args) {
            if (experimental::current_callbacks.parse_args == nullptr) {
                return;
            }
            using strvec_t = std::vector<std::string>;
            auto tokenize = [](const std::string &line, const std::string &delimiters) {
                strvec_t _result{};
                std::size_t _bidx = 0;  // position that is the beginning of the new string
                std::size_t _didx = 0;  // position of the delimiter in the string
                while (_bidx < line.length() && _didx < line.length()) {
                    // find the first character (starting at _didx) that is not a delimiter
                    _bidx = line.find_first_not_of(delimiters, _didx);
                    // if no more non-delimiter chars, done
                    if (_bidx == std::string::npos) break;
                    // starting at the position of the new string, find the next delimiter
                    _didx = line.find_first_of(delimiters, _bidx);
                    // starting at the position of the new string, get the characters
                    // between this position and the next delimiter
                    std::string _tmp = line.substr(_bidx, _didx - _bidx);
                    // don't add empty strings
                    if (!_tmp.empty()) _result.emplace_back(_tmp);
                }
                return _result;
            };
            auto vargs = tokenize(args, " \t");
            if (vargs.size() == 0) return;
            auto _argc = static_cast<int>(vargs.size());
            char **_argv = new char *[_argc + 1];
            _argv[vargs.size()] = nullptr;
            for (int i = 0; i < _argc; ++i) {
                auto &_str = vargs.at(i);
                _argv[i] = new char[_str.length() + 1];
                std::memcpy(_argv[i], _str.c_str(), _str.length() * sizeof(char));
                _argv[i][_str.length()] = '\0';
            }
            parseArgs(_argc, _argv);
            for (int i = 0; i < _argc; ++i) {
                delete[] _argv[i];
            }
            delete[] _argv;
        }

        SpaceHandle make_space_handle(const char *space_name) {
            SpaceHandle handle;
            strncpy(handle.name, space_name, 63);
            return handle;
        }

        template<typename Callback>
        void lookup_function(void *dlopen_handle, const std::string &basename,
                             Callback &callback) {
#ifdef FLARE_TOOLS_ENABLE_LIBDL
            // dlsym returns a pointer to an object, while we want to assign to
            // pointer to function A direct cast will give warnings hence, we have to
            // workaround the issue by casting pointer to pointers.
            void* p  = dlsym(dlopen_handle, basename.c_str());
            callback = *reinterpret_cast<Callback*>(&p);
#else
            (void) dlopen_handle;
            (void) basename;
            (void) callback;
#endif
        }

        void initialize(const std::string &profileLibrary) {
            // Make sure initialize calls happens only once
            static int is_initialized = 0;
            if (is_initialized) return;
            is_initialized = 1;

            auto invoke_init_callbacks = []() {
                experimental::invoke_flare_profile_callback(
                        experimental::MayRequireGlobalFencing::No,
                        experimental::current_callbacks.init, 0,
                        (uint64_t) FLARE_PROFILING_INTERFACE_VERSION, (uint32_t) 0, nullptr);

                experimental::tool_requirements.requires_global_fencing = true;

                experimental::invoke_flare_profile_callback(
                        experimental::MayRequireGlobalFencing::No,
                        experimental::current_callbacks.request_tool_settings, 1,
                        &experimental::tool_requirements);

                experimental::ToolProgrammingInterface actions;
                actions.fence = &experimental::detail::tool_invoked_fence;

                experimental::invoke_flare_profile_callback(
                        experimental::MayRequireGlobalFencing::No,
                        experimental::current_callbacks.provide_tool_programming_interface, 1,
                        actions);
            };

#ifdef FLARE_TOOLS_ENABLE_LIBDL
            void* firstProfileLibrary = nullptr;

            if ((profileLibrary.empty()) ||
                (profileLibrary == InitArguments::unset_string_option)) {
              invoke_init_callbacks();
              return;
            }

            if (auto end_first_library = profileLibrary.find(';');
                end_first_library != 0) {
              auto profileLibraryName = profileLibrary.substr(0, end_first_library);
              firstProfileLibrary =
                  dlopen(profileLibraryName.c_str(), RTLD_NOW | RTLD_GLOBAL);

              if (firstProfileLibrary == nullptr) {
                std::cerr << "Error: Unable to load flare profile library: "
                          << profileLibraryName << std::endl;
                std::cerr << "dlopen(" << profileLibraryName
                          << ", RTLD_NOW | RTLD_GLOBAL) failed with " << dlerror()
                          << '\n';
              } else {
                lookup_function(firstProfileLibrary, "flare_profile_begin_parallel_scan",
                                experimental::current_callbacks.begin_parallel_scan);
                lookup_function(firstProfileLibrary, "flare_profile_begin_parallel_for",
                                experimental::current_callbacks.begin_parallel_for);
                lookup_function(firstProfileLibrary, "flare_profile_begin_parallel_reduce",
                                experimental::current_callbacks.begin_parallel_reduce);
                lookup_function(firstProfileLibrary, "flare_profile_end_parallel_scan",
                                experimental::current_callbacks.end_parallel_scan);
                lookup_function(firstProfileLibrary, "flare_profile_end_parallel_for",
                                experimental::current_callbacks.end_parallel_for);
                lookup_function(firstProfileLibrary, "flare_profile_end_parallel_reduce",
                                experimental::current_callbacks.end_parallel_reduce);

                lookup_function(firstProfileLibrary, "flare_profile_init_library",
                                experimental::current_callbacks.init);
                lookup_function(firstProfileLibrary, "flare_profile_finalize_library",
                                experimental::current_callbacks.finalize);

                lookup_function(firstProfileLibrary, "flare_profile_push_profile_region",
                                experimental::current_callbacks.push_region);
                lookup_function(firstProfileLibrary, "flare_profile_pop_profile_region",
                                experimental::current_callbacks.pop_region);
                lookup_function(firstProfileLibrary, "flare_profile_allocate_data",
                                experimental::current_callbacks.allocate_data);
                lookup_function(firstProfileLibrary, "flare_profile_deallocate_data",
                                experimental::current_callbacks.deallocate_data);

                lookup_function(firstProfileLibrary, "flare_profile_begin_deep_copy",
                                experimental::current_callbacks.begin_deep_copy);
                lookup_function(firstProfileLibrary, "flare_profile_end_deep_copy",
                                experimental::current_callbacks.end_deep_copy);
                lookup_function(firstProfileLibrary, "flare_profile_begin_fence",
                                experimental::current_callbacks.begin_fence);
                lookup_function(firstProfileLibrary, "flare_profile_end_fence",
                                experimental::current_callbacks.end_fence);
                lookup_function(firstProfileLibrary, "flare_profile_dual_tensor_sync",
                                experimental::current_callbacks.sync_dual_tensor);
                lookup_function(firstProfileLibrary, "flare_profile_dual_tensor_modify",
                                experimental::current_callbacks.modify_dual_tensor);

                lookup_function(firstProfileLibrary, "flare_profile_declare_metadata",
                                experimental::current_callbacks.declare_metadata);
                lookup_function(firstProfileLibrary, "flare_profile_create_profile_section",
                                experimental::current_callbacks.create_profile_section);
                lookup_function(firstProfileLibrary, "flare_profile_start_profile_section",
                                experimental::current_callbacks.start_profile_section);
                lookup_function(firstProfileLibrary, "flare_profile_stop_profile_section",
                                experimental::current_callbacks.stop_profile_section);
                lookup_function(firstProfileLibrary, "flare_profile_destroy_profile_section",
                                experimental::current_callbacks.destroy_profile_section);

                lookup_function(firstProfileLibrary, "flare_profile_profile_event",
                                experimental::current_callbacks.profile_event);
#ifdef FLARE_ENABLE_TUNING
                lookup_function(firstProfileLibrary, "flare_profile_declare_output_type",
                                experimental::current_callbacks.declare_output_type);

                lookup_function(firstProfileLibrary, "flare_profile_declare_input_type",
                                experimental::current_callbacks.declare_input_type);
                lookup_function(firstProfileLibrary, "flare_profile_request_values",
                                experimental::current_callbacks.request_output_values);
                lookup_function(firstProfileLibrary, "flare_profile_end_context",
                                experimental::current_callbacks.end_tuning_context);
                lookup_function(firstProfileLibrary, "flare_profile_begin_context",
                                experimental::current_callbacks.begin_tuning_context);
                lookup_function(
                    firstProfileLibrary, "flare_profile_declare_optimization_goal",
                    experimental::current_callbacks.declare_optimization_goal);
#endif  // FLARE_ENABLE_TUNING

                lookup_function(firstProfileLibrary, "flare_profile_print_help",
                                experimental::current_callbacks.print_help);
                lookup_function(firstProfileLibrary, "flare_profile_parse_args",
                                experimental::current_callbacks.parse_args);
                lookup_function(
                    firstProfileLibrary, "flare_profile_provide_tool_programming_interface",
                    experimental::current_callbacks.provide_tool_programming_interface);
                lookup_function(firstProfileLibrary, "flare_profile_request_tool_settings",
                                experimental::current_callbacks.request_tool_settings);
              }
            }
#else
            (void) profileLibrary;
#endif  // FLARE_ENABLE_LIBDL

            invoke_init_callbacks();

#ifdef FLARE_ENABLE_TUNING
            experimental::VariableInfo kernel_name;
            kernel_name.type = experimental::ValueType::flare_value_string;
            kernel_name.category =
                experimental::StatisticalCategory::flare_value_categorical;
            kernel_name.valueQuantity =
                experimental::CandidateValueType::flare_value_unbounded;

            std::array<std::string, 4> candidate_values = {
                "parallel_for",
                "parallel_reduce",
                "parallel_scan",
                "parallel_copy",
            };

            experimental::SetOrRange kernel_type_variable_candidates =
                experimental::make_candidate_set(4, candidate_values.data());

            experimental::kernel_name_context_variable_id =
                experimental::declare_input_type("flare.kernel_name", kernel_name);

            experimental::VariableInfo kernel_type;
            kernel_type.type = experimental::ValueType::flare_value_string;
            kernel_type.category =
                experimental::StatisticalCategory::flare_value_categorical;
            kernel_type.valueQuantity =
                experimental::CandidateValueType::flare_value_set;
            kernel_type.candidates = kernel_type_variable_candidates;
            experimental::kernel_type_context_variable_id =
                experimental::declare_input_type("flare.kernel_type", kernel_type);

#endif

            experimental::no_profiling.init = nullptr;
            experimental::no_profiling.finalize = nullptr;

            experimental::no_profiling.begin_parallel_for = nullptr;
            experimental::no_profiling.begin_parallel_scan = nullptr;
            experimental::no_profiling.begin_parallel_reduce = nullptr;
            experimental::no_profiling.end_parallel_scan = nullptr;
            experimental::no_profiling.end_parallel_for = nullptr;
            experimental::no_profiling.end_parallel_reduce = nullptr;

            experimental::no_profiling.push_region = nullptr;
            experimental::no_profiling.pop_region = nullptr;
            experimental::no_profiling.allocate_data = nullptr;
            experimental::no_profiling.deallocate_data = nullptr;

            experimental::no_profiling.begin_deep_copy = nullptr;
            experimental::no_profiling.end_deep_copy = nullptr;

            experimental::no_profiling.create_profile_section = nullptr;
            experimental::no_profiling.start_profile_section = nullptr;
            experimental::no_profiling.stop_profile_section = nullptr;
            experimental::no_profiling.destroy_profile_section = nullptr;

            experimental::no_profiling.profile_event = nullptr;

            experimental::no_profiling.declare_input_type = nullptr;
            experimental::no_profiling.declare_output_type = nullptr;
            experimental::no_profiling.request_output_values = nullptr;
            experimental::no_profiling.end_tuning_context = nullptr;
        }

        void finalize() {
            // Make sure finalize calls happens only once
            static int is_finalized = 0;
            if (is_finalized) return;
            is_finalized = 1;

            if (experimental::current_callbacks.finalize != nullptr) {
                experimental::invoke_flare_profile_callback(
                        experimental::MayRequireGlobalFencing::No,
                        experimental::current_callbacks.finalize);

                experimental::pause_tools();
            }
#ifdef FLARE_ENABLE_TUNING
            // clean up string candidate set
            for (auto& metadata_pair : experimental::variable_metadata) {
              auto metadata = metadata_pair.second;
              if ((metadata.type == experimental::ValueType::flare_value_string) &&
                  (metadata.valueQuantity ==
                   experimental::CandidateValueType::flare_value_set)) {
                auto candidate_set = metadata.candidates.set;
                delete[] candidate_set.values.string_value;
              }
            }
#endif
        }

        void syncDualTensor(const std::string &label, const void *const ptr,
                          bool to_device) {
            experimental::invoke_flare_profile_callback(
                    experimental::MayRequireGlobalFencing::No,
                    experimental::current_callbacks.sync_dual_tensor, label.c_str(), ptr,
                    to_device);
        }

        void modifyDualTensor(const std::string &label, const void *const ptr,
                            bool on_device) {
            experimental::invoke_flare_profile_callback(
                    experimental::MayRequireGlobalFencing::No,
                    experimental::current_callbacks.modify_dual_tensor, label.c_str(), ptr,
                    on_device);
        }

        void declareMetadata(const std::string &key, const std::string &value) {
            experimental::invoke_flare_profile_callback(
                    experimental::MayRequireGlobalFencing::No,
                    experimental::current_callbacks.declare_metadata, key.c_str(),
                    value.c_str());
        }

    }  // namespace Tools

    namespace Tools {
        namespace experimental {
            void set_init_callback(initFunction callback) {
                current_callbacks.init = callback;
            }

            void set_finalize_callback(finalizeFunction callback) {
                current_callbacks.finalize = callback;
            }

            void set_parse_args_callback(parseArgsFunction callback) {
                current_callbacks.parse_args = callback;
            }

            void set_print_help_callback(printHelpFunction callback) {
                current_callbacks.print_help = callback;
            }

            void set_begin_parallel_for_callback(beginFunction callback) {
                current_callbacks.begin_parallel_for = callback;
            }

            void set_end_parallel_for_callback(endFunction callback) {
                current_callbacks.end_parallel_for = callback;
            }

            void set_begin_parallel_reduce_callback(beginFunction callback) {
                current_callbacks.begin_parallel_reduce = callback;
            }

            void set_end_parallel_reduce_callback(endFunction callback) {
                current_callbacks.end_parallel_reduce = callback;
            }

            void set_begin_parallel_scan_callback(beginFunction callback) {
                current_callbacks.begin_parallel_scan = callback;
            }

            void set_end_parallel_scan_callback(endFunction callback) {
                current_callbacks.end_parallel_scan = callback;
            }

            void set_push_region_callback(pushFunction callback) {
                current_callbacks.push_region = callback;
            }

            void set_pop_region_callback(popFunction callback) {
                current_callbacks.pop_region = callback;
            }

            void set_allocate_data_callback(allocateDataFunction callback) {
                current_callbacks.allocate_data = callback;
            }

            void set_deallocate_data_callback(deallocateDataFunction callback) {
                current_callbacks.deallocate_data = callback;
            }

            void set_create_profile_section_callback(
                    createProfileSectionFunction callback) {
                current_callbacks.create_profile_section = callback;
            }

            void set_start_profile_section_callback(startProfileSectionFunction callback) {
                current_callbacks.start_profile_section = callback;
            }

            void set_stop_profile_section_callback(stopProfileSectionFunction callback) {
                current_callbacks.stop_profile_section = callback;
            }

            void set_destroy_profile_section_callback(
                    destroyProfileSectionFunction callback) {
                current_callbacks.destroy_profile_section = callback;
            }

            void set_profile_event_callback(profileEventFunction callback) {
                current_callbacks.profile_event = callback;
            }

            void set_begin_deep_copy_callback(beginDeepCopyFunction callback) {
                current_callbacks.begin_deep_copy = callback;
            }

            void set_end_deep_copy_callback(endDeepCopyFunction callback) {
                current_callbacks.end_deep_copy = callback;
            }

            void set_begin_fence_callback(beginFenceFunction callback) {
                current_callbacks.begin_fence = callback;
            }

            void set_end_fence_callback(endFenceFunction callback) {
                current_callbacks.end_fence = callback;
            }

            void set_dual_tensor_sync_callback(dualTensorSyncFunction callback) {
                current_callbacks.sync_dual_tensor = callback;
            }

            void set_dual_tensor_modify_callback(dualTensorModifyFunction callback) {
                current_callbacks.modify_dual_tensor = callback;
            }

            void set_declare_metadata_callback(declareMetadataFunction callback) {
                current_callbacks.declare_metadata = callback;
            }

            void set_request_tool_settings_callback(requestToolSettingsFunction callback) {
                current_callbacks.request_tool_settings = callback;
            }

            void set_provide_tool_programming_interface_callback(
                    provideToolProgrammingInterfaceFunction callback) {
                current_callbacks.provide_tool_programming_interface = callback;
            }

            void set_declare_output_type_callback(
                    experimental::outputTypeDeclarationFunction callback) {
                current_callbacks.declare_output_type = callback;
            }

            void set_declare_input_type_callback(
                    experimental::inputTypeDeclarationFunction callback) {
                current_callbacks.declare_input_type = callback;
            }

            void set_request_output_values_callback(
                    experimental::requestValueFunction callback) {
                current_callbacks.request_output_values = callback;
            }

            void set_end_context_callback(experimental::contextEndFunction callback) {
                current_callbacks.end_tuning_context = callback;
            }

            void set_begin_context_callback(experimental::contextBeginFunction callback) {
                current_callbacks.begin_tuning_context = callback;
            }

            void set_declare_optimization_goal_callback(
                    experimental::optimizationGoalDeclarationFunction callback) {
                current_callbacks.declare_optimization_goal = callback;
            }

            void pause_tools() {
                backup_callbacks = current_callbacks;
                current_callbacks = no_profiling;
            }

            void resume_tools() { current_callbacks = backup_callbacks; }

            flare::Tools::experimental::EventSet get_callbacks() {
                return current_callbacks;
            }

            void set_callbacks(flare::Tools::experimental::EventSet new_events) {
                current_callbacks = new_events;
            }
        }  // namespace experimental
    }  // namespace Tools

    namespace Profiling {
        bool profileLibraryLoaded() { return flare::Tools::profileLibraryLoaded(); }

        void beginParallelFor(const std::string &kernelPrefix, const uint32_t devID,
                              uint64_t *kernelID) {
            flare::Tools::beginParallelFor(kernelPrefix, devID, kernelID);
        }

        void beginParallelReduce(const std::string &kernelPrefix, const uint32_t devID,
                                 uint64_t *kernelID) {
            flare::Tools::beginParallelReduce(kernelPrefix, devID, kernelID);
        }

        void beginParallelScan(const std::string &kernelPrefix, const uint32_t devID,
                               uint64_t *kernelID) {
            flare::Tools::beginParallelScan(kernelPrefix, devID, kernelID);
        }

        void endParallelFor(const uint64_t kernelID) {
            flare::Tools::endParallelFor(kernelID);
        }

        void endParallelReduce(const uint64_t kernelID) {
            flare::Tools::endParallelReduce(kernelID);
        }

        void endParallelScan(const uint64_t kernelID) {
            flare::Tools::endParallelScan(kernelID);
        }

        void pushRegion(const std::string &kName) { flare::Tools::pushRegion(kName); }

        void popRegion() { flare::Tools::popRegion(); }

        void createProfileSection(const std::string &sectionName, uint32_t *secID) {
            flare::Tools::createProfileSection(sectionName, secID);
        }

        void destroyProfileSection(const uint32_t secID) {
            flare::Tools::destroyProfileSection(secID);
        }

        void startSection(const uint32_t secID) { flare::Tools::startSection(secID); }

        void stopSection(const uint32_t secID) { flare::Tools::stopSection(secID); }

        void markEvent(const std::string &eventName) {
            flare::Tools::markEvent(eventName);
        }

        void allocateData(const SpaceHandle handle, const std::string name,
                          const void *data, const uint64_t size) {
            flare::Tools::allocateData(handle, name, data, size);
        }

        void deallocateData(const SpaceHandle space, const std::string label,
                            const void *ptr, const uint64_t size) {
            flare::Tools::deallocateData(space, label, ptr, size);
        }

        void beginDeepCopy(const SpaceHandle dst_space, const std::string dst_label,
                           const void *dst_ptr, const SpaceHandle src_space,
                           const std::string src_label, const void *src_ptr,
                           const uint64_t size) {
            flare::Tools::beginDeepCopy(dst_space, dst_label, dst_ptr, src_space,
                                        src_label, src_ptr, size);
        }

        void endDeepCopy() { flare::Tools::endDeepCopy(); }

        void finalize() { flare::Tools::finalize(); }

        void initialize(const std::string &profileLibrary) {
            flare::Tools::initialize(profileLibrary);
        }

        bool printHelp(const std::string &args) {
            return flare::Tools::printHelp(args);
        }

        void parseArgs(const std::string &args) { flare::Tools::parseArgs(args); }

        void parseArgs(int _argc, char **_argv) {
            flare::Tools::parseArgs(_argc, _argv);
        }

        SpaceHandle make_space_handle(const char *space_name) {
            return flare::Tools::make_space_handle(space_name);
        }
    }  // namespace Profiling

// Tuning

    namespace Tools {
        namespace experimental {
            static size_t &get_context_counter() {
                static size_t x;
                return x;
            }

            static size_t &get_variable_counter() {
                static size_t x;
                return ++x;
            }

            size_t get_new_context_id() { return ++get_context_counter(); }

            size_t get_current_context_id() { return get_context_counter(); }

            void decrement_current_context_id() { --get_context_counter(); }

            size_t get_new_variable_id() { return get_variable_counter(); }

            size_t declare_output_type(const std::string &variableName, VariableInfo info) {
                size_t variableId = get_new_variable_id();
#ifdef FLARE_ENABLE_TUNING
                experimental::invoke_flare_profile_callback(
                    experimental::MayRequireGlobalFencing::No,
                    experimental::current_callbacks.declare_output_type, variableName.c_str(),
                    variableId, &info);
                variable_metadata[variableId] = info;
#else
                (void) variableName;
                (void) info;
#endif
                return variableId;
            }

            size_t declare_input_type(const std::string &variableName, VariableInfo info) {
                size_t variableId = get_new_variable_id();
#ifdef FLARE_ENABLE_TUNING
                experimental::invoke_flare_profile_callback(
                    experimental::MayRequireGlobalFencing::No,
                    experimental::current_callbacks.declare_input_type, variableName.c_str(),
                    variableId, &info);
                variable_metadata[variableId] = info;
#else
                (void) variableName;
                (void) info;
#endif
                return variableId;
            }

            void set_input_values(size_t contextId, size_t count, VariableValue *values) {
#ifdef FLARE_ENABLE_TUNING
                if (features_per_context.find(contextId) == features_per_context.end()) {
                  features_per_context[contextId] = std::unordered_set<size_t>();
                }
                for (size_t x = 0; x < count; ++x) {
                  values[x].metadata = &variable_metadata[values[x].type_id];
                  features_per_context[contextId].insert(values[x].type_id);
                  active_features.insert(values[x].type_id);
                  feature_values[values[x].type_id] = values[x];
                }
#else
                (void) contextId;
                (void) count;
                (void) values;
#endif
            }

#include <iostream>

            void request_output_values(size_t contextId, size_t count,
                                       VariableValue *values) {
#ifdef FLARE_ENABLE_TUNING
                std::vector<size_t> context_ids;
                std::vector<VariableValue> context_values;
                for (auto id : active_features) {
                  context_values.push_back(feature_values[id]);
                }
                if (experimental::current_callbacks.request_output_values != nullptr) {
                  for (size_t x = 0; x < count; ++x) {
                    values[x].metadata = &variable_metadata[values[x].type_id];
                  }
                  experimental::invoke_flare_profile_callback(
                      experimental::MayRequireGlobalFencing::No,
                      experimental::current_callbacks.request_output_values, contextId,
                      context_values.size(), context_values.data(), count, values);
                }
#else
                (void) contextId;
                (void) count;
                (void) values;
#endif
            }

#ifdef FLARE_ENABLE_TUNING
            static std::unordered_map<size_t, size_t> optimization_goals;
#endif

            void begin_context(size_t contextId) {
                experimental::invoke_flare_profile_callback(
                        experimental::MayRequireGlobalFencing::No,
                        experimental::current_callbacks.begin_tuning_context, contextId);
            }

            void end_context(size_t contextId) {
#ifdef FLARE_ENABLE_TUNING
                for (auto id : features_per_context[contextId]) {
                  active_features.erase(id);
                }
                experimental::invoke_flare_profile_callback(
                    experimental::MayRequireGlobalFencing::No,
                    experimental::current_callbacks.end_tuning_context, contextId,
                    feature_values[optimization_goals[contextId]]);
                optimization_goals.erase(contextId);
                decrement_current_context_id();
#else
                (void) contextId;
#endif
            }

            bool have_tuning_tool() {
#ifdef FLARE_ENABLE_TUNING
                return (experimental::current_callbacks.request_output_values != nullptr);
#else
                return false;
#endif
            }

            VariableValue make_variable_value(size_t id, int64_t val) {
                VariableValue variable_value;
                variable_value.type_id = id;
                variable_value.value.int_value = val;
                return variable_value;
            }

            VariableValue make_variable_value(size_t id, double val) {
                VariableValue variable_value;
                variable_value.type_id = id;
                variable_value.value.double_value = val;
                return variable_value;
            }

            VariableValue make_variable_value(size_t id, const std::string &val) {
                VariableValue variable_value;
                variable_value.type_id = id;
                strncpy(variable_value.value.string_value, val.c_str(),
                        FLARE_TOOLS_TUNING_STRING_LENGTH - 1);
                return variable_value;
            }

            SetOrRange make_candidate_set(size_t size, std::string *data) {
                SetOrRange value_set;
                value_set.set.values.string_value = new TuningString[size];
                for (size_t x = 0; x < size; ++x) {
                    strncpy(value_set.set.values.string_value[x], data[x].c_str(),
                            FLARE_TOOLS_TUNING_STRING_LENGTH - 1);
                }
                value_set.set.size = size;
                return value_set;
            }

            SetOrRange make_candidate_set(size_t size, int64_t *data) {
                SetOrRange value_set;
                value_set.set.size = size;
                value_set.set.values.int_value = data;
                return value_set;
            }

            SetOrRange make_candidate_set(size_t size, double *data) {
                SetOrRange value_set;
                value_set.set.size = size;
                value_set.set.values.double_value = data;
                return value_set;
            }

            SetOrRange make_candidate_range(double lower, double upper, double step,
                                            bool openLower = false,
                                            bool openUpper = false) {
                SetOrRange value_range;
                value_range.range.lower.double_value = lower;
                value_range.range.upper.double_value = upper;
                value_range.range.step.double_value = step;
                value_range.range.openLower = openLower;
                value_range.range.openUpper = openUpper;
                return value_range;
            }

            SetOrRange make_candidate_range(int64_t lower, int64_t upper, int64_t step,
                                            bool openLower = false,
                                            bool openUpper = false) {
                SetOrRange value_range;
                value_range.range.lower.int_value = lower;
                value_range.range.upper.int_value = upper;
                value_range.range.step.int_value = step;
                value_range.range.openLower = openLower;
                value_range.range.openUpper = openUpper;
                return value_range;
            }

            size_t get_new_context_id();

            size_t get_current_context_id();

            void decrement_current_context_id();

            size_t get_new_variable_id();

            void declare_optimization_goal(const size_t context,
                                           const OptimizationGoal &goal) {
#ifdef FLARE_ENABLE_TUNING
                experimental::invoke_flare_profile_callback(
                    experimental::MayRequireGlobalFencing::No,
                    experimental::current_callbacks.declare_optimization_goal, context, goal);
                optimization_goals[context] = goal.type_id;
#else
                (void) context;
                (void) goal;
#endif
            }
        }  // end namespace experimental
    }  // end namespace Tools

}  // end namespace flare
