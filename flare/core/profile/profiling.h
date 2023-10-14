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

#ifndef FLARE_IMPL_FLARE_PROFILING_HPP
#define FLARE_IMPL_FLARE_PROFILING_HPP

#include <flare/core/profile/interface.h>
#include <memory>
#include <iosfwd>
#include <unordered_map>
#include <map>
#include <string>
#include <type_traits>
#include <mutex>

namespace flare {

// forward declaration
    bool show_warnings() noexcept;

    bool tune_internals() noexcept;

    namespace Tools {

        struct InitArguments {
            // NOTE DZP: PossiblyUnsetOption was introduced
            // before C++17, std::optional is a better choice
            // for this long-term
            static const std::string unset_string_option;
            enum PossiblyUnsetOption {
                unset, off, on
            };
            PossiblyUnsetOption help = unset;
            std::string lib = unset_string_option;
            std::string args = unset_string_option;
        };

        namespace detail {

            struct InitializationStatus {
                enum InitializationResult {
                    success,
                    failure,
                    help_request,
                    environment_argument_mismatch
                };
                InitializationResult result;
                std::string error_message;
            };

            InitializationStatus initialize_tools_subsystem(
                    const flare::Tools::InitArguments &args);

            void parse_command_line_arguments(int &narg, char *arg[],
                                              InitArguments &arguments);

            flare::Tools::detail::InitializationStatus parse_environment_variables(
                    InitArguments &arguments);

        }  // namespace detail

        bool profileLibraryLoaded();

        void beginParallelFor(const std::string &kernelPrefix, const uint32_t devID,
                              uint64_t *kernelID);

        void endParallelFor(const uint64_t kernelID);

        void beginParallelScan(const std::string &kernelPrefix, const uint32_t devID,
                               uint64_t *kernelID);

        void endParallelScan(const uint64_t kernelID);

        void beginParallelReduce(const std::string &kernelPrefix, const uint32_t devID,
                                 uint64_t *kernelID);

        void endParallelReduce(const uint64_t kernelID);

        void pushRegion(const std::string &kName);

        void popRegion();

        void createProfileSection(const std::string &sectionName, uint32_t *secID);

        void startSection(const uint32_t secID);

        void stopSection(const uint32_t secID);

        void destroyProfileSection(const uint32_t secID);

        void markEvent(const std::string &evName);

        void allocateData(const SpaceHandle space, const std::string label,
                          const void *ptr, const uint64_t size);

        void deallocateData(const SpaceHandle space, const std::string label,
                            const void *ptr, const uint64_t size);

        void beginDeepCopy(const SpaceHandle dst_space, const std::string dst_label,
                           const void *dst_ptr, const SpaceHandle src_space,
                           const std::string src_label, const void *src_ptr,
                           const uint64_t size);

        void endDeepCopy();

        void beginFence(const std::string name, const uint32_t deviceId,
                        uint64_t *handle);

        void endFence(const uint64_t handle);

        /**
         * syncDualTensor declares to the tool that a given DualTensor
         * has been synced.
         *
         * Arguments:
         *
         * label:     name of the Tensor within the DualTensor
         * ptr:       that Tensor's data ptr
         * to_device: true if the data is being synchronized to the device
         * 		false otherwise
         */
        void syncDualTensor(const std::string &label, const void *const ptr,
                          bool to_device);

        /**
         * modifyDualTensor declares to the tool that a given DualTensor
         * has been modified. Note: this means that somebody *called*
         * modify on the DualTensor, this doesn't get called any time
         * somebody touches the data
         *
         * Arguments:
         *
         * label:     name of the Tensor within the DualTensor
         * ptr:       that Tensor's data ptr
         * on_device: true if the data is being modified on the device
         * 		false otherwise
         */
        void modifyDualTensor(const std::string &label, const void *const ptr,
                            bool on_device);

        void declareMetadata(const std::string &key, const std::string &value);

        void initialize(
                const std::string & = {});  // should rename to impl_initialize ASAP
        void initialize(const flare::Tools::InitArguments &);

        void initialize(int argc, char *argv[]);

        void finalize();

        bool printHelp(const std::string &);

        void parseArgs(const std::string &);

        flare_profiling_space_handle make_space_handle(const char *space_name);

        namespace experimental {

            namespace detail {
                struct DirectFenceIDHandle {
                    uint32_t value;
                };

//
                template<typename Space>
                uint32_t idForInstance(const uintptr_t instance) {
                    static std::mutex instance_mutex;
                    const std::lock_guard<std::mutex> lock(instance_mutex);
                    /** Needed to be a ptr due to initialization order problems*/
                    using map_type = std::map<uintptr_t, uint32_t>;

                    static std::shared_ptr<map_type> map;
                    if (map.get() == nullptr) {
                        map = std::make_shared<map_type>(map_type());
                    }

                    static uint32_t value = 0;
                    constexpr const uint32_t offset =
                            flare::Tools::experimental::NumReservedDeviceIDs;

                    auto find = map->find(instance);
                    if (find == map->end()) {
                        auto ret = offset + value++;
                        (*map)[instance] = ret;
                        return ret;
                    }

                    return find->second;
                }

                template<typename Space, typename FencingFunctor>
                void profile_fence_event(const std::string &name, DirectFenceIDHandle devIDTag,
                                         const FencingFunctor &func) {
                    uint64_t handle = 0;
                    flare::Tools::beginFence(
                            name,
                            flare::Tools::experimental::device_id_root<Space>() + devIDTag.value,
                            &handle);
                    func();
                    flare::Tools::endFence(handle);
                }

                inline uint32_t int_for_synchronization_reason(
                        flare::Tools::experimental::SpecialSynchronizationCases reason) {
                    switch (reason) {
                        case GlobalDeviceSynchronization:
                            return 0;
                        case DeepCopyResourceSynchronization:
                            return 0x00ffffff;
                    }
                    return 0;
                }

                template<typename Space, typename FencingFunctor>
                void profile_fence_event(
                        const std::string &name,
                        flare::Tools::experimental::SpecialSynchronizationCases reason,
                        const FencingFunctor &func) {
                    uint64_t handle = 0;
                    flare::Tools::beginFence(
                            name, device_id_root<Space>() + int_for_synchronization_reason(reason),
                            &handle);  // TODO: correct ID
                    func();
                    flare::Tools::endFence(handle);
                }
            }  // namespace detail
            void set_init_callback(initFunction callback);

            void set_finalize_callback(finalizeFunction callback);

            void set_parse_args_callback(parseArgsFunction callback);

            void set_print_help_callback(printHelpFunction callback);

            void set_begin_parallel_for_callback(beginFunction callback);

            void set_end_parallel_for_callback(endFunction callback);

            void set_begin_parallel_reduce_callback(beginFunction callback);

            void set_end_parallel_reduce_callback(endFunction callback);

            void set_begin_parallel_scan_callback(beginFunction callback);

            void set_end_parallel_scan_callback(endFunction callback);

            void set_push_region_callback(pushFunction callback);

            void set_pop_region_callback(popFunction callback);

            void set_allocate_data_callback(allocateDataFunction callback);

            void set_deallocate_data_callback(deallocateDataFunction callback);

            void set_create_profile_section_callback(createProfileSectionFunction callback);

            void set_start_profile_section_callback(startProfileSectionFunction callback);

            void set_stop_profile_section_callback(stopProfileSectionFunction callback);

            void set_destroy_profile_section_callback(
                    destroyProfileSectionFunction callback);

            void set_profile_event_callback(profileEventFunction callback);

            void set_begin_deep_copy_callback(beginDeepCopyFunction callback);

            void set_end_deep_copy_callback(endDeepCopyFunction callback);

            void set_begin_fence_callback(beginFenceFunction callback);

            void set_end_fence_callback(endFenceFunction callback);

            void set_dual_tensor_sync_callback(dualTensorSyncFunction callback);

            void set_dual_tensor_modify_callback(dualTensorModifyFunction callback);

            void set_declare_metadata_callback(declareMetadataFunction callback);

            void set_request_tool_settings_callback(requestToolSettingsFunction callback);

            void set_provide_tool_programming_interface_callback(
                    provideToolProgrammingInterfaceFunction callback);

            void set_declare_output_type_callback(outputTypeDeclarationFunction callback);

            void set_declare_input_type_callback(inputTypeDeclarationFunction callback);

            void set_request_output_values_callback(requestValueFunction callback);

            void set_declare_optimization_goal_callback(
                    optimizationGoalDeclarationFunction callback);

            void set_end_context_callback(contextEndFunction callback);

            void set_begin_context_callback(contextBeginFunction callback);

            void pause_tools();

            void resume_tools();

            EventSet get_callbacks();

            void set_callbacks(EventSet new_events);
        }  // namespace experimental

        namespace experimental {
            // forward declarations
            size_t get_new_context_id();

            size_t get_current_context_id();
        }  // namespace experimental

    }  // namespace Tools
    namespace Profiling {

        bool profileLibraryLoaded();

        void beginParallelFor(const std::string &kernelPrefix, const uint32_t devID,
                              uint64_t *kernelID);

        void beginParallelReduce(const std::string &kernelPrefix, const uint32_t devID,
                                 uint64_t *kernelID);

        void beginParallelScan(const std::string &kernelPrefix, const uint32_t devID,
                               uint64_t *kernelID);

        void endParallelFor(const uint64_t kernelID);

        void endParallelReduce(const uint64_t kernelID);

        void endParallelScan(const uint64_t kernelID);

        void pushRegion(const std::string &kName);

        void popRegion();

        void createProfileSection(const std::string &sectionName, uint32_t *secID);

        void destroyProfileSection(const uint32_t secID);

        void startSection(const uint32_t secID);

        void stopSection(const uint32_t secID);

        void markEvent(const std::string &eventName);

        void allocateData(const SpaceHandle handle, const std::string name,
                          const void *data, const uint64_t size);

        void deallocateData(const SpaceHandle space, const std::string label,
                            const void *ptr, const uint64_t size);

        void beginDeepCopy(const SpaceHandle dst_space, const std::string dst_label,
                           const void *dst_ptr, const SpaceHandle src_space,
                           const std::string src_label, const void *src_ptr,
                           const uint64_t size);

        void endDeepCopy();

        void finalize();

        void initialize(const std::string & = {});

        SpaceHandle make_space_handle(const char *space_name);

        namespace experimental {
            using flare::Tools::experimental::set_allocate_data_callback;
            using flare::Tools::experimental::set_begin_deep_copy_callback;
            using flare::Tools::experimental::set_begin_parallel_for_callback;
            using flare::Tools::experimental::set_begin_parallel_reduce_callback;
            using flare::Tools::experimental::set_begin_parallel_scan_callback;
            using flare::Tools::experimental::set_create_profile_section_callback;
            using flare::Tools::experimental::set_deallocate_data_callback;
            using flare::Tools::experimental::set_destroy_profile_section_callback;
            using flare::Tools::experimental::set_end_deep_copy_callback;
            using flare::Tools::experimental::set_end_parallel_for_callback;
            using flare::Tools::experimental::set_end_parallel_reduce_callback;
            using flare::Tools::experimental::set_end_parallel_scan_callback;
            using flare::Tools::experimental::set_finalize_callback;
            using flare::Tools::experimental::set_init_callback;
            using flare::Tools::experimental::set_parse_args_callback;
            using flare::Tools::experimental::set_pop_region_callback;
            using flare::Tools::experimental::set_print_help_callback;
            using flare::Tools::experimental::set_profile_event_callback;
            using flare::Tools::experimental::set_push_region_callback;
            using flare::Tools::experimental::set_start_profile_section_callback;
            using flare::Tools::experimental::set_stop_profile_section_callback;

            using flare::Tools::experimental::EventSet;

            using flare::Tools::experimental::pause_tools;
            using flare::Tools::experimental::resume_tools;

            using flare::Tools::experimental::get_callbacks;
            using flare::Tools::experimental::set_callbacks;

        }  // namespace experimental
    }  // namespace Profiling

    namespace Tools {
        namespace experimental {

            VariableValue make_variable_value(size_t id, int64_t val);

            VariableValue make_variable_value(size_t id, double val);

            VariableValue make_variable_value(size_t id, const std::string &val);

            SetOrRange make_candidate_set(size_t size, std::string *data);

            SetOrRange make_candidate_set(size_t size, int64_t *data);

            SetOrRange make_candidate_set(size_t size, double *data);

            SetOrRange make_candidate_range(double lower, double upper, double step,
                                            bool openLower, bool openUpper);

            SetOrRange make_candidate_range(int64_t lower, int64_t upper, int64_t step,
                                            bool openLower, bool openUpper);

            void declare_optimization_goal(const size_t context,
                                           const OptimizationGoal &goal);

            size_t declare_output_type(const std::string &typeName, VariableInfo info);

            size_t declare_input_type(const std::string &typeName, VariableInfo info);

            void set_input_values(size_t contextId, size_t count, VariableValue *values);

            void end_context(size_t contextId);

            void begin_context(size_t contextId);

            void request_output_values(size_t contextId, size_t count,
                                       VariableValue *values);

            bool have_tuning_tool();

            size_t get_new_context_id();

            size_t get_current_context_id();

            size_t get_new_variable_id();
        }  // namespace experimental
    }  // namespace Tools

}  // namespace flare

#endif
