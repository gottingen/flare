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

#ifndef FLARE_CORE_PROFILE_INTERFACE_H_
#define FLARE_CORE_PROFILE_INTERFACE_H_

#include <cinttypes>
#include <cstddef>
#include <climits>

#include <cstdlib>

// NOTE: in this flare::Profiling block, do not define anything that shouldn't
// exist should Profiling be disabled

namespace flare {
namespace Tools {
namespace experimental {

constexpr const uint32_t NumReservedDeviceIDs = 1;

enum SpecialSynchronizationCases : int {
  GlobalDeviceSynchronization     = 1,
  DeepCopyResourceSynchronization = 2,
};

enum struct DeviceType {
  Serial,
  OpenMP,
  Cuda,
  Threads,
  Unknown
};

struct ExecutionSpaceIdentifier {
  DeviceType type;
  uint32_t device_id;
  uint32_t instance_id;
};

constexpr const uint32_t num_type_bits     = 8;
constexpr const uint32_t num_device_bits   = 7;
constexpr const uint32_t num_instance_bits = 17;
constexpr const uint32_t num_avail_bits    = sizeof(uint32_t) * CHAR_BIT;

inline DeviceType devicetype_from_uint32t(const uint32_t in) {
  switch (in) {
    case 0: return DeviceType::Serial;
    case 1: return DeviceType::OpenMP;
    case 2: return DeviceType::Cuda;
    case 3: return DeviceType::Threads;
    default: return DeviceType::Unknown;  // TODO: error out?
  }
}

inline ExecutionSpaceIdentifier identifier_from_devid(const uint32_t in) {
  constexpr const uint32_t shift = num_avail_bits - num_type_bits;

  return {devicetype_from_uint32t(in >> shift), /*First 8 bits*/
          (~((uint32_t(-1)) << num_device_bits)) &
              (in >> num_instance_bits),                  /*Next 7 bits */
          (~((uint32_t(-1)) << num_instance_bits)) & in}; /*Last 17 bits*/
}

template <typename ExecutionSpace>
struct DeviceTypeTraits;

template <typename ExecutionSpace>
constexpr uint32_t device_id_root() {
  constexpr auto device_id =
      static_cast<uint32_t>(DeviceTypeTraits<ExecutionSpace>::id);
  return (device_id << (num_instance_bits + num_device_bits));
}
template <typename ExecutionSpace>
inline uint32_t device_id(ExecutionSpace const& space) noexcept {
  return device_id_root<ExecutionSpace>() +
         (DeviceTypeTraits<ExecutionSpace>::device_id(space)
          << num_instance_bits) +
         space.impl_instance_id();
}
}  // namespace experimental
}  // namespace Tools
}  // end namespace flare

#if defined(FLARE_ENABLE_LIBDL)
// We check at configure time that libdl is available.
#include <dlfcn.h>
#endif

#include <flare/core/profile/device_info.h>
#include <flare/core/profile/c_interface.h>

namespace flare {
namespace Tools {

using SpaceHandle = flare_profiling_space_handle;

}  // namespace Tools

namespace Tools {

namespace experimental {
using EventSet = flare_profiling_event_set;
static_assert(sizeof(EventSet) / sizeof(flare_tools_function_pointer) == 275,
              "sizeof EventSet has changed, this is an error on the part of a "
              "flare developer");
static_assert(sizeof(flare_tools_tool_settings) / sizeof(bool) == 256,
              "sizeof EventSet has changed, this is an error on the part of a "
              "flare developer");
static_assert(sizeof(flare_tools_tool_programming_interface) /
                      sizeof(flare_tools_function_pointer) ==
                  32,
              "sizeof EventSet has changed, this is an error on the part of a "
              "flare developer");

using toolInvokedFenceFunction = flare_tools_tool_invoked_fence_function;
using provideToolProgrammingInterfaceFunction =
    flare_tools_provide_tool_programming_interface_function;
using requestToolSettingsFunction = flare_tools_request_tool_settings_function;
using ToolSettings                = flare_tools_tool_settings;
using ToolProgrammingInterface    = flare_tools_tool_programming_interface;
}  // namespace experimental
using initFunction           = flare_profiling_init_function;
using finalizeFunction       = flare_profiling_finalize_function;
using parseArgsFunction      = flare_profiling_parse_args_function;
using printHelpFunction      = flare_profiling_print_help_function;
using beginFunction          = flare_profiling_begin_function;
using endFunction            = flare_profiling_end_function;
using pushFunction           = flare_profiling_push_function;
using popFunction            = flare_profiling_pop_function;
using allocateDataFunction   = flare_profiling_allocate_data_function;
using deallocateDataFunction = flare_profiling_deallocate_data_function;
using createProfileSectionFunction =
    flare_profiling_create_profile_section_function;
using startProfileSectionFunction =
    flare_profiling_start_profile_section_function;
using stopProfileSectionFunction = flare_profiling_stop_profile_section_function;
using destroyProfileSectionFunction =
    flare_profiling_destroy_profile_section_function;
using profileEventFunction    = flare_profiling_profile_event_function;
using beginDeepCopyFunction   = flare_profiling_begin_deep_copy_function;
using endDeepCopyFunction     = flare_profiling_end_deep_copy_function;
using beginFenceFunction      = flare_profiling_begin_fence_function;
using endFenceFunction        = flare_profiling_end_fence_function;
using dualTensorSyncFunction    = flare_profiling_dual_tensor_sync_function;
using dualTensorModifyFunction  = flare_profiling_dual_tensor_modify_function;
using declareMetadataFunction = flare_profiling_declare_metadata_function;

}  // namespace Tools

}  // namespace flare

// Profiling

namespace flare {

namespace Profiling {

/** The Profiling namespace is being renamed to Tools.
 * This is reexposing the contents of what used to be the Profiling
 * Interface with their original names, to avoid breaking old code
 */

namespace experimental {

using flare::Tools::experimental::device_id;
using flare::Tools::experimental::DeviceType;
using flare::Tools::experimental::DeviceTypeTraits;

}  // namespace experimental

using flare::Tools::allocateDataFunction;
using flare::Tools::beginDeepCopyFunction;
using flare::Tools::beginFunction;
using flare::Tools::createProfileSectionFunction;
using flare::Tools::deallocateDataFunction;
using flare::Tools::destroyProfileSectionFunction;
using flare::Tools::endDeepCopyFunction;
using flare::Tools::endFunction;
using flare::Tools::finalizeFunction;
using flare::Tools::initFunction;
using flare::Tools::parseArgsFunction;
using flare::Tools::popFunction;
using flare::Tools::printHelpFunction;
using flare::Tools::profileEventFunction;
using flare::Tools::pushFunction;
using flare::Tools::SpaceHandle;
using flare::Tools::startProfileSectionFunction;
using flare::Tools::stopProfileSectionFunction;

}  // namespace Profiling
}  // namespace flare

// Tuning

namespace flare {
namespace Tools {
namespace experimental {
using ValueSet            = flare_tools_value_set;
using ValueRange          = flare_tools_value_range;
using StatisticalCategory = flare_tools_variable_info_statistical_category;
using ValueType           = flare_tools_variable_info_value_type;
using CandidateValueType  = flare_tools_variable_info_candidate_value_type;
using SetOrRange          = flare_tools_variable_info_set_or_range;
using VariableInfo        = flare_tools_variable_info;
using OptimizationGoal    = flare_tools_optimzation_goal;
using TuningString        = flare_tools_tuning_string;
using VariableValue       = flare_tools_variable_value;

using outputTypeDeclarationFunction =
    flare_tools_output_type_declaration_function;
using inputTypeDeclarationFunction = flare_tools_input_type_declaration_function;
using requestValueFunction         = flare_tools_request_value_function;
using contextBeginFunction         = flare_tools_context_begin_function;
using contextEndFunction           = flare_tools_context_end_function;
using optimizationGoalDeclarationFunction =
    flare_tools_optimization_goal_declaration_function;
}  // end namespace experimental
}  // end namespace Tools

}  // end namespace flare

#endif  // FLARE_CORE_PROFILE_INTERFACE_H_
