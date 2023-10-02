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


#ifndef FLARE_CORE_PROFILE_C_INTERFACE_H_
#define FLARE_CORE_PROFILE_C_INTERFACE_H_

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
#else
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#endif

#define FLARE_PROFILING_INTERFACE_VERSION 20211015

// Profiling

struct flare_profiling_device_info {
  size_t deviceID;
};

struct flare_profiling_space_handle {
  char name[64];
};

// NOLINTNEXTLINE(modernize-use-using): C compatibility
typedef void (*flare_profiling_init_function)(
    const int, const uint64_t, const uint32_t,
    struct flare_profiling_device_info*);
// NOLINTNEXTLINE(modernize-use-using): C compatibility
typedef void (*flare_profiling_finalize_function)();
// NOLINTNEXTLINE(modernize-use-using): C compatibility
typedef void (*flare_profiling_parse_args_function)(int, char**);
// NOLINTNEXTLINE(modernize-use-using): C compatibility
typedef void (*flare_profiling_print_help_function)(char*);
// NOLINTNEXTLINE(modernize-use-using): C compatibility
typedef void (*flare_profiling_begin_function)(const char*, const uint32_t,
                                               uint64_t*);
// NOLINTNEXTLINE(modernize-use-using): C compatibility
typedef void (*flare_profiling_end_function)(uint64_t);

// NOLINTNEXTLINE(modernize-use-using): C compatibility
typedef void (*flare_profiling_push_function)(const char*);
// NOLINTNEXTLINE(modernize-use-using): C compatibility
typedef void (*flare_profiling_pop_function)();

// NOLINTNEXTLINE(modernize-use-using): C compatibility
typedef void (*flare_profiling_allocate_data_function)(
    const struct flare_profiling_space_handle, const char*, const void*,
    const uint64_t);
// NOLINTNEXTLINE(modernize-use-using): C compatibility
typedef void (*flare_profiling_deallocate_data_function)(
    const struct flare_profiling_space_handle, const char*, const void*,
    const uint64_t);

// NOLINTNEXTLINE(modernize-use-using): C compatibility
typedef void (*flare_profiling_create_profile_section_function)(const char*,
                                                              uint32_t*);
// NOLINTNEXTLINE(modernize-use-using): C compatibility
typedef void (*flare_profiling_start_profile_section_function)(const uint32_t);
// NOLINTNEXTLINE(modernize-use-using): C compatibility
typedef void (*flare_profiling_stop_profile_section_function)(const uint32_t);
// NOLINTNEXTLINE(modernize-use-using): C compatibility
typedef void (*flare_profiling_destroy_profile_section_function)(const uint32_t);

// NOLINTNEXTLINE(modernize-use-using): C compatibility
typedef void (*flare_profiling_profile_event_function)(const char*);

// NOLINTNEXTLINE(modernize-use-using): C compatibility
typedef void (*flare_profiling_begin_deep_copy_function)(
    struct flare_profiling_space_handle, const char*, const void*,
    struct flare_profiling_space_handle, const char*, const void*, uint64_t);
// NOLINTNEXTLINE(modernize-use-using): C compatibility
typedef void (*flare_profiling_end_deep_copy_function)();
typedef void (*flare_profiling_begin_fence_function)(const char*, const uint32_t,
                                                    uint64_t*);
typedef void (*flare_profiling_end_fence_function)(uint64_t);

// NOLINTNEXTLINE(modernize-use-using): C compatibility
typedef void (*flare_profiling_dual_view_sync_function)(const char*,
                                                      const void* const, bool);
// NOLINTNEXTLINE(modernize-use-using): C compatibility
typedef void (*flare_profiling_dual_view_modify_function)(const char*,
                                                        const void* const,
                                                        bool);

// NOLINTNEXTLINE(modernize-use-using): C compatibility
typedef void (*flare_profiling_declare_metadata_function)(const char*,
                                                         const char*);

// NOLINTNEXTLINE(modernize-use-using): C compatibility
typedef void (*flare_tools_tool_invoked_fence_function)(const uint32_t);

// NOLINTNEXTLINE(modernize-use-using): C compatibility
typedef void (*flare_tools_function_pointer)();
struct flare_tools_tool_programming_interface {
  flare_tools_tool_invoked_fence_function fence;
  // allow addition of more actions
  flare_tools_function_pointer padding[31];
};

struct flare_tools_tool_settings {
  bool requires_global_fencing;
  bool padding[255];
};

// NOLINTNEXTLINE(modernize-use-using): C compatibility
typedef void (*flare_tools_provide_tool_programming_interface_function)(
    const uint32_t, struct flare_tools_tool_programming_interface);
// NOLINTNEXTLINE(modernize-use-using): C compatibility
typedef void (*flare_tools_request_tool_settings_function)(
    const uint32_t, struct flare_tools_tool_settings*);

// Tuning

#define FLARE_TOOLS_TUNING_STRING_LENGTH 64
typedef char flare_tools_tuning_string[FLARE_TOOLS_TUNING_STRING_LENGTH];
union flare_tools_variable_value_union {
  int64_t int_value;
  double double_value;
  flare_tools_tuning_string string_value;
};

union flare_tools_variable_value_union_set {
  int64_t* int_value;
  double* double_value;
  flare_tools_tuning_string* string_value;
};

struct flare_tools_value_set {
  size_t size;
  union flare_tools_variable_value_union_set values;
};

enum flare_tools_optimization_type {
  flare_tools_minimize,
  flare_tools_maximize
};

struct flare_tools_optimzation_goal {
  size_t type_id;
  enum flare_tools_optimization_type goal;
};

struct flare_tools_value_range {
  union flare_tools_variable_value_union lower;
  union flare_tools_variable_value_union upper;
  union flare_tools_variable_value_union step;
  bool openLower;
  bool openUpper;
};

enum flare_tools_variable_info_value_type {
  flare_value_double,
  flare_value_int64,
  flare_value_string,
};

enum flare_tools_variable_info_statistical_category {
  flare_value_categorical,  // unordered distinct objects
  flare_value_ordinal,      // ordered distinct objects
  flare_value_interval,  // ordered distinct objects for which distance matters
  flare_value_ratio  // ordered distinct objects for which distance matters,
                      // division matters, and the concept of zero exists
};

enum flare_tools_variable_info_candidate_value_type {
  flare_value_set,       // I am one of [2,3,4,5]
  flare_value_range,     // I am somewhere in [2,12)
  flare_value_unbounded  // I am [text/int/float], but we don't know at
                          // declaration time what values are appropriate. Only
                          // valid for Context Variables
};

union flare_tools_variable_info_set_or_range {
  struct flare_tools_value_set set;
  struct flare_tools_value_range range;
};

struct flare_tools_variable_info {
  enum flare_tools_variable_info_value_type type;
  enum flare_tools_variable_info_statistical_category category;
  enum flare_tools_variable_info_candidate_value_type valueQuantity;
  union flare_tools_variable_info_set_or_range candidates;
  void* toolProvidedInfo;
};

struct flare_tools_variable_value {
  size_t type_id;
  union flare_tools_variable_value_union value;
  struct flare_tools_variable_info* metadata;
};

typedef void (*flare_tools_output_type_declaration_function)(
    const char*, const size_t, struct flare_tools_variable_info* info);
typedef void (*flare_tools_input_type_declaration_function)(
    const char*, const size_t, struct flare_tools_variable_info* info);

typedef void (*flare_tools_request_value_function)(
    const size_t, const size_t, const struct flare_tools_variable_value*,
    const size_t count, struct flare_tools_variable_value*);
typedef void (*flare_tools_context_begin_function)(const size_t);
typedef void (*flare_tools_context_end_function)(
    const size_t, struct flare_tools_variable_value);
typedef void (*flare_tools_optimization_goal_declaration_function)(
    const size_t, const struct flare_tools_optimzation_goal goal);

struct flare_profiling_event_set {
  flare_profiling_init_function init;
  flare_profiling_finalize_function finalize;
  flare_profiling_parse_args_function parse_args;
  flare_profiling_print_help_function print_help;
  flare_profiling_begin_function begin_parallel_for;
  flare_profiling_end_function end_parallel_for;
  flare_profiling_begin_function begin_parallel_reduce;
  flare_profiling_end_function end_parallel_reduce;
  flare_profiling_begin_function begin_parallel_scan;
  flare_profiling_end_function end_parallel_scan;
  flare_profiling_push_function push_region;
  flare_profiling_pop_function pop_region;
  flare_profiling_allocate_data_function allocate_data;
  flare_profiling_deallocate_data_function deallocate_data;
  flare_profiling_create_profile_section_function create_profile_section;
  flare_profiling_start_profile_section_function start_profile_section;
  flare_profiling_stop_profile_section_function stop_profile_section;
  flare_profiling_destroy_profile_section_function destroy_profile_section;
  flare_profiling_profile_event_function profile_event;
  flare_profiling_begin_deep_copy_function begin_deep_copy;
  flare_profiling_end_deep_copy_function end_deep_copy;
  flare_profiling_begin_fence_function begin_fence;
  flare_profiling_end_fence_function end_fence;
  flare_profiling_dual_view_sync_function sync_dual_view;
  flare_profiling_dual_view_modify_function modify_dual_view;
  flare_profiling_declare_metadata_function declare_metadata;
  flare_tools_provide_tool_programming_interface_function
      provide_tool_programming_interface;
  flare_tools_request_tool_settings_function request_tool_settings;
  char profiling_padding[9 * sizeof(flare_tools_function_pointer)];
  flare_tools_output_type_declaration_function declare_output_type;
  flare_tools_input_type_declaration_function declare_input_type;
  flare_tools_request_value_function request_output_values;
  flare_tools_context_begin_function begin_tuning_context;
  flare_tools_context_end_function end_tuning_context;
  flare_tools_optimization_goal_declaration_function declare_optimization_goal;
  char padding[232 *
               sizeof(
                   flare_tools_function_pointer)];  // allows us to add another
                                                    // 256 events to the Tools
                                                    // interface without
                                                    // changing struct layout
};

#endif  // FLARE_CORE_PROFILE_C_INTERFACE_H_
