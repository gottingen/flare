#
# Copyright 2023 The EA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/kernel/sort_by_key/sort_by_key_impl.cpp" FILESTRINGS)

foreach(STR ${FILESTRINGS})
    if(${STR} MATCHES "// SBK_TYPES")
        string(REPLACE "// SBK_TYPES:" "" TEMP ${STR})
        string(REPLACE " " ";" SBK_TYPES ${TEMP})
    endif()
endforeach()

add_library(cpu_sort_by_key INTERFACE)
foreach(SBK_TYPE ${SBK_TYPES})
  add_library(cpu_sort_by_key_${SBK_TYPE} OBJECT
        "${CMAKE_CURRENT_SOURCE_DIR}/kernel/sort_by_key/sort_by_key_impl.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernel/sort_by_key_impl.hpp"
    )
  set_target_properties(cpu_sort_by_key_${SBK_TYPE}
    PROPERTIES
      COMPILE_DEFINITIONS "TYPE=${SBK_TYPE};FLY_DLL;$<TARGET_PROPERTY:Boost::boost,INTERFACE_COMPILE_DEFINITIONS>"
      CXX_STANDARD 17
      CXX_EXTENSIONS OFF
      CXX_VISIBILITY_PRESET hidden
      FOLDER "Generated Targets")

  flare_set_default_cxx_flags(cpu_sort_by_key_${SBK_TYPE})

  target_include_directories(cpu_sort_by_key_${SBK_TYPE}
    PUBLIC
      .
      ../../api/c
      ${flare_SOURCE_DIR}/include
      ${flare_BINARY_DIR}/include
    PRIVATE
      ../common
      ..
      threads)

  target_include_directories(cpu_sort_by_key_${SBK_TYPE}
    SYSTEM PRIVATE
      $<TARGET_PROPERTY:Boost::boost,INTERFACE_INCLUDE_DIRECTORIES>)

  set_target_properties(cpu_sort_by_key_${SBK_TYPE} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_sources(cpu_sort_by_key
    INTERFACE $<TARGET_OBJECTS:cpu_sort_by_key_${SBK_TYPE}>)
endforeach(SBK_TYPE ${SBK_TYPES})
