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

add_library(opencl_sort_by_key INTERFACE)
foreach(SBK_TYPE ${SBK_TYPES})
    add_library(opencl_sort_by_key_${SBK_TYPE} OBJECT
        "${CMAKE_CURRENT_SOURCE_DIR}/kernel/sort_by_key/sort_by_key_impl.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernel/sort_by_key_impl.hpp")
    add_dependencies(opencl_sort_by_key_${SBK_TYPE}
                        ${cl_kernel_targets} Boost::boost)

    target_include_directories(opencl_sort_by_key_${SBK_TYPE}
      SYSTEM PRIVATE
        $<TARGET_PROPERTY:OpenCL::OpenCL,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:Boost::boost,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:collie::collie,INTERFACE_INCLUDE_DIRECTORIES>)

    target_include_directories(opencl_sort_by_key_${SBK_TYPE}
      PRIVATE
        .
        ..
        ../../api/c
        ../common
        ../../../include
        magma
        ${flare_BINARY_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR})

      target_include_directories(opencl_sort_by_key_${SBK_TYPE}
        SYSTEM INTERFACE
          $<TARGET_PROPERTY:theia::theia,INCLUDE_DIRECTORIES>
      )
      target_include_directories(opencl_sort_by_key_${SBK_TYPE}
        SYSTEM INTERFACE
          $<TARGET_PROPERTY:theia::glad,INTERFACE_INCLUDE_DIRECTORIES>
      )

    set_target_properties(opencl_sort_by_key_${SBK_TYPE}
      PROPERTIES
        CXX_STANDARD 17
        CXX_EXTENSIONS False
        CXX_VISIBILITY_PRESET hidden
        POSITION_INDEPENDENT_CODE ON
        FOLDER "Generated Targets")

    flare_set_default_cxx_flags(opencl_sort_by_key_${SBK_TYPE})

    target_compile_definitions(opencl_sort_by_key_${SBK_TYPE}
      PRIVATE
        ${opencl_compile_definitions}
        $<TARGET_PROPERTY:Boost::boost,INTERFACE_COMPILE_DEFINITIONS>
        TYPE=${SBK_TYPE} FLY_DLL)
    target_sources(opencl_sort_by_key
      INTERFACE $<TARGET_OBJECTS:opencl_sort_by_key_${SBK_TYPE}>)
endforeach(SBK_TYPE ${SBK_TYPES})
