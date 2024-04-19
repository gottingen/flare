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

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_by_key/scan_by_key_impl.cpp" FILESTRINGS)
find_package(OpenCL REQUIRED)

foreach(STR ${FILESTRINGS})
    if(${STR} MATCHES "// SBK_BINARY_OPS")
        string(REPLACE "// SBK_BINARY_OPS:" "" TEMP ${STR})
        string(REPLACE " " ";" SBK_BINARY_OPS ${TEMP})
    endif()
endforeach()

add_library(opencl_scan_by_key INTERFACE)

add_dependencies(opencl_scan_by_key ${cl_kernel_targets} cl2hpp)
foreach(SBK_BINARY_OP ${SBK_BINARY_OPS})
    add_library(opencl_scan_by_key_${SBK_BINARY_OP} OBJECT
        "${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_by_key/scan_by_key_impl.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_first_by_key_impl.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_dim_by_key_impl.hpp")

    add_dependencies(opencl_scan_by_key_${SBK_BINARY_OP}
                        ${cl_kernel_targets}  Boost::boost)

    target_include_directories(opencl_scan_by_key_${SBK_BINARY_OP}
      PRIVATE
        .
        ..
        magma
        ../../api/c
        ../common
        ../../../include
        ${CMAKE_CURRENT_BINARY_DIR}
        $<TARGET_PROPERTY:collie::collie,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:OpenCL::OpenCL,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:Boost::boost,INTERFACE_INCLUDE_DIRECTORIES>
        ${flare_BINARY_DIR}/include
      )
      target_include_directories(opencl_scan_by_key_${SBK_BINARY_OP}
        SYSTEM INTERFACE
        $<TARGET_PROPERTY:theia::theia,INCLUDE_DIRECTORIES>
      )
      target_include_directories(opencl_scan_by_key_${SBK_BINARY_OP}
        SYSTEM INTERFACE
        $<TARGET_PROPERTY:theia::glad,INTERFACE_INCLUDE_DIRECTORIES>
      )

    set_target_properties(opencl_scan_by_key_${SBK_BINARY_OP}
      PROPERTIES
        CXX_STANDARD 17
        CXX_EXTENSIONS False
        CXX_VISIBILITY_PRESET hidden
        POSITION_INDEPENDENT_CODE ON
        FOLDER "Generated Targets")

    flare_set_default_cxx_flags(opencl_scan_by_key_${SBK_BINARY_OP})
    target_compile_definitions(opencl_scan_by_key_${SBK_BINARY_OP}
      PRIVATE
        ${opencl_compile_definitions}
        $<TARGET_PROPERTY:Boost::boost,INTERFACE_COMPILE_DEFINITIONS>
        $<TARGET_PROPERTY:collie::collie,INTERFACE_COMPILE_DEFINITIONS>
        TYPE=${SBK_BINARY_OP} FLY_DLL)
    target_sources(opencl_scan_by_key
      INTERFACE $<TARGET_OBJECTS:opencl_scan_by_key_${SBK_BINARY_OP}>)
endforeach(SBK_BINARY_OP ${SBK_BINARY_OPS})
