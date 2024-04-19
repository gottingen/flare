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
# picked up original content from https://crascit.com/2016/04/09/using-ccache-with-cmake/

find_program(CCACHE_PROGRAM ccache)

set(CCACHE_FOUND OFF)
if(CCACHE_PROGRAM)
  set(CCACHE_FOUND ON)
endif()

option(FLY_USE_CCACHE "Use ccache when compiling" ${CCACHE_FOUND})

if(${FLY_USE_CCACHE})
  message(STATUS "ccache FOUND: ${CCACHE_PROGRAM}")
  # Set up wrapper scripts
  set(C_LAUNCHER   "${CCACHE_PROGRAM}")
  set(CXX_LAUNCHER "${CCACHE_PROGRAM}")
  set(NVCC_LAUNCHER "${CCACHE_PROGRAM}")
  configure_file(${flare_SOURCE_DIR}/CMakeModules/launch-c.in   launch-c)
  configure_file(${flare_SOURCE_DIR}/CMakeModules/launch-cxx.in launch-cxx)
  configure_file(${flare_SOURCE_DIR}/CMakeModules/launch-nvcc.in launch-nvcc)
  execute_process(COMMAND chmod a+rx
      "${flare_BINARY_DIR}/launch-c"
      "${flare_BINARY_DIR}/launch-cxx"
      "${flare_BINARY_DIR}/launch-nvcc"
    )
  if(CMAKE_GENERATOR STREQUAL "Xcode")
    # Set Xcode project attributes to route compilation and linking
    # through our scripts
    set(CMAKE_XCODE_ATTRIBUTE_CC         "${flare_BINARY_DIR}/launch-c")
    set(CMAKE_XCODE_ATTRIBUTE_CXX        "${flare_BINARY_DIR}/launch-cxx")
    set(CMAKE_XCODE_ATTRIBUTE_LD         "${flare_BINARY_DIR}/launch-c")
    set(CMAKE_XCODE_ATTRIBUTE_LDPLUSPLUS "${flare_BINARY_DIR}/launch-cxx")
  else()
    # Support Unix Makefiles and Ninja
    set(CMAKE_C_COMPILER_LAUNCHER   "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  endif()
endif()
mark_as_advanced(CCACHE_PROGRAM)
mark_as_advanced(FLY_USE_CCACHE)
