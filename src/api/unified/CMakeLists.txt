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

generate_product_version(fly_unified_ver_res_file
  FILE_NAME "fly"
  FILE_DESCRIPTION "Unified Backend Dynamic-link library"
)

add_library(fly "")
add_library(flare::fly ALIAS fly)

target_sources(fly
  PRIVATE
    ${fly_unified_ver_res_file}
    algorithm.cpp
    arith.cpp
    array.cpp
    blas.cpp
    data.cpp
    device.cpp
    error.cpp
    event.cpp
    features.cpp
    graphics.cpp
    image.cpp
    index.cpp
    internal.cpp
    jit_test_api.cpp
    lapack.cpp
    memory.cpp
    ml.cpp
    moments.cpp
    random.cpp
    signal.cpp
    sparse.cpp
    statistics.cpp
    symbol_manager.cpp
    symbol_manager.hpp
    util.cpp
    vision.cpp

    $<$<BOOL:${CUDA_FOUND}>: ${CMAKE_CURRENT_SOURCE_DIR}/cuda.cpp>

    ${flare_SOURCE_DIR}/src/api/c/type_util.cpp
    ${flare_SOURCE_DIR}/src/api/c/version.cpp
    ${flare_SOURCE_DIR}/src/backend/common/Logger.cpp
    ${flare_SOURCE_DIR}/src/backend/common/Logger.hpp
    ${flare_SOURCE_DIR}/src/backend/common/constants.cpp
    ${flare_SOURCE_DIR}/src/backend/common/dim4.cpp
    ${flare_SOURCE_DIR}/src/backend/common/err_common.cpp
    ${flare_SOURCE_DIR}/src/backend/common/util.cpp
    ${flare_SOURCE_DIR}/src/backend/common/util.hpp
    ${flare_SOURCE_DIR}/src/backend/common/deprecated.hpp
  )

if(WIN32)
  target_sources(fly
    PRIVATE
      ${flare_SOURCE_DIR}/src/backend/common/module_loading_windows.cpp)
else()
  target_sources(fly
    PRIVATE
      ${flare_SOURCE_DIR}/src/backend/common/module_loading_unix.cpp)
endif()

target_compile_definitions(fly PRIVATE FLY_UNIFIED)

target_include_directories(fly
  PUBLIC
    $<BUILD_INTERFACE:${flare_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${flare_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${FLY_INSTALL_INC_DIR}>
  PRIVATE
    ${flare_SOURCE_DIR}/src/api/c
    ${flare_SOURCE_DIR}/src/api/unified)

target_include_directories(fly
  SYSTEM PRIVATE
    $<TARGET_PROPERTY:flycommon_interface,INTERFACE_INCLUDE_DIRECTORIES>
    $<$<BOOL:${CUDA_FOUND}>:  ${CUDA_INCLUDE_DIRS}>
  )

target_link_libraries(fly
  PRIVATE
    cpp_api_interface
    Threads::Threads
    Boost::boost
    ${CMAKE_DL_LIBS}
  )


install(TARGETS fly
  EXPORT flareUnifiedTargets
  COMPONENT unified
  PUBLIC_HEADER DESTINATION fly
  RUNTIME DESTINATION ${FLY_INSTALL_BIN_DIR}
  LIBRARY DESTINATION ${FLY_INSTALL_LIB_DIR}
  ARCHIVE DESTINATION ${FLY_INSTALL_LIB_DIR}
  FRAMEWORK DESTINATION framework
  INCLUDES DESTINATION ${FLY_INSTALL_INC_DIR}
  )

fly_split_debug_info(fly ${FLY_INSTALL_LIB_DIR})
