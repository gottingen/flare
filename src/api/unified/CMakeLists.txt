# Copyright (c) 2022, Flare
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

generate_product_version(fly_unified_ver_res_file
  FILE_NAME "fly"
  FILE_DESCRIPTION "Unified Backend Dynamic-link library"
)

add_library(fly "")
add_library(Flare::fly ALIAS fly)

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

    $<$<BOOL:${OpenCL_FOUND}>: ${CMAKE_CURRENT_SOURCE_DIR}/opencl.cpp>
    $<$<BOOL:${CUDA_FOUND}>: ${CMAKE_CURRENT_SOURCE_DIR}/cuda.cpp>

    ${Flare_SOURCE_DIR}/src/api/c/type_util.cpp
    ${Flare_SOURCE_DIR}/src/api/c/version.cpp
    ${Flare_SOURCE_DIR}/src/backend/common/Logger.cpp
    ${Flare_SOURCE_DIR}/src/backend/common/Logger.hpp
    ${Flare_SOURCE_DIR}/src/backend/common/constants.cpp
    ${Flare_SOURCE_DIR}/src/backend/common/dim4.cpp
    ${Flare_SOURCE_DIR}/src/backend/common/err_common.cpp
    ${Flare_SOURCE_DIR}/src/backend/common/util.cpp
    ${Flare_SOURCE_DIR}/src/backend/common/util.hpp
    ${Flare_SOURCE_DIR}/src/backend/common/deprecated.hpp
  )

if(WIN32)
  target_sources(fly
    PRIVATE
      ${Flare_SOURCE_DIR}/src/backend/common/module_loading_windows.cpp)
else()
  target_sources(fly
    PRIVATE
      ${Flare_SOURCE_DIR}/src/backend/common/module_loading_unix.cpp)
endif()

target_compile_definitions(fly PRIVATE FLY_UNIFIED)

target_include_directories(fly
  PUBLIC
    $<BUILD_INTERFACE:${Flare_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${Flare_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${FLY_INSTALL_INC_DIR}>
  PRIVATE
    ${Flare_SOURCE_DIR}/src/api/c
    ${Flare_SOURCE_DIR}/src/api/unified)

target_include_directories(fly
  SYSTEM PRIVATE
    $<TARGET_PROPERTY:flycommon_interface,INTERFACE_INCLUDE_DIRECTORIES>
    $<$<BOOL:${OpenCL_FOUND}>: $<TARGET_PROPERTY:OpenCL::OpenCL,INTERFACE_INCLUDE_DIRECTORIES>>
    $<$<BOOL:${CUDA_FOUND}>:  ${CUDA_INCLUDE_DIRS}>
  )

target_link_libraries(fly
  PRIVATE
    fly_spdlog
    cpp_api_interface
    Threads::Threads
    Boost::boost
    ${CMAKE_DL_LIBS}
  )

if(TARGET fmt::fmt)
  target_link_libraries(fly
    PRIVATE
      fmt::fmt
  )
endif()

install(TARGETS fly
  EXPORT FlareUnifiedTargets
  COMPONENT unified
  PUBLIC_HEADER DESTINATION fly
  RUNTIME DESTINATION ${FLY_INSTALL_BIN_DIR}
  LIBRARY DESTINATION ${FLY_INSTALL_LIB_DIR}
  ARCHIVE DESTINATION ${FLY_INSTALL_LIB_DIR}
  FRAMEWORK DESTINATION framework
  INCLUDES DESTINATION ${FLY_INSTALL_INC_DIR}
  )

fly_split_debug_info(fly ${FLY_INSTALL_LIB_DIR})

# install(TARGETS fly EXPORT FLY DESTINATION "${FLY_INSTALL_LIB_DIR}"
#   COMPONENT libraries)
#
# if(APPLE)
#   INSTALL(SCRIPT "${PROJECT_SOURCE_DIR}/CMakeModules/osx_install/InstallTool.cmake")
# endif(APPLE)
#
# export(TARGETS fly FILE FlareUnified.cmake)
# install(EXPORT FLY DESTINATION "${FLY_INSTALL_CMAKE_DIR}"
#   COMPONENT cmake
#   FILE FlareUnified.cmake)

source_group(include REGULAR_EXPRESSION ${Flare_SOURCE_DIR}/include/*)
source_group(source REGULAR_EXPRESSION ${CMAKE_CURRENT_SOURCE_DIR}/*|${Flare_SOURCE_DIR}/src/backend/common/*)
source_group(api\\cpp REGULAR_EXPRESSION ${Flare_SOURCE_DIR}/src/api/cpp/*)
source_group(api\\c REGULAR_EXPRESSION ${Flare_SOURCE_DIR}/src/api/c/*)
source_group("" FILES CMakeLists.txt)
