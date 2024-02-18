# Copyright (c) 2019, Flare
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

set(FG_VERSION_MAJOR 1)
set(FG_VERSION_MINOR 0)
set(FG_VERSION_PATCH 8)
set(FG_VERSION "${FG_VERSION_MAJOR}.${FG_VERSION_MINOR}.${FG_VERSION_PATCH}")
set(FG_API_VERSION_CURRENT ${FG_VERSION_MAJOR}${FG_VERSION_MINOR})


file(COPY ${PROJECT_SOURCE_DIR}/3rd/forge-1.0.8 DESTINATION ${PROJECT_BINARY_DIR}/extern)

set(forge_source_dir ${PROJECT_BINARY_DIR}/extern/forge-1.0.8)
set(forge_binary_dir ${PROJECT_BINARY_DIR}/extern/forge-build)

set(af_FETCHCONTENT_BASE_DIR ${FETCHCONTENT_BASE_DIR})
set(af_FETCHCONTENT_QUIET ${FETCHCONTENT_QUIET})
set(af_FETCHCONTENT_FULLY_DISCONNECTED ${FETCHCONTENT_FULLY_DISCONNECTED})
set(af_FETCHCONTENT_UPDATES_DISCONNECTED ${FETCHCONTENT_UPDATES_DISCONNECTED})

set(FlareInstallPrefix ${CMAKE_INSTALL_PREFIX})
set(FlareBuildType ${CMAKE_BUILD_TYPE})
set(CMAKE_INSTALL_PREFIX ${forge_binary_dir}/extern/forge/package)
set(CMAKE_BUILD_TYPE Release)
set(FG_BUILD_EXAMPLES OFF CACHE BOOL "Used to build Forge examples")
set(FG_BUILD_DOCS OFF CACHE BOOL "Used to build Forge documentation")
set(FG_WITH_FREEIMAGE OFF CACHE BOOL "Turn on usage of freeimage dependency")

add_subdirectory(
    ${forge_source_dir} ${forge_binary_dir} EXCLUDE_FROM_ALL)
mark_as_advanced(
    FG_BUILD_EXAMPLES
    FG_BUILD_DOCS
    FG_WITH_FREEIMAGE
    FG_USE_WINDOW_TOOLKIT
    FG_RENDERING_BACKEND
    SPHINX_EXECUTABLE
    glfw3_DIR
    glm_DIR
    )
set(CMAKE_BUILD_TYPE ${FlareBuildType})
set(CMAKE_INSTALL_PREFIX ${FlareInstallPrefix})
set(FETCHCONTENT_BASE_DIR ${af_FETCHCONTENT_BASE_DIR})
set(FETCHCONTENT_QUIET ${af_FETCHCONTENT_QUIET})
set(FETCHCONTENT_FULLY_DISCONNECTED ${af_FETCHCONTENT_FULLY_DISCONNECTED})
set(FETCHCONTENT_UPDATES_DISCONNECTED ${af_FETCHCONTENT_UPDATES_DISCONNECTED})
install(FILES
    $<TARGET_FILE:forge>
    $<$<PLATFORM_ID:Linux>:$<TARGET_SONAME_FILE:forge>>
    $<$<PLATFORM_ID:Darwin>:$<TARGET_SONAME_FILE:forge>>
    $<$<PLATFORM_ID:Linux>:$<TARGET_LINKER_FILE:forge>>
    $<$<PLATFORM_ID:Darwin>:$<TARGET_LINKER_FILE:forge>>
    DESTINATION "${FLY_INSTALL_LIB_DIR}"
    COMPONENT common_backend_dependencies)

if(FLY_INSTALL_STANDALONE)
    cmake_minimum_required(VERSION 3.21)
    install(FILES
        $<TARGET_RUNTIME_DLLS:forge>
        DESTINATION "${FLY_INSTALL_LIB_DIR}"
        COMPONENT common_backend_dependencies)
endif(FLY_INSTALL_STANDALONE)

set_property(TARGET forge APPEND_STRING PROPERTY COMPILE_FLAGS " -w")

