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
# Make a version file that includes the Flare version and git revision
#
set(FLY_VERSION_MAJOR ${flare_VERSION_MAJOR})
set(FLY_VERSION_MINOR ${flare_VERSION_MINOR})
set(FLY_VERSION_PATCH ${flare_VERSION_PATCH})

set(FLY_VERSION ${flare_VERSION})
set(flare_API_VERSION_CURRENT ${flare_VERSION_MAJOR}${flare_VERSION_MINOR})

# From CMake 3.0.0 CMAKE_<LANG>_COMPILER_ID is AppleClang for OSX machines
# that use clang for compilations
if("${CMAKE_C_COMPILER_ID}" STREQUAL "AppleClang")
    set(COMPILER_NAME "AppleClang")
elseif("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang")
    set(COMPILER_NAME "LLVM Clang")
elseif("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(COMPILER_NAME "GNU Compiler Collection(GCC/G++)")
elseif("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
    set(COMPILER_NAME "Intel Compiler")
elseif("${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC")
    set(COMPILER_NAME "Microsoft Visual Studio")
endif()

set(COMPILER_VERSION "${CMAKE_C_COMPILER_VERSION}")
set(FLY_COMPILER_STRING "${COMPILER_NAME} ${COMPILER_VERSION}")

execute_process(
    COMMAND git log -1 --format=%h
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT GIT_COMMIT_HASH)
    message(STATUS "No git. Setting hash to default")
    set(GIT_COMMIT_HASH "default")
endif()

configure_file(
    ${flare_SOURCE_DIR}/include/fly/version.h.in
    ${flare_SOURCE_DIR}/include/fly/version.h
)

configure_file(
    ${flare_SOURCE_DIR}/src/backend/build_version.hpp.in
    ${flare_SOURCE_DIR}/src/backend/build_version.hpp
)
