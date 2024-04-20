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

set(ENV{VCPKG_FEATURE_FLAGS} "versions")
set(ENV{VCPKG_KEEP_ENV_VARS} "MKLROOT")
set(VCPKG_MANIFEST_NO_DEFAULT_FEATURES ON)

set(VCPKG_OVERLAY_TRIPLETS ${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules/vcpkg/vcpkg-triplets)
set(VCPKG_OVERLAY_PORTS ${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules/vcpkg/ports)

if(FLY_BUILD_CUDA)
  list(APPEND VCPKG_MANIFEST_FEATURES "cuda")
endif()


if(BUILD_TESTING)
  list(APPEND VCPKG_MANIFEST_FEATURES "tests")
endif()

if(FLY_COMPUTE_LIBRARY STREQUAL "Intel-MKL")
  list(APPEND VCPKG_MANIFEST_FEATURES "mkl")
else()
  list(APPEND VCPKG_MANIFEST_FEATURES "openblasfftw")
endif()

if(DEFINED VCPKG_ROOT AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
elseif(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()
