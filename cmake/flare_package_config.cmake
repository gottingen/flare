#
# Copyright 2023-2024 The EA Authors.
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
include(carbin_system_info)

set(CARBIN_PACKAGING_INSTALL_PREFIX "/opt/EA/inf")
set(CARBIN_PACKAGE_VENDOR "${PROJECT_NAME}")
set(CARBIN_PACKAGE_NAME "${PROJECT_NAME}")

set(CARBIN_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CARBIN_PACKAGE_DESCRIPTION
        "flare is a high performance software library for parallel computing
        with an easy-to-use API. Its array based function set makes parallel
        programming simple.

        flare's multiple backends (CUDA and native CPU) make it
        platform independent and highly portable.

        A few lines of code in flare can replace dozens of lines of parallel
        computing code, saving you valuable time and lowering development costs."
)

set(CARBIN_PACKAGE_MAINTAINER "Jeff.li")
set(CARBIN_PACKAGE_CONTACT "lijippy@163.com")
set(CARBIN_PACKAGE_HOMEPAGE_URL "https://github.com/gottingen/flare")

if (${CARBIN_PACKAGE_SYSTEM_NAME} MATCHES "unknown")
    set(CARBIN_PACKAGE_SYSTEM_NAME "linux") # default to linux  if not set
endif ()
set (TAR_FILE_NAME "${CARBIN_PACKAGE_NAME}-${CARBIN_PACKAGE_VERSION}-${CARBIN_PACKAGE_SYSTEM_NAME}-${CMAKE_HOST_SYSTEM_PROCESSOR}")


if (FLY_BUILD_CUDA AND DEFINED CUDA_VERSION)
    set (TAR_FILE_NAME "${TAR_FILE_NAME}-cu${CUDA_VERSION}")
elseif (FLY_BUILD_CUDA AND DEFINED CUDAToolkit_VERSION)
    set (TAR_FILE_NAME "${TAR_FILE_NAME}-cu${CUDAToolkit_VERSION}")
endif ()
set(CARBIN_PACKAGE_FILE_NAME "${TAR_FILE_NAME}")
message(STATUS "CARBIN_PACKAGE_FILE_NAME: ${CARBIN_PACKAGE_FILE_NAME}")
set(CARBIN_PACKAGE_DIRECTORY package)
