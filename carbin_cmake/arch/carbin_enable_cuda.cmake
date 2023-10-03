#
# Copyright 2023 The Carbin Authors.
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

find_package(CUDA 11.6)

option(CARBIN_CAN_BUILD_CUDA     "Build flare with a CUDA backend"       ${CUDA_FOUND})
if(CARBIN_CAN_BUILD_CUDA)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
    elseif(CUDA_NVCC_EXECUTABLE)
        message(STATUS "Using the FindCUDA script to search for the CUDA compiler")
        set(CMAKE_CUDA_COMPILER ${CUDA_NVCC_EXECUTABLE} CACHE INTERNAL "CUDA compiler executable")
        enable_language(CUDA)
    else()
        message(WARNING "No CUDA support")
    endif()
endif()