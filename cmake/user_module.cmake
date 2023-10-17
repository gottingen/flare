#
# Copyright 2023 The titan-search Authors.
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

include(user_deps)
include(user_cxx_config)

set(FLARE_BUILD_DEVICES)
set(BUILDIN_ES)

if (FLARE_BUILD_CUDA)
    list(APPEND FLARE_BUILD_DEVICES cuda)
    list(APPEND BUILDIN_ES "    Device Parallel: flare::Cuda")
    set(FLARE_ARCH_${CARBIN_ARCH} ON)
else ()
    list(APPEND BUILDIN_ES "    Device Parallel: NoTypeDefined")
endif ()

if (FLARE_BUILD_OPENMP)
    list(APPEND FLARE_BUILD_DEVICES openmp)
    list(APPEND BUILDIN_ES "    Host Parallel: flare::OpenMP")
endif ()

if (FLARE_BUILD_THREADS)
    list(APPEND FLARE_BUILD_DEVICES threads)
    if(FLARE_BUILD_OPENMP)
        carbin_error("Multiple host parallel execution spaces are not allowed!"
                     "Trying to enable execution space flare::Threads, but execution"
                     "space flare::OpenMP is already enabled. Remove the CMakeCache.txt file and re-configure.")
    endif ()
    list(APPEND BUILDIN_ES "    Host Parallel: flare::Threads")
endif ()
if (NOT FLARE_BUILD_OPENMP AND NOT FLARE_BUILD_THREADS)
    list(APPEND BUILDIN_ES "    Host Parallel: NoTypeDefined")
endif ()
if (FLARE_BUILD_SERIAL)
    list(APPEND FLARE_BUILD_DEVICES serial)
    list(APPEND BUILDIN_ES "    Host Serial: flare::Serial")
else ()
    list(APPEND BUILDIN_ES "      Host Serial: flare::NONE")
endif ()

if(NOT FLARE_BUILD_OPENMP AND NOT FLARE_BUILD_THREADS AND NOT FLARE_BUILD_SERIAL)
    carbin_error("At least one host execution space must be enabled, "
            "but no host parallel execution space was requested "
            "and flare_ENABLE_SERIAL=OFF.")
endif ()

set(FLARE_ENABLE_OPENMP ${FLARE_BUILD_OPENMP})

set(FLARE_ENABLE_SERIAL ${FLARE_BUILD_SERIAL})

set(FLARE_ENABLE_CUDA ${FLARE_BUILD_CUDA})

set(FLARE_ENABLE_IMPL_CUDA_MALLOC_ASYNC ON)
set(FLARE_ARCH_AVX2 ON)

option(FLARE_INST_ORDINAL_INT "" ON)
option(FLARE_INST_ORDINAL_INT64_T "" OFF)
option(FLARE_INST_LAYOUT_LEFT "" ON)
option(FLARE_INST_LAYOUT_RIGHT "" OFF)
option(FLARE_INST_OFFSET_INT "" OFF)
option(FLARE_INST_OFFSET_SIZE_T "" ON)
option(FLARE_INST_DOUBLE "" ON)
option(FLARE_INST_FLOAT "" OFF)
option(FLARE_INST_HALF "" OFF)
option(FLARE_INST_BHALF "" OFF)
option(FLARE_ENABLE_PRAGMA_UNROLL "" OFF)

option(FLARE_ENABLE_CORE_TEST "" OFF)
option(FLARE_ENABLE_KERNEL_TEST "" OFF)
option(FLARE_ENABLE_SIMD_TEST "" OFF)
option(FLARE_ENABLE_TASKFLOW_TEST "" ON)

set(FLARE_DEBUG_LEVEL 1)
set(FLARE_BLAS_OPTIMIZATION_LEVEL_AXPBY 2)

carbin_print_list_label("FLARE_BUILD_DEVICES" FLARE_BUILD_DEVICES)


carbin_print_list_label("Built-in Execution Spaces" BUILDIN_ES)