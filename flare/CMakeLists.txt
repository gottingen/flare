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
###########################################################################
# below are examples, after initialization, modify it to your own.
###########################################################################
###carbin_example

set(FLARE_SRC
        ${PROJECT_SOURCE_DIR}/flare/core/common/command_line_parsing.cc
        ${PROJECT_SOURCE_DIR}/flare/core/common/cpu_discovery.cc
        ${PROJECT_SOURCE_DIR}/flare/core/common/exec_policy.cc
        ${PROJECT_SOURCE_DIR}/flare/core/common/host_thread_team.cc
        ${PROJECT_SOURCE_DIR}/flare/core/common/stacktrace.cc
        ${PROJECT_SOURCE_DIR}/flare/core/common/core.cc
        ${PROJECT_SOURCE_DIR}/flare/core/common/error.cc
        ${PROJECT_SOURCE_DIR}/flare/core/common/host_barrier.cc
        ${PROJECT_SOURCE_DIR}/flare/core/common/spinwait.cc
        ${PROJECT_SOURCE_DIR}/flare/core/memory/hbw_space.cc
        ${PROJECT_SOURCE_DIR}/flare/core/memory/host_space.cc
        ${PROJECT_SOURCE_DIR}/flare/core/memory/host_space_deepcopy.cc
        ${PROJECT_SOURCE_DIR}/flare/core/memory/hwloc.cc
        ${PROJECT_SOURCE_DIR}/flare/core/memory/memory_pool.cc
        ${PROJECT_SOURCE_DIR}/flare/core/memory/memory_space.cc
        ${PROJECT_SOURCE_DIR}/flare/core/memory/shared_alloc.cc
        ${PROJECT_SOURCE_DIR}/flare/core/profile/profiling.cc
        ${PROJECT_SOURCE_DIR}/flare/simd/dummy.cc
        ${PROJECT_SOURCE_DIR}/flare/containers/unordered_map_impl.cc
        ${PROJECT_SOURCE_DIR}/flare/kernel/blas/abs_impl.cc
)

if(FLARE_BUILD_OPENMP)
    set(FLARE_SRC
            ${FLARE_SRC}
            ${PROJECT_SOURCE_DIR}/flare/backend/openmp/openmp.cc
            ${PROJECT_SOURCE_DIR}/flare/backend/openmp/openmp_instance.cc
    )
endif (FLARE_BUILD_OPENMP)
if(FLARE_BUILD_SERIAL)
    set(FLARE_SRC
            ${FLARE_SRC}
            ${PROJECT_SOURCE_DIR}/flare/backend/serial/serial.cc
    )
endif (FLARE_BUILD_SERIAL)

if(FLARE_BUILD_CUDA)
    SET(DESUL_ATOMICS_ENABLE_CUDA ON)
    set(FLARE_SRC
            ${FLARE_SRC}
            ${PROJECT_SOURCE_DIR}/flare/backend/cuda/cuda_instance.cu
            ${PROJECT_SOURCE_DIR}/flare/backend/cuda/cuda_space.cu
            ${PROJECT_SOURCE_DIR}/flare/core/atomic/lock_array_cuda.cu
            ${PROJECT_SOURCE_DIR}/flare/core/common/core.cu
            ${PROJECT_SOURCE_DIR}/flare/core/common/error.cu
            ${PROJECT_SOURCE_DIR}/flare/kernel/blas/abs_impl.cu
    )
    include_directories(${CUDA_INCLUDE_DIRS})
endif ()


CONFIGURE_FILE(${PROJECT_SOURCE_DIR}/flare/core/core_config.h.in ${PROJECT_BINARY_DIR}/flare/core/core_config.h @ONLY)
message(STATUS ${CUDA_CUDA_LIBRARY})
carbin_cc_library(
        NAMESPACE flare
        NAME flare
        SOURCES
        ${FLARE_SRC}
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        ${CUDA_CUDA_LIBRARY}
        PUBLIC
        CUDA
)