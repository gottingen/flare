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

add_library(flycommon_interface INTERFACE)

target_sources(flycommon_interface
        INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/jit/BinaryNode.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/jit/BinaryNode.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/jit/ModdimNode.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/jit/NaryNode.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/jit/Node.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/jit/Node.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/jit/NodeIO.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/jit/NodeIterator.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/jit/ScalarNode.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/jit/UnaryNode.hpp
)

target_sources(flycommon_interface
        INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/AllocatorInterface.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ArrayInfo.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ArrayInfo.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/FlareTypesIO.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/DefaultMemoryManager.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/DefaultMemoryManager.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/DependencyModule.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/DependencyModule.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/FFTPlanCache.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/HandleBase.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/InteropManager.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/KernelInterface.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Logger.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Logger.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/MemoryManagerBase.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/MersenneTwister.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ModuleInterface.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Source.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/SparseArray.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/SparseArray.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/TemplateArg.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/TemplateTypename.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Version.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/blas_headers.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cast.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cast.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cblas.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/compile_module.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/complex.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/constants.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/defines.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/deterministicHash.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/deterministicHash.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/dim4.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/dispatch.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/dispatch.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/err_common.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/err_common.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/graphics_common.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/graphics_common.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/half.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/half.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/host_memory.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/host_memory.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/internal_enums.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel_cache.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel_cache.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel_type.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/moddims.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/moddims.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/module_loading.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_helpers.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/traits.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/unique_handle.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/util.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/util.hpp
)

if (WIN32)
    target_sources(flycommon_interface INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/module_loading_windows.cpp)
else ()
    target_sources(flycommon_interface INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/module_loading_unix.cpp)
endif ()

target_link_libraries(flycommon_interface
        INTERFACE
        Boost::boost
        ${CMAKE_DL_LIBS}
)

if (TARGET fmt::fmt)
    target_link_libraries(flycommon_interface
            INTERFACE
            fmt::fmt
    )
endif ()

target_link_libraries(flycommon_interface INTERFACE theia::glad)


target_include_directories(flycommon_interface
        SYSTEM INTERFACE
        $<$<PLATFORM_ID:Darwin>:${OPENGL_INCLUDE_DIR}>)
target_include_directories(flycommon_interface
        INTERFACE
        ${collie_INCLUDE_DIR})
target_include_directories(flycommon_interface
        INTERFACE
        ${flare_SOURCE_DIR}/src/backend
        ${flare_BINARY_DIR}/src/backend)

target_include_directories(flycommon_interface
        SYSTEM INTERFACE
        $<TARGET_PROPERTY:theia::theia,INCLUDE_DIRECTORIES>
)


if (APPLE AND NOT USE_MKL)
    target_sources(flycommon_interface
            INTERFACE
            ${CMAKE_CURRENT_SOURCE_DIR}/lapacke.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/lapacke.hpp)
endif ()
