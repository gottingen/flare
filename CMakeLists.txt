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

cmake_minimum_required(VERSION 3.24)
include(CheckLanguage)

project(flare VERSION 1.0.3 LANGUAGES C CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/find")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/carbin_cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/carbin_cmake/arch")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/carbin_cmake/copts")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/carbin_cmake/modules")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/carbin_cmake/package")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/carbin_cmake/recipes")

include(fly_configure_deps_vars)
include(fly_build_configurations)
include(fly_install_dirs)
include(CMakeDependentOption)
include(InternalUtils)
include(Version)
include(platform)
include(GetPrerequisites)
include(CheckCXXCompilerFlag)
include(CheckSymbolExists)
include(SplitDebugInfo)

# Use the function generate_product_version on Windows
# to attach version info in dll file attributes.
# Make sure to pass appropriate arguments for each backend
# to generate the correct resource file
include(generate_product_version)

set_policies(
        TYPE NEW
        POLICIES CMP0073
        CMP0074
        CMP0077
        CMP0079)
flare_set_cmake_default_variables()

option(FLY_WITH_EXTERNAL_PACKAGES_ONLY "Build Flare with External packages only" OFF)
if (FLY_WITH_EXTERNAL_PACKAGES_ONLY)
    set(FLY_REQUIRED REQUIRED)
endif ()
if (CMAKE_SYCL_COMPILER)
    get_filename_component(SYCL_COMPILER_NAME ${CMAKE_SYCL_COMPILER} NAME)
endif ()
if (SYCL_COMPILER_NAME STREQUAL "dpcpp" OR SYCL_COMPILER_NAME STREQUAL "dpcpp.exe"
        OR SYCL_COMPILER_NAME STREQUAL "icpx" OR SYCL_COMPILER_NAME STREQUAL "icx.exe")
    set(MKL_THREAD_LAYER "TBB" CACHE STRING "The thread layer to choose for MKL")
    set(MKL_INTERFACE "ilp64")
    set(MKL_INTERFACE_INTEGER_SIZE 8)
else ()
    set(MKL_THREAD_LAYER "Intel OpenMP" CACHE STRING "The thread layer to choose for MKL")
    set(MKL_INTERFACE "lp64")
    set(MKL_INTERFACE_INTEGER_SIZE 4)
endif ()

if (EXISTS ${PROJECT_SOURCE_DIR}/carbin)
    list(APPEND CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/carbin)
endif ()
find_package(CUDA 10.2)
find_package(cuDNN 4.0)
find_package(OpenGL)
find_package(FreeImage)
find_package(Threads)
find_package(FFTW)
find_package(CBLAS)
find_package(LAPACKE)
find_package(Doxygen)
find_package(FLY_MKL)
find_package(GTest REQUIRED)
find_package(CLBlast QUIET)
find_package(Boost 1.70 REQUIRED)

# CLFFT used in Flare requires a specific fork
#find_package(clFFT QUIET)

include(boost_package)
include(config_ccache)

option(FLY_BUILD_CPU "Build Flare with a CPU backend" ON)
option(FLY_BUILD_CUDA "Build Flare with a CUDA backend" ${CUDA_FOUND})
option(FLY_BUILD_UNIFIED "Build Backend-Independent Flare API" ON)
option(FLY_BUILD_DOCS "Create Flare Documentation" ${DOXYGEN_FOUND})
option(FLY_BUILD_EXAMPLES "Build Examples" ON)
option(FLY_WITH_CUDNN "Use cuDNN for convolveNN functions" ${cuDNN_FOUND})

option(FLY_WITH_NONFREE "Build Flare nonfree algorithms" OFF)
option(FLY_WITH_LOGGING "Build Flare with logging support" ON)
option(FLY_WITH_STACKTRACE "Add stacktraces to the error messages." ON)
option(FLY_CACHE_KERNELS_TO_DISK "Enable caching kernels to disk" ON)
option(FLY_WITH_STATIC_MKL "Link against static Intel MKL libraries" OFF)
option(FLY_WITH_STATIC_CUDA_NUMERIC_LIBS "Link libflycuda with static numeric libraries(cublas, cufft, etc.)" OFF)
option(FLY_WITH_FAST_MATH "Use lower precision but high performance numeric optimizations" OFF)
option(FLY_CTEST_SEPARATED "Run tests separately when called from ctest(increases test times)" OFF)

if (FLY_WITH_STATIC_CUDA_NUMERIC_LIBS)
    option(FLY_WITH_PRUNE_STATIC_CUDA_NUMERIC_LIBS "Prune CUDA static libraries to reduce binary size.(WARNING: May break some libs on older CUDA toolkits for some compute arch)" OFF)
endif ()

set(default_compute_library "FFTW/LAPACK/BLAS")
if (MKL_FOUND)
    set(default_compute_library "Intel-MKL")
endif ()

if (FLY_WITH_STATIC_MKL)
    set(MKL_LINK static)
endif ()
if (MKL_THREAD_LAYER STREQUAL "Sequential")
    set(MKL_THREADING "sequential")
elseif (MKL_THREAD_LAYER STREQUAL "GNU OpenMP")
    set(MKL_THREADING "gnu_thread")
elseif (MKL_THREAD_LAYER STREQUAL "Intel OpenMP")
    set(MKL_THREADING "intel_thread")
elseif (MKL_THREAD_LAYER STREQUAL "TBB")
    set(MKL_THREADING "tbb_thread")
else ()
endif ()

if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.13)
    # VCPKG overrides the find_package command and the PATH parameter is currently
    # broken with the current version of VCPKG so we are setting the MKL_ROOT
    # directory to the MKLROOT environment variable.
    if (DEFINED ENV{MKLROOT} AND NOT DEFINED MKL_ROOT)
        set(MKL_ROOT "$ENV{MKLROOT}")
    endif ()
    set(DPCPP_COMPILER ON)
    set(MKL_THREADING "tbb_thread")
    set(MKL_INTERFACE "ilp64")
    find_package(MKL 2023.1)
endif ()

fly_multiple_option(NAME FLY_COMPUTE_LIBRARY
        DEFAULT ${default_compute_library}
        DESCRIPTION "Compute library for signal processing and linear algebra routines"
        OPTIONS "Intel-MKL" "FFTW/LAPACK/BLAS")

if (WIN32)
    fly_multiple_option(NAME FLY_STACKTRACE_TYPE
            DEFAULT "Windbg"
            DESCRIPTION "The type of backtrace features. Windbg(simple), None"
            OPTIONS "Windbg" "None")
else ()
    fly_multiple_option(NAME FLY_STACKTRACE_TYPE
            DEFAULT "Basic"
            DESCRIPTION "The type of backtrace features. Basic(simple), libbacktrace(fancy), addr2line(fancy), None"
            OPTIONS "Basic" "libbacktrace" "addr2line" "None")
endif ()

option(FLY_INSTALL_STANDALONE "Build installers that include all dependencies" OFF)

cmake_dependent_option(FLY_WITH_RELATIVE_TEST_DIR "Use relative paths for the test data directory(For continious integration(CI) purposes only)" OFF
        "BUILD_TESTING" OFF)

cmake_dependent_option(FLY_WITH_IMAGEIO "Build Flare with Image IO support" ${FreeImage_FOUND}
        "FreeImage_FOUND" OFF)
cmake_dependent_option(FLY_BUILD_FRAMEWORK "Build an Flare framework for Apple platforms.(Experimental)" OFF
        "APPLE" OFF)

option(FLY_WITH_STATIC_FREEIMAGE "Use Static FreeImage Lib" OFF)

set(FLY_WITH_CPUID ON CACHE BOOL "Build with CPUID integration")

if (FLY_BUILD_CUDA)
    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
    elseif (CUDA_NVCC_EXECUTABLE)
        message(STATUS "Using the FindCUDA script to search for the CUDA compiler")
        set(CMAKE_CUDA_COMPILER ${CUDA_NVCC_EXECUTABLE} CACHE INTERNAL "CUDA compiler executable")
        enable_language(CUDA)
    else ()
        message(WARNING "No CUDA support")
    endif ()
endif ()

fly_deprecate(BUILD_CPU FLY_BUILD_CPU)
fly_deprecate(BUILD_CUDA FLY_BUILD_CUDA)
fly_deprecate(BUILD_UNIFIED FLY_BUILD_UNIFIED)
fly_deprecate(BUILD_DOCS FLY_BUILD_DOCS)
fly_deprecate(BUILD_NONFREE FLY_WITH_NONFREE)
fly_deprecate(BUILD_EXAMPLES FLY_BUILD_EXAMPLES)
fly_deprecate(USE_RELATIVE_TEST_DIR FLY_WITH_RELATIVE_TEST_DIR)
fly_deprecate(USE_FREEIMAGE_STATIC FLY_WITH_STATIC_FREEIMAGE)
fly_deprecate(USE_CPUID FLY_WITH_CPUID)
if (DEFINED USE_CPU_MKL)
    # Cannot use fly_deprecated as it expects the new and old variables to store values of
    # same type. In this case, USE_*_MKL variables are BOOLs and FLY_COMPUTE_LIBRARY is a STRING
    message(DEPRECATION
            "Variables USE_CPU_MKL are deprecated. Use FLY_COMPUTE_LIBRARY instead.")
    message(WARNING
            "USE_CPU_MKL defined. These values take precendence over the value of
    FLY_COMPUTE_LIBRARY until they are removed to preserve existing build behavior.")
    # Until USE_CPU_MKL are removed, if they are defined, they take
    # precendence and cmake will check and report error if Intel-MKL is not found
    if (USE_CPU_MKL)
        get_property(doc CACHE FLY_COMPUTE_LIBRARY PROPERTY HELPSTRING)
        set(FLY_COMPUTE_LIBRARY "Intel-MKL" CACHE STRING "${doc}" FORCE)
    endif ()
endif ()

if (FLY_COMPUTE_LIBRARY STREQUAL "Intel-MKL")
    set(BLA_VENDOR "Intel10_64lp")
    if (MKL_THREAD_LAYER STREQUAL "Sequential")
        set(BLA_VENDOR "${BLA_VENDOR}_seq")
    endif ()
endif ()
find_package(BLAS)
find_package(LAPACK)

# IF: the old USE_CPU_MKL flags are present,
# THEN Irrespective of FLY_COMPUTE_LIBRARY value, continue with MKL to preserve old
#      behavior. Once the deprecated USE_CPU_MKL are removed in later
#      versions FLY_COMPUTE_LIBRARY will take over total control of selecting CPU
#      compute backend.
#
# Note that the default value of FLY_COMPUTE_LIBRARY is Intel-MKL.
# Also, cmake doesn't have short-circuit of OR/AND conditions in if
if (${FLY_BUILD_CPU})
    if ("${FLY_COMPUTE_LIBRARY}" STREQUAL "Intel-MKL"
            OR "${FLY_COMPUTE_LIBRARY}" STREQUAL "MKL")
        fly_mkl_batch_check()
        dependency_check(MKL_Shared_FOUND "Please ensure Intel-MKL / oneAPI-oneMKL is installed")
        set(BUILD_WITH_MKL ON)
    elseif ("${FLY_COMPUTE_LIBRARY}" STREQUAL "FFTW/LAPACK/BLAS")
        dependency_check(FFTW_FOUND "FFTW not found")
        dependency_check(CBLAS_FOUND "CBLAS not found")
        if (UNIX AND NOT APPLE)
            dependency_check(LAPACK_FOUND "LAPACK not found")
        endif ()
    endif ()
endif ()


#include(fly_configure_forge_dep)
find_package(collie REQUIRED PATHS ${PROJECT_SOURCE_DIR}/carbin)
find_package(theia REQUIRED PATHS ${PROJECT_SOURCE_DIR}/carbin)


set(ASSETS_DIR ${PROJECT_SOURCE_DIR}/share/flare/asserts)

# when crosscompiling use the bin2cpp file from the native bin directory
if (CMAKE_CROSSCOMPILING)
    set(NATIVE_BIN_DIR "NATIVE_BIN_DIR-NOTFOUND"
            CACHE FILEPATH "Path to the Native build directory.")
    if (NATIVE_BIN_DIR)
        include(${NATIVE_BIN_DIR}/ImportExecutables.cmake)
    else ()
        message(SEND_ERROR "Native Directory not found. Run cmake in a separate"
                "directory and build the bin2cpp target.")
    endif ()
else ()
    add_executable(bin2cpp CMakeModules/bin2cpp.cpp
            src/backend/common/deterministicHash.cpp
            src/backend/common/deterministicHash.hpp
            src/backend/common/Source.hpp)
    set_target_properties(bin2cpp
            PROPERTIES
            CXX_STANDARD 17)

    if (WIN32)
        target_compile_definitions(bin2cpp PRIVATE OS_WIN)
    elseif (APPLE)
        target_compile_definitions(bin2cpp PRIVATE OS_MAC)
    elseif (UNIX)
        target_compile_definitions(bin2cpp PRIVATE OS_LNX)
    endif ()
    target_include_directories(bin2cpp PRIVATE
            ${flare_SOURCE_DIR}/include
            ${flare_BINARY_DIR}/include
            ${flare_SOURCE_DIR}/src/backend)
    export(TARGETS bin2cpp FILE ${CMAKE_BINARY_DIR}/ImportExecutables.cmake)
endif ()


if (NOT LAPACK_FOUND)
    if (APPLE)
        # UNSET THE VARIABLES FROM LAPACKE
        unset(LAPACKE_LIB CACHE)
        unset(LAPACK_LIB CACHE)
        unset(LAPACKE_INCLUDES CACHE)
        unset(LAPACKE_ROOT_DIR CACHE)
    endif ()
endif ()

add_subdirectory(src/backend/common)
add_subdirectory(src/api/c)
add_subdirectory(src/api/cpp)

conditional_directory(FLY_BUILD_CPU src/backend/cpu)
conditional_directory(FLY_BUILD_CUDA src/backend/cuda)
conditional_directory(FLY_BUILD_UNIFIED src/api/unified)

if (TARGET fly)
    list(APPEND built_backends fly)
endif ()

if (TARGET flycpu)
    list(APPEND built_backends flycpu)
endif ()

if (TARGET flycuda)
    list(APPEND built_backends flycuda)
endif ()


set_target_properties(${built_backends} PROPERTIES
        CXX_STANDARD 17
        CXX_EXTENSIONS OFF
        CXX_VISIBILITY_PRESET hidden
        VERSION "${flare_VERSION}"
        SOVERSION "${flare_VERSION_MAJOR}")

if (FLY_INSTALL_STANDALONE)

    # This flag enables the use of RUNPATH instead of RPATH which is the
    # preferred method to set the runtime lookup. Only doind this for
    # standalone builds because we include all libraries with the installers
    # and they are included in the same directory so the RUNPATH is set to
    # $ORIGIN. This avoid setting the linker path in ld.so.conf.d
    check_cxx_compiler_flag("-Wl,--enable-new-dtags" HAS_RUNPATH_FLAG)
    if (HAS_RUNPATH_FLAG)
        set_target_properties(${built_backends} PROPERTIES
                INSTALL_RPATH "$ORIGIN"
                LINK_OPTIONS "-Wl,--enable-new-dtags")
    endif ()
endif ()

# On some distributions the linker will not add a library to the ELF header if
# the symbols are not needed when the library was first parsed by the linker.
# This causes undefined references issues when linking with libraries which have
# circular dependencies.
if (UNIX AND NOT APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set_target_properties(${built_backends} PROPERTIES
            LINK_FLAGS "-Wl,--no-as-needed")
endif ()


find_library(Backtrace_LIBRARY backtrace
        DOC "libbacktrace.so file for more informative stacktraces. https://github.com/ianlancetaylor/libbacktrace")
find_program(ADDR2LINE_PROGRAM addr2line
        DOC "The path to the addr2line program for informative stacktraces")

check_cxx_compiler_flag(-Wno-ignored-attributes has_ignored_attributes_flag)
check_cxx_compiler_flag(-Wall has_all_warnings_flag)

foreach (backend ${built_backends})
    flare_set_default_cxx_flags(${backend})
endforeach ()

if (FLY_BUILD_FRAMEWORK)
    set_target_properties(${built_backends}
            PROPERTIES
            FRAMEWORK TRUE
            FRAMEWORK_VERSION A
            MACOSX_FRAMEWORK_IDENTIFIER com.flare.flareFramework
            #MACOSX_FRAMEWORK_INFO_PLIST Info.plist
            #PUBLIC_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/include/flare.h;${fly_headers}"
            #XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "iPhone Developer"
    )
endif ()

install(DIRECTORY include/ DESTINATION ${FLY_INSTALL_INC_DIR}
        COMPONENT headers
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp"
        PATTERN ".gitignore" EXCLUDE
)

## The Flare version file is generated and won't be included above, install
## it separately.
#[[
install(FILES ${flare_SOURCE_DIR}/include/fly/version.h
        ${flare_SOURCE_DIR}/include/fly/compilers.h
        DESTINATION "${FLY_INSTALL_INC_DIR}/fly/"
        COMPONENT headers)
        ]]

# install the examples irrespective of the FLY_BUILD_EXAMPLES value
# only the examples source files are installed, so the installation of these
# source files does not depend on FLY_BUILD_EXAMPLES
# when FLY_BUILD_EXAMPLES is OFF, the examples source is installed without
# building the example executables
install(DIRECTORY examples/ #NOTE The slash at the end is important
        DESTINATION ${FLY_INSTALL_EXAMPLE_DIR}
        COMPONENT examples)

install(DIRECTORY ${ASSETS_DIR}/examples/ #NOTE The slash at the end is important
        DESTINATION ${FLY_INSTALL_ASSERTS_DIR}/examples
        COMPONENT examples)

install(DIRECTORY "${flare_SOURCE_DIR}/LICENSES/"
        DESTINATION LICENSES
        COMPONENT licenses)

install(DIRECTORY "${flare_SOURCE_DIR}/python"
        DESTINATION share/flare
        COMPONENT python)

install(FILES "${flare_SOURCE_DIR}/tools/flare_install_py"
        DESTINATION bin
        PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
        GROUP_EXECUTE GROUP_READ GROUP_EXECUTE
        WORLD_EXECUTE WORLD_READ WORLD_EXECUTE
        COMPONENT python)

foreach (backend CPU CUDA Unified)
    string(TOUPPER ${backend} upper_backend)
    string(TOLOWER ${backend} lower_backend)
    if (FLY_BUILD_${upper_backend})
        install(EXPORT flare${backend}Targets
                NAMESPACE flare::
                DESTINATION ${FLY_INSTALL_CMAKE_DIR}
                COMPONENT ${lower_backend}_dev)

        export(EXPORT flare${backend}Targets
                NAMESPACE flare::
                FILE cmake/flare${backend}Targets.cmake)
    endif ()
endforeach ()

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
        "${flare_BINARY_DIR}/flareConfigVersion.cmake"
        COMPATIBILITY SameMajorVersion
)

# This config file will be installed so we need to set the install_destination
# path relitive to the install path
set(INCLUDE_DIRS include)
set(CMAKE_DIR ${FLY_INSTALL_CMAKE_DIR})
configure_package_config_file(
        ${flare_SOURCE_DIR}/CMakeModules/flareConfig.cmake.in
        cmake/install/flareConfig.cmake
        INSTALL_DESTINATION "${FLY_INSTALL_CMAKE_DIR}"
        PATH_VARS INCLUDE_DIRS CMAKE_DIR
)

install(FILES ${flare_BINARY_DIR}/cmake/install/flareConfig.cmake
        ${flare_BINARY_DIR}/flareConfigVersion.cmake
        DESTINATION ${FLY_INSTALL_CMAKE_DIR}
        COMPONENT cmake)

if (BUILD_WITH_MKL AND FLY_INSTALL_STANDALONE)
    if (TARGET MKL::ThreadingLibrary)
        get_filename_component(mkl_tl ${MKL_ThreadingLibrary_LINK_LIBRARY} REALPATH)
        install(FILES
                $<TARGET_FILE:MKL::ThreadingLibrary>
                ${mkl_tl}
                DESTINATION ${FLY_INSTALL_LIB_DIR}
                COMPONENT mkl_dependencies)
    endif ()

    if (NOT FLY_WITH_STATIC_MKL AND TARGET MKL::Shared)
        if (NOT WIN32)
            get_filename_component(mkl_int ${MKL_Interface_LINK_LIBRARY} REALPATH)
            install(FILES
                    $<TARGET_FILE:MKL::Interface>
                    ${mkl_int}
                    DESTINATION ${FLY_INSTALL_LIB_DIR}
                    COMPONENT mkl_dependencies)
        endif ()

        get_filename_component(mkl_rnt ${MKL_RT_LINK_LIBRARY} REALPATH)
        get_filename_component(mkl_shd ${MKL_Core_LINK_LIBRARY} REALPATH)
        get_filename_component(mkl_tly ${MKL_ThreadLayer_LINK_LIBRARY} REALPATH)
        install(FILES
                ${mkl_rnt}
                ${mkl_shd}
                ${mkl_tly}
                $<TARGET_FILE:MKL::RT>
                $<TARGET_FILE:MKL::Shared>
                $<TARGET_FILE:MKL::ThreadLayer>
                ${MKL_RUNTIME_KERNEL_LIBRARIES}

                # This variable is used to add tbb.so.2 library because the main lib
                # is a linker script and not a symlink so it cant be resolved using
                # get_filename_component
                ${FLY_ADDITIONAL_MKL_LIBRARIES}
                DESTINATION ${FLY_INSTALL_LIB_DIR}
                COMPONENT mkl_dependencies)
    endif ()
endif ()

# This file will be used to create the config file for the build directory.
# These config files will be used by the examples to find the Flare
# libraries
set(INCLUDE_DIRS "${flare_SOURCE_DIR}/include" "${flare_BINARY_DIR}/include")
set(CMAKE_DIR "${flare_BINARY_DIR}/cmake")
configure_package_config_file(
        ${flare_SOURCE_DIR}/CMakeModules/flareConfig.cmake.in
        flareConfig.cmake
        INSTALL_DESTINATION "${flare_BINARY_DIR}"
        PATH_VARS INCLUDE_DIRS CMAKE_DIR
        INSTALL_PREFIX "${flare_BINARY_DIR}"
)

# Registers the current build directory with the user's cmake config. This will
# create a file at $HOME/.cmake/packages/Flare which will point to this source
# build directory.
# TODO(umar): Disable for now. Causing issues with builds on windows.
#export(PACKAGE Flare)

# Unset the visibility to avoid setting policy commands for older versions of
# CMake for examples and tests.
unset(CMAKE_CXX_VISIBILITY_PRESET)

configure_file(
        ${flare_SOURCE_DIR}/CMakeModules/CTestCustom.cmake
        ${flare_BINARY_DIR}/CTestCustom.cmake)

include(CTest)

# Handle depricated BUILD_TEST variable if found.
if (BUILD_TEST)
    set(BUILD_TESTING ${BUILD_TEST})
endif ()

conditional_directory(BUILD_TESTING test)

conditional_directory(FLY_BUILD_EXAMPLES examples)
include(flare_package_config)
include(carbin_cpack)
