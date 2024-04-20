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

set(FLY_TEST_WITH_MTX_FILES
        ON CACHE BOOL
        "Download and run tests on large matrices form sparse.tamu.edu")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")

if (FLY_CTEST_SEPARATED)
    include(GoogleTest)
endif ()

if (FLY_TEST_WITH_MTX_FILES)
    include(download_sparse_datasets)
endif ()
file(COPY ${PROJECT_SOURCE_DIR}/3rd/matrixmarket DESTINATION ${CMAKE_BINARY_DIR}/extern)

if (WIN32)
    set(gtest_force_shared_crt ON
            CACHE INTERNAL "Required so that the libs Runtime is not set to MT DLL")
    set(BUILD_SHARED_LIBS OFF)
endif ()

if (NOT TARGET mmio)
    add_subdirectory(mmio)
endif ()


# Registers test with ctest
#
# Parameters
#  target: The target associated with this test
#  backend: The backend associated with this test
#  is_serial: If true the test will be serialized
function(fly_add_test target backend is_serial)
    if (FLY_CTEST_SEPARATED)
        gtest_discover_tests(${target}
                TEST_PREFIX $<UPPER_CASE:${backend}>.
                DISCOVERY_TIMEOUT 40)
    else ()
        add_test(NAME ${target} COMMAND ${target})
        if (${is_serial})
            set_tests_properties(${target}
                    PROPERTIES
                    ENVIRONMENT FLY_PRINT_ERRORS=1
                    TIMEOUT 900
                    RUN_SERIAL ON)
        endif (${is_serial})
    endif ()
endfunction()

# Reset the CXX flags for tests
set(CMAKE_CXX_STANDARD 11)

set(TESTDATA_SOURCE_DIR "${PROJECT_SOURCE_DIR}/test/test-data")

if (FLY_BUILD_CPU)
    list(APPEND enabled_backends "cpu")
endif (FLY_BUILD_CPU)

if (FLY_BUILD_CUDA)
    list(APPEND enabled_backends "cuda")
endif (FLY_BUILD_CUDA)

if (FLY_BUILD_UNIFIED)
    list(APPEND enabled_backends "unified")
endif (FLY_BUILD_UNIFIED)

add_library(flare_test STATIC
        testHelpers.hpp
        flare_test.cpp)

target_include_directories(flare_test
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${flare_SOURCE_DIR}/include
        ${flare_BINARY_DIR}/include)

target_include_directories(flare_test
        SYSTEM PRIVATE
        ${flare_SOURCE_DIR}/extern/half/include
)

# The tautological-constant-compare warning is always thrown for std::nan
# and std::info calls. Its unnecessarily verbose.
target_compile_options(flare_test
        PUBLIC
        # Intel compilers use fast math by default and ignore special floating point
        # values like NaN and Infs.
        $<$<COMPILE_LANGUAGE:CXX>:
        $<$<BOOL:${has_cxx_fp_model}>:-fp-model precise>
        $<$<BOOL:${has_cxx_unqualified_std_cast_call}>:-Wno-unqualified-std-cast-call>>
        PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>: /bigobj
        /EHsc>
)
if (WIN32)
    target_compile_definitions(flare_test
            PRIVATE
            WIN32_LEAN_AND_MEAN
            NOMINMAX)
endif ()

target_compile_definitions(flare_test
        PUBLIC
        $<$<BOOL:${FLY_WITH_FAST_MATH}>:FLY_WITH_FAST_MATH>
        PRIVATE
        TEST_RESULT_IMAGE_DIR="${CMAKE_BINARY_DIR}/test/"
        USE_MTX)

target_link_libraries(flare_test
        PRIVATE
        mmio
        PUBLIC
        GTest::gtest
        Boost::boost
)

# Creates tests for all backends
#
# Creates a standard test for all backends. Most of the time you only need to
# specify the name of the source file to create a test.
#
# Parameters
# ----------
# 'CXX11'       If set the tests will be compiled using c++11. Tests should strive
#               to be C++98 compliant
# 'SRC'         The source files for the test
# 'LIBRARIES'   Libraries other than Flare that need to be linked
# 'DEFINITIONS' Definitions that need to be defined
# 'BACKENDS'    Backends to target for this test. If not set then the test will
#               compiled againat all backends
function(make_test)
    set(options CXX11 SERIAL USE_MMIO NO_FLARE_TEST)
    set(single_args SRC)
    set(multi_args LIBRARIES DEFINITIONS BACKENDS)
    cmake_parse_arguments(mt_args "${options}" "${single_args}" "${multi_args}" ${ARGN})

    get_filename_component(src_name ${mt_args_SRC} NAME_WE)
    foreach (backend ${enabled_backends})
        if (NOT "${mt_args_BACKENDS}" STREQUAL "" AND
                NOT ${backend} IN_LIST mt_args_BACKENDS)
            continue()
        endif ()
        set(target "test_${src_name}_${backend}")

        add_executable(${target} ${mt_args_SRC})
        target_include_directories(${target}
                PRIVATE
                ${CMAKE_SOURCE_DIR}
                ${CMAKE_CURRENT_SOURCE_DIR})
        target_include_directories(${target}
                SYSTEM PRIVATE
                ${flare_SOURCE_DIR}/extern/half/include
        )
        target_link_libraries(${target}
                PRIVATE
                ${mt_args_LIBRARIES}
                flare_test
        )

        target_compile_options(${target}
                PRIVATE
                $<$<CXX_COMPILER_ID:MSVC>: /bigobj
                /EHsc>
        )

        if (${backend} STREQUAL "unified")
            target_link_libraries(${target}
                    PRIVATE
                    fly)
        else ()
            target_link_libraries(${target}
                    PRIVATE
                    fly${backend}
            )
        endif ()

        if (${mt_args_CXX11})
            set_target_properties(${target}
                    PROPERTIES
                    CXX_STANDARD 11)
        endif (${mt_args_CXX11})

        set_target_properties(${target}
                PROPERTIES
                FOLDER "Tests"
                OUTPUT_NAME "${src_name}_${backend}")

        target_compile_definitions(${target}
                PRIVATE
                TEST_DIR="${TESTDATA_SOURCE_DIR}"
                FLY_$<UPPER_CASE:${backend}>
                ${mt_args_DEFINITIONS}
        )
        target_link_libraries(${target} PRIVATE mmio)
        if (FLY_TEST_WITH_MTX_FILES AND ${mt_args_USE_MMIO})
            target_compile_definitions(${target}
                    PRIVATE
                    MTX_TEST_DIR="${flare_BINARY_DIR}/extern/matrixmarket/"
            )
        endif ()
        if (WIN32)
            target_compile_definitions(${target}
                    PRIVATE
                    WIN32_LEAN_AND_MEAN
                    NOMINMAX)
        endif ()

        # TODO(umar): Create this executable separately
        if (NOT ${backend} STREQUAL "unified" OR ${target} STREQUAL "backend_unified")
            fly_add_test(${target} ${backend} ${mt_args_SERIAL})
        endif ()

    endforeach ()
endfunction(make_test)

make_test(SRC anisotropic_diffusion.cpp)
make_test(SRC approx1.cpp)
make_test(SRC approx2.cpp)
make_test(SRC array.cpp CXX11)
make_test(SRC array_death_tests.cpp CXX11 SERIAL)
make_test(SRC arrayio.cpp)
make_test(SRC assign.cpp CXX11)
make_test(SRC backend.cpp CXX11)
make_test(SRC basic.cpp)
make_test(SRC bilateral.cpp)
make_test(SRC binary.cpp CXX11)
make_test(SRC blas.cpp)
make_test(SRC canny.cpp)
make_test(SRC cast.cpp)
make_test(SRC cholesky_dense.cpp SERIAL)
make_test(SRC clamp.cpp)
make_test(SRC compare.cpp)
make_test(SRC complex.cpp)
make_test(SRC confidence_connected.cpp CXX11)
make_test(SRC constant.cpp)
make_test(SRC convolve.cpp CXX11)
make_test(SRC corrcoef.cpp)
make_test(SRC covariance.cpp)
make_test(SRC diagonal.cpp)
make_test(SRC diff1.cpp)
make_test(SRC diff2.cpp)
make_test(SRC dog.cpp)
make_test(SRC dot.cpp)
make_test(SRC empty.cpp)
make_test(SRC event.cpp CXX11)
make_test(SRC fast.cpp)
make_test(SRC fft.cpp)
make_test(SRC fft_large.cpp)
make_test(SRC fft_real.cpp)
make_test(SRC fftconvolve.cpp)
make_test(SRC flat.cpp)
make_test(SRC flip.cpp)
make_test(SRC gaussiankernel.cpp)
make_test(SRC gen_assign.cpp)
make_test(SRC gen_index.cpp CXX11)
make_test(SRC getting_started.cpp)
make_test(SRC gfor.cpp)
make_test(SRC gradient.cpp)
make_test(SRC gray_rgb.cpp)
make_test(SRC half.cpp)
make_test(SRC hamming.cpp)
make_test(SRC harris.cpp)
make_test(SRC histogram.cpp)
make_test(SRC homography.cpp)
make_test(SRC hsv_rgb.cpp)
make_test(SRC iir.cpp)
make_test(SRC imageio.cpp)
make_test(SRC index.cpp CXX11)
make_test(SRC info.cpp)
make_test(SRC internal.cpp)
make_test(SRC inverse_deconv.cpp)
make_test(SRC inverse_dense.cpp SERIAL)
make_test(SRC iota.cpp)
make_test(SRC ireduce.cpp)
make_test(SRC iterative_deconv.cpp)
make_test(SRC jit.cpp CXX11)
make_test(SRC join.cpp)
make_test(SRC lu_dense.cpp SERIAL)
#make_test(manual_memory_test.cpp)
make_test(SRC match_template.cpp)
make_test(SRC math.cpp CXX11)
make_test(SRC matrix_manipulation.cpp)
make_test(SRC mean.cpp)
make_test(SRC meanshift.cpp)
make_test(SRC meanvar.cpp CXX11)
make_test(SRC medfilt.cpp)
make_test(SRC median.cpp)
make_test(SRC memory.cpp CXX11)
make_test(SRC memory_lock.cpp)
make_test(SRC missing.cpp)
make_test(SRC moddims.cpp)
make_test(SRC moments.cpp)
make_test(SRC morph.cpp)
make_test(SRC nearest_neighbour.cpp CXX11)
make_test(SRC nodevice.cpp CXX11)

if (CUDA_FOUND)
    include(fly_cuda_helpers)
    foreach (backend ${enabled_backends})
        set(cuda_test_backends "cuda" "unified")
        if (${backend} IN_LIST cuda_test_backends)
            set(target test_cuda_${backend})
            add_executable(${target} cuda.cu)
            target_include_directories(${target}
                    PRIVATE
                    ${CMAKE_SOURCE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR})
            target_include_directories(${target}
                    SYSTEM PRIVATE
                    ${flare_SOURCE_DIR}/extern/half/include)
            if (${backend} STREQUAL "unified")
                target_link_libraries(${target}
                        flare::fly)
            else ()
                target_link_libraries(${target}
                        flare::fly${backend})
            endif ()
            target_link_libraries(${target}
                    mmio
                    flare_test)

            # Couldn't get Threads::Threads to work with this cuda binary. The import
            # target would not add the -pthread flag which is required for this
            # executable (on Ubuntu 18.04 anyway)
            check_cxx_compiler_flag(-pthread pthread_flag)
            if (pthread_flag)
                target_link_libraries(${target} -pthread)
            endif ()

            fly_detect_and_set_cuda_architectures(${target})

            set_target_properties(${target}
                    PROPERTIES
                    FOLDER "Tests"
                    OUTPUT_NAME "cuda_${backend}")

            if (NOT ${backend} STREQUAL "unified")
                fly_add_test(${target} ${backend} ON)
            endif ()
        endif ()
    endforeach ()
endif ()


make_test(SRC orb.cpp)
make_test(SRC pad_borders.cpp CXX11)
make_test(SRC pinverse.cpp SERIAL)
make_test(SRC qr_dense.cpp SERIAL)
make_test(SRC random.cpp)
make_test(SRC rng_quality.cpp BACKENDS "cuda" SERIAL)
make_test(SRC range.cpp)
make_test(SRC rank_dense.cpp SERIAL)
make_test(SRC reduce.cpp CXX11)
make_test(SRC regions.cpp)
make_test(SRC reorder.cpp)
make_test(SRC replace.cpp CXX11)
make_test(SRC resize.cpp)
make_test(SRC rng_match.cpp CXX11 BACKENDS "unified")
make_test(SRC rotate.cpp)
make_test(SRC rotate_linear.cpp)
make_test(SRC sat.cpp)
make_test(SRC scan.cpp)
make_test(SRC scan_by_key.cpp)
make_test(SRC select.cpp CXX11)
make_test(SRC set.cpp CXX11)
make_test(SRC shift.cpp)
make_test(SRC gloh.cpp)
make_test(SRC sift.cpp)
make_test(SRC sobel.cpp)
make_test(SRC solve_dense.cpp CXX11 SERIAL)
make_test(SRC sort.cpp)
make_test(SRC sort_by_key.cpp)
make_test(SRC sort_index.cpp)
make_test(SRC sparse.cpp SERIAL)
make_test(SRC sparse_arith.cpp USE_MMIO)
make_test(SRC sparse_convert.cpp)
make_test(SRC stdev.cpp)
make_test(SRC susan.cpp)
make_test(SRC svd_dense.cpp SERIAL)
make_test(SRC threading.cpp CXX11 SERIAL)
make_test(SRC tile.cpp)
make_test(SRC topk.cpp CXX11)
make_test(SRC transform.cpp)
make_test(SRC transform_coordinates.cpp)
make_test(SRC translate.cpp)
make_test(SRC transpose.cpp)
make_test(SRC transpose_inplace.cpp)
make_test(SRC triangle.cpp)
make_test(SRC unwrap.cpp)
make_test(SRC var.cpp)
make_test(SRC where.cpp)
make_test(SRC wrap.cpp)
make_test(SRC write.cpp)
make_test(SRC ycbcr_rgb.cpp)

foreach (backend ${enabled_backends})
    set(target "basic_c_${backend}")
    add_executable(${target} basic_c.c)
    if (${backend} STREQUAL "unified")
        target_link_libraries(${target}
                PRIVATE
                flare::fly)
    else ()
        target_link_libraries(${target}
                PRIVATE
                flare::fly${backend})
    endif ()
    add_test(NAME ${target} COMMAND ${target})
endforeach ()

if (FLY_TEST_WITH_MTX_FILES)
    make_test(SRC matrixmarket.cpp USE_MMIO)
endif ()

add_executable(print_info print_info.cpp)
if (FLY_BUILD_UNIFIED)
    target_link_libraries(print_info flare::fly)
elseif (FLY_BUILD_CUDA)
    target_link_libraries(print_info flare::flycuda)
elseif (FLY_BUILD_CPU)
    target_link_libraries(print_info flare::flycpu)
endif ()

make_test(SRC jit_test_api.cpp)
