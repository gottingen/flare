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

include(InternalUtils)

generate_product_version(fly_cpu_ver_res_file
        FILE_NAME "flycpu"
        FILE_DESCRIPTION "CPU Backend Dynamic-link library"
)

add_library(flycpu "")
add_library(flare::flycpu ALIAS flycpu)

# CPU backend source files
target_sources(flycpu
        PRIVATE
        $<$<PLATFORM_ID:Windows>:${fly_cpu_ver_res_file}>
        Array.cpp
        Array.hpp
        anisotropic_diffusion.cpp
        anisotropic_diffusion.hpp
        approx.cpp
        approx.hpp
        arith.hpp
        assign.cpp
        assign.hpp
        backend.hpp
        binary.hpp
        bilateral.cpp
        bilateral.hpp
        blas.cpp
        blas.hpp
        canny.cpp
        canny.hpp
        cast.hpp
        cholesky.cpp
        cholesky.hpp
        complex.hpp
        convolve.cpp
        convolve.hpp
        copy.cpp
        copy.hpp
        device_manager.cpp
        device_manager.hpp
        diagonal.cpp
        diagonal.hpp
        diff.cpp
        diff.hpp
        err_cpu.hpp
        Event.cpp
        Event.hpp
        exampleFunction.cpp
        exampleFunction.hpp
        fast.cpp
        fast.hpp
        fft.cpp
        fft.hpp
        fftconvolve.cpp
        fftconvolve.hpp
        flood_fill.hpp
        flood_fill.cpp
        gradient.cpp
        gradient.hpp
        harris.cpp
        harris.hpp
        hist_graphics.cpp
        hist_graphics.hpp
        histogram.cpp
        histogram.hpp
        homography.cpp
        homography.hpp
        hsv_rgb.cpp
        hsv_rgb.hpp
        identity.cpp
        identity.hpp
        iir.cpp
        iir.hpp
        image.cpp
        image.hpp
        index.cpp
        index.hpp
        inverse.cpp
        inverse.hpp
        iota.cpp
        iota.hpp
        ireduce.cpp
        ireduce.hpp
        join.cpp
        join.hpp
        lapack_helper.hpp
        logic.hpp
        lookup.cpp
        lookup.hpp
        lu.cpp
        lu.hpp
        match_template.cpp
        match_template.hpp
        math.cpp
        math.hpp
        mean.cpp
        mean.hpp
        meanshift.cpp
        meanshift.hpp
        medfilt.cpp
        medfilt.hpp
        memory.cpp
        memory.hpp
        moments.cpp
        moments.hpp
        morph.cpp
        morph.hpp
        nearest_neighbour.cpp
        nearest_neighbour.hpp
        orb.cpp
        orb.hpp
        ParamIterator.hpp
        platform.cpp
        platform.hpp
        plot.cpp
        plot.hpp
        print.hpp
        qr.cpp
        qr.hpp
        queue.hpp
        random_engine.cpp
        random_engine.hpp
        range.cpp
        range.hpp
        reduce.cpp
        reduce.hpp
        regions.cpp
        regions.hpp
        reorder.cpp
        reorder.hpp
        resize.cpp
        resize.hpp
        reshape.cpp
        rotate.cpp
        rotate.hpp
        scan.cpp
        scan.hpp
        scan_by_key.cpp
        scan_by_key.hpp
        select.cpp
        select.hpp
        set.cpp
        set.hpp
        shift.cpp
        shift.hpp
        sift.cpp
        sift.hpp
        sobel.cpp
        sobel.hpp
        solve.cpp
        solve.hpp
        sort.cpp
        sort.hpp
        sort_by_key.cpp
        sort_by_key.hpp
        sort_index.cpp
        sort_index.hpp
        sparse.cpp
        sparse.hpp
        sparse_arith.cpp
        sparse_arith.hpp
        sparse_blas.cpp
        sparse_blas.hpp
        surface.cpp
        surface.hpp
        susan.cpp
        susan.hpp
        svd.cpp
        svd.hpp
        tile.cpp
        tile.hpp
        topk.cpp
        topk.hpp
        traits.hpp
        transform.cpp
        transform.hpp
        transpose.cpp
        transpose.hpp
        triangle.cpp
        triangle.hpp
        types.hpp
        unary.hpp
        unwrap.cpp
        unwrap.hpp
        utility.hpp
        vector_field.cpp
        vector_field.hpp
        where.cpp
        where.hpp
        wrap.cpp
        wrap.hpp
)

# CPU backend kernel files
target_sources(flycpu
        PRIVATE
        kernel/Array.hpp
        kernel/anisotropic_diffusion.hpp
        kernel/approx.hpp
        kernel/assign.hpp
        kernel/bilateral.hpp
        kernel/canny.hpp
        kernel/convolve.hpp
        kernel/copy.hpp
        kernel/diagonal.hpp
        kernel/diff.hpp
        kernel/dot.hpp
        kernel/exampleFunction.hpp
        kernel/fast.hpp
        kernel/fftconvolve.hpp
        kernel/flood_fill.hpp
        kernel/gradient.hpp
        kernel/harris.hpp
        kernel/histogram.hpp
        kernel/hsv_rgb.hpp
        kernel/identity.hpp
        kernel/iir.hpp
        kernel/index.hpp
        kernel/interp.hpp
        kernel/iota.hpp
        kernel/ireduce.hpp
        kernel/join.hpp
        kernel/lookup.hpp
        kernel/lu.hpp
        kernel/match_template.hpp
        kernel/meanshift.hpp
        kernel/medfilt.hpp
        kernel/moments.hpp
        kernel/morph.hpp
        kernel/nearest_neighbour.hpp
        kernel/orb.hpp
        kernel/pad_array_borders.hpp
        kernel/random_engine.hpp
        kernel/random_engine_mersenne.hpp
        kernel/random_engine_philox.hpp
        kernel/random_engine_threefry.hpp
        kernel/range.hpp
        kernel/reduce.hpp
        kernel/regions.hpp
        kernel/reorder.hpp
        kernel/resize.hpp
        kernel/rotate.hpp
        kernel/scan.hpp
        kernel/scan_by_key.hpp
        kernel/select.hpp
        kernel/shift.hpp
        kernel/sift.hpp
        kernel/sobel.hpp
        kernel/sort.hpp
        kernel/sort_by_key.hpp
        kernel/sort_helper.hpp
        kernel/sparse.hpp
        kernel/sparse_arith.hpp
        kernel/susan.hpp
        kernel/tile.hpp
        kernel/transform.hpp
        kernel/transpose.hpp
        kernel/triangle.hpp
        kernel/unwrap.hpp
        kernel/wrap.hpp
)

if (FLY_WITH_CPUID)
    target_compile_definitions(flycpu PRIVATE -DFLY_WITH_CPUID)
endif (FLY_WITH_CPUID)

target_sources(flycpu
        PRIVATE
        ${PROJECT_SOURCE_DIR}/include/fly/threads/async_queue.hpp
        ${PROJECT_SOURCE_DIR}/include/fly/threads/event.hpp
)

include("${CMAKE_CURRENT_SOURCE_DIR}/kernel/sort_by_key/CMakeLists.txt")

target_include_directories(flycpu
        PUBLIC
        $<BUILD_INTERFACE:${flare_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${flare_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${FLY_INSTALL_INC_DIR}>
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})

target_include_directories(flycpu
        SYSTEM PRIVATE
        ${CBLAS_INCLUDE_DIR})

target_compile_definitions(flycpu
        PRIVATE
        FLY_CPU
)

target_link_libraries(flycpu
        PRIVATE
        c_api_interface
        cpp_api_interface
        flycommon_interface
        cpu_sort_by_key
        Threads::Threads
)
if (BUILD_WITH_MKL)
    target_compile_definitions(flycpu PRIVATE USE_MKL)

    if (MKL_BATCH)
        target_compile_definitions(flycpu PRIVATE FLY_USE_MKL_BATCH)
    endif ()

    if (FLY_WITH_STATIC_MKL)
        target_link_libraries(flycpu PRIVATE MKL::Static)
        target_compile_definitions(flycpu PRIVATE USE_STATIC_MKL)
    else ()
        target_link_libraries(flycpu PRIVATE MKL::RT)
    endif ()
else ()
    target_link_libraries(flycpu
            PRIVATE
            ${CBLAS_LIBRARIES}
            FFTW::FFTW
            FFTW::FFTWF
    )
    if (LAPACK_FOUND AND LAPACKE_FOUND)
        target_link_libraries(flycpu PRIVATE LAPACKE::LAPACKE ${LAPACK_LIBRARIES})
    endif ()
endif ()

if (LAPACK_FOUND OR BUILD_WITH_MKL)
    target_compile_definitions(flycpu PRIVATE WITH_LINEAR_ALGEBRA)
endif ()

fly_split_debug_info(flycpu ${FLY_INSTALL_LIB_DIR})

install(TARGETS flycpu
        EXPORT flareCPUTargets
        COMPONENT cpu
        PUBLIC_HEADER DESTINATION fly
        RUNTIME DESTINATION ${FLY_INSTALL_BIN_DIR}
        LIBRARY DESTINATION ${FLY_INSTALL_LIB_DIR}
        ARCHIVE DESTINATION ${FLY_INSTALL_LIB_DIR}
        FRAMEWORK DESTINATION framework
        INCLUDES DESTINATION ${FLY_INSTALL_INC_DIR}
)

source_group(include REGULAR_EXPRESSION ${flare_SOURCE_DIR}/include/*)
source_group(api\\cpp REGULAR_EXPRESSION ${flare_SOURCE_DIR}/src/api/cpp/*)
source_group(api\\c REGULAR_EXPRESSION ${flare_SOURCE_DIR}/src/api/c/*)
source_group(backend REGULAR_EXPRESSION ${flare_SOURCE_DIR}/src/backend/common/*|${CMAKE_CURRENT_SOURCE_DIR}/*)
source_group(backend\\kernel REGULAR_EXPRESSION ${CMAKE_CURRENT_SOURCE_DIR}/kernel/*)
source_group("" FILES CMakeLists.txt)
