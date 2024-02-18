# Copyright (c) 2018, Flare
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

add_library(c_api_interface INTERFACE)

target_sources(c_api_interface
  INTERFACE
  ${Flare_SOURCE_DIR}/include/flare.h
  ${Flare_SOURCE_DIR}/include/fly/algorithm.h
  ${Flare_SOURCE_DIR}/include/fly/arith.h
  ${Flare_SOURCE_DIR}/include/fly/array.h
  ${Flare_SOURCE_DIR}/include/fly/backend.h
  ${Flare_SOURCE_DIR}/include/fly/blas.h
  ${Flare_SOURCE_DIR}/include/fly/compatible.h
  ${Flare_SOURCE_DIR}/include/fly/complex.h
  ${Flare_SOURCE_DIR}/include/fly/constants.h
  ${Flare_SOURCE_DIR}/include/fly/cuda.h
  ${Flare_SOURCE_DIR}/include/fly/data.h
  ${Flare_SOURCE_DIR}/include/fly/defines.h
  ${Flare_SOURCE_DIR}/include/fly/device.h
  ${Flare_SOURCE_DIR}/include/fly/dim4.hpp
  ${Flare_SOURCE_DIR}/include/fly/event.h
  ${Flare_SOURCE_DIR}/include/fly/exception.h
  ${Flare_SOURCE_DIR}/include/fly/features.h
  ${Flare_SOURCE_DIR}/include/fly/gfor.h
  ${Flare_SOURCE_DIR}/include/fly/graphics.h
  ${Flare_SOURCE_DIR}/include/fly/image.h
  ${Flare_SOURCE_DIR}/include/fly/index.h
  ${Flare_SOURCE_DIR}/include/fly/internal.h
  ${Flare_SOURCE_DIR}/include/fly/lapack.h
  ${Flare_SOURCE_DIR}/include/fly/macros.h
  ${Flare_SOURCE_DIR}/include/fly/ml.h
  ${Flare_SOURCE_DIR}/include/fly/memory.h
  ${Flare_SOURCE_DIR}/include/fly/opencl.h
  ${Flare_SOURCE_DIR}/include/fly/random.h
  ${Flare_SOURCE_DIR}/include/fly/seq.h
  ${Flare_SOURCE_DIR}/include/fly/signal.h
  ${Flare_SOURCE_DIR}/include/fly/sparse.h
  ${Flare_SOURCE_DIR}/include/fly/statistics.h
  ${Flare_SOURCE_DIR}/include/fly/timing.h
  ${Flare_SOURCE_DIR}/include/fly/traits.hpp
  ${Flare_SOURCE_DIR}/include/fly/util.h
  ${Flare_SOURCE_DIR}/include/fly/vision.h
  ${Flare_BINARY_DIR}/include/fly/version.h
  )

target_sources(c_api_interface
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/anisotropic_diffusion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/approx.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/array.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/assign.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bilateral.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/binary.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/blas.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/canny.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cholesky.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/clamp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/colorspace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/complex.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/confidence_connected.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/convolve.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/corrcoef.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/covariance.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/deconvolution.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/det.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/diff.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dog.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/events.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/events.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/error.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/exampleFunction.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/features.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/features.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fft.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fft_common.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fftconvolve.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/filters.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/flip.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gaussian_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gradient.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hamming.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/handle.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/handle.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/harris.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hist.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/histeq.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/histogram.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/homography.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hsv_rgb.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/iir.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/image.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/imageio.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/imageio2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/implicit.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/implicit.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/imgproc_common.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/index.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/internal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/inverse.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/jit_test_api.h
    ${CMAKE_CURRENT_SOURCE_DIR}/jit_test_api.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/join.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/match_template.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mean.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/meanshift.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/median.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/memory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/memoryapi.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moddims.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moments.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/morph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nearest_neighbour.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/norm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/optypes.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/orb.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pinverse.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/print.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/qr.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/random.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rank.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/regions.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reorder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/replace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/resize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rgb_gray.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rotate.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sat.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/select.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/set.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shift.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sift.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sobel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/solve.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sort.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sparse.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sparse_handle.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/stdev.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/stream.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/surface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/susan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/svd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/topk.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/transform.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/transform_coordinates.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/transpose.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/type_util.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/type_util.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unary.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unwrap.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/var.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/vector_field.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/version.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/where.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/window.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/wrap.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ycbcr_rgb.cpp
    )

if(FreeImage_FOUND AND FLY_WITH_IMAGEIO)
  target_compile_definitions(c_api_interface INTERFACE WITH_FREEIMAGE)
  if (FLY_WITH_STATIC_FREEIMAGE)
    target_compile_definitions(c_api_interface INTERFACE FREEIMAGE_STATIC)
    target_link_libraries(c_api_interface INTERFACE FreeImage::FreeImage_STATIC)
  else ()
    target_include_directories(c_api_interface SYSTEM INTERFACE $<TARGET_PROPERTY:FreeImage::FreeImage,INTERFACE_INCLUDE_DIRECTORIES>)
    if (WIN32 AND FLY_INSTALL_STANDALONE)
      install(FILES $<TARGET_FILE:FreeImage::FreeImage>
        DESTINATION ${FLY_INSTALL_BIN_DIR}
        COMPONENT common_backend_dependencies)
    endif ()
  endif ()
endif()

if(BUILD_WITH_MKL)
  target_compile_definitions(c_api_interface
    INTERFACE
      FLY_MKL_INTERFACE_SIZE=${MKL_INTERFACE_INTEGER_SIZE}
    )
  # Create mkl thread layer compile option based on cmake cache variable
  if(MKL_THREAD_LAYER STREQUAL "Sequential")
    target_compile_definitions(c_api_interface INTERFACE FLY_MKL_THREAD_LAYER=0)
  elseif(MKL_THREAD_LAYER STREQUAL "GNU OpenMP")
    target_compile_definitions(c_api_interface INTERFACE FLY_MKL_THREAD_LAYER=1)
  elseif(MKL_THREAD_LAYER STREQUAL "Intel OpenMP")
    target_compile_definitions(c_api_interface INTERFACE FLY_MKL_THREAD_LAYER=2)
  else() #default Intel Thread Layer for Flare
    target_compile_definitions(c_api_interface INTERFACE FLY_MKL_THREAD_LAYER=3)
  endif()
endif()

target_include_directories(c_api_interface
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/backend
    ${CMAKE_SOURCE_DIR}/include
    $<TARGET_PROPERTY:flycommon_interface,INTERFACE_INCLUDE_DIRECTORIES>
    )
