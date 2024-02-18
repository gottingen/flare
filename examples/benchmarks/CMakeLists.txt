# Copyright (c) 2018, Flare
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

cmake_minimum_required(VERSION 3.5)
project(Flare-Example-Benchmarks
  VERSION 3.5.0
  LANGUAGES CXX)

find_package(Flare)

if(Flare_CPU_FOUND)
  add_executable(blas_cpu blas.cpp)
  target_link_libraries(blas_cpu Flare::flycpu)

  add_executable(cg_cpu cg.cpp)
  target_link_libraries(cg_cpu Flare::flycpu)

  add_executable(fft_cpu fft.cpp)
  target_link_libraries(fft_cpu Flare::flycpu)

  add_executable(pi_cpu pi.cpp)
  target_link_libraries(pi_cpu Flare::flycpu)
endif()

if(Flare_CUDA_FOUND)
  add_executable(blas_cuda blas.cpp)
  target_link_libraries(blas_cuda Flare::flycuda)

  add_executable(cg_cuda cg.cpp)
  target_link_libraries(cg_cuda Flare::flycuda)

  add_executable(fft_cuda fft.cpp)
  target_link_libraries(fft_cuda Flare::flycuda)

  add_executable(pi_cuda pi.cpp)
  target_link_libraries(pi_cuda Flare::flycuda)
endif()

if(Flare_OpenCL_FOUND)
  add_executable(blas_opencl blas.cpp)
  target_link_libraries(blas_opencl Flare::flyopencl)

  add_executable(cg_opencl cg.cpp)
  target_link_libraries(cg_opencl Flare::flyopencl)

  add_executable(fft_opencl fft.cpp)
  target_link_libraries(fft_opencl Flare::flyopencl)

  add_executable(pi_opencl pi.cpp)
  target_link_libraries(pi_opencl Flare::flyopencl)
endif()

if(Flare_oneAPI_FOUND)
  add_executable(blas_oneapi blas.cpp)
  target_link_libraries(blas_oneapi Flare::flyoneapi)

  add_executable(cg_oneapi cg.cpp)
  target_link_libraries(cg_oneapi Flare::flyoneapi)

  add_executable(fft_oneapi fft.cpp)
  target_link_libraries(fft_oneapi Flare::flyoneapi)

  add_executable(pi_oneapi pi.cpp)
  target_link_libraries(pi_oneapi Flare::flyoneapi)
endif()
