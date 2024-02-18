# Copyright (c) 2018, Flare
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

cmake_minimum_required(VERSION 3.5)
project(Flare-Example-Getting-Started
  VERSION 3.5.0
  LANGUAGES CXX)

find_package(Flare)

if(Flare_CPU_FOUND)
  # Convolve examples
  add_executable(convolve_cpu convolve.cpp)
  target_link_libraries(convolve_cpu Flare::flycpu)

  # Integer examples
  add_executable(integer_cpu integer.cpp)
  target_link_libraries(integer_cpu Flare::flycpu)

  # Rainfall examples
  add_executable(rainfall_cpu rainfall.cpp)
  target_link_libraries(rainfall_cpu Flare::flycpu)

  # Vectorization examples
  add_executable(vectorize_cpu vectorize.cpp)
  target_link_libraries(vectorize_cpu Flare::flycpu)
endif()

if(Flare_CUDA_FOUND)
  add_executable(convolve_cuda convolve.cpp)
  target_link_libraries(convolve_cuda Flare::flycuda)

  add_executable(integer_cuda integer.cpp)
  target_link_libraries(integer_cuda Flare::flycuda)

  add_executable(rainfall_cuda rainfall.cpp)
  target_link_libraries(rainfall_cuda Flare::flycuda)

  add_executable(vectorize_cuda vectorize.cpp)
  target_link_libraries(vectorize_cuda Flare::flycuda)
endif()

if(Flare_OpenCL_FOUND)
  add_executable(convolve_opencl convolve.cpp)
  target_link_libraries(convolve_opencl Flare::flyopencl)

  add_executable(integer_opencl integer.cpp)
  target_link_libraries(integer_opencl Flare::flyopencl)

  add_executable(rainfall_opencl rainfall.cpp)
  target_link_libraries(rainfall_opencl Flare::flyopencl)

  add_executable(vectorize_opencl vectorize.cpp)
  target_link_libraries(vectorize_opencl Flare::flyopencl)
endif()

if(Flare_oneAPI_FOUND)
  add_executable(convolve_oneapi convolve.cpp)
  target_link_libraries(convolve_oneapi Flare::flyoneapi)

  add_executable(integer_oneapi integer.cpp)
  target_link_libraries(integer_oneapi Flare::flyoneapi)

  add_executable(rainfall_oneapi rainfall.cpp)
  target_link_libraries(rainfall_oneapi Flare::flyoneapi)

  add_executable(vectorize_oneapi vectorize.cpp)
  target_link_libraries(vectorize_oneapi Flare::flyoneapi)
endif()
