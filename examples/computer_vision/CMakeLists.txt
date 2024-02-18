# Copyright (c) 2018, Flare
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

cmake_minimum_required(VERSION 3.5)
project(Flare-Example-Computer-Vision
  VERSION 3.5.0
  LANGUAGES CXX)

find_package(Flare)

add_definitions("-DASSETS_DIR=\"${ASSETS_DIR}\"")

if (Flare_CPU_FOUND)
  # FAST examples
  add_executable(fast_cpu fast.cpp)
  target_link_libraries(fast_cpu Flare::flycpu)

  # Harris corner detector examples
  add_executable(harris_cpu harris.cpp)
  target_link_libraries(harris_cpu Flare::flycpu)

  # Template Matching examples
  add_executable(matching_cpu matching.cpp)
  target_link_libraries(matching_cpu Flare::flycpu)

  # Template Matching examples
  add_executable(susan_cpu susan.cpp)
  target_link_libraries(susan_cpu Flare::flycpu)
endif()

if (Flare_CUDA_FOUND)
  add_executable(fast_cuda fast.cpp)
  target_link_libraries(fast_cuda Flare::flycuda)

  add_executable(harris_cuda harris.cpp)
  target_link_libraries(harris_cuda Flare::flycuda)

  add_executable(matching_cuda matching.cpp)
  target_link_libraries(matching_cuda Flare::flycuda)

  add_executable(susan_cuda susan.cpp)
  target_link_libraries(susan_cuda Flare::flycuda)
endif()

if (Flare_OpenCL_FOUND)
  add_executable(fast_opencl fast.cpp)
  target_link_libraries(fast_opencl Flare::flyopencl)

  add_executable(harris_opencl harris.cpp)
  target_link_libraries(harris_opencl Flare::flyopencl)

  add_executable(matching_opencl matching.cpp)
  target_link_libraries(matching_opencl Flare::flyopencl)

  add_executable(susan_opencl susan.cpp)
  target_link_libraries(susan_opencl Flare::flyopencl)
endif()

if (Flare_oneAPI_FOUND)
  add_executable(fast_oneapi fast.cpp)
  target_link_libraries(fast_oneapi Flare::flyoneapi)

  add_executable(harris_oneapi harris.cpp)
  target_link_libraries(harris_oneapi Flare::flyoneapi)

  add_executable(matching_oneapi matching.cpp)
  target_link_libraries(matching_oneapi Flare::flyoneapi)

  add_executable(susan_oneapi susan.cpp)
  target_link_libraries(susan_oneapi Flare::flyoneapi)
endif()
