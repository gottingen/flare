# Copyright (c) 2018, Flare
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

cmake_minimum_required(VERSION 3.5)
project(Flare-Example-PDE
  VERSION 3.5.0
  LANGUAGES CXX)

find_package(Flare)

if(Flare_CPU_FOUND)
  # Shallow Water simulation example
  add_executable(swe_cpu swe.cpp)
  target_link_libraries(swe_cpu Flare::flycpu)
endif()

if(Flare_CUDA_FOUND)
  add_executable(swe_cuda swe.cpp)
  target_link_libraries(swe_cuda Flare::flycuda)
endif()

if(Flare_OpenCL_FOUND)
  add_executable(swe_opencl swe.cpp)
  target_link_libraries(swe_opencl Flare::flyopencl)
endif()

if(Flare_oneAPI_FOUND)
  add_executable(swe_oneapi swe.cpp)
  target_link_libraries(swe_oneapi Flare::flyoneapi)
endif()
