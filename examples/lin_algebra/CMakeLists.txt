# Copyright (c) 2018, Flare
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

cmake_minimum_required(VERSION 3.5)
project(Flare-Example-Linear-Algebra
  VERSION 3.5.0
  LANGUAGES CXX)

find_package(Flare)

if(Flare_CPU_FOUND)
  # Cholesky example
  add_executable(cholesky_cpu cholesky.cpp)
  target_link_libraries(cholesky_cpu Flare::flycpu)

  # LU example
  add_executable(lu_cpu lu.cpp)
  target_link_libraries(lu_cpu Flare::flycpu)

  # QR example
  add_executable(qr_cpu qr.cpp)
  target_link_libraries(qr_cpu Flare::flycpu)

  # SVD example
  add_executable(svd_cpu svd.cpp)
  target_link_libraries(svd_cpu Flare::flycpu)
endif()

if(Flare_CUDA_FOUND)
  add_executable(cholesky_cuda cholesky.cpp)
  target_link_libraries(cholesky_cuda Flare::flycuda)

  add_executable(lu_cuda lu.cpp)
  target_link_libraries(lu_cuda Flare::flycuda)

  add_executable(qr_cuda qr.cpp)
  target_link_libraries(qr_cuda Flare::flycuda)

  add_executable(svd_cuda svd.cpp)
  target_link_libraries(svd_cuda Flare::flycuda)
endif()

if(Flare_OpenCL_FOUND)
  add_executable(cholesky_opencl cholesky.cpp)
  target_link_libraries(cholesky_opencl Flare::flyopencl)

  add_executable(lu_opencl lu.cpp)
  target_link_libraries(lu_opencl Flare::flyopencl)

  add_executable(qr_opencl qr.cpp)
  target_link_libraries(qr_opencl Flare::flyopencl)

  add_executable(svd_opencl svd.cpp)
  target_link_libraries(svd_opencl Flare::flyopencl)
endif()

if(Flare_oneAPI_FOUND)
  add_executable(cholesky_oneapi cholesky.cpp)
  target_link_libraries(cholesky_oneapi Flare::flyoneapi)

  add_executable(lu_oneapi lu.cpp)
  target_link_libraries(lu_oneapi Flare::flyoneapi)

  add_executable(qr_oneapi qr.cpp)
  target_link_libraries(qr_oneapi Flare::flyoneapi)

  add_executable(svd_oneapi svd.cpp)
  target_link_libraries(svd_oneapi Flare::flyoneapi)
endif()
