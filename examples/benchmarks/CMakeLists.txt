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

cmake_minimum_required(VERSION 3.5)
project(Flare-Example-Benchmarks
  VERSION 3.5.0
  LANGUAGES CXX)

find_package(flare)

if(flare_CPU_FOUND)
  add_executable(blas_cpu blas.cpp)
  target_link_libraries(blas_cpu flare::flycpu)

  add_executable(cg_cpu cg.cpp)
  target_link_libraries(cg_cpu flare::flycpu)

  add_executable(fft_cpu fft.cpp)
  target_link_libraries(fft_cpu flare::flycpu)

  add_executable(pi_cpu pi.cpp)
  target_link_libraries(pi_cpu flare::flycpu)
endif()

if(flare_CUDA_FOUND)
  add_executable(blas_cuda blas.cpp)
  target_link_libraries(blas_cuda flare::flycuda)

  add_executable(cg_cuda cg.cpp)
  target_link_libraries(cg_cuda flare::flycuda)

  add_executable(fft_cuda fft.cpp)
  target_link_libraries(fft_cuda flare::flycuda)

  add_executable(pi_cuda pi.cpp)
  target_link_libraries(pi_cuda flare::flycuda)
endif()

if(flare_OpenCL_FOUND)
  add_executable(blas_opencl blas.cpp)
  target_link_libraries(blas_opencl flare::flyopencl)

  add_executable(cg_opencl cg.cpp)
  target_link_libraries(cg_opencl flare::flyopencl)

  add_executable(fft_opencl fft.cpp)
  target_link_libraries(fft_opencl flare::flyopencl)

  add_executable(pi_opencl pi.cpp)
  target_link_libraries(pi_opencl flare::flyopencl)
endif()

if(flare_oneAPI_FOUND)
  add_executable(blas_oneapi blas.cpp)
  target_link_libraries(blas_oneapi flare::flyoneapi)

  add_executable(cg_oneapi cg.cpp)
  target_link_libraries(cg_oneapi flare::flyoneapi)

  add_executable(fft_oneapi fft.cpp)
  target_link_libraries(fft_oneapi flare::flyoneapi)

  add_executable(pi_oneapi pi.cpp)
  target_link_libraries(pi_oneapi flare::flyoneapi)
endif()
