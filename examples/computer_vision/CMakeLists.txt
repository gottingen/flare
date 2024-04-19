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
project(Flare-Example-Computer-Vision
  VERSION 3.5.0
  LANGUAGES CXX)

find_package(flare)

add_definitions("-DASSETS_DIR=\"${ASSETS_DIR}\"")

if (flare_CPU_FOUND)
  # FAST examples
  add_executable(fast_cpu fast.cpp)
  target_link_libraries(fast_cpu flare::flycpu)

  # Harris corner detector examples
  add_executable(harris_cpu harris.cpp)
  target_link_libraries(harris_cpu flare::flycpu)

  # Template Matching examples
  add_executable(matching_cpu matching.cpp)
  target_link_libraries(matching_cpu flare::flycpu)

  # Template Matching examples
  add_executable(susan_cpu susan.cpp)
  target_link_libraries(susan_cpu flare::flycpu)
endif()

if (flare_CUDA_FOUND)
  add_executable(fast_cuda fast.cpp)
  target_link_libraries(fast_cuda flare::flycuda)

  add_executable(harris_cuda harris.cpp)
  target_link_libraries(harris_cuda flare::flycuda)

  add_executable(matching_cuda matching.cpp)
  target_link_libraries(matching_cuda flare::flycuda)

  add_executable(susan_cuda susan.cpp)
  target_link_libraries(susan_cuda flare::flycuda)
endif()

if (flare_OpenCL_FOUND)
  add_executable(fast_opencl fast.cpp)
  target_link_libraries(fast_opencl flare::flyopencl)

  add_executable(harris_opencl harris.cpp)
  target_link_libraries(harris_opencl flare::flyopencl)

  add_executable(matching_opencl matching.cpp)
  target_link_libraries(matching_opencl flare::flyopencl)

  add_executable(susan_opencl susan.cpp)
  target_link_libraries(susan_opencl flare::flyopencl)
endif()

if (flare_oneAPI_FOUND)
  add_executable(fast_oneapi fast.cpp)
  target_link_libraries(fast_oneapi flare::flyoneapi)

  add_executable(harris_oneapi harris.cpp)
  target_link_libraries(harris_oneapi flare::flyoneapi)

  add_executable(matching_oneapi matching.cpp)
  target_link_libraries(matching_oneapi flare::flyoneapi)

  add_executable(susan_oneapi susan.cpp)
  target_link_libraries(susan_oneapi flare::flyoneapi)
endif()
