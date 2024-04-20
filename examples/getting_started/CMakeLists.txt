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
project(Flare-Example-Getting-Started
  VERSION 3.5.0
  LANGUAGES CXX)

find_package(flare)

if(flare_CPU_FOUND)
  # Convolve examples
  add_executable(convolve_cpu convolve.cpp)
  target_link_libraries(convolve_cpu flare::flycpu)

  # Integer examples
  add_executable(integer_cpu integer.cpp)
  target_link_libraries(integer_cpu flare::flycpu)

  # Rainfall examples
  add_executable(rainfall_cpu rainfall.cpp)
  target_link_libraries(rainfall_cpu flare::flycpu)

  # Vectorization examples
  add_executable(vectorize_cpu vectorize.cpp)
  target_link_libraries(vectorize_cpu flare::flycpu)
endif()

if(flare_CUDA_FOUND)
  add_executable(convolve_cuda convolve.cpp)
  target_link_libraries(convolve_cuda flare::flycuda)

  add_executable(integer_cuda integer.cpp)
  target_link_libraries(integer_cuda flare::flycuda)

  add_executable(rainfall_cuda rainfall.cpp)
  target_link_libraries(rainfall_cuda flare::flycuda)

  add_executable(vectorize_cuda vectorize.cpp)
  target_link_libraries(vectorize_cuda flare::flycuda)
endif()
