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
project(Flare-Example-HelloWorld
  VERSION 3.5.0
  LANGUAGES CXX)

find_package(flare)

if(flare_CPU_FOUND)
  # Hello World example
  add_executable(helloworld_cpu helloworld.cpp)
  target_link_libraries(helloworld_cpu flare::flycpu)
endif()

if(flare_CUDA_FOUND)
  add_executable(helloworld_cuda helloworld.cpp)
  target_link_libraries(helloworld_cuda flare::flycuda)
endif()
