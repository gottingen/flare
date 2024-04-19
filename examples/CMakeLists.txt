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
cmake_policy(VERSION 3.5)
project(Flare-Examples
  VERSION 3.7.0
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
if(NOT EXISTS "${flare_SOURCE_DIR}/CMakeLists.txt")
  set(ASSETS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../asserts")
endif()

file(TO_NATIVE_PATH ${ASSETS_DIR} ASSETS_DIR)

if(WIN32)
  string(REPLACE "\\" "\\\\" ASSETS_DIR  ${ASSETS_DIR})
  # - WIN32_LEAN_AND_MEAN & VC_EXTRALEAN reduces the number of
  #   windows headers being included.
  # - NOMINMAX is required for Flare code that uses
  #   functions fly::min & fly::max. Having a namespace doesn't help also.
  add_definitions(-DWIN32_LEAN_AND_MEAN -DVC_EXTRALEAN -DNOMINMAX)
  unset(CMAKE_RUNTIME_OUTPUT_DIRECTORY)
endif()

add_subdirectory(benchmarks)
add_subdirectory(computer_vision)
add_subdirectory(financial)
add_subdirectory(getting_started)
add_subdirectory(graphics)
add_subdirectory(helloworld)
add_subdirectory(image_processing)
add_subdirectory(lin_algebra)
add_subdirectory(machine_learning)
add_subdirectory(pde)
add_subdirectory(unified)
