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
project(Flare-Example-Financial
  VERSION 3.5.0
  LANGUAGES CXX)

find_package(flare)

if(flare_CPU_FOUND)
  # Black-Scholes Options
  add_executable(black_scholes_options_cpu black_scholes_options.cpp input.h)
  target_link_libraries(black_scholes_options_cpu flare::flycpu)

  # Heston Model
  add_executable(heston_model_cpu heston_model.cpp)
  target_link_libraries(heston_model_cpu flare::flycpu)

  # Monte Carlo Options
  add_executable(monte_carlo_options_cpu monte_carlo_options.cpp)
  target_link_libraries(monte_carlo_options_cpu flare::flycpu)
endif()

if(flare_CUDA_FOUND)
  add_executable(black_scholes_options_cuda black_scholes_options.cpp input.h)
  target_link_libraries(black_scholes_options_cuda flare::flycuda)

  add_executable(heston_model_cuda heston_model.cpp)
  target_link_libraries(heston_model_cuda flare::flycuda)

  add_executable(monte_carlo_options_cuda monte_carlo_options.cpp)
  target_link_libraries(monte_carlo_options_cuda flare::flycuda)
endif()

