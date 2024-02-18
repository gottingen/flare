# Copyright (c) 2018, Flare
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

cmake_minimum_required(VERSION 3.5)
project(Flare-Example-Financial
  VERSION 3.5.0
  LANGUAGES CXX)

find_package(Flare)

if(Flare_CPU_FOUND)
  # Black-Scholes Options
  add_executable(black_scholes_options_cpu black_scholes_options.cpp input.h)
  target_link_libraries(black_scholes_options_cpu Flare::flycpu)

  # Heston Model
  add_executable(heston_model_cpu heston_model.cpp)
  target_link_libraries(heston_model_cpu Flare::flycpu)

  # Monte Carlo Options
  add_executable(monte_carlo_options_cpu monte_carlo_options.cpp)
  target_link_libraries(monte_carlo_options_cpu Flare::flycpu)
endif()

if(Flare_CUDA_FOUND)
  add_executable(black_scholes_options_cuda black_scholes_options.cpp input.h)
  target_link_libraries(black_scholes_options_cuda Flare::flycuda)

  add_executable(heston_model_cuda heston_model.cpp)
  target_link_libraries(heston_model_cuda Flare::flycuda)

  add_executable(monte_carlo_options_cuda monte_carlo_options.cpp)
  target_link_libraries(monte_carlo_options_cuda Flare::flycuda)
endif()

if(Flare_OpenCL_FOUND)
  add_executable(monte_carlo_options_opencl monte_carlo_options.cpp)
  target_link_libraries(monte_carlo_options_opencl Flare::flyopencl)

  add_executable(black_scholes_options_opencl black_scholes_options.cpp input.h)
  target_link_libraries(black_scholes_options_opencl Flare::flyopencl)

  add_executable(heston_model_opencl heston_model.cpp)
  target_link_libraries(heston_model_opencl Flare::flyopencl)
endif()

if(Flare_oneAPI_FOUND)
  add_executable(monte_carlo_options_oneapi monte_carlo_options.cpp)
  target_link_libraries(monte_carlo_options_oneapi Flare::flyoneapi)

  add_executable(black_scholes_options_oneapi black_scholes_options.cpp input.h)
  target_link_libraries(black_scholes_options_oneapi Flare::flyoneapi)

  add_executable(heston_model_oneapi heston_model.cpp)
  target_link_libraries(heston_model_oneapi Flare::flyoneapi)
endif()
