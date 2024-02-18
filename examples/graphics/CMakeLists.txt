# Copyright (c) 2018, Flare
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

cmake_minimum_required(VERSION 3.5)
project(Flare-Example-Graphics
  VERSION 3.5.0
  LANGUAGES CXX)

find_package(Flare)

add_definitions("-DASSETS_DIR=\"${ASSETS_DIR}\"")

if(Flare_CPU_FOUND)
  # Conway Game of Life
  add_executable(conway_cpu conway.cpp)
  target_link_libraries(conway_cpu Flare::flycpu)

  # Conway Game of Life with Color
  add_executable(conway_pretty_cpu conway_pretty.cpp)
  target_link_libraries(conway_pretty_cpu Flare::flycpu)

  # Vector fields example
  add_executable(field_cpu field.cpp)
  target_link_libraries(field_cpu Flare::flycpu)

  # Fractal example
  add_executable(fractal_cpu fractal.cpp)
  target_link_libraries(fractal_cpu Flare::flycpu)

  # Gravity Simulation example
  add_executable(gravity_sim_cpu gravity_sim.cpp gravity_sim_init.h)
  target_link_libraries(gravity_sim_cpu Flare::flycpu)

  # Histogram example
  add_executable(histogram_cpu histogram.cpp)
  target_compile_definitions(histogram_cpu PRIVATE "ASSETS_DIR=\"${ASSETS_DIR}\"")
  target_link_libraries(histogram_cpu Flare::flycpu)

  # Plot 2D example
  add_executable(plot2d_cpu plot2d.cpp)
  target_link_libraries(plot2d_cpu Flare::flycpu)

  # Plot 3 example
  add_executable(plot3_cpu plot3.cpp)
  target_link_libraries(plot3_cpu Flare::flycpu)

  # Surface example
  add_executable(surface_cpu surface.cpp)
  target_link_libraries(surface_cpu Flare::flycpu)
endif()

if(Flare_CUDA_FOUND)
  add_executable(conway_cuda conway.cpp)
  target_link_libraries(conway_cuda Flare::flycuda)

  add_executable(conway_pretty_cuda conway_pretty.cpp)
  target_link_libraries(conway_pretty_cuda Flare::flycuda)

  add_executable(field_cuda field.cpp)
  target_link_libraries(field_cuda Flare::flycuda)

  add_executable(fractal_cuda fractal.cpp)
  target_link_libraries(fractal_cuda Flare::flycuda)

  add_executable(gravity_sim_cuda gravity_sim.cpp gravity_sim_init.h)
  target_link_libraries(gravity_sim_cuda Flare::flycuda)

  add_executable(histogram_cuda histogram.cpp)
  target_compile_definitions(histogram_cuda PRIVATE "ASSETS_DIR=\"${ASSETS_DIR}\"")
  target_link_libraries(histogram_cuda Flare::flycuda)

  add_executable(plot2d_cuda plot2d.cpp)
  target_link_libraries(plot2d_cuda Flare::flycuda)
  add_executable(plot3_cuda plot3.cpp)
  target_link_libraries(plot3_cuda Flare::flycuda)

  add_executable(surface_cuda surface.cpp)
  target_link_libraries(surface_cuda Flare::flycuda)
endif()

if(Flare_OpenCL_FOUND)
  add_executable(conway_opencl conway.cpp)
  target_link_libraries(conway_opencl Flare::flyopencl)

  add_executable(conway_pretty_opencl conway_pretty.cpp)
  target_link_libraries(conway_pretty_opencl Flare::flyopencl)

  add_executable(field_opencl field.cpp)
  target_link_libraries(field_opencl Flare::flyopencl)

  add_executable(fractal_opencl fractal.cpp)
  target_link_libraries(fractal_opencl Flare::flyopencl)

  add_executable(gravity_sim_opencl gravity_sim.cpp gravity_sim_init.h)
  target_link_libraries(gravity_sim_opencl Flare::flyopencl)

  add_executable(histogram_opencl histogram.cpp)
  target_compile_definitions(histogram_opencl PRIVATE "ASSETS_DIR=\"${ASSETS_DIR}\"")
  target_link_libraries(histogram_opencl Flare::flyopencl)

  add_executable(plot2d_opencl plot2d.cpp)
  target_link_libraries(plot2d_opencl Flare::flyopencl)

  add_executable(plot3_opencl plot3.cpp)
  target_link_libraries(plot3_opencl Flare::flyopencl)

  add_executable(surface_opencl surface.cpp)
  target_link_libraries(surface_opencl Flare::flyopencl)
endif()

if(Flare_oneAPI_FOUND)
  add_executable(conway_oneapi conway.cpp)
  target_link_libraries(conway_oneapi Flare::flyoneapi)

  add_executable(conway_pretty_oneapi conway_pretty.cpp)
  target_link_libraries(conway_pretty_oneapi Flare::flyoneapi)

  add_executable(field_oneapi field.cpp)
  target_link_libraries(field_oneapi Flare::flyoneapi)

  add_executable(fractal_oneapi fractal.cpp)
  target_link_libraries(fractal_oneapi Flare::flyoneapi)

  add_executable(gravity_sim_oneapi gravity_sim.cpp gravity_sim_init.h)
  target_link_libraries(gravity_sim_oneapi Flare::flyoneapi)

  add_executable(histogram_oneapi histogram.cpp)
  target_compile_definitions(histogram_oneapi PRIVATE "ASSETS_DIR=\"${ASSETS_DIR}\"")
  target_link_libraries(histogram_oneapi Flare::flyoneapi)

  add_executable(plot2d_oneapi plot2d.cpp)
  target_link_libraries(plot2d_oneapi Flare::flyoneapi)

  add_executable(plot3_oneapi plot3.cpp)
  target_link_libraries(plot3_oneapi Flare::flyoneapi)

  add_executable(surface_oneapi surface.cpp)
  target_link_libraries(surface_oneapi Flare::flyoneapi)
endif()
