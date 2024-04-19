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
project(Flare-Example-Graphics
  VERSION 3.5.0
  LANGUAGES CXX)

find_package(flare)

add_definitions("-DASSETS_DIR=\"${ASSETS_DIR}\"")

if(flare_CPU_FOUND)
  # Conway Game of Life
  add_executable(conway_cpu conway.cpp)
  target_link_libraries(conway_cpu flare::flycpu)

  # Conway Game of Life with Color
  add_executable(conway_pretty_cpu conway_pretty.cpp)
  target_link_libraries(conway_pretty_cpu flare::flycpu)

  # Vector fields example
  add_executable(field_cpu field.cpp)
  target_link_libraries(field_cpu flare::flycpu)

  # Fractal example
  add_executable(fractal_cpu fractal.cpp)
  target_link_libraries(fractal_cpu flare::flycpu)

  # Gravity Simulation example
  add_executable(gravity_sim_cpu gravity_sim.cpp gravity_sim_init.h)
  target_link_libraries(gravity_sim_cpu flare::flycpu)

  # Histogram example
  add_executable(histogram_cpu histogram.cpp)
  target_compile_definitions(histogram_cpu PRIVATE "ASSETS_DIR=\"${ASSETS_DIR}\"")
  target_link_libraries(histogram_cpu flare::flycpu)

  # Plot 2D example
  add_executable(plot2d_cpu plot2d.cpp)
  target_link_libraries(plot2d_cpu flare::flycpu)

  # Plot 3 example
  add_executable(plot3_cpu plot3.cpp)
  target_link_libraries(plot3_cpu flare::flycpu)

  # Surface example
  add_executable(surface_cpu surface.cpp)
  target_link_libraries(surface_cpu flare::flycpu)
endif()

if(flare_CUDA_FOUND)
  add_executable(conway_cuda conway.cpp)
  target_link_libraries(conway_cuda flare::flycuda)

  add_executable(conway_pretty_cuda conway_pretty.cpp)
  target_link_libraries(conway_pretty_cuda flare::flycuda)

  add_executable(field_cuda field.cpp)
  target_link_libraries(field_cuda flare::flycuda)

  add_executable(fractal_cuda fractal.cpp)
  target_link_libraries(fractal_cuda flare::flycuda)

  add_executable(gravity_sim_cuda gravity_sim.cpp gravity_sim_init.h)
  target_link_libraries(gravity_sim_cuda flare::flycuda)

  add_executable(histogram_cuda histogram.cpp)
  target_compile_definitions(histogram_cuda PRIVATE "ASSETS_DIR=\"${ASSETS_DIR}\"")
  target_link_libraries(histogram_cuda flare::flycuda)

  add_executable(plot2d_cuda plot2d.cpp)
  target_link_libraries(plot2d_cuda flare::flycuda)
  add_executable(plot3_cuda plot3.cpp)
  target_link_libraries(plot3_cuda flare::flycuda)

  add_executable(surface_cuda surface.cpp)
  target_link_libraries(surface_cuda flare::flycuda)
endif()

if(flare_OpenCL_FOUND)
  add_executable(conway_opencl conway.cpp)
  target_link_libraries(conway_opencl flare::flyopencl)

  add_executable(conway_pretty_opencl conway_pretty.cpp)
  target_link_libraries(conway_pretty_opencl flare::flyopencl)

  add_executable(field_opencl field.cpp)
  target_link_libraries(field_opencl flare::flyopencl)

  add_executable(fractal_opencl fractal.cpp)
  target_link_libraries(fractal_opencl flare::flyopencl)

  add_executable(gravity_sim_opencl gravity_sim.cpp gravity_sim_init.h)
  target_link_libraries(gravity_sim_opencl flare::flyopencl)

  add_executable(histogram_opencl histogram.cpp)
  target_compile_definitions(histogram_opencl PRIVATE "ASSETS_DIR=\"${ASSETS_DIR}\"")
  target_link_libraries(histogram_opencl flare::flyopencl)

  add_executable(plot2d_opencl plot2d.cpp)
  target_link_libraries(plot2d_opencl flare::flyopencl)

  add_executable(plot3_opencl plot3.cpp)
  target_link_libraries(plot3_opencl flare::flyopencl)

  add_executable(surface_opencl surface.cpp)
  target_link_libraries(surface_opencl flare::flyopencl)
endif()

if(flare_oneAPI_FOUND)
  add_executable(conway_oneapi conway.cpp)
  target_link_libraries(conway_oneapi flare::flyoneapi)

  add_executable(conway_pretty_oneapi conway_pretty.cpp)
  target_link_libraries(conway_pretty_oneapi flare::flyoneapi)

  add_executable(field_oneapi field.cpp)
  target_link_libraries(field_oneapi flare::flyoneapi)

  add_executable(fractal_oneapi fractal.cpp)
  target_link_libraries(fractal_oneapi flare::flyoneapi)

  add_executable(gravity_sim_oneapi gravity_sim.cpp gravity_sim_init.h)
  target_link_libraries(gravity_sim_oneapi flare::flyoneapi)

  add_executable(histogram_oneapi histogram.cpp)
  target_compile_definitions(histogram_oneapi PRIVATE "ASSETS_DIR=\"${ASSETS_DIR}\"")
  target_link_libraries(histogram_oneapi flare::flyoneapi)

  add_executable(plot2d_oneapi plot2d.cpp)
  target_link_libraries(plot2d_oneapi flare::flyoneapi)

  add_executable(plot3_oneapi plot3.cpp)
  target_link_libraries(plot3_oneapi flare::flyoneapi)

  add_executable(surface_oneapi surface.cpp)
  target_link_libraries(surface_oneapi flare::flyoneapi)
endif()
