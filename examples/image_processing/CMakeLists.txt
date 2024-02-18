# Copyright (c) 2018, Flare
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

cmake_minimum_required(VERSION 3.5)
project(Flare-Example-Image-Processing
  VERSION 3.5.0
  LANGUAGES CXX)

find_package(Flare)

add_definitions("-DASSETS_DIR=\"${ASSETS_DIR}\"")

if(Flare_CPU_FOUND)
  # Adaptive Thresholding example
  add_executable(adaptive_thresholding_cpu adaptive_thresholding.cpp)
  target_link_libraries(adaptive_thresholding_cpu Flare::flycpu)

  # Binary Thresholding example
  add_executable(binary_thresholding_cpu binary_thresholding.cpp)
  target_link_libraries(binary_thresholding_cpu Flare::flycpu)

  # Brain Segmentation example
  add_executable(brain_segmentation_cpu brain_segmentation.cpp)
  target_link_libraries(brain_segmentation_cpu Flare::flycpu)

  # Confidence Connected Components example
  add_executable(confidence_connected_components_cpu
      confidence_connected_components.cpp)
  target_link_libraries(confidence_connected_components_cpu Flare::flycpu)

  # Edge detection example
  add_executable(edge_cpu edge.cpp)
  target_link_libraries(edge_cpu Flare::flycpu)

  # Filters example
  add_executable(filters_cpu filters.cpp)
  target_link_libraries(filters_cpu Flare::flycpu)

  # Image example
  add_executable(image_demo_cpu image_demo.cpp)
  target_link_libraries(image_demo_cpu Flare::flycpu)

  # Image Editing example
  add_executable(image_editing_cpu image_editing.cpp)
  target_link_libraries(image_editing_cpu Flare::flycpu)

  # Morph example
  add_executable(morphing_cpu morphing.cpp)
  target_link_libraries(morphing_cpu Flare::flycpu)

  # Optical Flow example
  add_executable(optical_flow_cpu optical_flow.cpp)
  target_link_libraries(optical_flow_cpu Flare::flycpu)

  # Pyramids example
  add_executable(pyramids_cpu pyramids.cpp)
  target_link_libraries(pyramids_cpu Flare::flycpu)

  # Gradient anisotropic diffusion example
  add_executable(gradient_diffusion_cpu gradient_diffusion.cpp)
  target_link_libraries(gradient_diffusion_cpu Flare::flycpu)

  #Image Deconvolution Example
  add_executable(deconvolution_cpu deconvolution.cpp)
  target_link_libraries(deconvolution_cpu Flare::flycpu)
endif()

if(Flare_CUDA_FOUND)
  add_executable(adaptive_thresholding_cuda adaptive_thresholding.cpp)
  target_link_libraries(adaptive_thresholding_cuda Flare::flycuda)

  add_executable(binary_thresholding_cuda binary_thresholding.cpp)
  target_link_libraries(binary_thresholding_cuda Flare::flycuda)

  add_executable(brain_segmentation_cuda brain_segmentation.cpp)
  target_link_libraries(brain_segmentation_cuda Flare::flycuda)

  add_executable(confidence_connected_components_cuda
      confidence_connected_components.cpp)
  target_link_libraries(confidence_connected_components_cuda Flare::flycuda)

  add_executable(edge_cuda edge.cpp)
  target_link_libraries(edge_cuda Flare::flycuda)

  add_executable(filters_cuda filters.cpp)
  target_link_libraries(filters_cuda Flare::flycuda)

  add_executable(image_demo_cuda image_demo.cpp)
  target_link_libraries(image_demo_cuda Flare::flycuda)

  add_executable(image_editing_cuda image_editing.cpp)
  target_link_libraries(image_editing_cuda Flare::flycuda)

  add_executable(morphing_cuda morphing.cpp)
  target_link_libraries(morphing_cuda Flare::flycuda)

  add_executable(optical_flow_cuda optical_flow.cpp)
  target_link_libraries(optical_flow_cuda Flare::flycuda)

  add_executable(pyramids_cuda pyramids.cpp)
  target_link_libraries(pyramids_cuda Flare::flycuda)

  # Gradient anisotropic diffusion example
  add_executable(gradient_diffusion_cuda gradient_diffusion.cpp)
  target_link_libraries(gradient_diffusion_cuda Flare::flycuda)

  #Image Deconvolution Example
  add_executable(deconvolution_cuda deconvolution.cpp)
  target_link_libraries(deconvolution_cuda Flare::flycuda)
endif()

if(Flare_OpenCL_FOUND)
  add_executable(adaptive_thresholding_opencl adaptive_thresholding.cpp)
  target_link_libraries(adaptive_thresholding_opencl Flare::flyopencl)

  add_executable(binary_thresholding_opencl binary_thresholding.cpp)
  target_link_libraries(binary_thresholding_opencl Flare::flyopencl)

  add_executable(brain_segmentation_opencl brain_segmentation.cpp)
  target_link_libraries(brain_segmentation_opencl Flare::flyopencl)

  add_executable(confidence_connected_components_opencl
      confidence_connected_components.cpp)
  target_link_libraries(confidence_connected_components_opencl Flare::flyopencl)

  add_executable(edge_opencl edge.cpp)
  target_link_libraries(edge_opencl Flare::flyopencl)

  add_executable(filters_opencl filters.cpp)
  target_link_libraries(filters_opencl Flare::flyopencl)

  add_executable(image_demo_opencl image_demo.cpp)
  target_link_libraries(image_demo_opencl Flare::flyopencl)

  add_executable(image_editing_opencl image_editing.cpp)
  target_link_libraries(image_editing_opencl Flare::flyopencl)

  add_executable(morphing_opencl morphing.cpp)
  target_link_libraries(morphing_opencl Flare::flyopencl)

  add_executable(optical_flow_opencl optical_flow.cpp)
  target_link_libraries(optical_flow_opencl Flare::flyopencl)

  add_executable(pyramids_opencl pyramids.cpp)
  target_link_libraries(pyramids_opencl Flare::flyopencl)

  # Gradient anisotropic diffusion example
  add_executable(gradient_diffusion_opencl gradient_diffusion.cpp)
  target_link_libraries(gradient_diffusion_opencl Flare::flyopencl)

  #Image Deconvolution Example
  add_executable(deconvolution_opencl deconvolution.cpp)
  target_link_libraries(deconvolution_opencl Flare::flyopencl)
endif()

if(Flare_oneAPI_FOUND)
  add_executable(adaptive_thresholding_oneapi adaptive_thresholding.cpp)
  target_link_libraries(adaptive_thresholding_oneapi Flare::flyoneapi)

  add_executable(binary_thresholding_oneapi binary_thresholding.cpp)
  target_link_libraries(binary_thresholding_oneapi Flare::flyoneapi)

  add_executable(brain_segmentation_oneapi brain_segmentation.cpp)
  target_link_libraries(brain_segmentation_oneapi Flare::flyoneapi)

  add_executable(confidence_connected_components_oneapi
      confidence_connected_components.cpp)
  target_link_libraries(confidence_connected_components_oneapi Flare::flyoneapi)

  add_executable(edge_oneapi edge.cpp)
  target_link_libraries(edge_oneapi Flare::flyoneapi)

  add_executable(filters_oneapi filters.cpp)
  target_link_libraries(filters_oneapi Flare::flyoneapi)

  add_executable(image_demo_oneapi image_demo.cpp)
  target_link_libraries(image_demo_oneapi Flare::flyoneapi)

  add_executable(image_editing_oneapi image_editing.cpp)
  target_link_libraries(image_editing_oneapi Flare::flyoneapi)

  add_executable(morphing_oneapi morphing.cpp)
  target_link_libraries(morphing_oneapi Flare::flyoneapi)

  add_executable(optical_flow_oneapi optical_flow.cpp)
  target_link_libraries(optical_flow_oneapi Flare::flyoneapi)

  add_executable(pyramids_oneapi pyramids.cpp)
  target_link_libraries(pyramids_oneapi Flare::flyoneapi)

  # Gradient anisotropic diffusion example
  add_executable(gradient_diffusion_oneapi gradient_diffusion.cpp)
  target_link_libraries(gradient_diffusion_oneapi Flare::flyoneapi)

  #Image Deconvolution Example
  add_executable(deconvolution_oneapi deconvolution.cpp)
  target_link_libraries(deconvolution_oneapi Flare::flyoneapi)
endif()
