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
project(Flare-Example-Image-Processing
  VERSION 3.5.0
  LANGUAGES CXX)

find_package(flare)

add_definitions("-DASSETS_DIR=\"${ASSETS_DIR}\"")

if(flare_CPU_FOUND)
  # Adaptive Thresholding example
  add_executable(adaptive_thresholding_cpu adaptive_thresholding.cpp)
  target_link_libraries(adaptive_thresholding_cpu flare::flycpu)

  # Binary Thresholding example
  add_executable(binary_thresholding_cpu binary_thresholding.cpp)
  target_link_libraries(binary_thresholding_cpu flare::flycpu)

  # Brain Segmentation example
  add_executable(brain_segmentation_cpu brain_segmentation.cpp)
  target_link_libraries(brain_segmentation_cpu flare::flycpu)

  # Confidence Connected Components example
  add_executable(confidence_connected_components_cpu
      confidence_connected_components.cpp)
  target_link_libraries(confidence_connected_components_cpu flare::flycpu)

  # Edge detection example
  add_executable(edge_cpu edge.cpp)
  target_link_libraries(edge_cpu flare::flycpu)

  # Filters example
  add_executable(filters_cpu filters.cpp)
  target_link_libraries(filters_cpu flare::flycpu)

  # Image example
  add_executable(image_demo_cpu image_demo.cpp)
  target_link_libraries(image_demo_cpu flare::flycpu)

  # Image Editing example
  add_executable(image_editing_cpu image_editing.cpp)
  target_link_libraries(image_editing_cpu flare::flycpu)

  # Morph example
  add_executable(morphing_cpu morphing.cpp)
  target_link_libraries(morphing_cpu flare::flycpu)

  # Optical Flow example
  add_executable(optical_flow_cpu optical_flow.cpp)
  target_link_libraries(optical_flow_cpu flare::flycpu)

  # Pyramids example
  add_executable(pyramids_cpu pyramids.cpp)
  target_link_libraries(pyramids_cpu flare::flycpu)

  # Gradient anisotropic diffusion example
  add_executable(gradient_diffusion_cpu gradient_diffusion.cpp)
  target_link_libraries(gradient_diffusion_cpu flare::flycpu)

  #Image Deconvolution Example
  add_executable(deconvolution_cpu deconvolution.cpp)
  target_link_libraries(deconvolution_cpu flare::flycpu)
endif()

if(flare_CUDA_FOUND)
  add_executable(adaptive_thresholding_cuda adaptive_thresholding.cpp)
  target_link_libraries(adaptive_thresholding_cuda flare::flycuda)

  add_executable(binary_thresholding_cuda binary_thresholding.cpp)
  target_link_libraries(binary_thresholding_cuda flare::flycuda)

  add_executable(brain_segmentation_cuda brain_segmentation.cpp)
  target_link_libraries(brain_segmentation_cuda flare::flycuda)

  add_executable(confidence_connected_components_cuda
      confidence_connected_components.cpp)
  target_link_libraries(confidence_connected_components_cuda flare::flycuda)

  add_executable(edge_cuda edge.cpp)
  target_link_libraries(edge_cuda flare::flycuda)

  add_executable(filters_cuda filters.cpp)
  target_link_libraries(filters_cuda flare::flycuda)

  add_executable(image_demo_cuda image_demo.cpp)
  target_link_libraries(image_demo_cuda flare::flycuda)

  add_executable(image_editing_cuda image_editing.cpp)
  target_link_libraries(image_editing_cuda flare::flycuda)

  add_executable(morphing_cuda morphing.cpp)
  target_link_libraries(morphing_cuda flare::flycuda)

  add_executable(optical_flow_cuda optical_flow.cpp)
  target_link_libraries(optical_flow_cuda flare::flycuda)

  add_executable(pyramids_cuda pyramids.cpp)
  target_link_libraries(pyramids_cuda flare::flycuda)

  # Gradient anisotropic diffusion example
  add_executable(gradient_diffusion_cuda gradient_diffusion.cpp)
  target_link_libraries(gradient_diffusion_cuda flare::flycuda)

  #Image Deconvolution Example
  add_executable(deconvolution_cuda deconvolution.cpp)
  target_link_libraries(deconvolution_cuda flare::flycuda)
endif()

if(flare_OpenCL_FOUND)
  add_executable(adaptive_thresholding_opencl adaptive_thresholding.cpp)
  target_link_libraries(adaptive_thresholding_opencl flare::flyopencl)

  add_executable(binary_thresholding_opencl binary_thresholding.cpp)
  target_link_libraries(binary_thresholding_opencl flare::flyopencl)

  add_executable(brain_segmentation_opencl brain_segmentation.cpp)
  target_link_libraries(brain_segmentation_opencl flare::flyopencl)

  add_executable(confidence_connected_components_opencl
      confidence_connected_components.cpp)
  target_link_libraries(confidence_connected_components_opencl flare::flyopencl)

  add_executable(edge_opencl edge.cpp)
  target_link_libraries(edge_opencl flare::flyopencl)

  add_executable(filters_opencl filters.cpp)
  target_link_libraries(filters_opencl flare::flyopencl)

  add_executable(image_demo_opencl image_demo.cpp)
  target_link_libraries(image_demo_opencl flare::flyopencl)

  add_executable(image_editing_opencl image_editing.cpp)
  target_link_libraries(image_editing_opencl flare::flyopencl)

  add_executable(morphing_opencl morphing.cpp)
  target_link_libraries(morphing_opencl flare::flyopencl)

  add_executable(optical_flow_opencl optical_flow.cpp)
  target_link_libraries(optical_flow_opencl flare::flyopencl)

  add_executable(pyramids_opencl pyramids.cpp)
  target_link_libraries(pyramids_opencl flare::flyopencl)

  # Gradient anisotropic diffusion example
  add_executable(gradient_diffusion_opencl gradient_diffusion.cpp)
  target_link_libraries(gradient_diffusion_opencl flare::flyopencl)

  #Image Deconvolution Example
  add_executable(deconvolution_opencl deconvolution.cpp)
  target_link_libraries(deconvolution_opencl flare::flyopencl)
endif()

if(flare_oneAPI_FOUND)
  add_executable(adaptive_thresholding_oneapi adaptive_thresholding.cpp)
  target_link_libraries(adaptive_thresholding_oneapi flare::flyoneapi)

  add_executable(binary_thresholding_oneapi binary_thresholding.cpp)
  target_link_libraries(binary_thresholding_oneapi flare::flyoneapi)

  add_executable(brain_segmentation_oneapi brain_segmentation.cpp)
  target_link_libraries(brain_segmentation_oneapi flare::flyoneapi)

  add_executable(confidence_connected_components_oneapi
      confidence_connected_components.cpp)
  target_link_libraries(confidence_connected_components_oneapi flare::flyoneapi)

  add_executable(edge_oneapi edge.cpp)
  target_link_libraries(edge_oneapi flare::flyoneapi)

  add_executable(filters_oneapi filters.cpp)
  target_link_libraries(filters_oneapi flare::flyoneapi)

  add_executable(image_demo_oneapi image_demo.cpp)
  target_link_libraries(image_demo_oneapi flare::flyoneapi)

  add_executable(image_editing_oneapi image_editing.cpp)
  target_link_libraries(image_editing_oneapi flare::flyoneapi)

  add_executable(morphing_oneapi morphing.cpp)
  target_link_libraries(morphing_oneapi flare::flyoneapi)

  add_executable(optical_flow_oneapi optical_flow.cpp)
  target_link_libraries(optical_flow_oneapi flare::flyoneapi)

  add_executable(pyramids_oneapi pyramids.cpp)
  target_link_libraries(pyramids_oneapi flare::flyoneapi)

  # Gradient anisotropic diffusion example
  add_executable(gradient_diffusion_oneapi gradient_diffusion.cpp)
  target_link_libraries(gradient_diffusion_oneapi flare::flyoneapi)

  #Image Deconvolution Example
  add_executable(deconvolution_oneapi deconvolution.cpp)
  target_link_libraries(deconvolution_oneapi flare::flyoneapi)
endif()
