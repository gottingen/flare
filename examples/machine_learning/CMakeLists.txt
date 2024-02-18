# Copyright (c) 2018, Flare
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

cmake_minimum_required(VERSION 3.5)
project(Flare-Example-Linear-Algebra
  VERSION 3.5.0
  LANGUAGES CXX)

find_package(Flare)

add_definitions("-DASSETS_DIR=\"${ASSETS_DIR}\"")

if(Flare_CPU_FOUND)
  # Bagging example
  add_executable(bagging_cpu bagging.cpp)
  target_link_libraries(bagging_cpu Flare::flycpu)

  # Deep Belief Network example
  add_executable(deep_belief_net_cpu deep_belief_net.cpp)
  target_link_libraries(deep_belief_net_cpu Flare::flycpu)

  # Genetic Algorithm example
  add_executable(geneticalgorithm_cpu geneticalgorithm.cpp)
  target_link_libraries(geneticalgorithm_cpu Flare::flycpu)

  # k Means example
  add_executable(kmeans_cpu kmeans.cpp)
  target_link_libraries(kmeans_cpu Flare::flycpu)

  # Logistic Regression example
  add_executable(logistic_regression_cpu logistic_regression.cpp)
  target_link_libraries(logistic_regression_cpu Flare::flycpu)

  # Naive Bayes example
  add_executable(naive_bayes_cpu naive_bayes.cpp)
  target_link_libraries(naive_bayes_cpu Flare::flycpu)

  # Neural Network example
  add_executable(neural_network_cpu neural_network.cpp)
  target_link_libraries(neural_network_cpu Flare::flycpu)

  # Preceptron example
  add_executable(perceptron_cpu perceptron.cpp)
  target_link_libraries(perceptron_cpu Flare::flycpu)

  # Restricted Boltsmann Machine example
  add_executable(rbm_cpu rbm.cpp)
  target_link_libraries(rbm_cpu Flare::flycpu)

  # Softmax Regression example
  add_executable(softmax_regression_cpu softmax_regression.cpp)
  target_link_libraries(softmax_regression_cpu Flare::flycpu)
endif()

if(Flare_CUDA_FOUND)
  add_executable(bagging_cuda bagging.cpp)
  target_link_libraries(bagging_cuda Flare::flycuda)

  add_executable(deep_belief_net_cuda deep_belief_net.cpp)
  target_link_libraries(deep_belief_net_cuda Flare::flycuda)

  add_executable(geneticalgorithm_cuda geneticalgorithm.cpp)
  target_link_libraries(geneticalgorithm_cuda Flare::flycuda)

  add_executable(kmeans_cuda kmeans.cpp)
  target_link_libraries(kmeans_cuda Flare::flycuda)

  add_executable(logistic_regression_cuda logistic_regression.cpp)
  target_link_libraries(logistic_regression_cuda Flare::flycuda)

  add_executable(naive_bayes_cuda naive_bayes.cpp)
  target_link_libraries(naive_bayes_cuda Flare::flycuda)

  add_executable(neural_network_cuda neural_network.cpp)
  target_link_libraries(neural_network_cuda Flare::flycuda)

  add_executable(perceptron_cuda perceptron.cpp)
  target_link_libraries(perceptron_cuda Flare::flycuda)

  add_executable(rbm_cuda rbm.cpp)
  target_link_libraries(rbm_cuda Flare::flycuda)

  add_executable(softmax_regression_cuda softmax_regression.cpp)
  target_link_libraries(softmax_regression_cuda Flare::flycuda)
endif()

if(Flare_OpenCL_FOUND)
  add_executable(bagging_opencl bagging.cpp)
  target_link_libraries(bagging_opencl Flare::flyopencl)

  add_executable(deep_belief_net_opencl deep_belief_net.cpp)
  target_link_libraries(deep_belief_net_opencl Flare::flyopencl)

  add_executable(geneticalgorithm_opencl geneticalgorithm.cpp)
  target_link_libraries(geneticalgorithm_opencl Flare::flyopencl)

  add_executable(kmeans_opencl kmeans.cpp)
  target_link_libraries(kmeans_opencl Flare::flyopencl)

  add_executable(logistic_regression_opencl logistic_regression.cpp)
  target_link_libraries(logistic_regression_opencl Flare::flyopencl)

  add_executable(naive_bayes_opencl naive_bayes.cpp)
  target_link_libraries(naive_bayes_opencl Flare::flyopencl)

  add_executable(neural_network_opencl neural_network.cpp)
  target_link_libraries(neural_network_opencl Flare::flyopencl)

  add_executable(perceptron_opencl perceptron.cpp)
  target_link_libraries(perceptron_opencl Flare::flyopencl)

  add_executable(rbm_opencl rbm.cpp)
  target_link_libraries(rbm_opencl Flare::flyopencl)

  add_executable(softmax_regression_opencl softmax_regression.cpp)
  target_link_libraries(softmax_regression_opencl Flare::flyopencl)
endif()

if(Flare_oneAPI_FOUND)
  add_executable(bagging_oneapi bagging.cpp)
  target_link_libraries(bagging_oneapi Flare::flyoneapi)

  add_executable(deep_belief_net_oneapi deep_belief_net.cpp)
  target_link_libraries(deep_belief_net_oneapi Flare::flyoneapi)

  add_executable(geneticalgorithm_oneapi geneticalgorithm.cpp)
  target_link_libraries(geneticalgorithm_oneapi Flare::flyoneapi)

  add_executable(kmeans_oneapi kmeans.cpp)
  target_link_libraries(kmeans_oneapi Flare::flyoneapi)

  add_executable(logistic_regression_oneapi logistic_regression.cpp)
  target_link_libraries(logistic_regression_oneapi Flare::flyoneapi)

  add_executable(naive_bayes_oneapi naive_bayes.cpp)
  target_link_libraries(naive_bayes_oneapi Flare::flyoneapi)

  add_executable(neural_network_oneapi neural_network.cpp)
  target_link_libraries(neural_network_oneapi Flare::flyoneapi)

  add_executable(perceptron_oneapi perceptron.cpp)
  target_link_libraries(perceptron_oneapi Flare::flyoneapi)

  add_executable(rbm_oneapi rbm.cpp)
  target_link_libraries(rbm_oneapi Flare::flyoneapi)

  add_executable(softmax_regression_oneapi softmax_regression.cpp)
  target_link_libraries(softmax_regression_oneapi Flare::flyoneapi)
endif()
