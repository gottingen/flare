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
project(Flare-Example-Linear-Algebra
  VERSION 3.5.0
  LANGUAGES CXX)

find_package(flare)

add_definitions("-DASSETS_DIR=\"${ASSETS_DIR}\"")

if(flare_CPU_FOUND)
  # Bagging example
  add_executable(bagging_cpu bagging.cpp)
  target_link_libraries(bagging_cpu flare::flycpu)

  # Deep Belief Network example
  add_executable(deep_belief_net_cpu deep_belief_net.cpp)
  target_link_libraries(deep_belief_net_cpu flare::flycpu)

  # Genetic Algorithm example
  add_executable(geneticalgorithm_cpu geneticalgorithm.cpp)
  target_link_libraries(geneticalgorithm_cpu flare::flycpu)

  # k Means example
  add_executable(kmeans_cpu kmeans.cpp)
  target_link_libraries(kmeans_cpu flare::flycpu)

  # Logistic Regression example
  add_executable(logistic_regression_cpu logistic_regression.cpp)
  target_link_libraries(logistic_regression_cpu flare::flycpu)

  # Naive Bayes example
  add_executable(naive_bayes_cpu naive_bayes.cpp)
  target_link_libraries(naive_bayes_cpu flare::flycpu)

  # Neural Network example
  add_executable(neural_network_cpu neural_network.cpp)
  target_link_libraries(neural_network_cpu flare::flycpu)

  # Preceptron example
  add_executable(perceptron_cpu perceptron.cpp)
  target_link_libraries(perceptron_cpu flare::flycpu)

  # Restricted Boltsmann Machine example
  add_executable(rbm_cpu rbm.cpp)
  target_link_libraries(rbm_cpu flare::flycpu)

  # Softmax Regression example
  add_executable(softmax_regression_cpu softmax_regression.cpp)
  target_link_libraries(softmax_regression_cpu flare::flycpu)
endif()

if(flare_CUDA_FOUND)
  add_executable(bagging_cuda bagging.cpp)
  target_link_libraries(bagging_cuda flare::flycuda)

  add_executable(deep_belief_net_cuda deep_belief_net.cpp)
  target_link_libraries(deep_belief_net_cuda flare::flycuda)

  add_executable(geneticalgorithm_cuda geneticalgorithm.cpp)
  target_link_libraries(geneticalgorithm_cuda flare::flycuda)

  add_executable(kmeans_cuda kmeans.cpp)
  target_link_libraries(kmeans_cuda flare::flycuda)

  add_executable(logistic_regression_cuda logistic_regression.cpp)
  target_link_libraries(logistic_regression_cuda flare::flycuda)

  add_executable(naive_bayes_cuda naive_bayes.cpp)
  target_link_libraries(naive_bayes_cuda flare::flycuda)

  add_executable(neural_network_cuda neural_network.cpp)
  target_link_libraries(neural_network_cuda flare::flycuda)

  add_executable(perceptron_cuda perceptron.cpp)
  target_link_libraries(perceptron_cuda flare::flycuda)

  add_executable(rbm_cuda rbm.cpp)
  target_link_libraries(rbm_cuda flare::flycuda)

  add_executable(softmax_regression_cuda softmax_regression.cpp)
  target_link_libraries(softmax_regression_cuda flare::flycuda)
endif()