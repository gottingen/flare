// Copyright 2023 The EA Authors.
// part of Elastic AI Search
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#pragma once

#include <common/DependencyModule.hpp>

#include <cudnn.h>

#include <memory>
#include <tuple>

#if CUDNN_VERSION > 4000
// This function is not available on versions greater than v4
cudnnStatus_t cudnnSetFilter4dDescriptor_v4(
    cudnnFilterDescriptor_t filterDesc,
    cudnnDataType_t dataType,  // image data type
    cudnnTensorFormat_t format,
    int k,   // number of output feature maps
    int c,   // number of input feature maps
    int h,   // height of each input filter
    int w);  // width of  each input filter
#else
// This function is only available on newer versions of cudnn
size_t cudnnGetCudartVersion(void);
#endif

#if CUDNN_VERSION >= 8000
typedef enum {
    CUDNN_CONVOLUTION_FWD_NO_WORKSPACE            = 0,
    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST          = 1,
    CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT = 2,
} cudnnConvolutionFwdPreference_t;

typedef enum {
    CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE            = 0,
    CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST          = 1,
    CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT = 2,
} cudnnConvolutionBwdFilterPreference_t;

cudnnStatus_t cudnnGetConvolutionForwardAlgorithm(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc,
    cudnnConvolutionFwdPreference_t preference, size_t memoryLimitInBytes,
    cudnnConvolutionFwdAlgo_t* algo);

cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc,
    cudnnConvolutionBwdFilterPreference_t preference, size_t memoryLimitInBytes,
    cudnnConvolutionBwdFilterAlgo_t* algo);
#endif

namespace flare {
namespace cuda {

class cudnnModule {
    common::DependencyModule module;

   public:
    cudnnModule();
    MODULE_MEMBER(cudnnConvolutionBackwardData);
    MODULE_MEMBER(cudnnConvolutionBackwardFilter);
    MODULE_MEMBER(cudnnConvolutionForward);
    MODULE_MEMBER(cudnnCreate);
    MODULE_MEMBER(cudnnCreateConvolutionDescriptor);
    MODULE_MEMBER(cudnnCreateFilterDescriptor);
    MODULE_MEMBER(cudnnCreateTensorDescriptor);
    MODULE_MEMBER(cudnnDestroy);
    MODULE_MEMBER(cudnnDestroyConvolutionDescriptor);
    MODULE_MEMBER(cudnnDestroyFilterDescriptor);
    MODULE_MEMBER(cudnnDestroyTensorDescriptor);
    MODULE_MEMBER(cudnnGetConvolutionBackwardDataWorkspaceSize);
    MODULE_MEMBER(cudnnGetConvolutionForwardAlgorithmMaxCount);
    MODULE_MEMBER(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount);
    MODULE_MEMBER(cudnnFindConvolutionForwardAlgorithm);
    MODULE_MEMBER(cudnnFindConvolutionBackwardFilterAlgorithm);
    MODULE_MEMBER(cudnnGetConvolutionForwardWorkspaceSize);
    MODULE_MEMBER(cudnnGetConvolutionBackwardFilterWorkspaceSize);
    MODULE_MEMBER(cudnnGetConvolutionForwardAlgorithm);
    MODULE_MEMBER(cudnnGetConvolutionBackwardFilterAlgorithm);
    MODULE_MEMBER(cudnnGetConvolutionNdForwardOutputDim);
    MODULE_MEMBER(cudnnSetConvolution2dDescriptor);
    MODULE_MEMBER(cudnnSetFilter4dDescriptor);
    MODULE_MEMBER(cudnnSetFilter4dDescriptor_v4);
    MODULE_MEMBER(cudnnGetVersion);
    MODULE_MEMBER(cudnnGetCudartVersion);
    MODULE_MEMBER(cudnnSetStream);
    MODULE_MEMBER(cudnnSetTensor4dDescriptor);

    clog::logger* getLogger() const noexcept;

    /// Returns the version of the cuDNN loaded at runtime
    common::Version getVersion() const noexcept { return module.getVersion(); }

    bool isLoaded() const noexcept { return module.isLoaded(); }
};

cudnnModule& getCudnnPlugin() noexcept;

}  // namespace cuda
}  // namespace flare
