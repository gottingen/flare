// Copyright 2023 The Elastic-AI Authors.
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

#ifndef FLARE_BACKEND_CUDA_CUDA_NVIDIA_GPU_ARCHITECTURES_H_
#define FLARE_BACKEND_CUDA_CUDA_NVIDIA_GPU_ARCHITECTURES_H_

#if defined(FLARE_ARCH_KEPLER30)
#define FLARE_IMPL_ARCH_NVIDIA_GPU 30
#elif defined(FLARE_ARCH_KEPLER32)
#define FLARE_IMPL_ARCH_NVIDIA_GPU 32
#elif defined(FLARE_ARCH_KEPLER35)
#define FLARE_IMPL_ARCH_NVIDIA_GPU 35
#elif defined(FLARE_ARCH_KEPLER37)
#define FLARE_IMPL_ARCH_NVIDIA_GPU 37
#elif defined(FLARE_ARCH_MAXWELL50)
#define FLARE_IMPL_ARCH_NVIDIA_GPU 50
#elif defined(FLARE_ARCH_MAXWELL52)
#define FLARE_IMPL_ARCH_NVIDIA_GPU 52
#elif defined(FLARE_ARCH_MAXWELL53)
#define FLARE_IMPL_ARCH_NVIDIA_GPU 53
#elif defined(FLARE_ARCH_PASCAL60)
#define FLARE_IMPL_ARCH_NVIDIA_GPU 60
#elif defined(FLARE_ARCH_PASCAL61)
#define FLARE_IMPL_ARCH_NVIDIA_GPU 61
#elif defined(FLARE_ARCH_VOLTA70)
#define FLARE_IMPL_ARCH_NVIDIA_GPU 70
#elif defined(FLARE_ARCH_VOLTA72)
#define FLARE_IMPL_ARCH_NVIDIA_GPU 72
#elif defined(FLARE_ARCH_TURING75)
#define FLARE_IMPL_ARCH_NVIDIA_GPU 75
#elif defined(FLARE_ARCH_AMPERE80)
#define FLARE_IMPL_ARCH_NVIDIA_GPU 80
#elif defined(FLARE_ARCH_AMPERE86)
#define FLARE_IMPL_ARCH_NVIDIA_GPU 86
#elif defined(FLARE_ARCH_ADA89)
#define FLARE_IMPL_ARCH_NVIDIA_GPU 89
#elif defined(FLARE_ARCH_HOPPER90)
#define FLARE_IMPL_ARCH_NVIDIA_GPU 90
#elif defined(FLARE_ON_CUDA_DEVICE)
// do not raise an error on other backends that may run on NVIDIA GPUs
#error NVIDIA GPU arch not recognized
#endif

#endif  // FLARE_BACKEND_CUDA_CUDA_NVIDIA_GPU_ARCHITECTURES_H_
