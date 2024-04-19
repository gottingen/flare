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

#ifndef __KPARAM_H
#define __KPARAM_H

// #ifndef __OPENCL_VERSION__
//  Only define dim_t in host code. dim_t is defined when setting the program
//  options in program.cpp
#include <fly/defines.h>
// #endif

// Defines the size and shape of the data in the OpenCL buffer
typedef struct {
    dim_t dims[4];
    dim_t strides[4];
    dim_t offset;
} KParam;

#endif
