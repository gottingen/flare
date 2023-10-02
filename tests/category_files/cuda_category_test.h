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

#ifndef CUDA_CATEGORY_TEST_H_
#define CUDA_CATEGORY_TEST_H_

#define TEST_CATEGORY cuda
#define TEST_CATEGORY_NUMBER 5
#define TEST_CATEGORY_DEATH cuda_DeathTest
#define TEST_EXECSPACE flare::Cuda
#define TEST_CATEGORY_FIXTURE(name) cuda_##name

#endif  // CUDA_CATEGORY_TEST_H_
