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

#ifndef SUBTENSOR_11_TEST_H_
#define SUBTENSOR_11_TEST_H_

#include <tensor/tensor_subtensor_test.h>

namespace Test {

    TEST_CASE("TEST_CATEGORY, tensor_subtensor_3d_from_5d_right_atomic") {
        TestTensorSubtensor::test_3d_subtensor_5d_right<
                TEST_EXECSPACE, flare::MemoryTraits<flare::Atomic> >();
    }

}  // namespace Test
#endif  // SUBTENSOR_11_TEST_H_
