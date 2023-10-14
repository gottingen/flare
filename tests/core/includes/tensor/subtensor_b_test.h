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

#ifndef SUBTENSOR_B_TEST_H_
#define SUBTENSOR_B_TEST_H_

#include <tensor/tensor_subtensor_test.h>

namespace Test {

    TEST_CASE("TEST_CATEGORY, tensor_subtensor_layoutleft_to_layoutleft") {
        TestTensorSubtensor::test_layoutleft_to_layoutleft<TEST_EXECSPACE>();
        TestTensorSubtensor::test_layoutleft_to_layoutleft<
                TEST_EXECSPACE, flare::MemoryTraits<flare::Atomic> >();
        TestTensorSubtensor::test_layoutleft_to_layoutleft<
                TEST_EXECSPACE, flare::MemoryTraits<flare::RandomAccess> >();
    }

    TEST_CASE("TEST_CATEGORY, tensor_subtensor_layoutright_to_layoutright") {
        TestTensorSubtensor::test_layoutright_to_layoutright<TEST_EXECSPACE>();
        TestTensorSubtensor::test_layoutright_to_layoutright<
                TEST_EXECSPACE, flare::MemoryTraits<flare::Atomic> >();
        TestTensorSubtensor::test_layoutright_to_layoutright<
                TEST_EXECSPACE, flare::MemoryTraits<flare::RandomAccess> >();
    }

}  // namespace Test
#endif  // SUBTENSOR_B_TEST_H_
