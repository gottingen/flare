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

#include <atomic/atomic_operations_test.h>
#include <doctest.h>

namespace Test {
TEST_CASE("TEST_CATEGORY, atomic_operations_int") {
  const int start = 1;  // Avoid zero for division.
  const int end   = 11;
  for (int i = start; i < end; ++i) {
      REQUIRE((TestAtomicOperations::AtomicOperationsTestIntegralType<
                 int, TEST_EXECSPACE>(start, end - i, 1)));
      REQUIRE((TestAtomicOperations::AtomicOperationsTestIntegralType<
                 int, TEST_EXECSPACE>(start, end - i, 2)));
      REQUIRE((TestAtomicOperations::AtomicOperationsTestIntegralType<
                 int, TEST_EXECSPACE>(start, end - i, 3)));
      REQUIRE((TestAtomicOperations::AtomicOperationsTestIntegralType<
                 int, TEST_EXECSPACE>(start, end - i, 4)));
      REQUIRE((TestAtomicOperations::AtomicOperationsTestIntegralType<
                 int, TEST_EXECSPACE>(start, end - i, 5)));
      REQUIRE((TestAtomicOperations::AtomicOperationsTestIntegralType<
                 int, TEST_EXECSPACE>(start, end - i, 6)));
      REQUIRE((TestAtomicOperations::AtomicOperationsTestIntegralType<
                 int, TEST_EXECSPACE>(start, end - i, 7)));
      REQUIRE((TestAtomicOperations::AtomicOperationsTestIntegralType<
                 int, TEST_EXECSPACE>(start, end - i, 8)));
      REQUIRE((TestAtomicOperations::AtomicOperationsTestIntegralType<
                 int, TEST_EXECSPACE>(start, end - i, 9)));
      REQUIRE((TestAtomicOperations::AtomicOperationsTestIntegralType<
                 int, TEST_EXECSPACE>(start, end - i, 11)));
      REQUIRE((TestAtomicOperations::AtomicOperationsTestIntegralType<
                 int, TEST_EXECSPACE>(start, end - i, 12)));
      REQUIRE((TestAtomicOperations::AtomicOperationsTestIntegralType<
                 int, TEST_EXECSPACE>(start, end - i, 13)));
  }
}
}  // namespace Test
