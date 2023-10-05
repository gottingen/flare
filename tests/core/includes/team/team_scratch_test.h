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

#ifndef FLARE_TEST_TEAM_SCRATCH_HPP
#define FLARE_TEST_TEAM_SCRATCH_HPP
#include <team_test.h>

namespace Test {

TEST(TEST_CATEGORY, team_shared_request) {
  TestSharedTeam<TEST_EXECSPACE, flare::Schedule<flare::Static> >();
  TestSharedTeam<TEST_EXECSPACE, flare::Schedule<flare::Dynamic> >();
}

TEST(TEST_CATEGORY, team_scratch_request) {
  TestScratchTeam<TEST_EXECSPACE, flare::Schedule<flare::Static> >();
  TestScratchTeam<TEST_EXECSPACE, flare::Schedule<flare::Dynamic> >();
}

#if defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)
TEST(TEST_CATEGORY, team_lambda_shared_request) {
  TestLambdaSharedTeam<flare::HostSpace, TEST_EXECSPACE,
                       flare::Schedule<flare::Static> >();
  TestLambdaSharedTeam<flare::HostSpace, TEST_EXECSPACE,
                       flare::Schedule<flare::Dynamic> >();
}
TEST(TEST_CATEGORY, scratch_align) { TestScratchAlignment<TEST_EXECSPACE>(); }
#endif

TEST(TEST_CATEGORY, shmem_size) { TestShmemSize<TEST_EXECSPACE>(); }

TEST(TEST_CATEGORY, multi_level_scratch) {
  TestMultiLevelScratchTeam<TEST_EXECSPACE,
                            flare::Schedule<flare::Static> >();
  TestMultiLevelScratchTeam<TEST_EXECSPACE,
                            flare::Schedule<flare::Dynamic> >();
}

}  // namespace Test
#endif
