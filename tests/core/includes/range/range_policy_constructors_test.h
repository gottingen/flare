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

#include <doctest.h>

#include <flare/core.h>

namespace {

TEST_CASE("TEST_CATEGORY, range_policy_runtime_parameters") {
  using Policy     = flare::RangePolicy<>;
  using Index      = Policy::index_type;
  Index work_begin = 5;
  Index work_end   = 15;
  Index chunk_size = 10;
  {
    Policy p(work_begin, work_end);
    REQUIRE_EQ(p.begin(), work_begin);
    REQUIRE_EQ(p.end(), work_end);
  }
  {
    Policy p(flare::DefaultExecutionSpace(), work_begin, work_end);
    REQUIRE_EQ(p.begin(), work_begin);
    REQUIRE_EQ(p.end(), work_end);
  }
  {
    Policy p(work_begin, work_end, flare::ChunkSize(chunk_size));
    REQUIRE_EQ(p.begin(), work_begin);
    REQUIRE_EQ(p.end(), work_end);
    REQUIRE_EQ(p.chunk_size(), chunk_size);
  }
  {
    Policy p(flare::DefaultExecutionSpace(), work_begin, work_end,
             flare::ChunkSize(chunk_size));
    REQUIRE_EQ(p.begin(), work_begin);
    REQUIRE_EQ(p.end(), work_end);
    REQUIRE_EQ(p.chunk_size(), chunk_size);
  }
  {
    Policy p;  // default-constructed
    REQUIRE_EQ(p.begin(), Index(0));
    REQUIRE_EQ(p.end(), Index(0));
    REQUIRE_EQ(p.chunk_size(), Index(0));

    // copy-assigned
    p = Policy(work_begin, work_end, flare::ChunkSize(chunk_size));
    REQUIRE_EQ(p.begin(), work_begin);
    REQUIRE_EQ(p.end(), work_end);
    REQUIRE_EQ(p.chunk_size(), chunk_size);
  }
  {
    Policy p1(work_begin, work_end, flare::ChunkSize(chunk_size));
    Policy p2(p1);  // copy-constructed
    REQUIRE_EQ(p1.begin(), p2.begin());
    REQUIRE_EQ(p1.end(), p2.end());
    REQUIRE_EQ(p1.chunk_size(), p2.chunk_size());
  }
}

}  // namespace
