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

#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <fly/data.h>

namespace fly {
int getMaxJitLen(void);

void setMaxJitLen(const int jitLen);
}  // namespace fly

TEST(JIT, UnitMaxHeight) {
    const int oldMaxJitLen = fly::getMaxJitLen();
    fly::setMaxJitLen(1);
    fly::array a = fly::constant(1, 10);
    fly::array b = fly::constant(2, 10);
    fly::array c = a * b;
    fly::array d = b * c;
    c.eval();
    d.eval();
    fly::setMaxJitLen(oldMaxJitLen);
}

TEST(JIT, ZeroMaxHeight) {
    EXPECT_THROW({ fly::setMaxJitLen(0); }, fly::exception);
}
