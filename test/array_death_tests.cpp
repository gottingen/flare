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

#include <flare.h>
#include <gtest/gtest.h>

#include <cstdlib>

using fly::array;
using fly::constant;
using fly::dim4;
using fly::end;
using fly::fft;
using fly::info;
using fly::randu;
using fly::scan;
using fly::seq;
using fly::setDevice;
using fly::sin;
using fly::sort;

template<typename T>
class ArrayDeathTest : public ::testing::Test {};

void deathTest() {
    info();
    setDevice(0);

    array A = randu(5, 3, f32);

    array B = sin(A) + 1.5;

    B(seq(0, 2), 1) = B(seq(0, 2), 1) * -1;

    array C = fft(B);

    array c = C.row(end);

    dim4 dims(16, 4, 1, 1);
    array r = constant(2, dims);

    array S = scan(r, 0, FLY_BINARY_MUL);

    float d[] = {1, 2, 3, 4, 5, 6};
    array D(2, 3, d, flyHost);

    D.col(0) = D.col(end);

    array vals, inds;
    sort(vals, inds, A);

    _exit(0);
}

TEST(ArrayDeathTest, ProxyMoveAssignmentOperator) {
    EXPECT_EXIT(deathTest(), ::testing::ExitedWithCode(0), "");
}
