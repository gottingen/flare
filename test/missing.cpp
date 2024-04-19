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
#include <fly/arith.h>
#include <fly/array.h>
#include <fly/data.h>
#include <fly/image.h>
#include <fly/lapack.h>
#include <fly/random.h>

using namespace fly;

TEST(MissingFunctionTests, Dummy) {
    array A = randu(10, 10, f32);
    fly_print(A);
    fly_print(arg(A));
    fly_print(arg(complex(A, A)));
    fly_print(trunc(3 * A));
    fly_print(factorial(ceil(2 * A)));
    fly_print(pow2(A));
    fly_print(root(2, A));
    fly_print(A - 0.5);
    fly_print(sign(A - 0.5));
    fly_print(minfilt(A, 3, 3) - erode(A, constant(1, 3, 3)));
    fly_print(maxfilt(A, 3, 3) - dilate(A, constant(1, 3, 3)));
    printf("%lf\n", norm(A));
}
