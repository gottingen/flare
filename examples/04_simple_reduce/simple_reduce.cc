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

#include <flare/core.h>
#include <cstdio>

//
// First reduction (parallel_reduce) example:
//   1. Start up flare
//   2. Execute a parallel_reduce loop in the default execution space,
//      using a functor to define the loop body
//   3. Shut down flare
//
// Compare this example to 02_simple_reduce_lambda, which uses a C++11
// lambda to define the loop body of the parallel_reduce.
//

// Reduction functor for computing the sum of squares.
//
// More advanced reduction examples will show how to control the
// reduction's "join" operator.  If the join operator is not provided,
// it defaults to binary operator+ (adding numbers together).
struct squaresum {
    // Specify the type of the reduction value with a "value_type"
    // alias.  In this case, the reduction value has type int.
    using value_type = int;

    // The reduction functor's operator() looks a little different than
    // the parallel_for functor's operator().  For the reduction, we
    // pass in both the loop index i, and the intermediate reduction
    // value lsum.  The latter MUST be passed in by nonconst reference.
    // (If the reduction type is an array like int[], indicating an
    // array reduction result, then the second argument is just int[].)
    FLARE_INLINE_FUNCTION
    void operator()(const int i, int &lsum) const {
        lsum += i * i;  // compute the sum of squares
    }
};

int main(int argc, char *argv[]) {
    flare::initialize(argc, argv);
    const int n = 10;

    // Compute the sum of squares of integers from 0 to n-1, in
    // parallel, using flare.
    int sum = 0;
    flare::parallel_reduce(n, squaresum(), sum);
    printf(
            "Sum of squares of integers from 0 to %i, "
            "computed in parallel, is %i\n",
            n - 1, sum);

    // Compare to a sequential loop.
    int seqSum = 0;
    for (int i = 0; i < n; ++i) {
        seqSum += i * i;
    }
    printf(
            "Sum of squares of integers from 0 to %i, "
            "computed sequentially, is %i\n",
            n - 1, seqSum);
    flare::finalize();
    return (sum == seqSum) ? 0 : -1;
}
