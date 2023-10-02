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
//      using a C++11 lambda to define the loop body
//   3. Shut down flare
//
// This example only builds if C++11 is enabled.  Compare this example
// to 02_simple_reduce, which uses a functor to define the loop body
// of the parallel_reduce.
//

int main(int argc, char* argv[]) {
  flare::initialize(argc, argv);
  const int n = 10;

  // Compute the sum of squares of integers from 0 to n-1, in
  // parallel, using flare.  This time, use a lambda instead of a
  // functor.  The lambda takes the same arguments as the functor's
  // operator().
  int sum = 0;
// The FLARE_LAMBDA macro replaces the capture-by-value clause [=].
// It also handles any other syntax needed for CUDA.
// We also need to protect the usage of a lambda against compiling
// with a backend which doesn't support it (i.e. Cuda 6.5/7.0).
#if defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)
  flare::parallel_reduce(
      n, FLARE_LAMBDA(const int i, int& lsum) { lsum += i * i; }, sum);
#endif
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
#if defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)
  return (sum == seqSum) ? 0 : -1;
#else
  return 0;
#endif
}
