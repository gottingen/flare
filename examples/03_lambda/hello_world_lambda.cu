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
#include <typeinfo>

//
// "Hello world" parallel_for example:
//   1. Start up flare
//   2. Execute a parallel for loop in the default execution space,
//      using a C++11 lambda to define the loop body
//   3. Shut down flare
//
// This example only builds if C++11 is enabled.  Compare this example
// to 01_hello_world, which uses functors (explicitly defined classes)
// to define the loop body of the parallel_for.  Both functors and
// lambdas have their places.
//

int main(int argc, char* argv[]) {
  // You must call initialize() before you may call flare.
  //
  // With no arguments, this initializes the default execution space
  // (and potentially its host execution space) with default
  // parameters.  You may also pass in argc and argv, analogously to
  // MPI_Init().  It reads and removes command-line arguments that
  // start with "--flare-".
  flare::initialize(argc, argv);

  // Print the name of flare' default execution space.  We're using
  // typeid here, so the name might get a bit mangled by the linker,
  // but you should still be able to figure out what it is.
  printf("Hello World on flare execution space %s\n",
         typeid(flare::DefaultExecutionSpace).name());

  // Run lambda on the default flare execution space in parallel,
  // with a parallel for loop count of 15.  The lambda's argument is
  // an integer which is the parallel for's loop index.  As you learn
  // about different kinds of parallelism, you will find out that
  // there are other valid argument types as well.
  //
  // For a single level of parallelism, we prefer that you use the
  // FLARE_LAMBDA macro.  If CUDA is disabled, this just turns into
  // [=].  That captures variables from the surrounding scope by
  // value.  Do NOT capture them by reference!  If CUDA is enabled,
  // this macro may have a special definition that makes the lambda
  // work correctly with CUDA.  Compare to the FLARE_INLINE_FUNCTION
  // macro, which has a special meaning if CUDA is enabled.
  //
  // The following parallel_for would look like this if we were using
  // OpenMP by itself, instead of flare:
  //
  // #pragma omp parallel for
  // for (int i = 0; i < 15; ++i) {
  //   printf ("Hello from i = %i\n", i);
  // }
  //
  // You may notice that the printed numbers do not print out in
  // order.  Parallel for loops may execute in any order.
  // We also need to protect the usage of a lambda against compiling
  // with a backend which doesn't support it (i.e. Cuda 6.5/7.0).
#if defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)
  flare::parallel_for(
      15, FLARE_LAMBDA(const int i) {
        // printf works in a CUDA parallel kernel; std::ostream does not.
        printf("Hello from i = %i\n", i);
      });
#endif
  // You must call finalize() after you are done using flare.
  flare::finalize();
}
