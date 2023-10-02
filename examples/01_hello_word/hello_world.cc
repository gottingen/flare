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
//      using a functor to define the loop body
//   3. Shut down flare
//
// If flare was built with C++11 enabled, try comparing this example
// to 01_hello_world_lambda.  The latter uses C++11 lambdas (anonymous
// functions) to define the loop body of the parallel_for.  That makes
// the code much more concise and readable.  On the other hand,
// breaking out the loop body into an explicit functor makes it easier
// to test the loop independently of the parallel pattern.
//

// Functor that defines the parallel_for's loop body.
//
// A "functor" is just a class or struct with a public operator()
// instance method.
struct hello_world {
  // If a functor has an "execution_space" (or "execution_space", for
  // backwards compatibility) public alias, parallel_* will only run
  // the functor in that execution space.  That's a good way to mark a
  // functor as specific to an execution space.  If the functor lacks
  // this alias, parallel_for will run it in the default execution
  // space, unless you tell it otherwise (that's an advanced topic;
  // see "execution policies").

  // The functor's operator() defines the loop body.  It takes an
  // integer argument which is the parallel for loop index.  Other
  // arguments are possible; see the "hierarchical parallelism" part
  // of the tutorial.
  //
  // The operator() method must be const, and must be marked with the
  // FLARE_INLINE_FUNCTION macro.  If building with CUDA, this macro
  // will mark your method as suitable for running on the CUDA device
  // (as well as on the host).  If not building with CUDA, the macro
  // is unnecessary but harmless.
  FLARE_INLINE_FUNCTION
  void operator()(const int i) const {
    printf("Hello from i = %i\n", i);

  }
};

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

  // Run the above functor on the default flare execution space in
  // parallel, with a parallel for loop count of 15.
  //
  // The flare::DefaultExecutionSpace alias gives the default
  // execution space.  Depending on how flare was configured, this
  // could be OpenMP, Threads, Cuda, Serial, or even some other
  // execution space.
  //
  // The following line of code would look like this in OpenMP:
  //
  // #pragma omp parallel for
  // for (int i = 0; i < 15; ++i) {
  //   printf ("Hello from i = %i\n", i);
  // }
  //
  // You may notice that the printed numbers do not print out in
  // order.  Parallel for loops may execute in any order.
  flare::parallel_for("HelloWorld", 15, hello_world());

  // You must call finalize() after you are done using flare.
  flare::finalize();
}
