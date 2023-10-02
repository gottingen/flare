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

//
// First flare::View (multidimensional array) example:
//   1. Start up flare
//   2. Allocate a flare::View
//   3. Execute a parallel_for and a parallel_reduce over that View's data
//   4. Shut down flare
//
// Compare this example to 03_simple_view, which uses functors to
// define the loop bodies of the parallel_for and parallel_reduce.
//

#include <flare/core.h>
#include <cstdio>

// A flare::View is an array of zero or more dimensions.  The number
// of dimensions is specified at compile time, as part of the type of
// the View.  This array has two dimensions.  The first one
// (represented by the asterisk) is a run-time dimension, and the
// second (represented by [3]) is a compile-time dimension.  Thus,
// this View type is an N x 3 array of type double, where N is
// specified at run time in the View's constructor.
//
// The first dimension of the View is the dimension over which it is
// efficient for flare to parallelize.
using view_type = flare::View<double * [3]>;

int main(int argc, char* argv[]) {
  flare::initialize(argc, argv);

  {
    // Allocate the View.  The first dimension is a run-time parameter
    // N.  We set N = 10 here.  The second dimension is a compile-time
    // parameter, 3.  We don't specify it here because we already set it
    // by declaring the type of the View.
    //
    // Views get initialized to zero by default.  This happens in
    // parallel, using the View's memory space's default execution
    // space.  Parallel initialization ensures first-touch allocation.
    // There is a way to shut off default initialization.
    //
    // You may NOT allocate a View inside of a parallel_{for, reduce,
    // scan}.  Treat View allocation as a "thread collective."
    //
    // The string "A" is just the label; it only matters for debugging.
    // Different Views may have the same label.
    view_type a("A", 10);

// Fill the View with some data.  The parallel_for loop will iterate
// over the View's first dimension N.
//
// Note that the View is passed by value into the lambda.  The macro
// FLARE_LAMBDA includes the "capture by value" clause [=].  This
// tells the lambda to "capture all variables in the enclosing scope
// by value."  Views have "view semantics"; they behave like
// pointers, not like std::vector.  Passing them by value does a
// shallow copy.  A deep copy never happens unless you explicitly
// ask for one.
// We also need to protect the usage of a lambda against compiling
// with a backend which doesn't support it (i.e. Cuda 6.5/7.0).
#if defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)
    flare::parallel_for(
        10, FLARE_LAMBDA(const int i) {
          // Acesss the View just like a Fortran array.  The layout depends
          // on the View's memory space, so don't rely on the View's
          // physical memory layout unless you know what you're doing.
          a(i, 0) = 1.0 * i;
          a(i, 1) = 1.0 * i * i;
          a(i, 2) = 1.0 * i * i * i;
        });
    // Reduction functor that reads the View given to its constructor.
    double sum = 0;
    flare::parallel_reduce(
        10,
        FLARE_LAMBDA(const int i, double& lsum) {
          lsum += a(i, 0) * a(i, 1) / (a(i, 2) + 0.1);
        },
        sum);
    printf("Result: %f\n", sum);
#endif
  }
  flare::finalize();
}
