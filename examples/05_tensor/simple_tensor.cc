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
// First flare::Tensor (multidimensional array) example:
//   1. Start up flare
//   2. Allocate a flare::Tensor
//   3. Execute a parallel_for and a parallel_reduce over that Tensor's data
//   4. Shut down flare
//
// Compare this example to 03_simple_tensor_lambda, which uses C++11
// lambdas to define the loop bodies of the parallel_for and
// parallel_reduce.
//

#include <flare/core.h>
#include <cstdio>

// A flare::Tensor is an array of zero or more dimensions.  The number
// of dimensions is specified at compile time, as part of the type of
// the Tensor.  This array has two dimensions.  The first one
// (represented by the asterisk) is a run-time dimension, and the
// second (represented by [3]) is a compile-time dimension.  Thus,
// this Tensor type is an N x 3 array of type double, where N is
// specified at run time in the Tensor's constructor.
//
// The first dimension of the Tensor is the dimension over which it is
// efficient for flare to parallelize.
using tensor_type = flare::Tensor<double * [3]>;

// parallel_for functor that fills the Tensor given to its constructor.
// The Tensor must already have been allocated.
struct InitTensor {
  tensor_type a;

  // Tensors have "tensor semantics."  This means that they behave like
  // pointers, not like std::vector.  Their copy constructor and
  // operator= only do shallow copies.  Thus, you can pass Tensor
  // objects around by "value"; they won't do a deep copy unless you
  // explicitly ask for a deep copy.
  InitTensor(tensor_type a_) : a(a_) {}

  // Fill the Tensor with some data.  The parallel_for loop will iterate
  // over the Tensor's first dimension N.
  FLARE_INLINE_FUNCTION
  void operator()(const int i) const {
    // Acesss the Tensor just like a Fortran array.  The layout depends
    // on the Tensor's memory space, so don't rely on the Tensor's
    // physical memory layout unless you know what you're doing.
    a(i, 0) = 1.0 * i;
    a(i, 1) = 1.0 * i * i;
    a(i, 2) = 1.0 * i * i * i;
  }
};

// Reduction functor that reads the Tensor given to its constructor.
struct ReduceFunctor {
  tensor_type a;

  // Constructor takes Tensor by "value"; this does a shallow copy.
  ReduceFunctor(tensor_type a_) : a(a_) {}

  // If you write a functor to do a reduction, you must specify the
  // type of the reduction result via a public 'value_type' alias.
  using value_type = double;

  FLARE_INLINE_FUNCTION
  void operator()(int i, double& lsum) const {
    lsum += a(i, 0) * a(i, 1) / (a(i, 2) + 0.1);
  }
};

int main(int argc, char* argv[]) {
  flare::initialize(argc, argv);
  {
    const int N = 10;

    // Allocate the Tensor.  The first dimension is a run-time parameter
    // N.  We set N = 10 here.  The second dimension is a compile-time
    // parameter, 3.  We don't specify it here because we already set it
    // by declaring the type of the Tensor.
    //
    // Tensors get initialized to zero by default.  This happens in
    // parallel, using the Tensor's memory space's default execution
    // space.  Parallel initialization ensures first-touch allocation.
    // There is a way to shut off default initialization.
    //
    // You may NOT allocate a Tensor inside of a parallel_{for, reduce,
    // scan}.  Treat Tensor allocation as a "thread collective."
    //
    // The string "A" is just the label; it only matters for debugging.
    // Different Tensors may have the same label.
    tensor_type a("A", N);

    flare::parallel_for(N, InitTensor(a));
    double sum = 0;
    flare::parallel_reduce(N, ReduceFunctor(a), sum);
    printf("Result: %f\n", sum);
  }  // use this scope to ensure the lifetime of "A" ends before finalize
  flare::finalize();
}
