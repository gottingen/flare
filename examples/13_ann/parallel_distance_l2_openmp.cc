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
#include <flare/ann/distance_l2.h>
#include <flare/bench.h>
#include <chrono>
#include <flare/runtime/taskflow.h>
#include <flare/runtime/algorithm/for_each.h>

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
using tensor_type = flare::Tensor<float *, flare::Serial>;

// parallel_for functor that fills the Tensor given to its constructor.
// The Tensor must already have been allocated.
struct InitTensor1 {
    tensor_type a;

    // Tensors have "tensor semantics."  This means that they behave like
    // pointers, not like std::vector.  Their copy constructor and
    // operator= only do shallow copies.  Thus, you can pass Tensor
    // objects around by "value"; they won't do a deep copy unless you
    // explicitly ask for a deep copy.
    InitTensor1(tensor_type a_) : a(a_) {}

    // Fill the Tensor with some data.  The parallel_for loop will iterate
    // over the Tensor's first dimension N.
    FLARE_INLINE_FUNCTION
    void operator()(const int i) const {
        // Acesss the Tensor just like a Fortran array.  The layout depends
        // on the Tensor's memory space, so don't rely on the Tensor's
        // physical memory layout unless you know what you're doing.
        a(i) = i * 1.0;
    }
};

struct InitTensor2 {
    tensor_type a;

    // Tensors have "tensor semantics."  This means that they behave like
    // pointers, not like std::vector.  Their copy constructor and
    // operator= only do shallow copies.  Thus, you can pass Tensor
    // objects around by "value"; they won't do a deep copy unless you
    // explicitly ask for a deep copy.
    InitTensor2(tensor_type a_) : a(a_) {}

    // Fill the Tensor with some data.  The parallel_for loop will iterate
    // over the Tensor's first dimension N.
    FLARE_INLINE_FUNCTION
    void operator()(const int i) const {
        // Acesss the Tensor just like a Fortran array.  The layout depends
        // on the Tensor's memory space, so don't rely on the Tensor's
        // physical memory layout unless you know what you're doing.
        a(i) = i * 1.0 + 1.0;
    }
};

void for_each(flare::rt::Executor &executor, int loop, const tensor_type &a, const tensor_type &b, bool batch) {

    flare::rt::Taskflow taskflow;

    taskflow.for_each_index(0, loop, 1, [&] (int i) {
        flare::ann::distance_l2<tensor_type>(a,b, batch);
        //printf("for_each on container item: %d\n", i);
    }, flare::rt::StaticPartitioner());

    executor.run(taskflow).get();

}


int main(int argc, char* argv[]) {
    flare::initialize(argc, argv);
    {

        const int N = 512;
        const int L = 100*100*100;
        flare::rt::Executor executor(4);

        // Allocate the Tensor.  The first dimension is a run-time parameter
        // N.  We set N = 128 here.
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
        tensor_type b("B", N);

        flare::parallel_for(N, InitTensor1(a));
        flare::parallel_for(N, InitTensor2(b));
        double dis = flare::ann::distance_l2<tensor_type>(a,b, false);
        printf("Result flare::ann::distance_l1<tensor_type>(a,b, false): %f\n", dis);

        double dis1 = flare::ann::distance_l2<tensor_type>(a,b);
        printf("Result flare::ann::distance_l1<tensor_type>(a,b): %f\n", dis1);
        auto bencher = flare::Benchmarker<>{ 100, std::chrono::seconds { 100 } };

        auto stats_nor = bencher([&]() {
#pragma omp parallel for num_threads(4)
            for(auto i =0; i <  L; i++) flare::ann::distance_l2<tensor_type>(a,b, false); }
            );
        std::cout << '\n'
                  << "nor " << stats_nor << '\n';
        /*
        auto stats_batch = bencher([&]() {
            for(auto i =0; i <  L; i++) {
                flare::ann::distance_l2<tensor_type>(a, b);
            }
        });

        std::cout << '\n'
                  << flare::simd::default_arch::name()<<" batch " << stats_batch << '\n';

        auto stats_pf = bencher([&]()
                                 {  for_each(executor, L, a,b,false); });
        std::cout << '\n'
                  << "parallel nor " << stats_pf << '\n';

        auto stats_pt = bencher([&]()
                                {  for_each(executor, L, a, b,true); });
        std::cout << '\n'
                  << "parallel batch " << stats_pt << '\n';
                  */

    }  // use this scope to ensure the lifetime of "A" ends before finalize
    flare::finalize();
}
