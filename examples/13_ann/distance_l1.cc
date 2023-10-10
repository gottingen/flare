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
// Created by jeff on 23-10-11.
//

#include <flare/core.h>
#include <cstdio>
#include <flare/ann/distance_l1.h>

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
using view_type = flare::View<double *>;

// parallel_for functor that fills the View given to its constructor.
// The View must already have been allocated.
struct InitView1 {
    view_type a;

    // Views have "view semantics."  This means that they behave like
    // pointers, not like std::vector.  Their copy constructor and
    // operator= only do shallow copies.  Thus, you can pass View
    // objects around by "value"; they won't do a deep copy unless you
    // explicitly ask for a deep copy.
    InitView1(view_type a_) : a(a_) {}

    // Fill the View with some data.  The parallel_for loop will iterate
    // over the View's first dimension N.
    FLARE_INLINE_FUNCTION
    void operator()(const int i) const {
        // Acesss the View just like a Fortran array.  The layout depends
        // on the View's memory space, so don't rely on the View's
        // physical memory layout unless you know what you're doing.
        a(i) = i * 1.0;
    }
};

struct InitView2 {
    view_type a;

    // Views have "view semantics."  This means that they behave like
    // pointers, not like std::vector.  Their copy constructor and
    // operator= only do shallow copies.  Thus, you can pass View
    // objects around by "value"; they won't do a deep copy unless you
    // explicitly ask for a deep copy.
    InitView2(view_type a_) : a(a_) {}

    // Fill the View with some data.  The parallel_for loop will iterate
    // over the View's first dimension N.
    FLARE_INLINE_FUNCTION
    void operator()(const int i) const {
        // Acesss the View just like a Fortran array.  The layout depends
        // on the View's memory space, so don't rely on the View's
        // physical memory layout unless you know what you're doing.
        a(i) = i * 1.0 + 1.0;
    }
};

int main(int argc, char* argv[]) {
    flare::initialize(argc, argv);
    {
        const int N = 128;

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
        view_type a("A", N);
        view_type b("B", N);

        flare::parallel_for(N, InitView1(a));
        flare::parallel_for(N, InitView2(b));
        double dis = 0;
        dis = flare::ann::distance_l1<view_type>(a,b);
        printf("Result: %f\n", dis);
    }  // use this scope to ensure the lifetime of "A" ends before finalize
    flare::finalize();
}
