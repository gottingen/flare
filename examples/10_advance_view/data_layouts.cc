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
#include <flare/timer.h>
#include <cstdio>

// These two View types are both 2-D arrays of double.  However, they
// have different layouts in memory.  left_type has "layout left,"
// which means "column major," the same as in Fortran, the BLAS, or
// LAPACK.  right_type has "layout right," which means "row major,"
// the same as in C, C++, or Java.
using left_type = flare::View<double **, flare::LayoutLeft>;
using right_type = flare::View<double **, flare::LayoutRight>;
// This is a one-dimensional View, so the layout matters less.
// However, it still has a layout!  Since its layout is not specified
// explicitly in the type, its layout is a function of the memory
// space.  For example, the default Cuda layout is LayoutLeft, and the
// default Host layout is LayoutRight.
using view_type = flare::View<double *>;

// parallel_for functor that fills the given View with some data.  It
// expects to access the View by rows in parallel: each call i of
// operator() accesses a row.
template<class ViewType>
struct init_view {
    ViewType a;

    init_view(ViewType a_) : a(a_) {}

    using size_type = typename ViewType::size_type;

    FLARE_INLINE_FUNCTION
    void operator()(const typename ViewType::size_type i) const {
        // On CPUs this loop could be vectorized so j should do stride 1
        // access on a for optimal performance. I.e. a should be LayoutRight.
        // On GPUs threads should do coalesced loads and stores. That means
        // that i should be the stride one access for optimal performance.
        for (size_type j = 0; j < static_cast<size_type>(a.extent(1)); ++j) {
            a(i, j) = 1.0 * a.extent(0) * i + 1.0 * j;
        }
    }
};

// Compute a contraction of v1 and v2 into a:
//
//   a(i) := sum_j (v1(i,j) * v2(j,i))
//
// Since the functor is templated on the ViewTypes itself it doesn't matter what
// there layouts are. That means you can use different layouts on different
// architectures.
template<class ViewType1, class ViewType2>
struct contraction {
    view_type a;
    typename ViewType1::const_type v1;
    typename ViewType2::const_type v2;

    contraction(view_type a_, ViewType1 v1_, ViewType2 v2_)
            : a(a_), v1(v1_), v2(v2_) {}

    using size_type = typename view_type::size_type;

    // As with the initialization functor the performance of this operator
    // depends on the architecture and the chosen data layouts.
    // On CPUs optimal would be to vectorize the inner loop, so j should be the
    // stride 1 access. That means v1 should be LayoutRight and v2 LayoutLeft.
    // In order to get coalesced access on GPUs where i corresponds closely to
    // the thread Index, i must be the stride 1 dimension. That means v1 should be
    // LayoutLeft and v2 LayoutRight.
    FLARE_INLINE_FUNCTION
    void operator()(const view_type::size_type i) const {
        for (size_type j = 0; j < static_cast<size_type>(a.extent(1)); ++j) {
            a(i) = v1(i, j) * v2(j, i);
        }
    }
};

// Compute a dot product. This is used for result verification.
struct dot {
    view_type a;

    dot(view_type a_) : a(a_) {}

    using value_type = double;  // Specify type for reduction target, lsum
    FLARE_INLINE_FUNCTION
    void operator()(const view_type::size_type i, double &lsum) const {
        lsum += a(i) * a(i);
    }
};

int main(int narg, char *arg[]) {
    // When initializing flare, you may pass in command-line arguments,
    // just like with MPI_Init().  flare reserves the right to remove
    // arguments from the list that start with '--flare-'.
    flare::initialize(narg, arg);

    {
        int size = 10000;
        view_type a("A", size);

        // Define two views with LayoutLeft and LayoutRight.
        left_type l("L", size, 10000);
        right_type r("R", size, 10000);

        // Initialize the data in the views.
        flare::parallel_for(size, init_view<left_type>(l));
        flare::parallel_for(size, init_view<right_type>(r));
        flare::fence();

        // Measure time to execute the contraction kernel when giving it a
        // LayoutLeft view for v1 and a LayoutRight view for v2. This should be
        // fast on GPUs and slow on CPUs
        flare::Timer time1;
        flare::parallel_for(size, contraction<left_type, right_type>(a, l, r));
        flare::fence();
        double sec1 = time1.seconds();

        double sum1 = 0;
        flare::parallel_reduce(size, dot(a), sum1);
        flare::fence();

        // Measure time to execute the contraction kernel when giving it a
        // LayoutRight view for v1 and a LayoutLeft view for v2. This should be
        // fast on CPUs and slow on GPUs
        flare::Timer time2;
        flare::parallel_for(size, contraction<right_type, left_type>(a, r, l));
        flare::fence();
        double sec2 = time2.seconds();

        double sum2 = 0;
        flare::parallel_reduce(size, dot(a), sum2);

        // flare' reductions are deterministic.
        // The results should always be equal.
        printf("Result Left/Right %f Right/Left %f (equal result: %i)\n", sec1,
               sec2, sum2 == sum1);
    }

    flare::finalize();
}
