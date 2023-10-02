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
// MDRangePolicy example with parallel_for and parallel_reduce:
//   1. Start up flare
//   2. Execute a parallel_for loop in the default execution space,
//      using a functor to define the loop body
//   3. Shut down flare
//
// Two examples are provided:
// Example 1: Rank 2 case with minimal default parameters and arguments used
//            in the MDRangePolicy
//
// Example 2: Rank 3 case with additional outer/inner iterate pattern parameters
//            and tile dims passed to the ctor

// Simple functor for computing/storing the product of indices in a View v
template <class ViewType>
struct MDFunctor2D {
  using value_type = long;

  ViewType v;
  size_t size;

  MDFunctor2D(const ViewType& v_, const size_t size_) : v(v_), size(size_) {}

  // 2D case - used by parallel_for
  FLARE_INLINE_FUNCTION
  void operator()(const int i, const int j) const {
    v(i, j) = i * j;  // compute the product of indices
  }

  // 2D case - reduction
  FLARE_INLINE_FUNCTION
  void operator()(const int i, const int j, value_type& incorrect_count) const {
    if (v(i, j) != i * j) {
      incorrect_count += 1;
    }
  }
};

template <class ViewType>
struct MDFunctor3D {
  using value_type = long;

  ViewType v;
  size_t size;

  MDFunctor3D(const ViewType& v_, const size_t size_) : v(v_), size(size_) {}

  // 3D case - used by parallel_for
  FLARE_INLINE_FUNCTION
  void operator()(const int i, const int j, const int k) const {
    v(i, j, k) = i * j * k;  // compute the product of indices
  }

  // 3D case - reduction
  FLARE_INLINE_FUNCTION
  void operator()(const int i, const int j, const int k,
                  value_type& incorrect_count) const {
    if (v(i, j, k) != i * j * k) {
      incorrect_count += 1;
    }
  }
};

int main(int argc, char* argv[]) {
  flare::initialize(argc, argv);

  // Bound(s) for MDRangePolicy
  const int n = 100;

  // ViewType aliases for Rank<2>, Rank<3> for example usage
  using ScalarType  = double;
  using ViewType_2D = flare::View<ScalarType**>;
  using ViewType_3D = flare::View<ScalarType***>;

  /////////////////////////////////////////////////////////////////////////////
  // Explanation of MDRangePolicy usage, template parameters, constructor
  // arguments
  //
  // MDRangePolicy aliases for Rank<2>, Rank<3> cases
  // Required template parameters:
  //   flare::Rank<N>: where N=rank
  //
  // Optional template parameters to Rank<...>:
  //   flare::Iterate::{Default,Left,Right}: Outer iteration pattern across
  //   tiles;
  //     defaults based on the execution space similar to flare::Layout
  //   flare::Iterate::{Default,Left,Right}: Inner iteration pattern within
  //   tiles;
  //     defaults based on the execution space similar to flare::Layout
  //
  //   e.g. using rank2ll = Rank<2, Iterate::Left, Iterate::Left>;
  //
  //
  // Optional template parameters to MDRangePolicy:
  //   ExecutionSpace: flare::Serial, flare::OpenMP, flare::Cuda, etc.
  //
  //   flare::IndexType< T >: where T = int, long, unsigned int, etc.
  //
  //   struct Tag{}: A user-provided tag for tagging functor operators
  //
  //   e.g. 1:  MDRangePolicy< flare::Serial, Rank<2, Iterate::Left,
  //   Iterate::Left>, IndexType<int>, Tag > mdpolicy; e.g. 2:  MDRangePolicy<
  //   flare::Serial, rank2ll, IndexType<int>, Tag > mdpolicy;
  //
  //
  // Required arguments to ctor:
  //   {{ l0, l1, ... }}: Lower bounds, provided as flare::Array or
  //   std::initializer_list
  //   {{ u0, u1, ... }}: Upper bounds, provided as flare::Array or
  //   std::initializer_list
  //
  // Optional arguments to ctor:
  //   {{ t0, t1, ... }}: Tile dimensions, provided as flare::Array or
  //   std::initializer_list
  //                      defaults based on the execution space
  //
  //  e.g. mdpolicy( {{0,0}}, {{u0,u1}}, {{t0,t1}};
  //
  /////////////////////////////////////////////////////////////////////////////

  // Example 1:
  long incorrect_count_2d = 0;
  {
    // Rank<2> Case: Rank is provided, all other parameters are default
    using MDPolicyType_2D = flare::MDRangePolicy<flare::Rank<2> >;

    // Construct 2D MDRangePolicy: lower and upper bounds provided, tile dims
    // defaulted
    MDPolicyType_2D mdpolicy_2d({{0, 0}}, {{n, n}});

    // Construct a 2D view to store result of product of indices
    ViewType_2D v2("v2", n, n);

    // Execute parallel_for with rank 2 MDRangePolicy
    flare::parallel_for("md2d", mdpolicy_2d, MDFunctor2D<ViewType_2D>(v2, n));

    // Check results with a parallel_reduce using the MDRangePolicy
    flare::parallel_reduce("md2dredux", mdpolicy_2d,
                            MDFunctor2D<ViewType_2D>(v2, n),
                            incorrect_count_2d);

    printf("Rank 2 MDRangePolicy incorrect count: %ld\n",
           incorrect_count_2d);  // should be 0
  }

  // Example 2:
  long incorrect_count_3d = 0;
  {
    // Rank<3> Case: Rank, inner iterate pattern, outer iterate pattern provided
    using MDPolicyType_3D = flare::MDRangePolicy<
        flare::Rank<3, flare::Iterate::Left, flare::Iterate::Left> >;

    // Construct 3D MDRangePolicy: lower, upper bounds, tile dims provided
    MDPolicyType_3D mdpolicy_3d({{0, 0, 0}}, {{n, n, n}}, {{4, 4, 4}});

    // Construct a 3D view to store result of product of indices
    ViewType_3D v3("v3", n, n, n);

    // Execute parallel_for with rank 3 MDRangePolicy
    flare::parallel_for("md3d", mdpolicy_3d, MDFunctor3D<ViewType_3D>(v3, n));

    // Check results with a parallel_reduce using the MDRangePolicy
    flare::parallel_reduce("md3dredux", mdpolicy_3d,
                            MDFunctor3D<ViewType_3D>(v3, n),
                            incorrect_count_3d);

    printf("Rank 3 MDRangePolicy incorrect count: %ld\n",
           incorrect_count_3d);  // should be 0
  }

  flare::finalize();

  return (incorrect_count_2d == long(0) && incorrect_count_3d == long(0)) ? 0
                                                                          : -1;
}
