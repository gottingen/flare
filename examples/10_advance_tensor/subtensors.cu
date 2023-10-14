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

// This example simulates one timestep of an explicit
// finite-difference discretization of a time-dependent partial
// differential equation (PDE).  It shows how to take subtensors of the
// mesh in order to represent particular boundaries or the interior of
// the mesh.

#include <flare/core.h>
#include <flare/timer.h>
#include <cstdio>

using mesh_type = flare::Tensor<double***, flare::LayoutRight>;

// These Tensor types represent subtensors of the mesh.  Some of the Tensors
// have layout LayoutStride, meaning that they have run-time "strides"
// in each dimension which may differ from that dimension.  For
// example, inner_mesh_type (which represents the interior of the
// mesh) has to skip over the boundaries when computing its stride;
// the dimensions of the interior mesh differ from these strides.  You
// may safely always use a LayoutStride layout when taking a subtensor
// of a LayoutRight or LayoutLeft subtensor, but strided accesses may
// cost a bit more, especially for 1-D Tensors.
using xz_plane_type   = flare::Tensor<double**, flare::LayoutStride>;
using yz_plane_type   = flare::Tensor<double**, flare::LayoutRight>;
using xy_plane_type   = flare::Tensor<double**, flare::LayoutStride>;
using inner_mesh_type = flare::Tensor<double***, flare::LayoutStride>;

// Functor to set all entries of a boundary of the mesh to a constant
// value.  The functor is templated on TensorType because different
// boundaries may have different layouts.
template <class TensorType>
struct set_boundary {
  TensorType a;
  double value;

  set_boundary(TensorType a_, double value_) : a(a_), value(value_) {}

  using size_type = typename TensorType::size_type;

  FLARE_INLINE_FUNCTION
  void operator()(const size_type i) const {
    for (size_type j = 0; j < static_cast<size_type>(a.extent(1)); ++j) {
      a(i, j) = value;
    }
  }
};

// Functor to set all entries of a boundary of the mesh to a constant
// value.  The functor is templated on TensorType because different
// boundaries may have different layouts.
template <class TensorType>
struct set_inner {
  TensorType a;
  double value;

  set_inner(TensorType a_, double value_) : a(a_), value(value_) {}

  using size_type = typename TensorType::size_type;

  FLARE_INLINE_FUNCTION
  void operator()(const size_type i) const {
    for (size_type j = 0; j < static_cast<size_type>(a.extent(1)); ++j) {
      for (size_type k = 0; k < static_cast<size_type>(a.extent(2)); ++k) {
        a(i, j, k) = value;
      }
    }
  }
};

// Update the interior of the mesh.  This simulates one timestep of a
// finite-difference method.
template <class TensorType>
struct update {
  TensorType a;
  const double dt;

  update(TensorType a_, const double dt_) : a(a_), dt(dt_) {}

  using size_type = typename TensorType::size_type;

  FLARE_INLINE_FUNCTION
  void operator()(size_type i) const {
    i++;
    for (size_type j = 1; j < static_cast<size_type>(a.extent(1) - 1); j++) {
      for (size_type k = 1; k < static_cast<size_type>(a.extent(2) - 1); k++) {
        a(i, j, k) += dt * (a(i, j, k + 1) - a(i, j, k - 1) + a(i, j + 1, k) -
                            a(i, j - 1, k) + a(i + 1, j, k) - a(i - 1, j, k));
      }
    }
  }
};

int main(int narg, char* arg[]) {
  using flare::ALL;
  using flare::pair;
  using flare::parallel_for;
  using flare::subtensor;
  using size_type = mesh_type::size_type;

  flare::initialize(narg, arg);

  {
    // The number of mesh points along each dimension of the mesh, not
    // including boundaries.
    const size_type size = 100;

    // A is the full cubic 3-D mesh, including the boundaries.
    mesh_type A("A", size + 2, size + 2, size + 2);
    // Ai is the "inner" part of A, _not_ including the boundaries.
    //
    // A pair of indices in a particular dimension means the contiguous
    // zero-based index range in that dimension, including the first
    // entry of the pair but _not_ including the second entry.
    inner_mesh_type Ai = subtensor(A, pair<size_type, size_type>(1, size + 1),
                                 pair<size_type, size_type>(1, size + 1),
                                 pair<size_type, size_type>(1, size + 1));
    // A has six boundaries, one for each face of the cube.
    // Create a Tensor of each of these boundaries.
    // ALL() means "select all indices in that dimension."
    xy_plane_type Zneg_halo = subtensor(A, ALL(), ALL(), 0);
    xy_plane_type Zpos_halo = subtensor(A, ALL(), ALL(), 101);
    xz_plane_type Yneg_halo = subtensor(A, ALL(), 0, ALL());
    xz_plane_type Ypos_halo = subtensor(A, ALL(), 101, ALL());
    yz_plane_type Xneg_halo = subtensor(A, 0, ALL(), ALL());
    yz_plane_type Xpos_halo = subtensor(A, 101, ALL(), ALL());

    // Set the boundaries to their initial conditions.
    parallel_for(Zneg_halo.extent(0),
                 set_boundary<xy_plane_type>(Zneg_halo, 1));
    parallel_for(Zpos_halo.extent(0),
                 set_boundary<xy_plane_type>(Zpos_halo, -1));
    parallel_for(Yneg_halo.extent(0),
                 set_boundary<xz_plane_type>(Yneg_halo, 2));
    parallel_for(Ypos_halo.extent(0),
                 set_boundary<xz_plane_type>(Ypos_halo, -2));
    parallel_for(Xneg_halo.extent(0),
                 set_boundary<yz_plane_type>(Xneg_halo, 3));
    parallel_for(Xpos_halo.extent(0),
                 set_boundary<yz_plane_type>(Xpos_halo, -3));

    // Set the interior of the mesh to its initial condition.
    parallel_for(Ai.extent(0), set_inner<inner_mesh_type>(Ai, 0));

    // Update the interior of the mesh.
    // This simulates one timestep with dt = 0.1.
    parallel_for(Ai.extent(0), update<mesh_type>(A, 0.1));

    flare::fence();
    printf("Done\n");
  }
  flare::finalize();
}
