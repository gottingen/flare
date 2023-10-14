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
#include <cstdlib>

using tensor_type = flare::Tensor<double*>;
// flare::Tensors have an MemoryTraits template parameter which
// allows users to specify usage scenarios of a Tensor.
// Some of those act simply as hints, which can be used to insert
// optimal load and store paths, others change the symantics of the
// access. The trait flare::Atomic is one of the latter. A tensor with
// that MemoryTrait will perform any access atomicly (read, write, update).
//
// In this example we use a tensor with a usage hint for RandomAccess.
// flare::RandomAccess means that we expect to use this tensor
// with indirect indexing.
//
// In CUDA, RandomAccess allows accesses through the texture
// cache.  This only works if the Tensor is read-only, which we enforce
// through the first template parameter.
//
// Note that we are still talking about tensors of the data, its not a new
// allocation. For example you can have an atomic tensor of a default tensor. While
// you even could use both in the same kernel, this could lead to undefined
// behaviour because one of your access paths is not atomic. Think of it in the
// same way as you think of pointers to const data and pointers to non-const
// data (i.e. const double* and double*). While these pointers can point to the
// same data you should not use them together if that brakes the const guarantee
// of the first pointer.
using tensor_type_rnd =
    flare::Tensor<const double*, flare::MemoryTraits<flare::RandomAccess> >;
using idx_type      = flare::Tensor<int**>;
using idx_type_host = idx_type::HostMirror;

// We template this functor on the TensorTypes to show the effect of the
// RandomAccess trait.
template <class DestType, class SrcType>
struct localsum {
  idx_type::const_type idx;
  DestType dest;
  SrcType src;
  localsum(idx_type idx_, DestType dest_, SrcType src_)
      : idx(idx_), dest(dest_), src(src_) {}

  // Calculate a local sum of values
  FLARE_INLINE_FUNCTION
  void operator()(const int i) const {
    double tmp = 0.0;
    for (int j = 0; j < (int)idx.extent(1); ++j) {
      // This is an indirect access on src
      const double val = src(idx(i, j));
      tmp += val * val + 0.5 * (idx.extent(0) * val - idx.extent(1) * val);
    }
    dest(i) = tmp;
  }
};

int main(int narg, char* arg[]) {
  flare::initialize(narg, arg);

  {
    int size = 1000000;

    idx_type idx("Idx", size, 64);
    idx_type_host h_idx = flare::create_mirror_tensor(idx);

    tensor_type dest("Dest", size);
    tensor_type src("Src", size);

    srand(134231);

    using size_type = tensor_type::size_type;
    for (int i = 0; i < size; i++) {
      for (size_type j = 0; j < static_cast<size_type>(h_idx.extent(1)); ++j) {
        h_idx(i, j) = (size + i + (rand() % 500 - 250)) % size;
      }
    }

    // Deep copy the initial data to the device
    flare::deep_copy(idx, h_idx);
    // Run the first kernel to warmup caches
    flare::parallel_for(size,
                         localsum<tensor_type, tensor_type_rnd>(idx, dest, src));
    flare::fence();

    // Run the localsum functor using the RandomAccess trait. On CPUs there
    // should not be any different in performance to not using the RandomAccess
    // trait. On GPUs where can be a dramatic difference
    flare::Timer time1;
    flare::parallel_for(size,
                         localsum<tensor_type, tensor_type_rnd>(idx, dest, src));
    flare::fence();
    double sec1 = time1.seconds();

    flare::Timer time2;
    flare::parallel_for(size, localsum<tensor_type, tensor_type>(idx, dest, src));
    flare::fence();
    double sec2 = time2.seconds();

    printf("Time with Trait RandomAccess: %f with Plain: %f \n", sec1, sec2);
  }

  flare::finalize();
}
