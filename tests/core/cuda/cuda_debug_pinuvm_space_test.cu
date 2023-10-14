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
#include <cuda_category_test.h>
#include <doctest.h>

namespace Test {

template <class Tensor>
struct CopyFunctor {
  Tensor a;
  Tensor b;

  CopyFunctor(int N) : a(Tensor("A", N)), b(Tensor("B", N)) {}

  FLARE_INLINE_FUNCTION
  void operator()(int i) const { a(i) = b(i); }

  double time_copy(int R) {
    flare::parallel_for("CopyFunctor::time_copy", a.extent(0), *this);
    flare::fence();

    flare::Timer timer;
    for (int r = 0; r < R; r++)
      flare::parallel_for("CopyFunctor::time_copy", a.extent(0), *this);
    flare::fence();
    return timer.seconds();
  }
};

TEST_CASE("cuda, debug_pin_um_to_host") {
  double time_cuda_space;
  double time_cuda_host_pinned_space;
  double time_cuda_uvm_space_not_pinned_1;
  double time_cuda_uvm_space_pinned;
  double time_cuda_uvm_space_not_pinned_2;

  int N = 10000000;
  int R = 100;
  {
    CopyFunctor<flare::Tensor<int*, flare::CudaSpace>> f(N);
    time_cuda_space = f.time_copy(R);
  }
  {
    CopyFunctor<flare::Tensor<int*, flare::CudaHostPinnedSpace>> f(N);
    time_cuda_host_pinned_space = f.time_copy(R);
  }
  {
    CopyFunctor<flare::Tensor<int*, flare::CudaUVMSpace>> f(N);
    time_cuda_uvm_space_not_pinned_1 = f.time_copy(R);
  }
  {
#ifdef FLARE_IMPL_DEBUG_CUDA_PIN_UVM_TO_HOST
    flare_impl_cuda_set_pin_uvm_to_host(true);
#endif
    CopyFunctor<flare::Tensor<int*, flare::CudaUVMSpace>> f(N);
    time_cuda_uvm_space_pinned = f.time_copy(R);
#ifdef FLARE_IMPL_DEBUG_CUDA_PIN_UVM_TO_HOST
    flare_impl_cuda_set_pin_uvm_to_host(false);
#endif
  }
  {
    CopyFunctor<flare::Tensor<int*, flare::CudaUVMSpace>> f(N);
    time_cuda_uvm_space_not_pinned_2 = f.time_copy(R);
  }
  bool uvm_approx_cuda_1 =
      time_cuda_uvm_space_not_pinned_1 < time_cuda_space * 2.0;
  bool uvm_approx_cuda_2 =
      time_cuda_uvm_space_not_pinned_2 < time_cuda_space * 2.0;
  bool pinned_slower_cuda = time_cuda_host_pinned_space > time_cuda_space * 2.0;
  bool uvm_pinned_slower_cuda =
      time_cuda_uvm_space_pinned > time_cuda_space * 2.0;

  bool passed = uvm_approx_cuda_1 && uvm_approx_cuda_2 && pinned_slower_cuda &&
#ifdef FLARE_IMPL_DEBUG_CUDA_PIN_UVM_TO_HOST
                uvm_pinned_slower_cuda;
#else
                !uvm_pinned_slower_cuda;
#endif
  if (!passed)
    printf(
        "Time CudaSpace: %lf CudaUVMSpace_1: %lf CudaUVMSpace_2: %lf "
        "CudaPinnedHostSpace: %lf CudaUVMSpace_Pinned: %lf\n",
        time_cuda_space, time_cuda_uvm_space_not_pinned_1,
        time_cuda_uvm_space_not_pinned_2, time_cuda_host_pinned_space,
        time_cuda_uvm_space_pinned);
  REQUIRE(passed);
}

}  // namespace Test
