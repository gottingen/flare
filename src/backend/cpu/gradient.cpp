// Copyright 2023 The EA Authors.
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

#include <Array.hpp>
#include <err_cpu.hpp>
#include <gradient.hpp>
#include <kernel/gradient.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <stdexcept>

namespace flare {
namespace cpu {

template<typename T>
void gradient(Array<T> &grad0, Array<T> &grad1, const Array<T> &in) {
    getQueue().enqueue(kernel::gradient<T>, grad0, grad1, in);
}

#define INSTANTIATE(T)                                            \
    template void gradient<T>(Array<T> & grad0, Array<T> & grad1, \
                              const Array<T> &in);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)

}  // namespace cpu
}  // namespace flare
