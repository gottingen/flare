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

// header with cuda backend specific
// Array class implementation that inherits
// ArrayInfo base class
#include <Array.hpp>

#include <exampleFunction.hpp>  // cuda backend function header

// error check functions and Macros
// specific to cuda backend
#include <err_cuda.hpp>

// this header is under the folder src/cuda/kernel
// defines the CUDA kernel and its wrapper
// function to which the main computation of your
// algorithm should be relayed to
#include <kernel/exampleFunction.hpp>

using fly::dim4;

namespace flare {
namespace cuda {

template<typename T>
Array<T> exampleFunction(const Array<T> &a, const Array<T> &b,
                         const fly_someenum_t method) {
    dim4 outputDims;  // this should be '= in.dims();' in most cases
                      // but would definitely depend on the type of
                      // algorithm you are implementing.

    Array<T> out = createEmptyArray<T>(outputDims);
    // Please use the create***Array<T> helper
    // functions defined in Array.hpp to create
    // different types of Arrays. Please check the
    // file to know what are the different types you
    // can create.

    // Relay the actual computation to CUDA kernel wrapper
    kernel::exampleFunc<T>(out, a, b, method);

    return out;  // return the result
}

#define INSTANTIATE(T)                                                         \
    template Array<T> exampleFunction<T>(const Array<T> &a, const Array<T> &b, \
                                         const fly_someenum_t method);

// INSTANTIATIONS for all the types which
// are present in the switch case statement
// in src/api/c/exampleFunction.cpp should be available
INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)

}  // namespace cuda
}  // namespace flare
