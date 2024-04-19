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

#include <err_oneapi.hpp>
#include <identity.hpp>
#include <solve.hpp>

#if defined(WITH_LINEAR_ALGEBRA)
#include <platform.hpp>

namespace flare {
namespace oneapi {

template<typename T>
Array<T> inverse(const Array<T> &in) {
    Array<T> I = identity<T>(in.dims());
    return solve<T>(in, I);
}

#define INSTANTIATE(T) template Array<T> inverse<T>(const Array<T> &in);

INSTANTIATE(float)
INSTANTIATE(cfloat)
INSTANTIATE(double)
INSTANTIATE(cdouble)

}  // namespace oneapi
}  // namespace flare

#else  // WITH_LINEAR_ALGEBRA

namespace flare {
namespace oneapi {

template<typename T>
Array<T> inverse(const Array<T> &in) {
    ONEAPI_NOT_SUPPORTED("");
    FLY_ERROR("Linear Algebra is disabled on OneAPI backend",
             FLY_ERR_NOT_CONFIGURED);
}

#define INSTANTIATE(T) template Array<T> inverse<T>(const Array<T> &in);

INSTANTIATE(float)
INSTANTIATE(cfloat)
INSTANTIATE(double)
INSTANTIATE(cdouble)

}  // namespace oneapi
}  // namespace flare

#endif
