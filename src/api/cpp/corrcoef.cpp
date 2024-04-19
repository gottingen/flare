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

#include <fly/array.h>
#include <fly/statistics.h>
#include "error.hpp"

namespace fly {

#define INSTANTIATE_CORRCOEF(T)                               \
    template<>                                                \
    FLY_API T corrcoef(const array& X, const array& Y) {        \
        double real;                                          \
        FLY_THROW(fly_corrcoef(&real, NULL, X.get(), Y.get())); \
        return (T)real;                                       \
    }

INSTANTIATE_CORRCOEF(float);
INSTANTIATE_CORRCOEF(double);
INSTANTIATE_CORRCOEF(int);
INSTANTIATE_CORRCOEF(unsigned int);
INSTANTIATE_CORRCOEF(char);
INSTANTIATE_CORRCOEF(unsigned char);
INSTANTIATE_CORRCOEF(long long);
INSTANTIATE_CORRCOEF(unsigned long long);
INSTANTIATE_CORRCOEF(short);
INSTANTIATE_CORRCOEF(unsigned short);

#undef INSTANTIATE_CORRCOEF

}  // namespace fly
