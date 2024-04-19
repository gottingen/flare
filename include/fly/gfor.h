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

#pragma once
#include <fly/defines.h>
#include <fly/seq.h>


#ifdef __cplusplus
namespace fly
{
class array;
class dim4;

FLY_API bool gforToggle();
FLY_API bool gforGet();
FLY_API void gforSet(bool val);


#define gfor(var, ...) for (var = fly::seq(fly::seq(__VA_ARGS__), true); fly::gforToggle(); )

typedef array (*batchFunc_t)(const array &lhs, const array &rhs);
FLY_API array batchFunc(const array &lhs, const array &rhs, batchFunc_t func);

}
#endif
