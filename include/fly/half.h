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

typedef struct {
    union {
        unsigned short data_ : 16;
        struct {
            unsigned short fraction : 10;
            unsigned short exponent : 5;
            unsigned short sign : 1;
        };
    };
} fly_half;

#ifdef __cplusplus
namespace fly {
#endif
typedef fly_half half;
#ifdef __cplusplus
}
#endif
