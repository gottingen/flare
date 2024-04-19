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

#include <flare.h>

int main() {
    fly_array out = 0;
    dim_t s[]    = {10, 10, 1, 1};
    fly_err e     = fly_randu(&out, 4, s, f32);
    if (out != 0) fly_release_array(out);
    return (FLY_SUCCESS != e);
}
