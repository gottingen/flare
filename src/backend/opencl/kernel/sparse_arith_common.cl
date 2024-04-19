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

T _add_(T v1, T v2) { return v1 + v2; }

T _sub_(T v1, T v2) { return v1 - v2; }

#if IS_CPLX
T _mul_(T v1, T v2) {
    T out;
    out.x = v1.x * v2.x - v1.y * v2.y;
    out.y = v1.x * v2.y + v1.y * v2.x;
    return out;
}

T _div_(T v1, T v2) {
    T out;
    out.x = (v1.x * v2.x + v1.y * v2.y) / (v2.x * v2.x + v2.y * v2.y);
    out.y = (v1.y * v2.x - v1.x * v2.y) / (v2.x * v2.x + v2.y * v2.y);
    return out;
}
#else
T _mul_(T v1, T v2) { return v1 * v2; }

T _div_(T v1, T v2) { return v1 / v2; }
#endif

#define ADD _add_
#define SUB _sub_
#define MUL _mul_
#define DIV _div_
