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

#if CPLX
inline bool is_nan(T in) { return (in.x != in.x) || (in.y != in.y); }
#else
inline bool is_nan(T in) { return (in != in); }
#endif

#if CPLX
#define sabs(in) ((in.x) * (in.x) + (in.y) * (in.y))
#ifdef MIN_OP
void binOp(T *lhs, uint *lidx, T rhs, uint ridx) {
    if (((sabs(lhs[0]) > sabs(rhs)) ||
         (sabs(lhs[0]) == sabs(rhs) && *lidx < ridx))) {
        *lhs  = rhs;
        *lidx = ridx;
    }
}
#endif

#ifdef MAX_OP
void binOp(T *lhs, uint *lidx, T rhs, uint ridx) {
    if (((sabs(lhs[0]) < sabs(rhs)) ||
         (sabs(lhs[0]) == sabs(rhs) && *lidx > ridx))) {
        *lhs  = rhs;
        *lidx = ridx;
    }
}
#endif
#else
#ifdef MIN_OP
void binOp(T *lhs, uint *lidx, T rhs, uint ridx) {
    if (((*lhs > rhs) || (*lhs == rhs && *lidx < ridx))) {
        *lhs  = rhs;
        *lidx = ridx;
    }
}
#endif

#ifdef MAX_OP
void binOp(T *lhs, uint *lidx, T rhs, uint ridx) {
    if (((*lhs < rhs) || (*lhs == rhs && *lidx > ridx))) {
        *lhs  = rhs;
        *lidx = ridx;
    }
}
#endif
#endif
