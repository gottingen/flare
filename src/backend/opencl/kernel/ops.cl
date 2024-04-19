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

#define IS_NAN(in) !((in) == (in))

#ifdef ADD_OP
T binOp(T lhs, T rhs) { return lhs + rhs; }

To transform(Ti in) { return (To)(in); }
#endif

#ifdef MUL_OP
#if CPLX
T binOp(T lhs, T rhs) {
    T out;
    out.x = lhs.x * rhs.x - lhs.y * rhs.y;
    out.y = lhs.x * rhs.y + lhs.y * rhs.x;
    return out;
}
#else
T binOp(T lhs, T rhs) { return lhs * rhs; }
#endif

To transform(Ti in) { return (To)(in); }
#endif

#ifdef OR_OP
uchar binOp(uchar lhs, uchar rhs) { return lhs || rhs; }

#if CPLX
uchar transform(Ti in) { return (in.x != 0) || (in.y != 0); }
#else
uchar transform(Ti in) { return (in != 0); }
#endif
#endif

#ifdef AND_OP
uchar binOp(uchar lhs, uchar rhs) { return lhs && rhs; }

#if CPLX
uchar transform(Ti in) { return (in.x != 0) || (in.y != 0); }
#else
uchar transform(Ti in) { return (in != 0); }
#endif
#endif

#ifdef NOTZERO_OP
uint binOp(uint lhs, uint rhs) { return lhs + rhs; }

#if CPLX
uint transform(Ti in) { return (in.x != 0) || (in.y != 0); }
#else
uint transform(Ti in) { return (in != 0); }
#endif
#endif

#ifdef MIN_OP

#if CPLX
#undef IS_NAN
#define IS_NAN(in) !((in.x) == (in.x)) || !((in.y) == (in.y))
#endif

T transform(T in) {
    T val = init;
    return IS_NAN(in) ? (val) : (in);
}

#if CPLX
#define sabs(in) ((in.x) * (in.x) + (in.y) * (in.y))
#else
#define sabs(in) in
#endif

T binOp(T lhs, T rhs) { return sabs(lhs) < sabs(rhs) ? lhs : rhs; }
#endif

#ifdef MAX_OP

#if CPLX
#undef IS_NAN
#define IS_NAN(in) !((in.x) == (in.x)) || !((in.y) == (in.y))
#endif

T transform(T in) {
    T val = init;
    return IS_NAN(in) ? (val) : (in);
}

#if CPLX
#define sabs(in) ((in.x) * (in.x) + (in.y) * (in.y))
#else
#define sabs(in) in
#endif

T binOp(T lhs, T rhs) { return sabs(lhs) > sabs(rhs) ? lhs : rhs; }
#endif
