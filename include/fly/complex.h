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
#include "fly/defines.h"


#ifdef __cplusplus
#include <ostream>
#include <istream>

namespace fly{
#endif

#ifdef __cplusplus
extern "C" {
#endif
typedef struct fly_cfloat {
    float real;
    float imag;
#ifdef __cplusplus
    fly_cfloat(const float _real = 0, const float _imag = 0) :real(_real), imag(_imag) {}
#endif
} fly_cfloat;

typedef struct fly_cdouble {
    double real;
    double imag;
#ifdef __cplusplus
    fly_cdouble(const double _real = 0, const double _imag = 0) :real(_real), imag(_imag) {}
#endif
} fly_cdouble;
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
typedef fly::fly_cfloat   cfloat;
typedef fly::fly_cdouble  cdouble;

FLY_API float real(fly_cfloat val);
FLY_API double real(fly_cdouble val);

FLY_API float imag(fly_cfloat val);
FLY_API double imag(fly_cdouble val);

// +,-,*,/ for (cfloat, cfloat) and (cdouble, cdouble)
#define DEFINE_OP(OP)                                                               \
    FLY_API fly::cfloat  operator OP(const fly::cfloat  &lhs, const fly::cfloat  &rhs);  \
    FLY_API fly::cdouble operator OP(const fly::cdouble &lhs, const fly::cdouble &rhs);  \

DEFINE_OP(+)
DEFINE_OP(-)
DEFINE_OP(*)
DEFINE_OP(/)

#undef DEFINE_OP

// +,/ for (cfloat, double) and (cdouble, double)
#define DEFINE_OP(OP)                                                               \
    FLY_API fly::cfloat  operator OP(const fly::cfloat  &lhs, const     double  &rhs);  \
    FLY_API fly::cdouble operator OP(const fly::cdouble &lhs, const     double  &rhs);  \

DEFINE_OP(+)
DEFINE_OP(/)

#undef DEFINE_OP

// -,* for (cfloat, double) and (cdouble, double)
#define DEFINE_OP(OP)                                                               \
    FLY_API fly::cfloat  operator OP(const fly::cfloat  &lhs, const     double  &rhs);  \
    FLY_API fly::cdouble operator OP(const fly::cdouble &lhs, const     double  &rhs);  \

DEFINE_OP(-)
DEFINE_OP(*)

#undef DEFINE_OP

// +, -, *, / for (double, cfloat/cdouble) and (cfloat/cdouble, cdouble/cfloat)
#define DEFINE_OP(OP)                                                               \
    FLY_API fly::cfloat  operator OP(const double      &rhs, const fly::cfloat  &lhs);  \
    FLY_API fly::cdouble operator OP(const double      &rhs, const fly::cdouble &lhs);  \
    FLY_API fly::cdouble operator OP(const fly::cfloat  &lhs, const fly::cdouble &rhs);  \
    FLY_API fly::cdouble operator OP(const fly::cdouble &lhs, const fly::cfloat  &rhs);  \

DEFINE_OP(+)
DEFINE_OP(-)
DEFINE_OP(*)
DEFINE_OP(/)

#undef DEFINE_OP

FLY_API bool operator==(const cfloat &lhs, const cfloat &rhs);
FLY_API bool operator==(const cdouble &lhs, const cdouble &rhs);

FLY_API bool operator!=(const cfloat &lhs, const cfloat &rhs);
FLY_API bool operator!=(const cdouble &lhs, const cdouble &rhs);

FLY_API std::istream& operator>> (std::istream &is, cfloat &in);
FLY_API std::istream& operator>> (std::istream &is, cdouble &in);

FLY_API std::ostream& operator<< (std::ostream &os, const cfloat &in);
FLY_API std::ostream& operator<< (std::ostream &os, const cdouble &in);


FLY_API float abs(const cfloat &val);
FLY_API double abs(const cdouble &val);

FLY_API cfloat conj(const cfloat &val);
FLY_API cdouble conj(const cdouble &val);

}
#endif
