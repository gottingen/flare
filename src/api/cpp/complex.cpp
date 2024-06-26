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

#include <fly/complex.h>
#include <cmath>
#include <complex>
#include <istream>

namespace fly {
using std::complex;

float real(fly_cfloat val) { return val.real; }
double real(fly_cdouble val) { return val.real; }

float imag(fly_cfloat val) { return val.imag; }
double imag(fly_cdouble val) { return val.imag; }

cfloat operator+(const cfloat &lhs, const cfloat &rhs) {
    cfloat out(lhs.real + rhs.real, lhs.imag + rhs.imag);
    return out;
}

cdouble operator+(const cdouble &lhs, const cdouble &rhs) {
    cdouble out(lhs.real + rhs.real, lhs.imag + rhs.imag);
    return out;
}

cfloat operator*(const cfloat &lhs, const cfloat &rhs) {
    complex<float> clhs(lhs.real, lhs.imag);
    complex<float> crhs(rhs.real, rhs.imag);
    complex<float> out = clhs * crhs;
    return {out.real(), out.imag()};
}

cdouble operator*(const cdouble &lhs, const cdouble &rhs) {
    complex<double> clhs(lhs.real, lhs.imag);
    complex<double> crhs(rhs.real, rhs.imag);
    complex<double> out = clhs * crhs;
    return {out.real(), out.imag()};
}

cfloat operator-(const cfloat &lhs, const cfloat &rhs) {
    cfloat out(lhs.real - rhs.real, lhs.imag - rhs.imag);
    return out;
}

cdouble operator-(const cdouble &lhs, const cdouble &rhs) {
    cdouble out(lhs.real - rhs.real, lhs.imag - rhs.imag);
    return out;
}

cfloat operator/(const cfloat &lhs, const cfloat &rhs) {
    complex<float> clhs(lhs.real, lhs.imag);
    complex<float> crhs(rhs.real, rhs.imag);
    complex<float> out = clhs / crhs;
    return {out.real(), out.imag()};
}

cdouble operator/(const cdouble &lhs, const cdouble &rhs) {
    complex<double> clhs(lhs.real, lhs.imag);
    complex<double> crhs(rhs.real, rhs.imag);
    complex<double> out = clhs / crhs;
    return {out.real(), out.imag()};
}

#define IMPL_OP(OP)                                              \
    cfloat operator OP(const cfloat &lhs, const double &rhs) {   \
        return lhs OP cfloat(rhs);                               \
    }                                                            \
    cdouble operator OP(const cdouble &lhs, const double &rhs) { \
        return lhs OP cdouble(rhs);                              \
    }                                                            \
    cfloat operator OP(const double &lhs, const cfloat &rhs) {   \
        return cfloat(lhs) OP rhs;                               \
    }                                                            \
    cdouble operator OP(const double &lhs, const cdouble &rhs) { \
        return cdouble(lhs) OP rhs;                              \
    }                                                            \
    cdouble operator OP(const cfloat &lhs, const cdouble &rhs) { \
        return cdouble(real(lhs), imag(lhs)) OP rhs;             \
    }                                                            \
    cdouble operator OP(const cdouble &lhs, const cfloat &rhs) { \
        return lhs OP cdouble(real(rhs), imag(rhs));             \
    }

IMPL_OP(+)
IMPL_OP(-)
IMPL_OP(*)
IMPL_OP(/)

#undef IMPL_OP

bool operator!=(const cfloat &lhs, const cfloat &rhs) { return !(lhs == rhs); }

bool operator!=(const cdouble &lhs, const cdouble &rhs) {
    return !(lhs == rhs);
}

bool operator==(const cfloat &lhs, const cfloat &rhs) {
    return lhs.real == rhs.real && lhs.imag == rhs.imag;
}

bool operator==(const cdouble &lhs, const cdouble &rhs) {
    return lhs.real == rhs.real && lhs.imag == rhs.imag;
}

float abs(const cfloat &val) {
    std::complex<float> out(val.real, val.imag);
    return abs(out);
}

double abs(const cdouble &val) {
    std::complex<double> out(val.real, val.imag);
    return abs(out);
}

cfloat conj(const cfloat &val) { return {val.real, -val.imag}; }

cdouble conj(const cdouble &val) { return {val.real, -val.imag}; }

std::ostream &operator<<(std::ostream &os, const cfloat &in) {
    os << "(" << in.real << ", " << in.imag << ")";
    return os;
}

std::ostream &operator<<(std::ostream &os, const cdouble &in) {
    os << "(" << in.real << " " << in.imag << ")";
    return os;
}

std::istream &operator>>(std::istream &is, cfloat &in) {
    char trash;
    is >> trash;
    is >> in.real;
    is >> trash;
    is >> in.imag;
    is >> trash;
    return is;
}

std::istream &operator>>(std::istream &is, cdouble &in) {
    char trash;
    is >> trash;
    is >> in.real;
    is >> trash;
    is >> in.imag;
    is >> trash;
    return is;
}

}  // namespace fly
