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
#include <fly/lapack.h>
#include "error.hpp"

namespace fly {
void svd(array &u, array &s, array &vt, const array &in) {
    fly_array sl = 0, ul = 0, vtl = 0;
    FLY_THROW(fly_svd(&ul, &sl, &vtl, in.get()));
    s  = array(sl);
    u  = array(ul);
    vt = array(vtl);
}

void svdInPlace(array &u, array &s, array &vt, array &in) {
    fly_array sl = 0, ul = 0, vtl = 0;
    FLY_THROW(fly_svd_inplace(&ul, &sl, &vtl, in.get()));
    s  = array(sl);
    u  = array(ul);
    vt = array(vtl);
}

void lu(array &out, array &pivot, const array &in, const bool is_lapack_piv) {
    out        = in.copy();
    fly_array p = 0;
    FLY_THROW(fly_lu_inplace(&p, out.get(), is_lapack_piv));
    pivot = array(p);
}

void lu(array &lower, array &upper, array &pivot, const array &in) {
    fly_array l = 0, u = 0, p = 0;
    FLY_THROW(fly_lu(&l, &u, &p, in.get()));
    lower = array(l);
    upper = array(u);
    pivot = array(p);
}

void luInPlace(array &pivot, array &in, const bool is_lapack_piv) {
    fly_array p = 0;
    FLY_THROW(fly_lu_inplace(&p, in.get(), is_lapack_piv));
    pivot = array(p);
}

void qr(array &out, array &tau, const array &in) {
    out        = in.copy();
    fly_array t = 0;
    FLY_THROW(fly_qr_inplace(&t, out.get()));
    tau = array(t);
}

void qr(array &q, array &r, array &tau, const array &in) {
    fly_array q_ = 0, r_ = 0, t_ = 0;
    FLY_THROW(fly_qr(&q_, &r_, &t_, in.get()));
    q   = array(q_);
    r   = array(r_);
    tau = array(t_);
}

void qrInPlace(array &tau, array &in) {
    fly_array t = 0;
    FLY_THROW(fly_qr_inplace(&t, in.get()));
    tau = array(t);
}

int cholesky(array &out, const array &in, const bool is_upper) {
    int info = 0;
    fly_array res;
    FLY_THROW(fly_cholesky(&res, &info, in.get(), is_upper));
    out = array(res);
    return info;
}

int choleskyInPlace(array &in, const bool is_upper) {
    int info = 0;
    FLY_THROW(fly_cholesky_inplace(&info, in.get(), is_upper));
    return info;
}

array solve(const array &a, const array &b, const matProp options) {
    fly_array out;
    FLY_THROW(fly_solve(&out, a.get(), b.get(), options));
    return array(out);
}

array solveLU(const array &a, const array &piv, const array &b,
              const matProp options) {
    fly_array out;
    FLY_THROW(fly_solve_lu(&out, a.get(), piv.get(), b.get(), options));
    return array(out);
}

array inverse(const array &in, const matProp options) {
    fly_array out;
    FLY_THROW(fly_inverse(&out, in.get(), options));
    return array(out);
}

array pinverse(const array &in, const double tol, const matProp options) {
    fly_array out;
    FLY_THROW(fly_pinverse(&out, in.get(), tol, options));
    return array(out);
}

unsigned rank(const array &in, const double tol) {
    unsigned r = 0;
    FLY_THROW(fly_rank(&r, in.get(), tol));
    return r;
}

#define INSTANTIATE_DET(TR, TC)                   \
    template<>                                    \
    FLY_API TR det(const array &in) {               \
        double real;                              \
        double imag;                              \
        FLY_THROW(fly_det(&real, &imag, in.get())); \
        return real;                              \
    }                                             \
    template<>                                    \
    FLY_API TC det(const array &in) {               \
        double real;                              \
        double imag;                              \
        FLY_THROW(fly_det(&real, &imag, in.get())); \
        TC out((TR)real, (TR)imag);               \
        return out;                               \
    }

INSTANTIATE_DET(float, fly_cfloat)
INSTANTIATE_DET(double, fly_cdouble)

double norm(const array &in, const normType type, const double p,
            const double q) {
    double out;
    FLY_THROW(fly_norm(&out, in.get(), type, p, q));
    return out;
}

bool isLAPACKAvailable() {
    bool out = false;
    FLY_THROW(fly_is_lapack_available(&out));
    return out;
}
}  // namespace fly
