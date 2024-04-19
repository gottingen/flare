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

// This needs to be the first thing in the file
#if defined(_WIN32) || defined(_MSC_VER)
#define _USE_MATH_DEFINES
#endif
#include <cmath>

#include <arith.hpp>
#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <common/cast.hpp>
#include <common/err_common.hpp>
#include <common/half.hpp>
#include <complex.hpp>
#include <handle.hpp>
#include <implicit.hpp>
#include <logic.hpp>
#include <optypes.hpp>
#include <unary.hpp>
#include <fly/arith.h>
#include <fly/array.h>
#include <fly/data.h>
#include <fly/defines.h>

using fly::dim4;
using flare::common::half;
using detail::arithOp;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::cplx;
using detail::createValueArray;
using detail::imag;
using detail::intl;
using detail::logicOp;
using detail::real;
using detail::scalar;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

template<typename T, fly_op_t op>
static inline fly_array unaryOp(const fly_array in) {
    fly_array res = getHandle(unaryOp<T, op>(castArray<T>(in)));
    return res;
}

template<typename Tc, typename Tr, fly_op_t op>
struct unaryOpCplxFun;

template<typename Tc, typename Tr, fly_op_t op>
static inline Array<Tc> unaryOpCplx(const Array<Tc> &in) {
    return unaryOpCplxFun<Tc, Tr, op>()(in);
}

template<typename Tc, typename Tr, fly_op_t op>
static inline fly_array unaryOpCplx(const fly_array in) {
    return getHandle(unaryOpCplx<Tc, Tr, op>(castArray<Tc>(in)));
}

template<fly_op_t op>
static fly_err fly_unary(fly_array *out, const fly_array in) {
    try {
        const ArrayInfo &in_info = getInfo(in);
        ARG_ASSERT(1, in_info.isReal());

        fly_dtype in_type = in_info.getType();
        fly_array res;

        // Convert all inputs to floats / doubles
        fly_dtype type = implicit(in_type, f32);
        if (in_type == f16) { type = f16; }
        if (in_info.ndims() == 0) { return fly_retain_array(out, in); }

        switch (type) {
            case f16: res = unaryOp<half, op>(in); break;
            case f32: res = unaryOp<float, op>(in); break;
            case f64: res = unaryOp<double, op>(in); break;
            default: TYPE_ERROR(1, in_type); break;
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

template<fly_op_t op>
static fly_err fly_unary_complex(fly_array *out, const fly_array in) {
    try {
        const ArrayInfo &in_info = getInfo(in);

        fly_dtype in_type = in_info.getType();
        fly_array res;

        // Convert all inputs to floats / doubles
        fly_dtype type = implicit(in_type, f32);
        if (in_type == f16) { type = f16; }
        if (in_info.ndims() == 0) { return fly_retain_array(out, in); }

        switch (type) {
            case f32: res = unaryOp<float, op>(in); break;
            case f64: res = unaryOp<double, op>(in); break;
            case c32: res = unaryOpCplx<cfloat, float, op>(in); break;
            case c64: res = unaryOpCplx<cdouble, double, op>(in); break;
            case f16: res = unaryOp<half, op>(in); break;
            default: TYPE_ERROR(1, in_type); break;
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

#define UNARY_FN(name, opcode)                           \
    fly_err fly_##name(fly_array *out, const fly_array in) { \
        return fly_unary<fly_##opcode##_t>(out, in);       \
    }

#define UNARY(fn) UNARY_FN(fn, fn)

#define UNARY_COMPLEX(fn)                              \
    fly_err fly_##fn(fly_array *out, const fly_array in) { \
        return fly_unary_complex<fly_##fn##_t>(out, in); \
    }

UNARY(trunc)
UNARY_FN(sign, signbit)
UNARY(round)
UNARY(floor)
UNARY(ceil)

UNARY(sigmoid)
UNARY(expm1)
UNARY(erf)
UNARY(erfc)

UNARY(log10)
UNARY(log1p)
UNARY(log2)

UNARY(cbrt)
UNARY(rsqrt)

UNARY(tgamma)
UNARY(lgamma)

UNARY_COMPLEX(acosh)
UNARY_COMPLEX(acos)
UNARY_COMPLEX(asin)
UNARY_COMPLEX(asinh)
UNARY_COMPLEX(atan)
UNARY_COMPLEX(atanh)
UNARY_COMPLEX(cos)
UNARY_COMPLEX(cosh)
UNARY_COMPLEX(exp)
UNARY_COMPLEX(log)
UNARY_COMPLEX(sin)
UNARY_COMPLEX(sinh)
UNARY_COMPLEX(sqrt)
UNARY_COMPLEX(tan)
UNARY_COMPLEX(tanh)

template<typename Tc, typename Tr>
struct unaryOpCplxFun<Tc, Tr, fly_exp_t> {
    Array<Tc> operator()(const Array<Tc> &z) {
        // exp(a + ib)
        // --> exp(a) * exp(ib)
        // --> exp(a) * (cos(a) + i * sin(b))
        // --> exp(a) * cos(a) + i * exp(a) * sin(b)

        Array<Tr> a = real<Tr, Tc>(z);
        Array<Tr> b = imag<Tr, Tc>(z);

        Array<Tr> exp_a = unaryOp<Tr, fly_exp_t>(a);
        Array<Tr> cos_b = unaryOp<Tr, fly_cos_t>(b);
        Array<Tr> sin_b = unaryOp<Tr, fly_sin_t>(b);

        // exp(a) * cos(b)
        Array<Tr> a_out = arithOp<Tr, fly_mul_t>(exp_a, cos_b, exp_a.dims());
        // exp(a) * sin(b)
        Array<Tr> b_out = arithOp<Tr, fly_mul_t>(exp_a, sin_b, exp_a.dims());

        // exp(a) * cos(b) + i * exp(a) * sin(b)
        return cplx<Tc, Tr>(a_out, b_out, a_out.dims());
    }
};

template<typename Tc, typename Tr>
struct unaryOpCplxFun<Tc, Tr, fly_log_t> {
    Array<Tc> operator()(const Array<Tc> &z) {
        // log(a + ib)
        // using r = abs(a + ib), phi == arg(a + ib)
        // --> log(r * exp(i * phi))
        // --> log(r) + i * phi

        // convert cartesian to polar
        Array<Tr> a = real<Tr, Tc>(z);
        Array<Tr> b = imag<Tr, Tc>(z);

        // phi = arg(a + ib)
        // --> phi = atan2(b, a)
        Array<Tr> phi = arithOp<Tr, fly_atan2_t>(b, a, b.dims());

        Array<Tr> r = detail::abs<Tr>(z);

        // compute log
        // log(r)
        Array<Tr> a_out = unaryOp<Tr, fly_log_t>(r);
        // phi
        const Array<Tr> &b_out = phi;

        // log(r) + i * phi
        return cplx<Tc, Tr>(a_out, b_out, a_out.dims());
    }
};

template<typename Tc, typename Tr>
struct unaryOpCplxFun<Tc, Tr, fly_sin_t> {
    Array<Tc> operator()(const Array<Tc> &z) {
        // sin(a + ib)
        // --> sin(a) * cos(ib) + cos(a) * sin(ib)
        // --> sin(a) * cosh(b) + i * cos(a) * sinh(b)

        Array<Tr> a = real<Tr, Tc>(z);
        Array<Tr> b = imag<Tr, Tc>(z);

        // compute sin
        Array<Tr> sin_a  = unaryOp<Tr, fly_sin_t>(a);
        Array<Tr> cos_a  = unaryOp<Tr, fly_cos_t>(a);
        Array<Tr> sinh_b = unaryOp<Tr, fly_sinh_t>(b);
        Array<Tr> cosh_b = unaryOp<Tr, fly_cosh_t>(b);

        // sin(a) * cosh(b)
        Array<Tr> a_out = arithOp<Tr, fly_mul_t>(sin_a, cosh_b, sin_a.dims());
        // cos(a) * sinh(b)
        Array<Tr> b_out = arithOp<Tr, fly_mul_t>(cos_a, sinh_b, cos_a.dims());

        // sin(a) * cosh(b) + i * cos(a) * sinh(b)
        return cplx<Tc, Tr>(a_out, b_out, a_out.dims());
    }
};

template<typename Tc, typename Tr>
struct unaryOpCplxFun<Tc, Tr, fly_cos_t> {
    Array<Tc> operator()(const Array<Tc> &z) {
        // cos(a + ib)
        // --> cos(a) * cos(ib) - sin(a) * sin(ib)
        // --> cos(a) * cosh(b) - i * sin(a) * sinh(b)

        Array<Tr> a = real<Tr, Tc>(z);
        Array<Tr> b = imag<Tr, Tc>(z);

        // compute cos
        Array<Tr> sin_a  = unaryOp<Tr, fly_sin_t>(a);
        Array<Tr> cos_a  = unaryOp<Tr, fly_cos_t>(a);
        Array<Tr> sinh_b = unaryOp<Tr, fly_sinh_t>(b);
        Array<Tr> cosh_b = unaryOp<Tr, fly_cosh_t>(b);

        // cos(a) * cosh(b)
        Array<Tr> a_out = arithOp<Tr, fly_mul_t>(cos_a, cosh_b, sin_a.dims());
        // -1
        Array<Tr> neg_one = createValueArray<Tr>(a_out.dims(), -1);
        // sin(a) * sinh(b)
        Array<Tr> b_out_neg =
            arithOp<Tr, fly_mul_t>(sin_a, sinh_b, cos_a.dims());
        // -1 * sin(a) * sinh(b)
        Array<Tr> b_out =
            arithOp<Tr, fly_mul_t>(neg_one, b_out_neg, b_out_neg.dims());
        // cos(a) * cosh(b) - i * sin(a) * sinh(b)
        return cplx<Tc, Tr>(a_out, b_out, a_out.dims());
    }
};

template<typename Tc, typename Tr>
struct unaryOpCplxFun<Tc, Tr, fly_tan_t> {
    Array<Tc> operator()(const Array<Tc> &z) {
        // tan(a + ib) = sin(a + ib) / cos(a + ib)
        Array<Tc> sin_z = unaryOpCplx<Tc, Tr, fly_sin_t>(z);
        Array<Tc> cos_z = unaryOpCplx<Tc, Tr, fly_cos_t>(z);
        return arithOp<Tc, fly_div_t>(sin_z, cos_z, sin_z.dims());
    }
};

template<typename Tc, typename Tr>
struct unaryOpCplxFun<Tc, Tr, fly_sinh_t> {
    Array<Tc> operator()(const Array<Tc> &z) {
        // sinh(a + ib)
        // --> sinh(a) * cosh(ib) + cosh(a) * sinh(ib)
        // --> sinh(a) * cos(b) + i * cosh(a) * sin(b)

        Array<Tr> a = real<Tr, Tc>(z);
        Array<Tr> b = imag<Tr, Tc>(z);

        // compute sinh
        Array<Tr> sinh_a = unaryOp<Tr, fly_sinh_t>(a);
        Array<Tr> cosh_a = unaryOp<Tr, fly_cosh_t>(a);
        Array<Tr> sin_b  = unaryOp<Tr, fly_sin_t>(b);
        Array<Tr> cos_b  = unaryOp<Tr, fly_cos_t>(b);

        // sinh(a) * cos(b)
        Array<Tr> a_out = arithOp<Tr, fly_mul_t>(sinh_a, cos_b, sinh_a.dims());
        // cosh(a) * sin(b)
        Array<Tr> b_out = arithOp<Tr, fly_mul_t>(cosh_a, sin_b, cosh_a.dims());

        // sinh(a) * cos(b) + i * cosh(a) * sin(b)
        return cplx<Tc, Tr>(a_out, b_out, a_out.dims());
    }
};

template<typename Tc, typename Tr>
struct unaryOpCplxFun<Tc, Tr, fly_cosh_t> {
    Array<Tc> operator()(const Array<Tc> &z) {
        // cosh(a + ib)
        // --> cosh(a) * cosh(ib) + sinh(a) * sinh(ib)
        // --> cosh(a) * cos(b) + i * sinh(a) * sin(b)
        Array<Tr> a = real<Tr, Tc>(z);
        Array<Tr> b = imag<Tr, Tc>(z);

        // compute cosh
        Array<Tr> sinh_a = unaryOp<Tr, fly_sinh_t>(a);
        Array<Tr> cosh_a = unaryOp<Tr, fly_cosh_t>(a);
        Array<Tr> sin_b  = unaryOp<Tr, fly_sin_t>(b);
        Array<Tr> cos_b  = unaryOp<Tr, fly_cos_t>(b);

        // cosh(a) * cos(b)
        Array<Tr> a_out = arithOp<Tr, fly_mul_t>(cosh_a, cos_b, cosh_a.dims());
        // sinh(a) * sin(b)
        Array<Tr> b_out = arithOp<Tr, fly_mul_t>(sinh_a, sin_b, sinh_a.dims());

        // cosh(a) * cos(b) + i * sinh(a) * sin(b)
        return cplx<Tc, Tr>(a_out, b_out, a_out.dims());
    }
};

template<typename Tc, typename Tr>
struct unaryOpCplxFun<Tc, Tr, fly_tanh_t> {
    Array<Tc> operator()(const Array<Tc> &z) {
        // tanh(a + ib) = sinh(a + ib) / cosh(a + ib)
        Array<Tc> sinh_z = unaryOpCplx<Tc, Tr, fly_sinh_t>(z);
        Array<Tc> cosh_z = unaryOpCplx<Tc, Tr, fly_cosh_t>(z);
        return arithOp<Tc, fly_div_t>(sinh_z, cosh_z, sinh_z.dims());
    }
};

template<typename Tc, typename Tr>
struct unaryOpCplxFun<Tc, Tr, fly_acosh_t> {
    Array<Tc> operator()(const Array<Tc> &z) {
        // dont simplify this expression, as it might lead to branch cuts
        // acosh(z) = log(z+sqrt(z+1)*sqrt(z-1))

        Array<Tc> one = createValueArray<Tc>(z.dims(), scalar<Tc>(1.0));

        // (z + 1)
        Array<Tc> z_plus_one = arithOp<Tc, fly_add_t>(z, one, z.dims());
        // (z - 1)
        Array<Tc> z_minus_one = arithOp<Tc, fly_sub_t>(z, one, z.dims());
        // sqrt(z + 1)
        Array<Tc> sqrt_z_plus_one = unaryOpCplx<Tc, Tr, fly_sqrt_t>(z_plus_one);
        // sqrt(z - 1)
        Array<Tc> sqrt_z_minus_one =
            unaryOpCplx<Tc, Tr, fly_sqrt_t>(z_minus_one);
        // sqrt(z + 1) * sqrt(z - 1)
        Array<Tc> sqrt_prod = arithOp<Tc, fly_mul_t>(
            sqrt_z_plus_one, sqrt_z_minus_one, sqrt_z_plus_one.dims());
        // z + sqrt(z + 1) * sqrt(z - 1)
        Array<Tc> w = arithOp<Tc, fly_add_t>(z, sqrt_prod, z.dims());
        // log(z + sqrt(z + 1) * sqrt(z - 1))
        return unaryOpCplx<Tc, Tr, fly_log_t>(w);
    }
};

template<typename Tc, typename Tr>
struct unaryOpCplxFun<Tc, Tr, fly_asinh_t> {
    Array<Tc> operator()(const Array<Tc> &z) {
        // asinh(z) = log(z+sqrt(z^2+1))
        Array<Tc> one = createValueArray<Tc>(z.dims(), scalar<Tc>(1.0));

        // z^2
        Array<Tc> z2 = arithOp<Tc, fly_mul_t>(z, z, z.dims());
        // ((a + 1) + i * b) --> z^2 + 1
        Array<Tc> z2_plus_one = arithOp<Tc, fly_add_t>(z2, one, z.dims());
        // sqrt(z^2 + 1)
        Array<Tc> sqrt_z2_plus_one =
            unaryOpCplx<Tc, Tr, fly_sqrt_t>(z2_plus_one);
        // z + sqrt(z^2 + 1)
        Array<Tc> w = arithOp<Tc, fly_add_t>(z, sqrt_z2_plus_one, z.dims());
        // log(z + sqrt(z^2 + 1))
        return unaryOpCplx<Tc, Tr, fly_log_t>(w);
    }
};

template<typename Tc, typename Tr>
struct unaryOpCplxFun<Tc, Tr, fly_atanh_t> {
    Array<Tc> operator()(const Array<Tc> &z) {
        // atanh(z) = 0.5*(log(1+z)-log(1-z))
        Array<Tc> one =
            createValueArray<Tc>(z.dims(), scalar<Tc, Tr>(1.0, 0.0));
        Array<Tc> half =
            createValueArray<Tc>(z.dims(), scalar<Tc, Tr>(0.5, 0.0));

        // (1 + z)
        Array<Tc> one_plus_z = arithOp<Tc, fly_add_t>(one, z, one.dims());
        // (1 - z)
        Array<Tc> one_minus_z = arithOp<Tc, fly_sub_t>(one, z, one.dims());
        // log(1 + z)
        Array<Tc> log_one_plus_z = unaryOpCplx<Tc, Tr, fly_log_t>(one_plus_z);
        // log(1 - z)
        Array<Tc> log_one_minus_z = unaryOpCplx<Tc, Tr, fly_log_t>(one_minus_z);
        // (log(1 + z) - log(1 - z))
        Array<Tc> w = arithOp<Tc, fly_sub_t>(log_one_plus_z, log_one_minus_z,
                                            log_one_plus_z.dims());
        // 0.5 * (log(1 + z) - log(1 - z))
        return arithOp<Tc, fly_mul_t>(w, half, w.dims());
    }
};

template<typename Tc, typename Tr>
struct unaryOpCplxFun<Tc, Tr, fly_acos_t> {
    Array<Tc> operator()(const Array<Tc> &z) {
        // acos(z) = pi/2 + i*log(i*z+sqrt(1-z.^2))
        // --> pi/2 - asinz(z)

        Array<Tc> one =
            createValueArray<Tc>(z.dims(), scalar<Tc, Tr>(1.0, 0.0));

        Array<Tc> i = createValueArray<Tc>(z.dims(), scalar<Tc, Tr>(0.0, 1.0));
        Array<Tc> pi_half =
            createValueArray<Tc>(z.dims(), scalar<Tc, Tr>(M_PI_2, 0.0));

        // z^2
        Array<Tc> z2 = arithOp<Tc, fly_mul_t>(z, z, z.dims());
        // 1 - z^2
        Array<Tc> one_minus_z2 = arithOp<Tc, fly_sub_t>(one, z2, one.dims());
        // sqrt(1 - z^2)
        Array<Tc> sqrt_one_minus_z2 =
            unaryOpCplx<Tc, Tr, fly_sqrt_t>(one_minus_z2);
        // i*z
        Array<Tc> iz = arithOp<Tc, fly_mul_t>(i, z, z.dims());
        // (i*z - sqrt(1 - z^2))
        Array<Tc> w = arithOp<Tc, fly_add_t>(iz, sqrt_one_minus_z2, iz.dims());
        // log(i*z - sqrt(1 - z^2))
        Array<Tc> log_w = unaryOpCplx<Tc, Tr, fly_log_t>(w);
        // i*log(i*z - sqrt(1 - z^2))
        Array<Tc> i_log_w = arithOp<Tc, fly_mul_t>(i, log_w, i.dims());
        // pi/2 + i*log(i*z - sqrt(1 - z^2))
        return arithOp<Tc, fly_add_t>(pi_half, i_log_w, pi_half.dims());
    }
};

template<typename Tc, typename Tr>
struct unaryOpCplxFun<Tc, Tr, fly_asin_t> {
    Array<Tc> operator()(const Array<Tc> &z) {
        // asin(z) = -i*log(i*z+sqrt(1-z^2))

        Array<Tc> one =
            createValueArray<Tc>(z.dims(), scalar<Tc, Tr>(1.0, 0.0));
        Array<Tc> i = createValueArray<Tc>(z.dims(), scalar<Tc, Tr>(0.0, 1.0));
        Array<Tc> minus_i =
            createValueArray<Tc>(z.dims(), scalar<Tc, Tr>(0.0, -1.0));

        // z^2
        Array<Tc> z2 = arithOp<Tc, fly_mul_t>(z, z, z.dims());
        // 1 - z^2
        Array<Tc> one_minus_z2 = arithOp<Tc, fly_sub_t>(one, z2, one.dims());
        // sqrt(1 - z^2)
        Array<Tc> sqrt_one_minus_z2 =
            unaryOpCplx<Tc, Tr, fly_sqrt_t>(one_minus_z2);
        // i*z
        Array<Tc> iz = arithOp<Tc, fly_mul_t>(i, z, z.dims());
        // (i*z + sqrt(1 - z^2))
        Array<Tc> w = arithOp<Tc, fly_add_t>(iz, sqrt_one_minus_z2, iz.dims());
        // log(i*z + sqrt(1 - z^2))
        Array<Tc> log_w = unaryOpCplx<Tc, Tr, fly_log_t>(w);
        // i*log(i*z + sqrt(1 - z^2))
        return arithOp<Tc, fly_mul_t>(minus_i, log_w, minus_i.dims());
    }
};

template<typename Tc, typename Tr>
struct unaryOpCplxFun<Tc, Tr, fly_atan_t> {
    Array<Tc> operator()(const Array<Tc> &z) {
        // atan(z) = 0.5 * i * (log(1-i*z)-log(1+i*z))
        Array<Tc> one =
            createValueArray<Tc>(z.dims(), scalar<Tc, Tr>(1.0, 0.0));
        Array<Tc> i = createValueArray<Tc>(z.dims(), scalar<Tc, Tr>(0.0, 1.0));

        // 0.5 * i
        Array<Tc> i_half =
            createValueArray<Tc>(z.dims(), scalar<Tc, Tr>(0.0, 0.5));
        // i*z
        Array<Tc> iz = arithOp<Tc, fly_mul_t>(i, z, z.dims());
        // 1 - i*z
        Array<Tc> one_minus_iz = arithOp<Tc, fly_sub_t>(one, iz, z.dims());
        // 1 + i*z
        Array<Tc> one_plus_iz = arithOp<Tc, fly_add_t>(one, iz, z.dims());
        // log(1 - i*z)
        Array<Tc> log_minus = unaryOpCplx<Tc, Tr, fly_log_t>(one_minus_iz);
        // log(1 + i*z)
        Array<Tc> log_plus = unaryOpCplx<Tc, Tr, fly_log_t>(one_plus_iz);
        // log(1 - i*z) - log(1 + i*z)
        Array<Tc> log_diff =
            arithOp<Tc, fly_sub_t>(log_minus, log_plus, z.dims());
        // 0.5 * i * (log(1 - i*z) - log(1 + i*z))
        return arithOp<Tc, fly_mul_t>(i_half, log_diff, z.dims());
    }
};

template<typename Tc, typename Tr>
struct unaryOpCplxFun<Tc, Tr, fly_sqrt_t> {
    Array<Tc> operator()(const Array<Tc> &z) {
        // sqrt(a + ib)
        // using r = abs(a + ib), phi == arg(a + ib)
        // --> sqrt(r * exp(i * phi))
        // --> sqrt(r) * exp(i * phi / 2)
        // --> sqrt(r) * cos(phi/2) + i * sqrt(r) * sin(phi/2)

        // convert cartesian to polar
        Array<Tr> a = real<Tr, Tc>(z);
        Array<Tr> b = imag<Tr, Tc>(z);

        // phi = arg(a + ib)
        // --> phi = atan2(b, a)
        Array<Tr> phi = arithOp<Tr, fly_atan2_t>(b, a, b.dims());
        Array<Tr> r   = detail::abs<Tr>(z);

        // compute sqrt
        Array<Tr> two = createValueArray<Tr>(phi.dims(), 2.0);

        // sqrt(r)
        Array<Tr> r_out = unaryOp<Tr, fly_sqrt_t>(r);

        // phi/2
        Array<Tr> phi_out = arithOp<Tr, fly_div_t>(phi, two, phi.dims());

        // convert polar to cartesian
        // cos(phi/2)
        Array<Tr> a_out_unit = unaryOp<Tr, fly_cos_t>(phi_out);
        // sin(phi/2)
        Array<Tr> b_out_unit = unaryOp<Tr, fly_sin_t>(phi_out);
        // sqrt(r) * cos(phi/2)
        Array<Tr> a_out =
            arithOp<Tr, fly_mul_t>(r_out, a_out_unit, r_out.dims());
        // sqrt(r) * sin(phi/2)
        Array<Tr> b_out =
            arithOp<Tr, fly_mul_t>(r_out, b_out_unit, r_out.dims());

        // sqrt(r) * cos(phi/2) + i * sqrt(r) * sin(phi/2)
        return cplx<Tc, Tr>(a_out, b_out, a_out.dims());
    }
};

fly_err fly_not(fly_array *out, const fly_array in) {
    try {
        fly_array tmp;
        const ArrayInfo &in_info = getInfo(in);
        if (in_info.ndims() == 0) { return fly_retain_array(out, in); }

        FLY_CHECK(fly_constant(&tmp, 0, in_info.ndims(), in_info.dims().get(),
                             in_info.getType()));

        FLY_CHECK(fly_eq(out, in, tmp, false));

        FLY_CHECK(fly_release_array(tmp));
    }
    CATCHALL;

    return FLY_SUCCESS;
}

template<typename T>
static inline fly_array bitOpNot(const fly_array in) {
    return unaryOp<T, fly_bitnot_t>(in);
}

fly_err fly_bitnot(fly_array *out, const fly_array in) {
    try {
        const ArrayInfo &iinfo = getInfo(in);
        const fly_dtype type    = iinfo.getType();

        dim4 odims = iinfo.dims();

        if (odims.ndims() == 0) {
            return fly_create_handle(out, 0, nullptr, type);
        }

        fly_array res;
        switch (type) {
            case s32: res = bitOpNot<int>(in); break;
            case u32: res = bitOpNot<uint>(in); break;
            case u8: res = bitOpNot<uchar>(in); break;
            case b8: res = bitOpNot<char>(in); break;
            case s64: res = bitOpNot<intl>(in); break;
            case u64: res = bitOpNot<uintl>(in); break;
            case s16: res = bitOpNot<short>(in); break;
            case u16: res = bitOpNot<ushort>(in); break;
            default: TYPE_ERROR(0, type);
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_arg(fly_array *out, const fly_array in) {
    try {
        const ArrayInfo &in_info = getInfo(in);
        if (in_info.ndims() == 0) { return fly_retain_array(out, in); }

        if (!in_info.isComplex()) {
            return fly_constant(out, 0, in_info.ndims(), in_info.dims().get(),
                               in_info.getType());
        }

        fly_array real;
        fly_array imag;

        FLY_CHECK(fly_real(&real, in));
        FLY_CHECK(fly_imag(&imag, in));

        FLY_CHECK(fly_atan2(out, imag, real, false));

        FLY_CHECK(fly_release_array(real));
        FLY_CHECK(fly_release_array(imag));
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_pow2(fly_array *out, const fly_array in) {
    try {
        fly_array two;
        const ArrayInfo &in_info = getInfo(in);
        if (in_info.ndims() == 0) { return fly_retain_array(out, in); }

        FLY_CHECK(fly_constant(&two, 2, in_info.ndims(), in_info.dims().get(),
                             in_info.getType()));

        FLY_CHECK(fly_pow(out, two, in, false));

        FLY_CHECK(fly_release_array(two));
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_factorial(fly_array *out, const fly_array in) {
    try {
        fly_array one;
        const ArrayInfo &in_info = getInfo(in);
        if (in_info.ndims() == 0) { return fly_retain_array(out, in); }

        FLY_CHECK(fly_constant(&one, 1, in_info.ndims(), in_info.dims().get(),
                             in_info.getType()));

        fly_array inp1;
        FLY_CHECK(fly_add(&inp1, one, in, false));

        FLY_CHECK(fly_tgamma(out, inp1));

        FLY_CHECK(fly_release_array(one));
        FLY_CHECK(fly_release_array(inp1));
    }
    CATCHALL;

    return FLY_SUCCESS;
}

template<typename T, fly_op_t op>
static inline fly_array checkOp(const fly_array in) {
    fly_array res = getHandle(checkOp<T, op>(castArray<T>(in)));
    return res;
}

template<fly_op_t op>
struct cplxLogicOp {
    fly_array operator()(const Array<char> &resR, const Array<char> &resI,
                        const dim4 &dims) {
        return getHandle(logicOp<char, fly_or_t>(resR, resI, dims));
    }
};

template<>
struct cplxLogicOp<fly_iszero_t> {
    fly_array operator()(const Array<char> &resR, const Array<char> &resI,
                        const dim4 &dims) {
        return getHandle(logicOp<char, fly_and_t>(resR, resI, dims));
    }
};

template<typename T, typename BT, fly_op_t op>
static inline fly_array checkOpCplx(const fly_array in) {
    Array<BT> R = real<BT, T>(getArray<T>(in));
    Array<BT> I = imag<BT, T>(getArray<T>(in));

    Array<char> resR = checkOp<BT, op>(R);
    Array<char> resI = checkOp<BT, op>(I);

    const ArrayInfo &in_info = getInfo(in);
    const dim4 &dims         = in_info.dims();
    cplxLogicOp<op> cplxLogic;
    fly_array res = cplxLogic(resR, resI, dims);

    return res;
}

template<fly_op_t op>
static fly_err fly_check(fly_array *out, const fly_array in) {
    try {
        const ArrayInfo &in_info = getInfo(in);

        fly_dtype in_type = in_info.getType();
        fly_array res;

        // Convert all inputs to floats / doubles / complex
        fly_dtype type = implicit(in_type, f32);
        if (in_type == f16) { type = f16; }
        if (in_info.ndims() == 0) { return fly_retain_array(out, in); }

        switch (type) {
            case f32: res = checkOp<float, op>(in); break;
            case f64: res = checkOp<double, op>(in); break;
            case f16: res = checkOp<half, op>(in); break;
            case c32: res = checkOpCplx<cfloat, float, op>(in); break;
            case c64: res = checkOpCplx<cdouble, double, op>(in); break;
            default: TYPE_ERROR(1, in_type); break;
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

#define FN_CHECK(fn)                                      \
    fly_err fly_##fn(fly_array *out, const fly_array in) { \
        return fly_check<fly_##fn##_t>(out, in);         \
    }

FN_CHECK(isinf)
FN_CHECK(isnan)
FN_CHECK(iszero)
