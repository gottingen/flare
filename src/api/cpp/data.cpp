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
#include <fly/data.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wparentheses"
#include <fly/half.hpp>
#pragma GCC diagnostic pop

#include <fly/arith.h>
#include <fly/array.h>
#include <fly/complex.h>
#include <fly/defines.h>
#include <fly/gfor.h>
#include <fly/half.h>
#include <fly/traits.hpp>
#include "error.hpp"

#include <type_traits>

using fly::array;
using fly::dim4;
using fly::dtype;
using std::enable_if;

namespace {
// NOTE: we are repeating this here so that we don't need to access the
// is_complex types in backend/common. This is done to isolate the C++ API from
// the internal API
template<typename T>
struct is_complex {
    static const bool value = false;
};
template<>
struct is_complex<fly::cfloat> {
    static const bool value = true;
};
template<>
struct is_complex<fly::cdouble> {
    static const bool value = true;
};

array constant(fly_half val, const dim4 &dims, const dtype type) {
    fly_array res;
    UNUSED(val);
    FLY_THROW(fly_constant(&res, 0,  //(double)val,
                         dims.ndims(), dims.get(), type));
    return array(res);
}

template<typename T, typename = typename enable_if<
                         !static_cast<bool>(is_complex<T>::value), T>::type>
array constant(T val, const dim4 &dims, dtype type) {
    fly_array res;
    if (type != s64 && type != u64) {
        FLY_THROW(
            fly_constant(&res, (double)val, dims.ndims(), dims.get(), type));
    } else if (type == s64) {
        FLY_THROW(
            fly_constant_long(&res, (long long)val, dims.ndims(), dims.get()));
    } else {
        FLY_THROW(fly_constant_ulong(&res, (unsigned long long)val, dims.ndims(),
                                   dims.get()));
    }
    return array(res);
}

template<typename T>
typename enable_if<static_cast<bool>(is_complex<T>::value), array>::type
constant(T val, const dim4 &dims, const dtype type) {
    if (type != c32 && type != c64) {
        return ::constant(real(val), dims, type);
    }
    fly_array res;
    FLY_THROW(fly_constant_complex(&res, real(val), imag(val), dims.ndims(),
                                 dims.get(), type));
    return array(res);
}
}  // namespace

namespace fly {
template<typename T>
array constant(T val, const dim4 &dims, const fly::dtype type) {
    return ::constant(val, dims, type);
}

template<typename T>
array constant(T val, const dim_t d0, const fly::dtype ty) {
    return ::constant(val, dim4(d0), ty);
}

template<typename T>
array constant(T val, const dim_t d0, const dim_t d1, const fly::dtype ty) {
    return ::constant(val, dim4(d0, d1), ty);
}

template<typename T>
array constant(T val, const dim_t d0, const dim_t d1, const dim_t d2,
               const fly::dtype ty) {
    return ::constant(val, dim4(d0, d1, d2), ty);
}

template<typename T>
array constant(T val, const dim_t d0, const dim_t d1, const dim_t d2,
               const dim_t d3, const fly::dtype ty) {
    return ::constant(val, dim4(d0, d1, d2, d3), ty);
}

#define CONSTANT(TYPE)                                                       \
    template FLY_API array constant<TYPE>(TYPE val, const dim4 &dims,          \
                                        const fly::dtype ty);                 \
    template FLY_API array constant<TYPE>(TYPE val, const dim_t d0,            \
                                        const fly::dtype ty);                 \
    template FLY_API array constant<TYPE>(TYPE val, const dim_t d0,            \
                                        const dim_t d1, const fly::dtype ty); \
    template FLY_API array constant<TYPE>(TYPE val, const dim_t d0,            \
                                        const dim_t d1, const dim_t d2,      \
                                        const fly::dtype ty);                 \
    template FLY_API array constant<TYPE>(TYPE val, const dim_t d0,            \
                                        const dim_t d1, const dim_t d2,      \
                                        const dim_t d3, const fly::dtype ty);
CONSTANT(double);
CONSTANT(float);
CONSTANT(int);
CONSTANT(unsigned);
CONSTANT(char);
CONSTANT(unsigned char);
CONSTANT(cfloat);
CONSTANT(cdouble);
CONSTANT(long);
CONSTANT(unsigned long);
CONSTANT(long long);
CONSTANT(unsigned long long);
CONSTANT(bool);
CONSTANT(short);
CONSTANT(unsigned short);
CONSTANT(half);
CONSTANT(half_float::half);

#undef CONSTANT

array range(const dim4 &dims, const int seq_dim, const fly::dtype ty) {
    fly_array out;
    FLY_THROW(fly_range(&out, dims.ndims(), dims.get(), seq_dim, ty));
    return array(out);
}

array range(const dim_t d0, const dim_t d1, const dim_t d2, const dim_t d3,
            const int seq_dim, const fly::dtype ty) {
    return range(dim4(d0, d1, d2, d3), seq_dim, ty);
}

array iota(const dim4 &dims, const dim4 &tile_dims, const fly::dtype ty) {
    fly_array out;
    FLY_THROW(fly_iota(&out, dims.ndims(), dims.get(), tile_dims.ndims(),
                     tile_dims.get(), ty));
    return array(out);
}

array identity(const dim4 &dims, const fly::dtype type) {
    fly_array res;
    FLY_THROW(fly_identity(&res, dims.ndims(), dims.get(), type));
    return array(res);
}

array identity(const dim_t d0, const fly::dtype ty) {
    return identity(dim4(d0), ty);
}

array identity(const dim_t d0, const dim_t d1, const fly::dtype ty) {
    return identity(dim4(d0, d1), ty);
}

array identity(const dim_t d0, const dim_t d1, const dim_t d2,
               const fly::dtype ty) {
    return identity(dim4(d0, d1, d2), ty);
}

array identity(const dim_t d0, const dim_t d1, const dim_t d2, const dim_t d3,
               const fly::dtype ty) {
    return identity(dim4(d0, d1, d2, d3), ty);
}

array diag(const array &in, const int num, const bool extract) {
    fly_array res;
    if (extract) {
        FLY_THROW(fly_diag_extract(&res, in.get(), num));
    } else {
        FLY_THROW(fly_diag_create(&res, in.get(), num));
    }

    return array(res);
}

array moddims(const array &in, const unsigned ndims, const dim_t *const dims) {
    fly_array out = 0;
    FLY_THROW(fly_moddims(&out, in.get(), ndims, dims));
    return array(out);
}

array moddims(const array &in, const dim4 &dims) {
    return fly::moddims(in, dims.ndims(), dims.get());
}

array moddims(const array &in, const dim_t d0, const dim_t d1, const dim_t d2,
              const dim_t d3) {
    dim_t dims[4] = {d0, d1, d2, d3};
    return fly::moddims(in, 4, dims);
}

array flat(const array &in) {
    fly_array out = 0;
    FLY_THROW(fly_flat(&out, in.get()));
    return array(out);
}

array join(const int dim, const array &first, const array &second) {
    fly_array out = 0;
    FLY_THROW(fly_join(&out, dim, first.get(), second.get()));
    return array(out);
}

array join(const int dim, const array &first, const array &second,
           const array &third) {
    fly_array out       = 0;
    fly_array inputs[3] = {first.get(), second.get(), third.get()};
    FLY_THROW(fly_join_many(&out, dim, 3, inputs));
    return array(out);
}

array join(const int dim, const array &first, const array &second,
           const array &third, const array &fourth) {
    fly_array out       = 0;
    fly_array inputs[4] = {first.get(), second.get(), third.get(), fourth.get()};
    FLY_THROW(fly_join_many(&out, dim, 4, inputs));
    return array(out);
}

array tile(const array &in, const unsigned x, const unsigned y,
           const unsigned z, const unsigned w) {
    fly_array out = 0;
    FLY_THROW(fly_tile(&out, in.get(), x, y, z, w));
    return array(out);
}

array tile(const array &in, const fly::dim4 &dims) {
    fly_array out = 0;
    FLY_THROW(fly_tile(&out, in.get(), dims[0], dims[1], dims[2], dims[3]));
    return array(out);
}

array reorder(const array &in, const unsigned x, const unsigned y,
              const unsigned z, const unsigned w) {
    fly_array out = 0;
    FLY_THROW(fly_reorder(&out, in.get(), x, y, z, w));
    return array(out);
}

array shift(const array &in, const int x, const int y, const int z,
            const int w) {
    fly_array out = 0;
    FLY_THROW(fly_shift(&out, in.get(), x, y, z, w));
    return array(out);
}

array flip(const array &in, const unsigned dim) {
    fly_array out = 0;
    FLY_THROW(fly_flip(&out, in.get(), dim));
    return array(out);
}

array lower(const array &in, bool is_unit_diag) {
    fly_array res;
    FLY_THROW(fly_lower(&res, in.get(), is_unit_diag));
    return array(res);
}

array upper(const array &in, bool is_unit_diag) {
    fly_array res;
    FLY_THROW(fly_upper(&res, in.get(), is_unit_diag));
    return array(res);
}

array select(const array &cond, const array &a, const array &b) {
    fly_array res;
    FLY_THROW(fly_select(&res, cond.get(), a.get(), b.get()));
    return array(res);
}

array select(const array &cond, const array &a, const double &b) {
    fly_array res;
    FLY_THROW(fly_select_scalar_r(&res, cond.get(), a.get(), b));
    return array(res);
}

array select(const array &cond, const double &a, const array &b) {
    fly_array res;
    FLY_THROW(fly_select_scalar_l(&res, cond.get(), a, b.get()));
    return array(res);
}

void replace(array &a, const array &cond, const array &b) {
    FLY_THROW(fly_replace(a.get(), cond.get(), b.get()));
}

void replace(array &a, const array &cond, const double &b) {
    FLY_THROW(fly_replace_scalar(a.get(), cond.get(), b));
}

void replace(array &a, const array &cond, const long long b) {
    FLY_THROW(fly_replace_scalar_long(a.get(), cond.get(), b));
}

void replace(array &a, const array &cond, const unsigned long long b) {
    FLY_THROW(fly_replace_scalar_ulong(a.get(), cond.get(), b));
}

array select(const array &cond, const array &a, const long long b) {
    fly_array res;
    FLY_THROW(fly_select_scalar_r_long(&res, cond.get(), a.get(), b));
    return array(res);
}

array select(const array &cond, const array &a, const unsigned long long b) {
    fly_array res;
    FLY_THROW(fly_select_scalar_r_ulong(&res, cond.get(), a.get(), b));
    return array(res);
}

array select(const array &cond, const long long a, const array &b) {
    fly_array res;
    FLY_THROW(fly_select_scalar_l_long(&res, cond.get(), a, b.get()));
    return array(res);
}

array select(const array &cond, const unsigned long long a, const array &b) {
    fly_array res;
    FLY_THROW(fly_select_scalar_l_ulong(&res, cond.get(), a, b.get()));
    return array(res);
}

array pad(const array &in, const dim4 &beginPadding, const dim4 &endPadding,
          const borderType padFillType) {
    fly_array out = 0;
    // FIXME(pradeep) Cannot use dim4::ndims() since that will
    //               always return 0 if any one of dimensions
    //               has no padding completely
    FLY_THROW(fly_pad(&out, in.get(), 4, beginPadding.get(), 4, endPadding.get(),
                    padFillType));
    return array(out);
}

}  // namespace fly
