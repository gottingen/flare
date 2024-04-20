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

#include <fly/algorithm.h>
#include <fly/arith.h>
#include <fly/blas.h>
#include <fly/data.h>
#include <fly/device.h>
#include <fly/gfor.h>
#include <fly/half.h>
#include <fly/index.h>
#include <fly/internal.h>
#include <fly/traits.hpp>
#include <fly/util.h>
#include "error.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wparentheses"
#include <fly/half.hpp>  //note: NOT common. From extern/half/include/half.hpp
#pragma GCC diagnostic pop

#ifdef FLY_CUDA
// NOTE: Adding ifdef here to avoid copying code constructor in the cuda backend
#include <cuda_fp16.h>
#include <traits.hpp>
#endif

#ifdef FLY_UNIFIED
#include <symbol_manager.hpp>
#include <fly/backend.h>
using flare::common::getFunctionPointer;
#endif

#include <memory>
#include <stdexcept>
#include <vector>

using fly::calcDim;
using fly::dim4;
using std::copy;
using std::logic_error;
using std::vector;

namespace {
int gforDim(fly_index_t *indices) {
    for (int i = 0; i < FLY_MAX_DIMS; i++) {
        if (indices[i].isBatch) { return i; }
    }
    return -1;
}

fly_array gforReorder(const fly_array in, unsigned dim) {
    // This is here to stop gcc from complaining
    if (dim > 3) { FLY_THROW_ERR("GFor: Dimension is invalid", FLY_ERR_SIZE); }
    unsigned order[FLY_MAX_DIMS] = {0, 1, 2, dim};

    order[dim] = 3;
    fly_array out;
    FLY_THROW(fly_reorder(&out, in, order[0], order[1], order[2], order[3]));
    return out;
}

fly::dim4 seqToDims(fly_index_t *indices, fly::dim4 parentDims,
                   bool reorder = true) {
    try {
        fly::dim4 odims(1);
        for (int i = 0; i < FLY_MAX_DIMS; i++) {
            if (indices[i].isSeq) {
                odims[i] = calcDim(indices[i].idx.seq, parentDims[i]);
            } else {
                dim_t elems = 0;
                FLY_THROW(fly_get_elements(&elems, indices[i].idx.arr));
                odims[i] = elems;
            }
        }

        // Change the dimensions if inside GFOR
        if (reorder) {
            for (int i = 0; i < FLY_MAX_DIMS; i++) {
                if (indices[i].isBatch) {
                    int tmp  = odims[i];
                    odims[i] = odims[3];
                    odims[3] = tmp;
                    break;
                }
            }
        }
        return odims;
    } catch (const logic_error &err) { FLY_THROW_ERR(err.what(), FLY_ERR_SIZE); }
}

unsigned numDims(const fly_array arr) {
    unsigned nd;
    FLY_THROW(fly_get_numdims(&nd, arr));
    return nd;
}

dim4 getDims(const fly_array arr) {
    dim_t d0, d1, d2, d3;
    FLY_THROW(fly_get_dims(&d0, &d1, &d2, &d3, arr));
    return dim4(d0, d1, d2, d3);
}

fly_array initEmptyArray(fly::dtype ty, dim_t d0, dim_t d1 = 1, dim_t d2 = 1,
                        dim_t d3 = 1) {
    fly_array arr;
    dim_t my_dims[] = {d0, d1, d2, d3};
    FLY_THROW(fly_create_handle(&arr, FLY_MAX_DIMS, my_dims, ty));
    return arr;
}

fly_array initDataArray(const void *ptr, int ty, fly::source src, dim_t d0,
                       dim_t d1 = 1, dim_t d2 = 1, dim_t d3 = 1) {
    dim_t my_dims[] = {d0, d1, d2, d3};
    fly_array arr;
    switch (src) {
        case flyHost:
            FLY_THROW(fly_create_array(&arr, ptr, FLY_MAX_DIMS, my_dims,
                                     static_cast<fly_dtype>(ty)));
            break;
        case flyDevice:
            FLY_THROW(fly_device_array(&arr, const_cast<void *>(ptr), FLY_MAX_DIMS,
                                     my_dims, static_cast<fly_dtype>(ty)));
            break;
        default:
            FLY_THROW_ERR(
                "Can not create array from the requested source pointer",
                FLY_ERR_ARG);
    }
    return arr;
}
}  // namespace

namespace fly {

struct array::array_proxy::array_proxy_impl {
    // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
    array *parent_;  //< The original array
    // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
    fly_index_t indices_[4];  //< Indexing array or seq objects
    // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
    bool is_linear_;

    // if true the parent_ object will be deleted on distruction. This is
    // necessary only when calling indexing functions in array_proxy objects.
    // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
    bool delete_on_destruction_;
    array_proxy_impl(array &parent, fly_index_t *idx, bool linear)
        : parent_(&parent)
        , indices_()
        , is_linear_(linear)
        , delete_on_destruction_(false) {
        std::copy(idx, idx + FLY_MAX_DIMS, indices_);
    }

    void delete_on_destruction(bool val) { delete_on_destruction_ = val; }

    ~array_proxy_impl() {
        if (delete_on_destruction_) { delete parent_; }
    }

    array_proxy_impl(const array_proxy_impl &)            = delete;
    array_proxy_impl(const array_proxy_impl &&)           = delete;
    array_proxy_impl operator=(const array_proxy_impl &)  = delete;
    array_proxy_impl operator=(const array_proxy_impl &&) = delete;
};

array::array(const fly_array handle) : arr(handle) {}

array::array() : arr(initEmptyArray(f32, 0, 1, 1, 1)) {}

array::array(array &&other) noexcept : arr(other.arr) { other.arr = 0; }

array &array::operator=(array &&other) noexcept {
    fly_release_array(arr);
    arr       = other.arr;
    other.arr = 0;
    return *this;
}

array::array(const dim4 &dims, fly::dtype ty)
    : arr(initEmptyArray(ty, dims[0], dims[1], dims[2], dims[3])) {}

array::array(dim_t dim0, fly::dtype ty) : arr(initEmptyArray(ty, dim0)) {}

array::array(dim_t dim0, dim_t dim1, fly::dtype ty)
    : arr(initEmptyArray(ty, dim0, dim1)) {}

array::array(dim_t dim0, dim_t dim1, dim_t dim2, fly::dtype ty)
    : arr(initEmptyArray(ty, dim0, dim1, dim2)) {}

array::array(dim_t dim0, dim_t dim1, dim_t dim2, dim_t dim3, fly::dtype ty)
    : arr(initEmptyArray(ty, dim0, dim1, dim2, dim3)) {}

template<>
struct dtype_traits<half_float::half> {
    enum { fly_type = f16, ctype = f16 };
    using base_type = half;
    static const char *getName() { return "half"; }
};

#define INSTANTIATE(T)                                                         \
    template<>                                                                 \
    FLY_API array::array(const dim4 &dims, const T *ptr, fly::source src)         \
        : arr(initDataArray(ptr, dtype_traits<T>::fly_type, src, dims[0],       \
                            dims[1], dims[2], dims[3])) {}                     \
    template<>                                                                 \
    FLY_API array::array(dim_t dim0, const T *ptr, fly::source src)               \
        : arr(initDataArray(ptr, dtype_traits<T>::fly_type, src, dim0)) {}      \
    template<>                                                                 \
    FLY_API array::array(dim_t dim0, dim_t dim1, const T *ptr, fly::source src)   \
        : arr(initDataArray(ptr, dtype_traits<T>::fly_type, src, dim0, dim1)) { \
    }                                                                          \
    template<>                                                                 \
    FLY_API array::array(dim_t dim0, dim_t dim1, dim_t dim2, const T *ptr,       \
                       fly::source src)                                         \
        : arr(initDataArray(ptr, dtype_traits<T>::fly_type, src, dim0, dim1,    \
                            dim2)) {}                                          \
    template<>                                                                 \
    FLY_API array::array(dim_t dim0, dim_t dim1, dim_t dim2, dim_t dim3,         \
                       const T *ptr, fly::source src)                           \
        : arr(initDataArray(ptr, dtype_traits<T>::fly_type, src, dim0, dim1,    \
                            dim2, dim3)) {}

INSTANTIATE(cdouble)
INSTANTIATE(cfloat)
INSTANTIATE(double)
INSTANTIATE(float)
INSTANTIATE(unsigned)
INSTANTIATE(int)
INSTANTIATE(unsigned char)
INSTANTIATE(char)
INSTANTIATE(long long)
INSTANTIATE(unsigned long long)
INSTANTIATE(short)
INSTANTIATE(unsigned short)
INSTANTIATE(fly_half)
INSTANTIATE(half_float::half)
#ifdef FLY_CUDA
INSTANTIATE(__half);
#endif

#undef INSTANTIATE

array::~array() {
#ifdef FLY_UNIFIED
    using fly_release_array_ptr =
        std::add_pointer<decltype(fly_release_array)>::type;

    if (get()) {
        fly_backend backend = flare::unified::getActiveBackend();
        fly_err err         = fly_get_backend_id(&backend, get());
        if (!err) {
            switch (backend) {
                case FLY_BACKEND_CPU: {
                    static auto *cpu_handle =
                        flare::unified::getActiveHandle();
                    static auto release_func =
                        reinterpret_cast<fly_release_array_ptr>(
                            getFunctionPointer(cpu_handle, "fly_release_array"));
                    release_func(get());
                    break;
                }
                case FLY_BACKEND_CUDA: {
                    static auto *cuda_handle =
                        flare::unified::getActiveHandle();
                    static auto release_func =
                        reinterpret_cast<fly_release_array_ptr>(
                            getFunctionPointer(cuda_handle,
                                               "fly_release_array"));
                    release_func(get());
                    break;
                }
                case FLY_BACKEND_DEFAULT:
                    assert(1 != 1 &&
                           "FLY_BACKEND_DEFAULT cannot be set as a backend for "
                           "an array");
            }
        }
    }
#else
    // THOU SHALL NOT THROW IN DESTRUCTORS
    if (fly_array arr = get()) { fly_release_array(arr); }
#endif
}

fly::dtype array::type() const {
    fly::dtype my_type;
    FLY_THROW(fly_get_type(&my_type, arr));
    return my_type;
}

dim_t array::elements() const {
    dim_t elems;
    FLY_THROW(fly_get_elements(&elems, get()));
    return elems;
}

void array::host(void *ptr) const { FLY_THROW(fly_get_data_ptr(ptr, get())); }

fly_array array::get() { return arr; }

fly_array array::get() const { return const_cast<array *>(this)->get(); }

// Helper functions
dim4 array::dims() const { return getDims(get()); }

dim_t array::dims(unsigned dim) const { return dims()[dim]; }

unsigned array::numdims() const { return numDims(get()); }

size_t array::bytes() const {
    dim_t nElements;
    FLY_THROW(fly_get_elements(&nElements, get()));
    return nElements * getSizeOf(type());
}

size_t array::allocated() const {
    size_t result = 0;
    FLY_THROW(fly_get_allocated_bytes(&result, get()));
    return result;
}

array array::copy() const {
    fly_array other = nullptr;
    FLY_THROW(fly_copy_array(&other, get()));
    return array(other);
}

#undef INSTANTIATE
#define INSTANTIATE(fn)                    \
    bool array::is##fn() const {           \
        bool ret = false;                  \
        FLY_THROW(fly_is_##fn(&ret, get())); \
        return ret;                        \
    }

INSTANTIATE(empty)
INSTANTIATE(scalar)
INSTANTIATE(vector)
INSTANTIATE(row)
INSTANTIATE(column)
INSTANTIATE(complex)
INSTANTIATE(double)
INSTANTIATE(single)
INSTANTIATE(half)
INSTANTIATE(realfloating)
INSTANTIATE(floating)
INSTANTIATE(integer)
INSTANTIATE(bool)
INSTANTIATE(sparse)

#undef INSTANTIATE

static array::array_proxy gen_indexing(const array &ref, const index &s0,
                                       const index &s1, const index &s2,
                                       const index &s3, bool linear = false) {
    ref.eval();
    fly_index_t inds[FLY_MAX_DIMS];
    inds[0] = s0.get();
    inds[1] = s1.get();
    inds[2] = s2.get();
    inds[3] = s3.get();

    return array::array_proxy(const_cast<array &>(ref), inds, linear);
}

array::array_proxy array::operator()(const index &s0) {
    return const_cast<const array *>(this)->operator()(s0);
}

array::array_proxy array::operator()(const index &s0, const index &s1,
                                     const index &s2, const index &s3) {
    return const_cast<const array *>(this)->operator()(s0, s1, s2, s3);
}

// NOLINTNEXTLINE(readability-const-return-type)
const array::array_proxy array::operator()(const index &s0) const {
    index z = index(0);
    if (isvector()) {
        switch (numDims(this->arr)) {
            case 1: return gen_indexing(*this, s0, z, z, z);
            case 2: return gen_indexing(*this, z, s0, z, z);
            case 3: return gen_indexing(*this, z, z, s0, z);
            case 4: return gen_indexing(*this, z, z, z, s0);
            default: FLY_THROW_ERR("ndims for Array is invalid", FLY_ERR_SIZE);
        }
    } else {
        return gen_indexing(*this, s0, z, z, z, true);
    }
}

// NOLINTNEXTLINE(readability-const-return-type)
const array::array_proxy array::operator()(const index &s0, const index &s1,
                                           const index &s2,
                                           const index &s3) const {
    return gen_indexing(*this, s0, s1, s2, s3);
}

// NOLINTNEXTLINE(readability-const-return-type)
const array::array_proxy array::row(int index) const {
    return this->operator()(index, span, span, span);
}

array::array_proxy array::row(int index) {
    return const_cast<const array *>(this)->row(index);
}

// NOLINTNEXTLINE(readability-const-return-type)
const array::array_proxy array::col(int index) const {
    return this->operator()(span, index, span, span);
}

array::array_proxy array::col(int index) {
    return const_cast<const array *>(this)->col(index);
}

// NOLINTNEXTLINE(readability-const-return-type)
const array::array_proxy array::slice(int index) const {
    return this->operator()(span, span, index, span);
}

array::array_proxy array::slice(int index) {
    return const_cast<const array *>(this)->slice(index);
}

// NOLINTNEXTLINE(readability-const-return-type)
const array::array_proxy array::rows(int first, int last) const {
    seq idx(first, last, 1);
    return this->operator()(idx, span, span, span);
}

array::array_proxy array::rows(int first, int last) {
    return const_cast<const array *>(this)->rows(first, last);
}

// NOLINTNEXTLINE(readability-const-return-type)
const array::array_proxy array::cols(int first, int last) const {
    seq idx(first, last, 1);
    return this->operator()(span, idx, span, span);
}

array::array_proxy array::cols(int first, int last) {
    return const_cast<const array *>(this)->cols(first, last);
}

// NOLINTNEXTLINE(readability-const-return-type)
const array::array_proxy array::slices(int first, int last) const {
    seq idx(first, last, 1);
    return this->operator()(span, span, idx, span);
}

array::array_proxy array::slices(int first, int last) {
    return const_cast<const array *>(this)->slices(first, last);
}

// NOLINTNEXTLINE(readability-const-return-type)
const array array::as(fly::dtype type) const {
    fly_array out;
    FLY_THROW(fly_cast(&out, this->get(), type));
    return array(out);
}

array::array(const array &in) : arr(nullptr) {
    FLY_THROW(fly_retain_array(&arr, in.get()));
}

array::array(const array &input, const dim4 &dims) : arr(nullptr) {
    FLY_THROW(fly_moddims(&arr, input.get(), FLY_MAX_DIMS, dims.get()));
}

array::array(const array &input, const dim_t dim0, const dim_t dim1,
             const dim_t dim2, const dim_t dim3)
    : arr(nullptr) {
    dim_t dims[] = {dim0, dim1, dim2, dim3};
    FLY_THROW(fly_moddims(&arr, input.get(), FLY_MAX_DIMS, dims));
}

// Transpose and Conjugate Transpose
array array::T() const { return transpose(*this); }

array array::H() const { return transpose(*this, true); }

void array::set(fly_array tmp) {
    if (arr) { FLY_THROW(fly_release_array(arr)); }
    arr = tmp;
}

// Assign values to an array
array::array_proxy &fly::array::array_proxy::operator=(const array &other) {
    unsigned nd           = numDims(impl->parent_->get());
    const dim4 this_dims  = getDims(impl->parent_->get());
    const dim4 other_dims = other.dims();
    int dim               = gforDim(impl->indices_);
    fly_array other_arr    = other.get();

    bool batch_assign = false;
    bool is_reordered = false;
    if (dim >= 0) {
        // FIXME: Figure out a faster, cleaner way to do this
        dim4 out_dims = seqToDims(impl->indices_, this_dims, false);

        batch_assign = true;
        for (int i = 0; i < FLY_MAX_DIMS; i++) {
            if (this->impl->indices_[i].isBatch) {
                batch_assign &= (other_dims[i] == 1);
            } else {
                batch_assign &= (other_dims[i] == out_dims[i]);
            }
        }

        if (batch_assign) {
            fly_array out;
            FLY_THROW(fly_tile(&out, other_arr, out_dims[0] / other_dims[0],
                             out_dims[1] / other_dims[1],
                             out_dims[2] / other_dims[2],
                             out_dims[3] / other_dims[3]));
            other_arr = out;

        } else if (out_dims != other_dims) {
            // HACK: This is a quick check to see if other has been reordered
            // inside gfor
            // TODO(umar): Figure out if this breaks and implement a cleaner
            // method
            other_arr    = gforReorder(other_arr, dim);
            is_reordered = true;
        }
    }

    fly_array par_arr = 0;

    dim4 parent_dims = impl->parent_->dims();
    if (impl->is_linear_) {
        FLY_THROW(fly_flat(&par_arr, impl->parent_->get()));
        // The set call will dereference the impl->parent_ array. We are doing
        // this because the fly_flat call above increases the reference count of
        // the parent array which triggers a copy operation. This triggers a
        // copy operation inside the fly_assign_gen function below. The parent
        // array will be reverted to the original array and shape later in the
        // code.
        fly_array empty = 0;
        impl->parent_->set(empty);
        nd = 1;
    } else {
        par_arr = impl->parent_->get();
    }

    fly_array flat_res = 0;
    FLY_THROW(fly_assign_gen(&flat_res, par_arr, nd, impl->indices_, other_arr));

    fly_array res         = 0;
    fly_array unflattened = 0;
    if (impl->is_linear_) {
        FLY_THROW(
            fly_moddims(&res, flat_res, this_dims.ndims(), this_dims.get()));
        // Unflatten the fly_array and reset the original reference
        FLY_THROW(fly_moddims(&unflattened, par_arr, parent_dims.ndims(),
                            parent_dims.get()));
        impl->parent_->set(unflattened);
        FLY_THROW(fly_release_array(par_arr));
        FLY_THROW(fly_release_array(flat_res));
    } else {
        res = flat_res;
    }

    impl->parent_->set(res);

    if (dim >= 0 && (is_reordered || batch_assign)) {
        if (other_arr) { FLY_THROW(fly_release_array(other_arr)); }
    }
    return *this;
}

array::array_proxy &fly::array::array_proxy::operator=(
    const array::array_proxy &other) {
    if (this == &other) { return *this; }
    array out = other;
    *this     = out;
    return *this;
}

fly::array::array_proxy::array_proxy(array &par, fly_index_t *ssss, bool linear)
    : impl(new array_proxy_impl(par, ssss, linear)) {}

fly::array::array_proxy::array_proxy(const array_proxy &other)
    : impl(new array_proxy_impl(*other.impl->parent_, other.impl->indices_,
                                other.impl->is_linear_)) {}

// NOLINTNEXTLINE(performance-noexcept-move-constructor,hicpp-noexcept-move)
fly::array::array_proxy::array_proxy(array_proxy &&other) {
    impl       = other.impl;
    other.impl = nullptr;
}

// NOLINTNEXTLINE(performance-noexcept-move-constructor,hicpp-noexcept-move)
array::array_proxy &fly::array::array_proxy::operator=(array_proxy &&other) {
    array out = other;
    *this     = out;
    return *this;
}

fly::array::array_proxy::~array_proxy() { delete impl; }

array array::array_proxy::as(dtype type) const {
    array out = *this;
    return out.as(type);
}

dim_t array::array_proxy::dims(unsigned dim) const {
    array out = *this;
    return out.dims(dim);
}

void array::array_proxy::host(void *ptr) const {
    array out = *this;
    return out.host(ptr);
}

#define MEM_FUNC(PREFIX, FUNC)                \
    PREFIX array::array_proxy::FUNC() const { \
        array out = *this;                    \
        return out.FUNC();                    \
    }

MEM_FUNC(dim_t, elements)
MEM_FUNC(array, T)
MEM_FUNC(array, H)
MEM_FUNC(dtype, type)
MEM_FUNC(dim4, dims)
MEM_FUNC(unsigned, numdims)
MEM_FUNC(size_t, bytes)
MEM_FUNC(size_t, allocated)
MEM_FUNC(array, copy)
MEM_FUNC(bool, isempty)
MEM_FUNC(bool, isscalar)
MEM_FUNC(bool, isvector)
MEM_FUNC(bool, isrow)
MEM_FUNC(bool, iscolumn)
MEM_FUNC(bool, iscomplex)
MEM_FUNC(bool, isdouble)
MEM_FUNC(bool, issingle)
MEM_FUNC(bool, ishalf)
MEM_FUNC(bool, isrealfloating)
MEM_FUNC(bool, isfloating)
MEM_FUNC(bool, isinteger)
MEM_FUNC(bool, isbool)
MEM_FUNC(bool, issparse)
MEM_FUNC(void, eval)
MEM_FUNC(fly_array, get)
// MEM_FUNC(void                   , unlock)
#undef MEM_FUNC

#define ASSIGN_TYPE(TY, OP)                                                \
    array::array_proxy &array::array_proxy::operator OP(const TY &value) { \
        dim4 pdims = getDims(impl->parent_->get());                        \
        if (impl->is_linear_) pdims = dim4(pdims.elements());              \
        dim4 dims    = seqToDims(impl->indices_, pdims);                   \
        fly::dtype ty = impl->parent_->type();                              \
        array cst    = constant(value, dims, ty);                          \
        this->operator OP(cst);                                            \
        return *this;                                                      \
    }

#define ASSIGN_OP(OP, op1)              \
    ASSIGN_TYPE(double, OP)             \
    ASSIGN_TYPE(float, OP)              \
    ASSIGN_TYPE(cdouble, OP)            \
    ASSIGN_TYPE(cfloat, OP)             \
    ASSIGN_TYPE(int, OP)                \
    ASSIGN_TYPE(unsigned, OP)           \
    ASSIGN_TYPE(long, OP)               \
    ASSIGN_TYPE(unsigned long, OP)      \
    ASSIGN_TYPE(long long, OP)          \
    ASSIGN_TYPE(unsigned long long, OP) \
    ASSIGN_TYPE(char, OP)               \
    ASSIGN_TYPE(unsigned char, OP)      \
    ASSIGN_TYPE(bool, OP)               \
    ASSIGN_TYPE(short, OP)              \
    ASSIGN_TYPE(unsigned short, OP)

ASSIGN_OP(=, =)
ASSIGN_OP(+=, +)
ASSIGN_OP(-=, -)
ASSIGN_OP(*=, *)
ASSIGN_OP(/=, /)
#undef ASSIGN_OP

#undef ASSIGN_TYPE

#define SELF_OP(OP, op1)                                                      \
    array::array_proxy &array::array_proxy::operator OP(                      \
        const array_proxy &other) {                                           \
        *this = *this op1 other;                                              \
        return *this;                                                         \
    }                                                                         \
    array::array_proxy &array::array_proxy::operator OP(const array &other) { \
        *this = *this op1 other;                                              \
        return *this;                                                         \
    }

SELF_OP(+=, +)
SELF_OP(-=, -)
SELF_OP(*=, *)
SELF_OP(/=, /)
#undef SELF_OP

array::array_proxy::operator array() const {
    fly_array tmp = nullptr;
    fly_array arr = nullptr;

    if (impl->is_linear_) {
        FLY_THROW(fly_flat(&arr, impl->parent_->get()));
    } else {
        arr = impl->parent_->get();
    }

    FLY_THROW(fly_index_gen(&tmp, arr, FLY_MAX_DIMS, impl->indices_));
    if (impl->is_linear_) { FLY_THROW(fly_release_array(arr)); }

    int dim = gforDim(impl->indices_);
    if (tmp && dim >= 0) {
        arr = gforReorder(tmp, dim);
        if (tmp) { FLY_THROW(fly_release_array(tmp)); }
    } else {
        arr = tmp;
    }

    return array(arr);
}

array::array_proxy::operator array() {
    return const_cast<const array::array_proxy *>(this)->operator array();
}

#define MEM_INDEX(FUNC_SIG, USAGE)                                \
    array::array_proxy array::array_proxy::FUNC_SIG {             \
        array *out               = new array(*this);              \
        array::array_proxy proxy = out->USAGE;                    \
        proxy.impl->delete_on_destruction(true);                  \
        return proxy;                                             \
    }                                                             \
                                                                  \
    const array::array_proxy array::array_proxy::FUNC_SIG const { \
        const array *out         = new array(*this);              \
        array::array_proxy proxy = out->USAGE;                    \
        proxy.impl->delete_on_destruction(true);                  \
        return proxy;                                             \
    }
// NOLINTNEXTLINE(readability-const-return-type)
MEM_INDEX(row(int index), row(index));
// NOLINTNEXTLINE(readability-const-return-type)
MEM_INDEX(rows(int first, int last), rows(first, last));
// NOLINTNEXTLINE(readability-const-return-type)
MEM_INDEX(col(int index), col(index));
// NOLINTNEXTLINE(readability-const-return-type)
MEM_INDEX(cols(int first, int last), cols(first, last));
// NOLINTNEXTLINE(readability-const-return-type)
MEM_INDEX(slice(int index), slice(index));
// NOLINTNEXTLINE(readability-const-return-type)
MEM_INDEX(slices(int first, int last), slices(first, last));

#undef MEM_INDEX

///////////////////////////////////////////////////////////////////////////
// Operator =
///////////////////////////////////////////////////////////////////////////
array &array::operator=(const array &other) {
    if (this == &other || this->get() == other.get()) { return *this; }
    // TODO(umar): Unsafe. loses data if fly_weak_copy fails
    if (this->arr != nullptr) { FLY_THROW(fly_release_array(this->arr)); }

    fly_array temp = nullptr;
    FLY_THROW(fly_retain_array(&temp, other.get()));
    this->arr = temp;
    return *this;
}
#define ASSIGN_TYPE(TY, OP)                        \
    array &array::operator OP(const TY &value) {   \
        fly::dim4 dims = this->dims();              \
        fly::dtype ty  = this->type();              \
        array cst     = constant(value, dims, ty); \
        return operator OP(cst);                   \
    }

#define ASSIGN_OP(OP, op1)                                        \
    array &array::operator OP(const array &other) {               \
        fly_array out = 0;                                         \
        FLY_THROW(op1(&out, this->get(), other.get(), gforGet())); \
        this->set(out);                                           \
        return *this;                                             \
    }                                                             \
    ASSIGN_TYPE(double, OP)                                       \
    ASSIGN_TYPE(float, OP)                                        \
    ASSIGN_TYPE(cdouble, OP)                                      \
    ASSIGN_TYPE(cfloat, OP)                                       \
    ASSIGN_TYPE(int, OP)                                          \
    ASSIGN_TYPE(unsigned, OP)                                     \
    ASSIGN_TYPE(long, OP)                                         \
    ASSIGN_TYPE(unsigned long, OP)                                \
    ASSIGN_TYPE(long long, OP)                                    \
    ASSIGN_TYPE(unsigned long long, OP)                           \
    ASSIGN_TYPE(char, OP)                                         \
    ASSIGN_TYPE(unsigned char, OP)                                \
    ASSIGN_TYPE(bool, OP)                                         \
    ASSIGN_TYPE(short, OP)                                        \
    ASSIGN_TYPE(unsigned short, OP)

ASSIGN_OP(+=, fly_add)
ASSIGN_OP(-=, fly_sub)
ASSIGN_OP(*=, fly_mul)
ASSIGN_OP(/=, fly_div)

#undef ASSIGN_OP

#undef ASSIGN_TYPE

#define ASSIGN_TYPE(TY, OP)                        \
    array &array::operator OP(const TY &value) {   \
        fly::dim4 dims = this->dims();              \
        fly::dtype ty  = this->type();              \
        array cst     = constant(value, dims, ty); \
        operator OP(cst);                          \
        return *this;                              \
    }

#define ASSIGN_OP(OP)                   \
    ASSIGN_TYPE(double, OP)             \
    ASSIGN_TYPE(float, OP)              \
    ASSIGN_TYPE(cdouble, OP)            \
    ASSIGN_TYPE(cfloat, OP)             \
    ASSIGN_TYPE(int, OP)                \
    ASSIGN_TYPE(unsigned, OP)           \
    ASSIGN_TYPE(long, OP)               \
    ASSIGN_TYPE(unsigned long, OP)      \
    ASSIGN_TYPE(long long, OP)          \
    ASSIGN_TYPE(unsigned long long, OP) \
    ASSIGN_TYPE(char, OP)               \
    ASSIGN_TYPE(unsigned char, OP)      \
    ASSIGN_TYPE(bool, OP)               \
    ASSIGN_TYPE(short, OP)              \
    ASSIGN_TYPE(unsigned short, OP)

ASSIGN_OP(=)

#undef ASSIGN_OP

#undef ASSIGN_TYPE

fly::dtype implicit_dtype(fly::dtype scalar_type, fly::dtype array_type) {
    // If same, do not do anything
    if (scalar_type == array_type) { return scalar_type; }

    // If complex, return appropriate complex type
    if (scalar_type == c32 || scalar_type == c64) {
        if (array_type == f64 || array_type == c64) { return c64; }
        return c32;
    }

    // If 64 bit precision, do not lose precision
    if (array_type == f64 || array_type == c64 || array_type == f32 ||
        array_type == c32) {
        return array_type;
    }

    // If the array is f16 then avoid upcasting to float or double
    if ((scalar_type == f64 || scalar_type == f32) && (array_type == f16)) {
        return f16;
    }

    // Default to single precision by default when multiplying with scalar
    if ((scalar_type == f64 || scalar_type == c64) &&
        (array_type != f64 && array_type != c64)) {
        return f32;
    }

    // Punt to C api for everything else
    return scalar_type;
}

#define BINARY_TYPE(TY, OP, release_func, dty)                          \
    array operator OP(const array &plhs, const TY &value) {             \
        fly_array out;                                                   \
        fly::dtype cty = implicit_dtype(dty, plhs.type());               \
        array cst     = constant(value, plhs.dims(), cty);              \
        FLY_THROW(release_func(&out, plhs.get(), cst.get(), gforGet())); \
        return array(out);                                              \
    }                                                                   \
    array operator OP(const TY &value, const array &other) {            \
        const fly_array rhs = other.get();                               \
        fly_array out;                                                   \
        fly::dtype cty = implicit_dtype(dty, other.type());              \
        array cst     = constant(value, other.dims(), cty);             \
        FLY_THROW(release_func(&out, cst.get(), rhs, gforGet()));        \
        return array(out);                                              \
    }

#define BINARY_OP(OP, release_func)                                    \
    array operator OP(const array &lhs, const array &rhs) {            \
        fly_array out;                                                  \
        FLY_THROW(release_func(&out, lhs.get(), rhs.get(), gforGet())); \
        return array(out);                                             \
    }                                                                  \
    BINARY_TYPE(double, OP, release_func, f64)                         \
    BINARY_TYPE(float, OP, release_func, f32)                          \
    BINARY_TYPE(cdouble, OP, release_func, c64)                        \
    BINARY_TYPE(cfloat, OP, release_func, c32)                         \
    BINARY_TYPE(int, OP, release_func, s32)                            \
    BINARY_TYPE(unsigned, OP, release_func, u32)                       \
    BINARY_TYPE(long, OP, release_func, s64)                           \
    BINARY_TYPE(unsigned long, OP, release_func, u64)                  \
    BINARY_TYPE(long long, OP, release_func, s64)                      \
    BINARY_TYPE(unsigned long long, OP, release_func, u64)             \
    BINARY_TYPE(char, OP, release_func, b8)                            \
    BINARY_TYPE(unsigned char, OP, release_func, u8)                   \
    BINARY_TYPE(bool, OP, release_func, b8)                            \
    BINARY_TYPE(short, OP, release_func, s16)                          \
    BINARY_TYPE(unsigned short, OP, release_func, u16)

BINARY_OP(+, fly_add)
BINARY_OP(-, fly_sub)
BINARY_OP(*, fly_mul)
BINARY_OP(/, fly_div)
BINARY_OP(==, fly_eq)
BINARY_OP(!=, fly_neq)
BINARY_OP(<, fly_lt)
BINARY_OP(<=, fly_le)
BINARY_OP(>, fly_gt)
BINARY_OP(>=, fly_ge)
BINARY_OP(&&, fly_and)
BINARY_OP(||, fly_or)
BINARY_OP(%, fly_mod)
BINARY_OP(&, fly_bitand)
BINARY_OP(|, fly_bitor)
BINARY_OP(^, fly_bitxor)
BINARY_OP(<<, fly_bitshiftl)
BINARY_OP(>>, fly_bitshiftr)

#undef BINARY_OP

#undef BINARY_TYPE

array array::operator-() const {
    fly_array lhs = this->get();
    fly_array out;
    array cst = constant(0, this->dims(), this->type());
    FLY_THROW(fly_sub(&out, cst.get(), lhs, gforGet()));
    return array(out);
}

array array::operator!() const {
    fly_array lhs = this->get();
    fly_array out;
    FLY_THROW(fly_not(&out, lhs));
    return array(out);
}

array array::operator~() const {
    fly_array lhs = this->get();
    fly_array out = nullptr;
    FLY_THROW(fly_bitnot(&out, lhs));
    return array(out);
}

void array::eval() const { FLY_THROW(fly_eval(get())); }

// array instanciations
#define INSTANTIATE(T)                                                         \
    template<>                                                                 \
    FLY_API T *array::host() const {                                             \
        if (type() != (fly::dtype)dtype_traits<T>::fly_type) {                   \
            FLY_THROW_ERR("Requested type doesn't match with array",            \
                         FLY_ERR_TYPE);                                         \
        }                                                                      \
        void *res;                                                             \
        FLY_THROW(fly_alloc_host(&res, bytes()));                                \
        FLY_THROW(fly_get_data_ptr(res, get()));                                 \
                                                                               \
        return (T *)res;                                                       \
    }                                                                          \
    template<>                                                                 \
    FLY_API T array::scalar() const {                                            \
        fly_dtype type = (fly_dtype)fly::dtype_traits<T>::fly_type;                \
        if (type != this->type())                                              \
            FLY_THROW_ERR("Requested type doesn't match array type",            \
                         FLY_ERR_TYPE);                                         \
        T val;                                                                 \
        FLY_THROW(fly_get_scalar(&val, get()));                                  \
        return val;                                                            \
    }                                                                          \
    template<>                                                                 \
    FLY_API T *array::device() const {                                           \
        void *ptr = NULL;                                                      \
        FLY_THROW(fly_get_device_ptr(&ptr, get()));                              \
        return (T *)ptr;                                                       \
    }                                                                          \
    template<>                                                                 \
    FLY_API void array::write(const T *ptr, const size_t bytes,                  \
                            fly::source src) {                                  \
        if (src == flyHost) {                                                   \
            FLY_THROW(fly_write_array(get(), ptr, bytes, (fly::source)flyHost));   \
        }                                                                      \
        if (src == flyDevice) {                                                 \
            FLY_THROW(fly_write_array(get(), ptr, bytes, (fly::source)flyDevice)); \
        }                                                                      \
    }

INSTANTIATE(cdouble)
INSTANTIATE(cfloat)
INSTANTIATE(double)
INSTANTIATE(float)
INSTANTIATE(unsigned)
INSTANTIATE(int)
INSTANTIATE(unsigned char)
INSTANTIATE(char)
INSTANTIATE(long long)
INSTANTIATE(unsigned long long)
INSTANTIATE(short)
INSTANTIATE(unsigned short)
INSTANTIATE(fly_half)
INSTANTIATE(half_float::half)

template<>
FLY_API void array::write(const void *ptr, const size_t bytes, fly::source src) {
    FLY_THROW(fly_write_array(get(), ptr, bytes, src));
}

#undef INSTANTIATE

template<>
FLY_API void *array::device() const {
    void *ptr = nullptr;
    FLY_THROW(fly_get_device_ptr(&ptr, get()));
    return ptr;
}

// array_proxy instanciations
#define TEMPLATE_MEM_FUNC(TYPE, RETURN_TYPE, FUNC)       \
    template<>                                           \
    FLY_API RETURN_TYPE array::array_proxy::FUNC() const { \
        array out = *this;                               \
        return out.FUNC<TYPE>();                         \
    }

#define INSTANTIATE(T)              \
    TEMPLATE_MEM_FUNC(T, T *, host) \
    TEMPLATE_MEM_FUNC(T, T, scalar) \
    TEMPLATE_MEM_FUNC(T, T *, device)

INSTANTIATE(cdouble)
INSTANTIATE(cfloat)
INSTANTIATE(double)
INSTANTIATE(float)
INSTANTIATE(unsigned)
INSTANTIATE(int)
INSTANTIATE(unsigned char)
INSTANTIATE(char)
INSTANTIATE(long long)
INSTANTIATE(unsigned long long)
INSTANTIATE(short)
INSTANTIATE(unsigned short)
INSTANTIATE(fly_half)
INSTANTIATE(half_float::half)

#undef INSTANTIATE
#undef TEMPLATE_MEM_FUNC

// FIXME: These functions need to be implemented properly at a later point
void array::array_proxy::unlock() const {}
void array::array_proxy::lock() const {}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
bool array::array_proxy::isLocked() const { return false; }

int array::nonzeros() const { return count<int>(*this); }

void array::lock() const { FLY_THROW(fly_lock_array(get())); }

bool array::isLocked() const {
    bool res;
    FLY_THROW(fly_is_locked_array(&res, get()));
    return res;
}

void array::unlock() const { FLY_THROW(fly_unlock_array(get())); }

void eval(int num, array **arrays) {
    vector<fly_array> outputs(num);
    for (int i = 0; i < num; i++) { outputs[i] = arrays[i]->get(); }
    FLY_THROW(fly_eval_multiple(num, &outputs[0]));
}

void setManualEvalFlag(bool flag) { FLY_THROW(fly_set_manual_eval_flag(flag)); }

bool getManualEvalFlag() {
    bool flag;
    FLY_THROW(fly_get_manual_eval_flag(&flag));
    return flag;
}
}  // namespace fly
