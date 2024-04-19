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

#include <index.hpp>
#include <indexing_common.hpp>

#include <Array.hpp>
#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <common/err_common.hpp>
#include <common/moddims.hpp>
#include <handle.hpp>
#include <lookup.hpp>
#include <fly/arith.h>
#include <fly/array.h>
#include <fly/data.h>
#include <fly/index.h>

#include <array>
#include <cassert>
#include <cmath>
#include <vector>

using std::signbit;
using std::swap;
using std::vector;

using fly::dim4;
using flare::common::convert2Canonical;
using flare::common::createSpanIndex;
using flare::common::flat;
using flare::common::half;
using detail::cdouble;
using detail::cfloat;
using detail::index;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

namespace flare {
namespace common {
fly_index_t createSpanIndex() {
    static fly_index_t s = [] {
        fly_index_t s;
        s.idx.seq = fly_span;
        s.isSeq   = true;
        s.isBatch = false;
        return s;
    }();
    return s;
}

fly_seq convert2Canonical(const fly_seq s, const dim_t len) {
    double begin = signbit(s.begin) ? (len + s.begin) : s.begin;
    double end   = signbit(s.end) ? (len + s.end) : s.end;

    return fly_seq{begin, end, s.step};
}
}  // namespace common
}  // namespace flare

template<typename T>
static fly_array indexBySeqs(const fly_array& src,
                            const vector<fly_seq>& indicesV) {
    auto ndims        = static_cast<dim_t>(indicesV.size());
    const auto& input = getArray<T>(src);

    if (ndims == 1U && ndims != input.ndims()) {
        return getHandle(createSubArray(flat(input), indicesV));
    } else {
        return getHandle(createSubArray(input, indicesV));
    }
}

fly_err fly_index(fly_array* result, const fly_array in, const unsigned ndims,
                const fly_seq* indices) {
    try {
        ARG_ASSERT(2, (ndims > 0 && ndims <= FLY_MAX_DIMS));

        const ArrayInfo& inInfo = getInfo(in);
        fly_dtype type           = inInfo.getType();
        const dim4& iDims       = inInfo.dims();

        vector<fly_seq> indices_(ndims, fly_span);
        for (unsigned i = 0; i < ndims; ++i) {
            indices_[i] = convert2Canonical(indices[i], iDims[i]);

            ARG_ASSERT(3, (indices_[i].begin >= 0. && indices_[i].end >= 0.));
            if (signbit(indices_[i].step)) {
                ARG_ASSERT(3, indices_[i].begin >= indices_[i].end);
            } else {
                ARG_ASSERT(3, indices_[i].begin <= indices_[i].end);
            }
        }

        fly_array out = 0;

        switch (type) {
            case f32: out = indexBySeqs<float>(in, indices_); break;
            case c32: out = indexBySeqs<cfloat>(in, indices_); break;
            case f64: out = indexBySeqs<double>(in, indices_); break;
            case c64: out = indexBySeqs<cdouble>(in, indices_); break;
            case b8: out = indexBySeqs<char>(in, indices_); break;
            case s32: out = indexBySeqs<int>(in, indices_); break;
            case u32: out = indexBySeqs<unsigned>(in, indices_); break;
            case s16: out = indexBySeqs<short>(in, indices_); break;
            case u16: out = indexBySeqs<ushort>(in, indices_); break;
            case s64: out = indexBySeqs<intl>(in, indices_); break;
            case u64: out = indexBySeqs<uintl>(in, indices_); break;
            case u8: out = indexBySeqs<uchar>(in, indices_); break;
            case f16: out = indexBySeqs<half>(in, indices_); break;
            default: TYPE_ERROR(1, type);
        }
        swap(*result, out);
    }
    CATCHALL
    return FLY_SUCCESS;
}

template<typename T, typename idx_t>
inline fly_array lookup(const fly_array& in, const fly_array& idx,
                       const unsigned dim) {
    return getHandle(lookup(getArray<T>(in), getArray<idx_t>(idx), dim));
}

template<typename idx_t>
static fly_array lookup(const fly_array& in, const fly_array& idx,
                       const unsigned dim) {
    const ArrayInfo& inInfo = getInfo(in);
    fly_dtype inType         = inInfo.getType();

    switch (inType) {
        case f32: return lookup<float, idx_t>(in, idx, dim);
        case c32: return lookup<cfloat, idx_t>(in, idx, dim);
        case f64: return lookup<double, idx_t>(in, idx, dim);
        case c64: return lookup<cdouble, idx_t>(in, idx, dim);
        case s32: return lookup<int, idx_t>(in, idx, dim);
        case u32: return lookup<unsigned, idx_t>(in, idx, dim);
        case s64: return lookup<intl, idx_t>(in, idx, dim);
        case u64: return lookup<uintl, idx_t>(in, idx, dim);
        case s16: return lookup<short, idx_t>(in, idx, dim);
        case u16: return lookup<ushort, idx_t>(in, idx, dim);
        case u8: return lookup<uchar, idx_t>(in, idx, dim);
        case b8: return lookup<char, idx_t>(in, idx, dim);
        case f16: return lookup<half, idx_t>(in, idx, dim);
        default: TYPE_ERROR(1, inType);
    }
}

fly_err fly_lookup(fly_array* out, const fly_array in, const fly_array indices,
                 const unsigned dim) {
    try {
        const ArrayInfo& idxInfo = getInfo(indices);

        if (idxInfo.ndims() == 0) {
            *out = retain(indices);
            return FLY_SUCCESS;
        }

        ARG_ASSERT(3, (dim <= 3));
        ARG_ASSERT(2, idxInfo.isVector() || idxInfo.isScalar());

        fly_dtype idxType = idxInfo.getType();

        ARG_ASSERT(2, (idxType != c32));
        ARG_ASSERT(2, (idxType != c64));
        ARG_ASSERT(2, (idxType != b8));

        fly_array output = 0;

        switch (idxType) {
            case f32: output = lookup<float>(in, indices, dim); break;
            case f64: output = lookup<double>(in, indices, dim); break;
            case s32: output = lookup<int>(in, indices, dim); break;
            case u32: output = lookup<unsigned>(in, indices, dim); break;
            case s16: output = lookup<short>(in, indices, dim); break;
            case u16: output = lookup<ushort>(in, indices, dim); break;
            case s64: output = lookup<intl>(in, indices, dim); break;
            case u64: output = lookup<uintl>(in, indices, dim); break;
            case u8: output = lookup<uchar>(in, indices, dim); break;
            case f16: output = lookup<half>(in, indices, dim); break;
            default: TYPE_ERROR(1, idxType);
        }
        std::swap(*out, output);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

// idxrs parameter to the below static function
// expects 4 values which is handled appropriately
// by the C-API fly_index_gen
template<typename T>
static inline fly_array genIndex(const fly_array& in, const fly_index_t idxrs[]) {
    return getHandle<T>(index<T>(getArray<T>(in), idxrs));
}

fly_err fly_index_gen(fly_array* out, const fly_array in, const dim_t ndims,
                    const fly_index_t* indexs) {
    try {
        ARG_ASSERT(2, (ndims > 0 && ndims <= FLY_MAX_DIMS));
        ARG_ASSERT(3, (indexs != NULL));

        const ArrayInfo& iInfo = getInfo(in);
        const dim4& iDims      = iInfo.dims();
        fly_dtype inType        = getInfo(in).getType();

        if (iDims.ndims() <= 0) {
            *out = createHandle(dim4(0), inType);
            return FLY_SUCCESS;
        }

        if (ndims == 1 && ndims != static_cast<dim_t>(iInfo.ndims())) {
            fly_array in_ = 0;
            FLY_CHECK(fly_flat(&in_, in));
            FLY_CHECK(fly_index_gen(out, in_, ndims, indexs));
            FLY_CHECK(fly_release_array(in_));
            return FLY_SUCCESS;
        }

        int track = 0;
        std::array<fly_seq, FLY_MAX_DIMS> seqs{};
        seqs.fill(fly_span);
        for (dim_t i = 0; i < ndims; i++) {
            if (indexs[i].isSeq) {
                track++;
                seqs[i] = indexs[i].idx.seq;
            }
        }

        if (track == static_cast<int>(ndims)) {
            return fly_index(out, in, ndims, seqs.data());
        }

        std::array<fly_index_t, FLY_MAX_DIMS> idxrs{};

        for (dim_t i = 0; i < FLY_MAX_DIMS; ++i) {
            if (i < ndims) {
                bool isSeq = indexs[i].isSeq;
                if (!isSeq) {
                    // check if all fly_arrays have atleast one value
                    // to enable indexing along that dimension
                    const ArrayInfo& idxInfo = getInfo(indexs[i].idx.arr);
                    fly_dtype idxType         = idxInfo.getType();

                    ARG_ASSERT(3, (idxType != c32));
                    ARG_ASSERT(3, (idxType != c64));
                    ARG_ASSERT(3, (idxType != b8));

                    idxrs[i] = {{indexs[i].idx.arr}, isSeq, indexs[i].isBatch};
                } else {
                    // copy the fly_seq to local variable
                    fly_seq inSeq =
                        convert2Canonical(indexs[i].idx.seq, iDims[i]);
                    ARG_ASSERT(3, (inSeq.begin >= 0. || inSeq.end >= 0.));
                    if (signbit(inSeq.step)) {
                        ARG_ASSERT(3, inSeq.begin >= inSeq.end);
                    } else {
                        ARG_ASSERT(3, inSeq.begin <= inSeq.end);
                    }
                    idxrs[i].idx.seq = inSeq;
                    idxrs[i].isSeq   = isSeq;
                    idxrs[i].isBatch = indexs[i].isBatch;
                }
            } else {
                // set all dimensions above ndims to spanner
                idxrs[i] = createSpanIndex();
            }
        }
        fly_index_t* ptr = idxrs.data();

        fly_array output = 0;
        switch (inType) {
            case c64: output = genIndex<cdouble>(in, ptr); break;
            case f64: output = genIndex<double>(in, ptr); break;
            case c32: output = genIndex<cfloat>(in, ptr); break;
            case f32: output = genIndex<float>(in, ptr); break;
            case u64: output = genIndex<uintl>(in, ptr); break;
            case s64: output = genIndex<intl>(in, ptr); break;
            case u32: output = genIndex<uint>(in, ptr); break;
            case s32: output = genIndex<int>(in, ptr); break;
            case u16: output = genIndex<ushort>(in, ptr); break;
            case s16: output = genIndex<short>(in, ptr); break;
            case u8: output = genIndex<uchar>(in, ptr); break;
            case b8: output = genIndex<char>(in, ptr); break;
            case f16: output = genIndex<half>(in, ptr); break;
            default: TYPE_ERROR(1, inType);
        }
        std::swap(*out, output);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_seq fly_make_seq(double begin, double end, double step) {
    return fly_seq{begin, end, step};
}

fly_err fly_create_indexers(fly_index_t** indexers) {
    try {
        auto* out = new fly_index_t[FLY_MAX_DIMS];
        for (int i = 0; i < FLY_MAX_DIMS; ++i) {
            out[i].idx.seq = fly_span;
            out[i].isSeq   = true;
            out[i].isBatch = false;
        }
        std::swap(*indexers, out);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_set_array_indexer(fly_index_t* indexer, const fly_array idx,
                            const dim_t dim) {
    try {
        ARG_ASSERT(0, (indexer != NULL));
        ARG_ASSERT(1, (idx != NULL));
        ARG_ASSERT(2, (dim >= 0 && dim <= 3));
        indexer[dim] = fly_index_t{{idx}, false, false};
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_set_seq_indexer(fly_index_t* indexer, const fly_seq* idx,
                          const dim_t dim, const bool is_batch) {
    try {
        ARG_ASSERT(0, (indexer != NULL));
        ARG_ASSERT(1, (idx != NULL));
        ARG_ASSERT(2, (dim >= 0 && dim <= 3));
        indexer[dim].idx.seq = *idx;
        indexer[dim].isSeq   = true;
        indexer[dim].isBatch = is_batch;
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_set_seq_param_indexer(fly_index_t* indexer, const double begin,
                                const double end, const double step,
                                const dim_t dim, const bool is_batch) {
    try {
        ARG_ASSERT(0, (indexer != NULL));
        ARG_ASSERT(4, (dim >= 0 && dim <= 3));
        fly_seq s             = fly_make_seq(begin, end, step);
        indexer[dim].idx.seq = s;
        indexer[dim].isSeq   = true;
        indexer[dim].isBatch = is_batch;
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_release_indexers(fly_index_t* indexers) {
    try {
        delete[] indexers;
    }
    CATCHALL;
    return FLY_SUCCESS;
}
