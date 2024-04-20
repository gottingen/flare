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

#include <fly/algorithm.h>
#include <fly/array.h>
#include <fly/index.h>
#include "common.hpp"
#include "error.hpp"

namespace fly {

array lookup(const array &in, const array &idx, const int dim) {
    fly_array out = 0;
    FLY_THROW(fly_lookup(&out, in.get(), idx.get(), getFNSD(dim, in.dims())));
    return array(out);
}

void copy(array &dst, const array &src, const index &idx0, const index &idx1,
          const index &idx2, const index &idx3) {
    unsigned nd = dst.numdims();

    fly_index_t indices[] = {idx0.get(), idx1.get(), idx2.get(), idx3.get()};

    fly_array lhs       = dst.get();
    const fly_array rhs = src.get();
    FLY_THROW(fly_assign_gen(&lhs, lhs, nd, indices, rhs));
}

index::index() : impl{} {
    impl.idx.seq = fly_span;
    impl.isSeq   = true;
    impl.isBatch = false;
}

index::index(const int idx) : impl{} {
    impl.idx.seq = fly_make_seq(idx, idx, 1);
    impl.isSeq   = true;
    impl.isBatch = false;
}

index::index(const fly::seq &s0) : impl{} {
    impl.idx.seq = s0.s;
    impl.isSeq   = true;
    impl.isBatch = s0.m_gfor;
}

index::index(const fly_seq &s0) : impl{} {
    impl.idx.seq = s0;
    impl.isSeq   = true;
    impl.isBatch = false;
}

index::index(const fly::array &idx0) : impl{} {
    array idx    = idx0.isbool() ? where(idx0) : idx0;
    fly_array arr = 0;
    FLY_THROW(fly_retain_array(&arr, idx.get()));
    impl.idx.arr = arr;

    impl.isSeq   = false;
    impl.isBatch = false;
}

index::index(const fly::index &idx0) : impl{idx0.impl} {
    if (!impl.isSeq && impl.idx.arr) {
        // increment reference count to avoid double free
        // when/if idx0 is destroyed
        FLY_THROW(fly_retain_array(&impl.idx.arr, impl.idx.arr));
    }
}

// NOLINTNEXTLINE(hicpp-noexcept-move, performance-noexcept-move-constructor)
index::index(index &&idx0) : impl{idx0.impl} { idx0.impl.idx.arr = nullptr; }

index::~index() {
    if (!impl.isSeq && impl.idx.arr) { fly_release_array(impl.idx.arr); }
}

index &index::operator=(const index &idx0) {
    if (this == &idx0) { return *this; }

    impl = idx0.get();
    if (!impl.isSeq && impl.idx.arr) {
        // increment reference count to avoid double free
        // when/if idx0 is destroyed
        FLY_THROW(fly_retain_array(&impl.idx.arr, impl.idx.arr));
    }
    return *this;
}

// NOLINTNEXTLINE(hicpp-noexcept-move, performance-noexcept-move-constructor)
index &index::operator=(index &&idx0) {
    impl              = idx0.impl;
    idx0.impl.idx.arr = nullptr;
    return *this;
}

static bool operator==(const fly_seq &lhs, const fly_seq &rhs) {
    return lhs.begin == rhs.begin && lhs.end == rhs.end && lhs.step == rhs.step;
}

bool index::isspan() const { return impl.isSeq && impl.idx.seq == fly_span; }

const fly_index_t &index::get() const { return impl; }

}  // namespace fly
