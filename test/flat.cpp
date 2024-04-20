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

#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <fly/arith.h>
#include <fly/array.h>
#include <fly/data.h>
#include <fly/device.h>
#include <fly/random.h>

#include <vector>

using fly::array;
using fly::dim4;
using fly::flat;
using fly::freeHost;
using fly::randu;
using fly::seq;
using fly::span;

using std::vector;

TEST(FlatTests, Test_flat_1D) {
    const int num = 10000;
    array in      = randu(num);
    array out     = flat(in);

    ASSERT_ARRAYS_EQ(in, out);
}

TEST(FlatTests, Test_flat_2D_Half) {
    if (noHalfTests(f16)) return;
    const int num = 10;
    array in      = randu(num, num, f16);
    array out     = flat(in);

    vector<half_float::half> gold(num * num);
    in.host(&gold[0]);

    ASSERT_VEC_ARRAY_EQ(gold, dim4(num * num), out);
}

TEST(FlatTests, Test_flat_2D) {
    const int nx = 200;
    const int ny = 200;

    array in  = randu(nx, ny);
    array out = flat(in);

    vector<float> h_in_flat(in.elements());
    in.host(h_in_flat.data());
    dim4 h_in_flat_dims = dim4(nx * ny);
    ASSERT_VEC_ARRAY_EQ(h_in_flat, h_in_flat_dims, out);
}

TEST(FlatTests, Test_flat_1D_index) {
    const int num = 10000;
    const int st  = 101;
    const int en  = 5000;

    array in  = randu(num);
    array tmp = in(seq(st, en));
    array out = flat(tmp);

    float *h_in  = in.host<float>();
    float *h_out = out.host<float>();

    // TODO: Use ASSERT_ARRAYS_EQUAL
    for (int i = st; i <= en; i++) { ASSERT_EQ(h_in[i], h_out[i - st]); }

    freeHost(h_in);
    freeHost(h_out);
}

TEST(FlatTests, Test_flat_2D_index0) {
    const int nx  = 200;
    const int ny  = 200;
    const int st  = 21;
    const int en  = 180;
    const int nxo = (en - st + 1);

    array in  = randu(nx, ny);
    array tmp = in(seq(st, en), span);
    array out = flat(tmp);

    float *h_in  = in.host<float>();
    float *h_out = out.host<float>();

    // TODO: Use ASSERT_ARRAYS_EQUAL
    for (int j = 0; j < ny; j++) {
        const int in_off  = j * nx;
        const int out_off = j * nxo;
        for (int i = st; i <= en; i++) {
            ASSERT_EQ(h_in[i + in_off], h_out[i - st + out_off])
                << "at (" << i << "," << j << ")";
        }
    }

    freeHost(h_in);
    freeHost(h_out);
}

TEST(FlatTests, Test_flat_2D_index1) {
    const int nx = 200;
    const int ny = 200;
    const int st = 21;
    const int en = 180;

    array in  = randu(nx, ny);
    array tmp = in(span, seq(st, en));
    array out = flat(tmp);

    float *h_in  = in.host<float>();
    float *h_out = out.host<float>();

    // TODO: Use ASSERT_ARRAYS_EQUAL
    for (int j = st; j <= en; j++) {
        const int in_off  = j * nx;
        const int out_off = (j - st) * nx;

        for (int i = 0; i < nx; i++) {
            ASSERT_EQ(h_in[i + in_off], h_out[i + out_off])
                << "at (" << i << "," << j << ")";
        }
    }

    freeHost(h_in);
    freeHost(h_out);
}
