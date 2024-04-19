
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

#include <flare.h>
#include <gtest/gtest.h>
#include <vector>

using fly::array;
using fly::join;
using fly::randu;
using fly::tile;
using std::vector;

TEST(MatrixManipulation, SNIPPET_matrix_manipulation_tile) {
    //! [ex_matrix_manipulation_tile]
    float h[]       = {1, 2, 3, 4};
    array small_arr = array(2, 2, h);  // 2x2 matrix
    fly_print(small_arr);
    array large_arr =
        tile(small_arr, 2, 3);  // produces 4x6 matrix: (2*2)x(2*3)
    fly_print(large_arr);
    //! [ex_matrix_manipulation_tile]

    ASSERT_EQ(4, large_arr.dims(0));
    ASSERT_EQ(6, large_arr.dims(1));

    vector<float> h_large_arr(large_arr.elements());
    large_arr.host(&h_large_arr.front());

    unsigned fdim = large_arr.dims(0);
    unsigned sdim = large_arr.dims(1);
    for (unsigned i = 0; i < sdim; i++) {
        for (unsigned j = 0; j < fdim; j++) {
            ASSERT_FLOAT_EQ(h[(i % 2) * 2 + (j % 2)],
                            h_large_arr[i * fdim + j]);
        }
    }
}

TEST(MatrixManipulation, SNIPPET_matrix_manipulation_join) {
    //! [ex_matrix_manipulation_join]
    float hA[] = {1, 2, 3, 4, 5, 6};
    float hB[] = {10, 20, 30, 40, 50, 60, 70, 80, 90};
    array A    = array(3, 2, hA);
    array B    = array(3, 3, hB);

    fly_print(join(1, A, B));  // 3x5 matrix
    // array result = join(0, A, B); // fail: dimension mismatch
    //! [ex_matrix_manipulation_join]

    array out = join(1, A, B);
    vector<float> h_out(out.elements());
    out.host(&h_out.front());
    fly_print(out);

    ASSERT_EQ(3, out.dims(0));
    ASSERT_EQ(5, out.dims(1));

    unsigned fdim = out.dims(0);
    unsigned sdim = out.dims(1);
    for (unsigned i = 0; i < sdim; i++) {
        for (unsigned j = 0; j < fdim; j++) {
            if (i < 2) {
                ASSERT_FLOAT_EQ(hA[i * fdim + j], h_out[i * fdim + j])
                    << "At [" << i << ", " << j << "]";
            } else {
                ASSERT_FLOAT_EQ(hB[(i - 2) * fdim + j], h_out[i * fdim + j])
                    << "At [" << i << ", " << j << "]";
            }
        }
    }
}

TEST(MatrixManipulation, SNIPPET_matrix_manipulation_mesh) {
    //! [ex_matrix_manipulation_mesh]
    float hx[] = {1, 2, 3, 4};
    float hy[] = {5, 6};

    array x = array(4, hx);
    array y = array(2, hy);

    fly_print(tile(x, 1, 2));
    fly_print(tile(y.T(), 4, 1));
    //! [ex_matrix_manipulation_mesh]

    array outx = tile(x, 1, 2);
    array outy = tile(y.T(), 4, 1);

    ASSERT_EQ(4, outx.dims(0));
    ASSERT_EQ(4, outy.dims(0));
    ASSERT_EQ(2, outx.dims(1));
    ASSERT_EQ(2, outy.dims(1));

    vector<float> houtx(outx.elements());
    outx.host(&houtx.front());
    vector<float> houty(outy.elements());
    outy.host(&houty.front());

    for (unsigned i = 0; i < houtx.size(); i++)
        ASSERT_EQ(hx[i % 4], houtx[i]) << "At [" << i << "]";
    for (unsigned i = 0; i < houty.size(); i++)
        ASSERT_EQ(hy[i > 3], houty[i]) << "At [" << i << "]";
}

TEST(MatrixManipulation, SNIPPET_matrix_manipulation_moddims) {
    //! [ex_matrix_manipulation_moddims]
    int hA[] = {1, 2, 3, 4, 5, 6};
    array A  = array(3, 2, hA);

    fly_print(A);                 // 2x3 matrix
    fly_print(moddims(A, 2, 3));  // 2x3 matrix
    fly_print(moddims(A, 6, 1));  // 6x1 column vector

    // moddims(A, 2, 2); // fail: wrong number of elements
    // moddims(A, 8, 8); // fail: wrong number of elements
    //! [ex_matrix_manipulation_moddims]
}

TEST(MatrixManipulation, SNIPPET_matrix_manipulation_transpose) {
    //! [ex_matrix_manipulation_transpose]
    array x = randu(2, 2, f32);
    fly_print(x.T());  // transpose (real)

    array c = randu(2, 2, c32);
    fly_print(c.T());  // transpose (complex)
    fly_print(c.H());  // Hermitian (conjugate) transpose
    //! [ex_matrix_manipulation_transpose]
}
