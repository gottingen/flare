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

TEST(Sparse, ReadRealMTXFile) {
    fly::array out;
    std::string file(MTX_TEST_DIR "HB/bcsstm02/bcsstm02.mtx");
    ASSERT_TRUE(mtxReadSparseMatrix(out, file.c_str()));
}

TEST(Sparse, ReadComplexMTXFile) {
    fly::array out;
    std::string file(MTX_TEST_DIR "HB/young4c/young4c.mtx");
    ASSERT_TRUE(mtxReadSparseMatrix(out, file.c_str()));
}

TEST(Sparse, FailIntegerMTXRead) {
    fly::array out;
    std::string file(MTX_TEST_DIR "JGD_Kocay/Trec4/Trec4.mtx");
    ASSERT_FALSE(mtxReadSparseMatrix(out, file.c_str()));
}
