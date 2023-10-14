// Copyright 2023 The Elastic-AI Authors.
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

#include <doctest.h>

#include <sstream>
#include <iostream>
#include <time.h>

#include <flare/core.h>

namespace Test {

TEST_CASE("TEST_CATEGORY, view_layoutstride_left_to_layoutleft_assignment") {
  using exec_space = TEST_EXECSPACE;

  srand(123456);  // arbitrary seed for random generator

  {  // Assignment of rank-1 LayoutLeft = LayoutStride
    int ndims   = 1;
    int dims[]  = {10};
    int order[] = {0};
    flare::LayoutStride layout =
        flare::LayoutStride::order_dimensions(ndims, order, dims);
    flare::Tensor<double*, flare::LayoutStride, exec_space> src("LayoutStride",
                                                                layout);

    flare::Tensor<double*, flare::LayoutStride, exec_space>::HostMirror h_src =
        flare::create_mirror_tensor(src);

    for (size_t i = 0; i < src.span(); i++)
      h_src.data()[i] = (double)rand() / RAND_MAX * (100);

    flare::deep_copy(src, h_src);

    flare::Tensor<double*, flare::LayoutLeft, exec_space> dst = src;

    flare::Tensor<double*, flare::LayoutLeft, exec_space>::HostMirror h_dst =
        flare::create_mirror_tensor(dst);

    flare::deep_copy(h_dst, dst);

    bool test = true;
    for (size_t i = 0; i < src.span(); i++) {
      if (h_src.data()[i] != h_dst.data()[i]) {
        test = false;
        break;
      }
    }
    REQUIRE_EQ(dst.span(), src.span());
    REQUIRE_EQ(test, true);
  }
  {  // Assignment of rank-2 LayoutLeft = LayoutStride
    int ndims   = 2;
    int dims[]  = {10, 9};
    int order[] = {0, 1};
    flare::LayoutStride layout =
        flare::LayoutStride::order_dimensions(ndims, order, dims);
    flare::Tensor<double**, flare::LayoutStride, exec_space> src("LayoutStride",
                                                                 layout);

    flare::Tensor<double**, flare::LayoutStride, exec_space>::HostMirror h_src =
        flare::create_mirror_tensor(src);

    for (size_t i = 0; i < src.span(); i++)
      h_src.data()[i] = (double)rand() / RAND_MAX * (100);

    flare::deep_copy(src, h_src);

    flare::Tensor<double**, flare::LayoutLeft, exec_space> dst = src;

    flare::Tensor<double**, flare::LayoutLeft, exec_space>::HostMirror h_dst =
        flare::create_mirror_tensor(dst);

    flare::deep_copy(h_dst, dst);

    bool test = true;
    for (size_t i = 0; i < src.span(); i++) {
      if (h_src.data()[i] != h_dst.data()[i]) {
        test = false;
        break;
      }
    }
    REQUIRE_EQ(dst.span(), src.span());
    REQUIRE_EQ(test, true);
  }
  {  // Assignment of rank-3 LayoutLeft = LayoutStride
    int ndims   = 3;
    int dims[]  = {10, 9, 8};
    int order[] = {0, 1, 2};
    flare::LayoutStride layout =
        flare::LayoutStride::order_dimensions(ndims, order, dims);
    flare::Tensor<double***, flare::LayoutStride, exec_space> src(
        "LayoutStride", layout);

    flare::Tensor<double***, flare::LayoutStride, exec_space>::HostMirror
        h_src = flare::create_mirror_tensor(src);

    for (size_t i = 0; i < src.span(); i++)
      h_src.data()[i] = (double)rand() / RAND_MAX * (100);

    flare::deep_copy(src, h_src);

    flare::Tensor<double***, flare::LayoutLeft, exec_space> dst = src;

    flare::Tensor<double***, flare::LayoutLeft, exec_space>::HostMirror h_dst =
        flare::create_mirror_tensor(dst);

    flare::deep_copy(h_dst, dst);

    bool test = true;
    for (size_t i = 0; i < src.span(); i++) {
      if (h_src.data()[i] != h_dst.data()[i]) {
        test = false;
        break;
      }
    }
    REQUIRE_EQ(dst.span(), src.span());
    REQUIRE_EQ(test, true);
  }
  {  // Assignment of rank-4 LayoutLeft = LayoutStride
    int ndims   = 4;
    int dims[]  = {10, 9, 8, 7};
    int order[] = {0, 1, 2, 3};
    flare::LayoutStride layout =
        flare::LayoutStride::order_dimensions(ndims, order, dims);
    flare::Tensor<double****, flare::LayoutStride, exec_space> src(
        "LayoutStride", layout);

    flare::Tensor<double****, flare::LayoutStride, exec_space>::HostMirror
        h_src = flare::create_mirror_tensor(src);

    for (size_t i = 0; i < src.span(); i++)
      h_src.data()[i] = (double)rand() / RAND_MAX * (100);

    flare::deep_copy(src, h_src);

    flare::Tensor<double****, flare::LayoutLeft, exec_space> dst = src;

    flare::Tensor<double****, flare::LayoutLeft, exec_space>::HostMirror h_dst =
        flare::create_mirror_tensor(dst);

    flare::deep_copy(h_dst, dst);

    bool test = true;
    for (size_t i = 0; i < src.span(); i++) {
      if (h_src.data()[i] != h_dst.data()[i]) {
        test = false;
        break;
      }
    }
    REQUIRE_EQ(dst.span(), src.span());
    REQUIRE_EQ(test, true);
  }
  {  // Assignment of rank-5 LayoutLeft = LayoutStride
    int ndims   = 5;
    int dims[]  = {10, 9, 8, 7, 6};
    int order[] = {0, 1, 2, 3, 4};
    flare::LayoutStride layout =
        flare::LayoutStride::order_dimensions(ndims, order, dims);
    flare::Tensor<double*****, flare::LayoutStride, exec_space> src(
        "LayoutStride", layout);

    flare::Tensor<double*****, flare::LayoutStride, exec_space>::HostMirror
        h_src = flare::create_mirror_tensor(src);

    for (size_t i = 0; i < src.span(); i++)
      h_src.data()[i] = (double)rand() / RAND_MAX * (100);

    flare::deep_copy(src, h_src);

    flare::Tensor<double*****, flare::LayoutLeft, exec_space> dst = src;

    flare::Tensor<double*****, flare::LayoutLeft, exec_space>::HostMirror
        h_dst = flare::create_mirror_tensor(dst);

    flare::deep_copy(h_dst, dst);

    bool test = true;
    for (size_t i = 0; i < src.span(); i++) {
      if (h_src.data()[i] != h_dst.data()[i]) {
        test = false;
        break;
      }
    }
    REQUIRE_EQ(dst.span(), src.span());
    REQUIRE_EQ(test, true);
  }
  {  // Assignment of rank-6 LayoutLeft = LayoutStride
    int ndims   = 6;
    int dims[]  = {10, 9, 8, 7, 6, 5};
    int order[] = {0, 1, 2, 3, 4, 5};
    flare::LayoutStride layout =
        flare::LayoutStride::order_dimensions(ndims, order, dims);
    flare::Tensor<double******, flare::LayoutStride, exec_space> src(
        "LayoutStride", layout);

    flare::Tensor<double******, flare::LayoutStride, exec_space>::HostMirror
        h_src = flare::create_mirror_tensor(src);

    for (size_t i = 0; i < src.span(); i++)
      h_src.data()[i] = (double)rand() / RAND_MAX * (100);

    flare::deep_copy(src, h_src);

    flare::Tensor<double******, flare::LayoutLeft, exec_space> dst = src;

    flare::Tensor<double******, flare::LayoutLeft, exec_space>::HostMirror
        h_dst = flare::create_mirror_tensor(dst);

    flare::deep_copy(h_dst, dst);

    bool test = true;
    for (size_t i = 0; i < src.span(); i++) {
      if (h_src.data()[i] != h_dst.data()[i]) {
        test = false;
        break;
      }
    }
    REQUIRE_EQ(dst.span(), src.span());
    REQUIRE_EQ(test, true);
  }
  {  // Assignment of rank-7 LayoutLeft = LayoutStride
    int ndims   = 7;
    int dims[]  = {10, 9, 8, 7, 6, 5, 4};
    int order[] = {0, 1, 2, 3, 4, 5, 6};
    flare::LayoutStride layout =
        flare::LayoutStride::order_dimensions(ndims, order, dims);
    flare::Tensor<double*******, flare::LayoutStride, exec_space> src(
        "LayoutStride", layout);

    flare::Tensor<double*******, flare::LayoutStride, exec_space>::HostMirror
        h_src = flare::create_mirror_tensor(src);

    for (size_t i = 0; i < src.span(); i++)
      h_src.data()[i] = (double)rand() / RAND_MAX * (100);

    flare::deep_copy(src, h_src);

    flare::Tensor<double*******, flare::LayoutLeft, exec_space> dst = src;

    flare::Tensor<double*******, flare::LayoutLeft, exec_space>::HostMirror
        h_dst = flare::create_mirror_tensor(dst);

    flare::deep_copy(h_dst, dst);

    bool test = true;
    for (size_t i = 0; i < src.span(); i++) {
      if (h_src.data()[i] != h_dst.data()[i]) {
        test = false;
        break;
      }
    }
    REQUIRE_EQ(dst.span(), src.span());
    REQUIRE_EQ(test, true);
  }
  {  // Assignment of rank-8 LayoutLeft = LayoutStride
    int ndims   = 8;
    int dims[]  = {10, 9, 8, 7, 6, 5, 4, 3};
    int order[] = {0, 1, 2, 3, 4, 5, 6, 7};
    flare::LayoutStride layout =
        flare::LayoutStride::order_dimensions(ndims, order, dims);
    flare::Tensor<double********, flare::LayoutStride, exec_space> src(
        "LayoutStride", layout);

    flare::Tensor<double********, flare::LayoutStride, exec_space>::HostMirror
        h_src = flare::create_mirror_tensor(src);

    for (size_t i = 0; i < src.span(); i++)
      h_src.data()[i] = (double)rand() / RAND_MAX * (100);

    flare::deep_copy(src, h_src);

    flare::Tensor<double********, flare::LayoutLeft, exec_space> dst = src;

    flare::Tensor<double********, flare::LayoutLeft, exec_space>::HostMirror
        h_dst = flare::create_mirror_tensor(dst);

    flare::deep_copy(h_dst, dst);

    bool test = true;
    for (size_t i = 0; i < src.span(); i++) {
      if (h_src.data()[i] != h_dst.data()[i]) {
        test = false;
        break;
      }
    }
    REQUIRE_EQ(dst.span(), src.span());
    REQUIRE_EQ(test, true);
  }
}

TEST_CASE("TEST_CATEGORY, view_layoutstride_right_to_layoutright_assignment") {
  using exec_space = TEST_EXECSPACE;

  srand(123456);  // arbitrary seed for random generator

  {  // Assignment of rank-1 LayoutRight = LayoutStride
    int ndims   = 1;
    int dims[]  = {10};
    int order[] = {0};
    flare::LayoutStride layout =
        flare::LayoutStride::order_dimensions(ndims, order, dims);
    flare::Tensor<double*, flare::LayoutStride, exec_space> src("LayoutStride",
                                                                layout);

    flare::Tensor<double*, flare::LayoutStride, exec_space>::HostMirror h_src =
        flare::create_mirror_tensor(src);

    for (size_t i = 0; i < src.span(); i++)
      h_src.data()[i] = (double)rand() / RAND_MAX * (100);

    flare::deep_copy(src, h_src);

    flare::Tensor<double*, flare::LayoutRight, exec_space> dst = src;

    flare::Tensor<double*, flare::LayoutRight, exec_space>::HostMirror h_dst =
        flare::create_mirror_tensor(dst);

    flare::deep_copy(h_dst, dst);

    bool test = true;
    for (size_t i = 0; i < src.span(); i++) {
      if (h_src.data()[i] != h_dst.data()[i]) {
        test = false;
        break;
      }
    }
    REQUIRE_EQ(dst.span(), src.span());
    REQUIRE_EQ(test, true);
  }
  {  // Assignment of rank-2 LayoutRight = LayoutStride
    int ndims   = 2;
    int dims[]  = {10, 9};
    int order[] = {1, 0};
    flare::LayoutStride layout =
        flare::LayoutStride::order_dimensions(ndims, order, dims);
    flare::Tensor<double**, flare::LayoutStride, exec_space> src("LayoutStride",
                                                                 layout);

    flare::Tensor<double**, flare::LayoutStride, exec_space>::HostMirror h_src =
        flare::create_mirror_tensor(src);

    for (size_t i = 0; i < src.span(); i++)
      h_src.data()[i] = (double)rand() / RAND_MAX * (100);

    flare::deep_copy(src, h_src);

    flare::Tensor<double**, flare::LayoutRight, exec_space> dst = src;

    flare::Tensor<double**, flare::LayoutRight, exec_space>::HostMirror h_dst =
        flare::create_mirror_tensor(dst);

    flare::deep_copy(h_dst, dst);

    bool test = true;
    for (size_t i = 0; i < src.span(); i++) {
      if (h_src.data()[i] != h_dst.data()[i]) {
        test = false;
        break;
      }
    }
    REQUIRE_EQ(dst.span(), src.span());
    REQUIRE_EQ(test, true);
  }
  {  // Assignment of rank-3 LayoutRight = LayoutStride
    int ndims   = 3;
    int dims[]  = {10, 9, 8};
    int order[] = {2, 1, 0};
    flare::LayoutStride layout =
        flare::LayoutStride::order_dimensions(ndims, order, dims);
    flare::Tensor<double***, flare::LayoutStride, exec_space> src(
        "LayoutStride", layout);

    flare::Tensor<double***, flare::LayoutStride, exec_space>::HostMirror
        h_src = flare::create_mirror_tensor(src);

    for (size_t i = 0; i < src.span(); i++)
      h_src.data()[i] = (double)rand() / RAND_MAX * (100);

    flare::deep_copy(src, h_src);

    flare::Tensor<double***, flare::LayoutRight, exec_space> dst = src;

    flare::Tensor<double***, flare::LayoutRight, exec_space>::HostMirror h_dst =
        flare::create_mirror_tensor(dst);

    flare::deep_copy(h_dst, dst);

    bool test = true;
    for (size_t i = 0; i < src.span(); i++) {
      if (h_src.data()[i] != h_dst.data()[i]) {
        test = false;
        break;
      }
    }
    REQUIRE_EQ(dst.span(), src.span());
    REQUIRE_EQ(test, true);
  }
  {  // Assignment of rank-4 LayoutRight = LayoutStride
    int ndims   = 4;
    int dims[]  = {10, 9, 8, 7};
    int order[] = {3, 2, 1, 0};
    flare::LayoutStride layout =
        flare::LayoutStride::order_dimensions(ndims, order, dims);
    flare::Tensor<double****, flare::LayoutStride, exec_space> src(
        "LayoutStride", layout);

    flare::Tensor<double****, flare::LayoutStride, exec_space>::HostMirror
        h_src = flare::create_mirror_tensor(src);

    for (size_t i = 0; i < src.span(); i++)
      h_src.data()[i] = (double)rand() / RAND_MAX * (100);

    flare::deep_copy(src, h_src);

    flare::Tensor<double****, flare::LayoutRight, exec_space> dst = src;

    flare::Tensor<double****, flare::LayoutRight, exec_space>::HostMirror
        h_dst = flare::create_mirror_tensor(dst);

    flare::deep_copy(h_dst, dst);

    bool test = true;
    for (size_t i = 0; i < src.span(); i++) {
      if (h_src.data()[i] != h_dst.data()[i]) {
        test = false;
        break;
      }
    }
    REQUIRE_EQ(dst.span(), src.span());
    REQUIRE_EQ(test, true);
  }
  {  // Assignment of rank-5 LayoutRight = LayoutStride
    int ndims   = 5;
    int dims[]  = {10, 9, 8, 7, 6};
    int order[] = {4, 3, 2, 1, 0};
    flare::LayoutStride layout =
        flare::LayoutStride::order_dimensions(ndims, order, dims);
    flare::Tensor<double*****, flare::LayoutStride, exec_space> src(
        "LayoutStride", layout);

    flare::Tensor<double*****, flare::LayoutStride, exec_space>::HostMirror
        h_src = flare::create_mirror_tensor(src);

    for (size_t i = 0; i < src.span(); i++)
      h_src.data()[i] = (double)rand() / RAND_MAX * (100);

    flare::deep_copy(src, h_src);

    flare::Tensor<double*****, flare::LayoutRight, exec_space> dst = src;

    flare::Tensor<double*****, flare::LayoutRight, exec_space>::HostMirror
        h_dst = flare::create_mirror_tensor(dst);

    flare::deep_copy(h_dst, dst);

    bool test = true;
    for (size_t i = 0; i < src.span(); i++) {
      if (h_src.data()[i] != h_dst.data()[i]) {
        test = false;
        break;
      }
    }
    REQUIRE_EQ(dst.span(), src.span());
    REQUIRE_EQ(test, true);
  }
  {  // Assignment of rank-6 LayoutRight = LayoutStride
    int ndims   = 6;
    int dims[]  = {10, 9, 8, 7, 6, 5};
    int order[] = {5, 4, 3, 2, 1, 0};
    flare::LayoutStride layout =
        flare::LayoutStride::order_dimensions(ndims, order, dims);
    flare::Tensor<double******, flare::LayoutStride, exec_space> src(
        "LayoutStride", layout);

    flare::Tensor<double******, flare::LayoutStride, exec_space>::HostMirror
        h_src = flare::create_mirror_tensor(src);

    for (size_t i = 0; i < src.span(); i++)
      h_src.data()[i] = (double)rand() / RAND_MAX * (100);

    flare::deep_copy(src, h_src);

    flare::Tensor<double******, flare::LayoutRight, exec_space> dst = src;

    flare::Tensor<double******, flare::LayoutRight, exec_space>::HostMirror
        h_dst = flare::create_mirror_tensor(dst);

    flare::deep_copy(h_dst, dst);

    bool test = true;
    for (size_t i = 0; i < src.span(); i++) {
      if (h_src.data()[i] != h_dst.data()[i]) {
        test = false;
        break;
      }
    }
    REQUIRE_EQ(dst.span(), src.span());
    REQUIRE_EQ(test, true);
  }
  {  // Assignment of rank-7 LayoutRight = LayoutStride
    int ndims   = 7;
    int dims[]  = {10, 9, 8, 7, 6, 5, 4};
    int order[] = {6, 5, 4, 3, 2, 1, 0};
    flare::LayoutStride layout =
        flare::LayoutStride::order_dimensions(ndims, order, dims);
    flare::Tensor<double*******, flare::LayoutStride, exec_space> src(
        "LayoutStride", layout);

    flare::Tensor<double*******, flare::LayoutStride, exec_space>::HostMirror
        h_src = flare::create_mirror_tensor(src);

    for (size_t i = 0; i < src.span(); i++)
      h_src.data()[i] = (double)rand() / RAND_MAX * (100);

    flare::deep_copy(src, h_src);

    flare::Tensor<double*******, flare::LayoutRight, exec_space> dst = src;

    flare::Tensor<double*******, flare::LayoutRight, exec_space>::HostMirror
        h_dst = flare::create_mirror_tensor(dst);

    flare::deep_copy(h_dst, dst);

    bool test = true;
    for (size_t i = 0; i < src.span(); i++) {
      if (h_src.data()[i] != h_dst.data()[i]) {
        test = false;
        break;
      }
    }
    REQUIRE_EQ(dst.span(), src.span());
    REQUIRE_EQ(test, true);
  }
  {  // Assignment of rank-8 LayoutRight = LayoutStride
    int ndims   = 8;
    int dims[]  = {10, 9, 8, 7, 6, 5, 4, 3};
    int order[] = {7, 6, 5, 4, 3, 2, 1, 0};
    flare::LayoutStride layout =
        flare::LayoutStride::order_dimensions(ndims, order, dims);
    flare::Tensor<double********, flare::LayoutStride, exec_space> src(
        "LayoutStride", layout);

    flare::Tensor<double********, flare::LayoutStride, exec_space>::HostMirror
        h_src = flare::create_mirror_tensor(src);

    for (size_t i = 0; i < src.span(); i++)
      h_src.data()[i] = (double)rand() / RAND_MAX * (100);

    flare::deep_copy(src, h_src);

    flare::Tensor<double********, flare::LayoutRight, exec_space> dst = src;

    flare::Tensor<double********, flare::LayoutRight, exec_space>::HostMirror
        h_dst = flare::create_mirror_tensor(dst);

    flare::deep_copy(h_dst, dst);

    bool test = true;
    for (size_t i = 0; i < src.span(); i++) {
      if (h_src.data()[i] != h_dst.data()[i]) {
        test = false;
        break;
      }
    }
    REQUIRE_EQ(dst.span(), src.span());
    REQUIRE_EQ(test, true);
  }
}

TEST_CASE("TEST_CATEGORY_DEATH, view_layoutstride_right_to_layoutleft_assignment") {
  using exec_space = TEST_EXECSPACE;

  srand(123456);  // arbitrary seed for random generator

  {  // Assignment of rank-1 LayoutLeft = LayoutStride (LayoutRight compatible)
    int ndims   = 1;
    int dims[]  = {10};
    int order[] = {0};
    flare::LayoutStride layout =
        flare::LayoutStride::order_dimensions(ndims, order, dims);
    flare::Tensor<double*, flare::LayoutStride, exec_space> src("LayoutStride",
                                                                layout);

    flare::Tensor<double*, flare::LayoutStride, exec_space>::HostMirror h_src =
        flare::create_mirror_tensor(src);

    for (size_t i = 0; i < src.span(); i++)
      h_src.data()[i] = (double)rand() / RAND_MAX * (100);

    flare::deep_copy(src, h_src);

    flare::Tensor<double*, flare::LayoutLeft, exec_space> dst;

    dst = src;

    flare::Tensor<double*, flare::LayoutLeft, exec_space>::HostMirror h_dst =
        flare::create_mirror_tensor(dst);

    flare::deep_copy(h_dst, dst);

    bool test = true;
    for (size_t i = 0; i < src.span(); i++) {
      if (h_src.data()[i] != h_dst.data()[i]) {
        test = false;
        break;
      }
    }
    REQUIRE_EQ(dst.span(), src.span());
    REQUIRE_EQ(test, true);
  }
  {  // Assignment of rank-2 LayoutLeft = LayoutStride (LayoutRight compatible)
    int ndims   = 2;
    int dims[]  = {10, 9};
    int order[] = {1, 0};
    flare::LayoutStride layout =
        flare::LayoutStride::order_dimensions(ndims, order, dims);
    flare::Tensor<double**, flare::LayoutStride, exec_space> src("LayoutStride",
                                                                 layout);

    flare::Tensor<double**, flare::LayoutLeft, exec_space> dst;

    ASSERT_DEATH({ dst = src; },
                 "Tensor assignment must have compatible layouts");
  }
  {  // Assignment of rank-3 LayoutLeft = LayoutStride (LayoutRight compatible)
    int ndims   = 3;
    int dims[]  = {10, 9, 8};
    int order[] = {2, 1, 0};
    flare::LayoutStride layout =
        flare::LayoutStride::order_dimensions(ndims, order, dims);
    flare::Tensor<double***, flare::LayoutStride, exec_space> src(
        "LayoutStride", layout);

    flare::Tensor<double***, flare::LayoutLeft, exec_space> dst;

    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    ASSERT_DEATH({ dst = src; },
                 "Tensor assignment must have compatible layouts");
  }
  {  // Assignment of rank-4 LayoutLeft = LayoutStride (LayoutRight compatible)
    int ndims   = 4;
    int dims[]  = {10, 9, 8, 7};
    int order[] = {3, 2, 1, 0};
    flare::LayoutStride layout =
        flare::LayoutStride::order_dimensions(ndims, order, dims);
    flare::Tensor<double****, flare::LayoutStride, exec_space> src(
        "LayoutStride", layout);

    flare::Tensor<double****, flare::LayoutLeft, exec_space> dst;

    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    ASSERT_DEATH({ dst = src; },
                 "Tensor assignment must have compatible layouts");
  }
  {  // Assignment of rank-5 LayoutLeft = LayoutStride (LayoutRight compatible)
    int ndims   = 5;
    int dims[]  = {10, 9, 8, 7, 6};
    int order[] = {4, 3, 2, 1, 0};
    flare::LayoutStride layout =
        flare::LayoutStride::order_dimensions(ndims, order, dims);
    flare::Tensor<double*****, flare::LayoutStride, exec_space> src(
        "LayoutStride", layout);

    flare::Tensor<double*****, flare::LayoutLeft, exec_space> dst;

    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    ASSERT_DEATH({ dst = src; },
                 "Tensor assignment must have compatible layouts");
  }
  {  // Assignment of rank-6 LayoutLeft = LayoutStride (LayoutRight compatible)
    int ndims   = 6;
    int dims[]  = {10, 9, 8, 7, 6, 5};
    int order[] = {5, 4, 3, 2, 1, 0};
    flare::LayoutStride layout =
        flare::LayoutStride::order_dimensions(ndims, order, dims);
    flare::Tensor<double******, flare::LayoutStride, exec_space> src(
        "LayoutStride", layout);

    flare::Tensor<double******, flare::LayoutLeft, exec_space> dst;

    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    ASSERT_DEATH({ dst = src; },
                 "Tensor assignment must have compatible layouts");
  }
  {  // Assignment of rank-7 LayoutLeft = LayoutStride (LayoutRight compatible)
    int ndims   = 7;
    int dims[]  = {10, 9, 8, 7, 6, 5, 4};
    int order[] = {6, 5, 4, 3, 2, 1, 0};
    flare::LayoutStride layout =
        flare::LayoutStride::order_dimensions(ndims, order, dims);
    flare::Tensor<double*******, flare::LayoutStride, exec_space> src(
        "LayoutStride", layout);

    flare::Tensor<double*******, flare::LayoutLeft, exec_space> dst;

    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    ASSERT_DEATH({ dst = src; },
                 "Tensor assignment must have compatible layouts");
  }
  {  // Assignment of rank-8 LayoutLeft = LayoutStride (LayoutRight compatible)
    int ndims   = 8;
    int dims[]  = {10, 9, 8, 7, 6, 5, 4, 3};
    int order[] = {7, 6, 5, 4, 3, 2, 1, 0};
    flare::LayoutStride layout =
        flare::LayoutStride::order_dimensions(ndims, order, dims);
    flare::Tensor<double********, flare::LayoutStride, exec_space> src(
        "LayoutStride", layout);

    flare::Tensor<double********, flare::LayoutLeft, exec_space> dst;

    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    ASSERT_DEATH({ dst = src; },
                 "Tensor assignment must have compatible layouts");
  }
}

TEST_CASE("TEST_CATEGORY_DEATH, view_layoutstride_left_to_layoutright_assignment") {
  using exec_space = TEST_EXECSPACE;

  srand(123456);  // arbitrary seed for random generator

  {  // Assignment of rank-1 LayoutRight = LayoutStride (LayoutLeft compatible)
    int ndims   = 1;
    int dims[]  = {10};
    int order[] = {0};
    flare::LayoutStride layout =
        flare::LayoutStride::order_dimensions(ndims, order, dims);
    flare::Tensor<double*, flare::LayoutStride, exec_space> src("LayoutStride",
                                                                layout);

    flare::Tensor<double*, flare::LayoutStride, exec_space>::HostMirror h_src =
        flare::create_mirror_tensor(src);

    for (size_t i = 0; i < src.span(); i++)
      h_src.data()[i] = (double)rand() / RAND_MAX * (100);

    flare::deep_copy(src, h_src);

    flare::Tensor<double*, flare::LayoutRight, exec_space> dst;

    dst = src;

    flare::Tensor<double*, flare::LayoutRight, exec_space>::HostMirror h_dst =
        flare::create_mirror_tensor(dst);

    flare::deep_copy(h_dst, dst);

    bool test = true;
    for (size_t i = 0; i < src.span(); i++) {
      if (h_src.data()[i] != h_dst.data()[i]) {
        test = false;
        break;
      }
    }
    REQUIRE_EQ(dst.span(), src.span());
    REQUIRE_EQ(test, true);
  }
  {  // Assignment of rank-2 LayoutRight = LayoutStride (LayoutLeft compatible)
    int ndims   = 2;
    int dims[]  = {10, 9};
    int order[] = {0, 1};
    flare::LayoutStride layout =
        flare::LayoutStride::order_dimensions(ndims, order, dims);
    flare::Tensor<double**, flare::LayoutStride, exec_space> src("LayoutStride",
                                                                 layout);

    flare::Tensor<double**, flare::LayoutRight, exec_space> dst;

    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    ASSERT_DEATH({ dst = src; },
                 "Tensor assignment must have compatible layouts");
  }
  {  // Assignment of rank-3 LayoutRight = LayoutStride (LayoutLeft compatible)
    int ndims   = 3;
    int dims[]  = {10, 9, 8};
    int order[] = {0, 1, 2};
    flare::LayoutStride layout =
        flare::LayoutStride::order_dimensions(ndims, order, dims);
    flare::Tensor<double***, flare::LayoutStride, exec_space> src(
        "LayoutStride", layout);

    flare::Tensor<double***, flare::LayoutRight, exec_space> dst;

    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    ASSERT_DEATH({ dst = src; },
                 "Tensor assignment must have compatible layouts");
  }
  {  // Assignment of rank-4 LayoutRight = LayoutStride (LayoutLeft compatible)
    int ndims   = 4;
    int dims[]  = {10, 9, 8, 7};
    int order[] = {0, 1, 2, 3};
    flare::LayoutStride layout =
        flare::LayoutStride::order_dimensions(ndims, order, dims);
    flare::Tensor<double****, flare::LayoutStride, exec_space> src(
        "LayoutStride", layout);

    flare::Tensor<double****, flare::LayoutRight, exec_space> dst;

    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    ASSERT_DEATH({ dst = src; },
                 "Tensor assignment must have compatible layouts");
  }
  {  // Assignment of rank-5 LayoutRight = LayoutStride (LayoutLeft compatible)
    int ndims   = 5;
    int dims[]  = {10, 9, 8, 7, 6};
    int order[] = {0, 1, 2, 3, 4};
    flare::LayoutStride layout =
        flare::LayoutStride::order_dimensions(ndims, order, dims);
    flare::Tensor<double*****, flare::LayoutStride, exec_space> src(
        "LayoutStride", layout);

    flare::Tensor<double*****, flare::LayoutRight, exec_space> dst;

    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    ASSERT_DEATH({ dst = src; },
                 "Tensor assignment must have compatible layouts");
  }
  {  // Assignment of rank-6 LayoutRight = LayoutStride (LayoutLeft compatible)
    int ndims   = 6;
    int dims[]  = {10, 9, 8, 7, 6, 5};
    int order[] = {0, 1, 2, 3, 4, 5};
    flare::LayoutStride layout =
        flare::LayoutStride::order_dimensions(ndims, order, dims);
    flare::Tensor<double******, flare::LayoutStride, exec_space> src(
        "LayoutStride", layout);

    flare::Tensor<double******, flare::LayoutRight, exec_space> dst;

    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    ASSERT_DEATH({ dst = src; },
                 "Tensor assignment must have compatible layouts");
  }
  {  // Assignment of rank-7 LayoutRight = LayoutStride (LayoutLeft compatible)
    int ndims   = 7;
    int dims[]  = {10, 9, 8, 7, 6, 5, 4};
    int order[] = {0, 1, 2, 3, 4, 5, 6};
    flare::LayoutStride layout =
        flare::LayoutStride::order_dimensions(ndims, order, dims);
    flare::Tensor<double*******, flare::LayoutStride, exec_space> src(
        "LayoutStride", layout);

    flare::Tensor<double*******, flare::LayoutRight, exec_space> dst;

    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    ASSERT_DEATH({ dst = src; },
                 "Tensor assignment must have compatible layouts");
  }
  {  // Assignment of rank-8 LayoutRight = LayoutStride (LayoutLeft compatible)
    int ndims   = 8;
    int dims[]  = {10, 9, 8, 7, 6, 5, 4, 3};
    int order[] = {0, 1, 2, 3, 4, 5, 6, 7};
    flare::LayoutStride layout =
        flare::LayoutStride::order_dimensions(ndims, order, dims);
    flare::Tensor<double********, flare::LayoutStride, exec_space> src(
        "LayoutStride", layout);

    flare::Tensor<double********, flare::LayoutRight, exec_space> dst;

    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    ASSERT_DEATH({ dst = src; },
                 "Tensor assignment must have compatible layouts");
  }
}

}  // namespace Test

#include <tensor/irregular_layout_test.h>
