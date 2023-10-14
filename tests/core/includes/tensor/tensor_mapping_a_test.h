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
#include <cstddef>
#include <sstream>
#include <iostream>

#include <flare/core.h>

namespace Test {

template <class Space>
void test_tensor_mapping() {
  using ExecSpace = typename Space::execution_space;

  using dim_0        = flare::detail::TensorDimension<>;
  using dim_s2       = flare::detail::TensorDimension<2>;
  using dim_s2_s3    = flare::detail::TensorDimension<2, 3>;
  using dim_s2_s3_s4 = flare::detail::TensorDimension<2, 3, 4>;

  using dim_s0       = flare::detail::TensorDimension<0>;
  using dim_s0_s3    = flare::detail::TensorDimension<0, 3>;
  using dim_s0_s3_s4 = flare::detail::TensorDimension<0, 3, 4>;

  using dim_s0_s0    = flare::detail::TensorDimension<0, 0>;
  using dim_s0_s0_s4 = flare::detail::TensorDimension<0, 0, 4>;

  using dim_s0_s0_s0          = flare::detail::TensorDimension<0, 0, 0>;
  using dim_s0_s0_s0_s0       = flare::detail::TensorDimension<0, 0, 0, 0>;
  using dim_s0_s0_s0_s0_s0    = flare::detail::TensorDimension<0, 0, 0, 0, 0>;
  using dim_s0_s0_s0_s0_s0_s0 = flare::detail::TensorDimension<0, 0, 0, 0, 0, 0>;
  using dim_s0_s0_s0_s0_s0_s0_s0 =
      flare::detail::TensorDimension<0, 0, 0, 0, 0, 0, 0>;
  using dim_s0_s0_s0_s0_s0_s0_s0_s0 =
      flare::detail::TensorDimension<0, 0, 0, 0, 0, 0, 0, 0>;

// Fully static dimensions should not be larger than an int.
#ifndef _WIN32  // For some reason on Windows the first test here fails with
                // size being 7 bytes on windows???
  REQUIRE_LE(sizeof(dim_0), sizeof(int));
  REQUIRE_LE(sizeof(dim_s2), sizeof(int));
  REQUIRE_LE(sizeof(dim_s2_s3), sizeof(int));
  REQUIRE_LE(sizeof(dim_s2_s3_s4), sizeof(int));

  // Rank 1 is size_t.
  REQUIRE_EQ(sizeof(dim_s0), sizeof(size_t));
  REQUIRE_EQ(sizeof(dim_s0_s3), sizeof(size_t));
  REQUIRE_EQ(sizeof(dim_s0_s3_s4), sizeof(size_t));

  // Allow for padding.
  REQUIRE_LE(sizeof(dim_s0_s0), 2 * sizeof(size_t));
  REQUIRE_LE(sizeof(dim_s0_s0_s4), 2 * sizeof(size_t));

  REQUIRE_LE(sizeof(dim_s0_s0_s0), 4 * sizeof(size_t));
  REQUIRE_EQ(sizeof(dim_s0_s0_s0_s0), 4 * sizeof(unsigned));
  REQUIRE_LE(sizeof(dim_s0_s0_s0_s0_s0), 6 * sizeof(unsigned));
  REQUIRE_EQ(sizeof(dim_s0_s0_s0_s0_s0_s0), 6 * sizeof(unsigned));
  REQUIRE_LE(sizeof(dim_s0_s0_s0_s0_s0_s0_s0), 8 * sizeof(unsigned));
  REQUIRE_EQ(sizeof(dim_s0_s0_s0_s0_s0_s0_s0_s0), 8 * sizeof(unsigned));
#endif
  static_assert(int(dim_0::rank) == int(0), "");
  static_assert(int(dim_0::rank_dynamic) == int(0), "");
  static_assert(int(dim_0::ArgN0) == 1, "");
  static_assert(int(dim_0::ArgN1) == 1, "");
  static_assert(int(dim_0::ArgN2) == 1, "");

  static_assert(int(dim_s2::rank) == int(1), "");
  static_assert(int(dim_s2::rank_dynamic) == int(0), "");
  static_assert(int(dim_s2::ArgN0) == 2, "");
  static_assert(int(dim_s2::ArgN1) == 1, "");

  static_assert(int(dim_s2_s3::rank) == int(2), "");
  static_assert(int(dim_s2_s3::rank_dynamic) == int(0), "");
  static_assert(int(dim_s2_s3::ArgN0) == 2, "");
  static_assert(int(dim_s2_s3::ArgN1) == 3, "");
  static_assert(int(dim_s2_s3::ArgN2) == 1, "");

  static_assert(int(dim_s2_s3_s4::rank) == int(3), "");
  static_assert(int(dim_s2_s3_s4::rank_dynamic) == int(0), "");
  static_assert(int(dim_s2_s3_s4::ArgN0) == 2, "");
  static_assert(int(dim_s2_s3_s4::ArgN1) == 3, "");
  static_assert(int(dim_s2_s3_s4::ArgN2) == 4, "");
  static_assert(int(dim_s2_s3_s4::ArgN3) == 1, "");

  static_assert(int(dim_s0::rank) == int(1), "");
  static_assert(int(dim_s0::rank_dynamic) == int(1), "");

  static_assert(int(dim_s0_s3::rank) == int(2), "");
  static_assert(int(dim_s0_s3::rank_dynamic) == int(1), "");
  static_assert(int(dim_s0_s3::ArgN0) == 0, "");
  static_assert(int(dim_s0_s3::ArgN1) == 3, "");

  static_assert(int(dim_s0_s3_s4::rank) == int(3), "");
  static_assert(int(dim_s0_s3_s4::rank_dynamic) == int(1), "");
  static_assert(int(dim_s0_s3_s4::ArgN0) == 0, "");
  static_assert(int(dim_s0_s3_s4::ArgN1) == 3, "");
  static_assert(int(dim_s0_s3_s4::ArgN2) == 4, "");

  static_assert(int(dim_s0_s0_s4::rank) == int(3), "");
  static_assert(int(dim_s0_s0_s4::rank_dynamic) == int(2), "");
  static_assert(int(dim_s0_s0_s4::ArgN0) == 0, "");
  static_assert(int(dim_s0_s0_s4::ArgN1) == 0, "");
  static_assert(int(dim_s0_s0_s4::ArgN2) == 4, "");

  static_assert(int(dim_s0_s0_s0::rank) == int(3), "");
  static_assert(int(dim_s0_s0_s0::rank_dynamic) == int(3), "");

  static_assert(int(dim_s0_s0_s0_s0::rank) == int(4), "");
  static_assert(int(dim_s0_s0_s0_s0::rank_dynamic) == int(4), "");

  static_assert(int(dim_s0_s0_s0_s0_s0::rank) == int(5), "");
  static_assert(int(dim_s0_s0_s0_s0_s0::rank_dynamic) == int(5), "");

  static_assert(int(dim_s0_s0_s0_s0_s0_s0::rank) == int(6), "");
  static_assert(int(dim_s0_s0_s0_s0_s0_s0::rank_dynamic) == int(6), "");

  static_assert(int(dim_s0_s0_s0_s0_s0_s0_s0::rank) == int(7), "");
  static_assert(int(dim_s0_s0_s0_s0_s0_s0_s0::rank_dynamic) == int(7), "");

  static_assert(int(dim_s0_s0_s0_s0_s0_s0_s0_s0::rank) == int(8), "");
  static_assert(int(dim_s0_s0_s0_s0_s0_s0_s0_s0::rank_dynamic) == int(8), "");

  dim_s0 d1(2, 3, 4, 5, 6, 7, 8, 9);
  dim_s0_s0 d2(2, 3, 4, 5, 6, 7, 8, 9);
  dim_s0_s0_s0 d3(2, 3, 4, 5, 6, 7, 8, 9);
  dim_s0_s0_s0_s0 d4(2, 3, 4, 5, 6, 7, 8, 9);

  REQUIRE_EQ(d1.N0, 2u);
  REQUIRE_EQ(d2.N0, 2u);
  REQUIRE_EQ(d3.N0, 2u);
  REQUIRE_EQ(d4.N0, 2u);

  REQUIRE_EQ(d1.N1, 1u);
  REQUIRE_EQ(d2.N1, 3u);
  REQUIRE_EQ(d3.N1, 3u);
  REQUIRE_EQ(d4.N1, 3u);

  REQUIRE_EQ(d1.N2, 1u);
  REQUIRE_EQ(d2.N2, 1u);
  REQUIRE_EQ(d3.N2, 4u);
  REQUIRE_EQ(d4.N2, 4u);

  REQUIRE_EQ(d1.N3, 1u);
  REQUIRE_EQ(d2.N3, 1u);
  REQUIRE_EQ(d3.N3, 1u);
  REQUIRE_EQ(d4.N3, 5u);

  //----------------------------------------

  using stride_s0_s0_s0 =
      flare::detail::TensorOffset<dim_s0_s0_s0, flare::LayoutStride>;

  //----------------------------------------
  // Static dimension.
  {
    using left_s2_s3_s4 =
        flare::detail::TensorOffset<dim_s2_s3_s4, flare::LayoutLeft>;

    REQUIRE_EQ(sizeof(left_s2_s3_s4), sizeof(dim_s2_s3_s4));

    left_s2_s3_s4 off3;

    stride_s0_s0_s0 stride3(off3);

    REQUIRE_EQ(off3.stride_0(), 1u);
    REQUIRE_EQ(off3.stride_1(), 2u);
    REQUIRE_EQ(off3.stride_2(), 6u);
    REQUIRE_EQ(off3.span(), 24u);

    REQUIRE_EQ(off3.stride_0(), stride3.stride_0());
    REQUIRE_EQ(off3.stride_1(), stride3.stride_1());
    REQUIRE_EQ(off3.stride_2(), stride3.stride_2());
    REQUIRE_EQ(off3.span(), stride3.span());

    unsigned offset = 0;

    for (int k = 0; k < 4; ++k)
      for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 2; ++i, ++offset) {
          REQUIRE_EQ(off3(i, j, k), offset);
          REQUIRE_EQ(stride3(i, j, k), off3(i, j, k));
        }
  }

  //----------------------------------------
  // Small dimension is unpadded.
  {
    using left_s0_s0_s4 =
        flare::detail::TensorOffset<dim_s0_s0_s4, flare::LayoutLeft>;

    left_s0_s0_s4 dyn_off3(std::integral_constant<unsigned, sizeof(int)>(),
                           flare::LayoutLeft(2, 3, 0, 0, 0, 0, 0, 0));

    stride_s0_s0_s0 stride3(dyn_off3);

    REQUIRE_EQ(dyn_off3.m_dim.rank, 3u);
    REQUIRE_EQ(dyn_off3.m_dim.N0, 2u);
    REQUIRE_EQ(dyn_off3.m_dim.N1, 3u);
    REQUIRE_EQ(dyn_off3.m_dim.N2, 4u);
    REQUIRE_EQ(dyn_off3.m_dim.N3, 1u);
    REQUIRE_EQ(dyn_off3.size(), (size_t)2 * 3 * 4);

    const flare::LayoutLeft layout = dyn_off3.layout();

    REQUIRE_EQ(layout.dimension[0], 2u);
    REQUIRE_EQ(layout.dimension[1], 3u);
    REQUIRE_EQ(layout.dimension[2], 4u);
    REQUIRE_EQ(layout.dimension[3], FLARE_INVALID_INDEX);
    REQUIRE_EQ(layout.dimension[4], FLARE_INVALID_INDEX);
    REQUIRE_EQ(layout.dimension[5], FLARE_INVALID_INDEX);
    REQUIRE_EQ(layout.dimension[6], FLARE_INVALID_INDEX);
    REQUIRE_EQ(layout.dimension[7], FLARE_INVALID_INDEX);

    REQUIRE_EQ(stride3.m_dim.rank, 3u);
    REQUIRE_EQ(stride3.m_dim.N0, 2u);
    REQUIRE_EQ(stride3.m_dim.N1, 3u);
    REQUIRE_EQ(stride3.m_dim.N2, 4u);
    REQUIRE_EQ(stride3.m_dim.N3, 1u);
    REQUIRE_EQ(stride3.size(), (size_t)2 * 3 * 4);

    size_t offset = 0;

    for (int k = 0; k < 4; ++k)
      for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 2; ++i, ++offset) {
          REQUIRE_EQ(offset, dyn_off3(i, j, k));
          REQUIRE_EQ(stride3(i, j, k), dyn_off3(i, j, k));
        }

    REQUIRE_EQ(dyn_off3.span(), offset);
    REQUIRE_EQ(stride3.span(), dyn_off3.span());
  }

  //----------------------------------------
  // Large dimension is likely padded.
  {
    constexpr size_t N0 = 2000;
    constexpr size_t N1 = 300;

    using left_s0_s0_s4 =
        flare::detail::TensorOffset<dim_s0_s0_s4, flare::LayoutLeft>;

    left_s0_s0_s4 dyn_off3(std::integral_constant<unsigned, sizeof(int)>(),
                           flare::LayoutLeft(N0, N1, 0, 0, 0, 0, 0, 0));

    stride_s0_s0_s0 stride3(dyn_off3);

    REQUIRE_EQ(dyn_off3.m_dim.rank, 3u);
    REQUIRE_EQ(dyn_off3.m_dim.N0, N0);
    REQUIRE_EQ(dyn_off3.m_dim.N1, N1);
    REQUIRE_EQ(dyn_off3.m_dim.N2, 4u);
    REQUIRE_EQ(dyn_off3.m_dim.N3, 1u);
    REQUIRE_EQ(dyn_off3.size(), N0 * N1 * 4);

    REQUIRE_EQ(stride3.m_dim.rank, 3u);
    REQUIRE_EQ(stride3.m_dim.N0, N0);
    REQUIRE_EQ(stride3.m_dim.N1, N1);
    REQUIRE_EQ(stride3.m_dim.N2, 4u);
    REQUIRE_EQ(stride3.m_dim.N3, 1u);
    REQUIRE_EQ(stride3.size(), N0 * N1 * 4);
    REQUIRE_EQ(stride3.span(), dyn_off3.span());

    size_t offset = 0;

    for (size_t k = 0; k < 4; ++k)
      for (size_t j = 0; j < N1; ++j)
        for (size_t i = 0; i < N0; ++i) {
          REQUIRE_LE(offset, dyn_off3(i, j, k));
          REQUIRE_EQ(stride3(i, j, k), dyn_off3(i, j, k));
          offset = dyn_off3(i, j, k) + 1;
        }

    REQUIRE_LE(offset, dyn_off3.span());
  }

  //----------------------------------------
  // Static dimension.
  {
    using right_s2_s3_s4 =
        flare::detail::TensorOffset<dim_s2_s3_s4, flare::LayoutRight>;

    REQUIRE_EQ(sizeof(right_s2_s3_s4), sizeof(dim_s2_s3_s4));

    right_s2_s3_s4 off3;

    stride_s0_s0_s0 stride3(off3);

    REQUIRE_EQ(off3.stride_0(), 12u);
    REQUIRE_EQ(off3.stride_1(), 4u);
    REQUIRE_EQ(off3.stride_2(), 1u);

    REQUIRE_EQ(off3.dimension_0(), stride3.dimension_0());
    REQUIRE_EQ(off3.dimension_1(), stride3.dimension_1());
    REQUIRE_EQ(off3.dimension_2(), stride3.dimension_2());
    REQUIRE_EQ(off3.stride_0(), stride3.stride_0());
    REQUIRE_EQ(off3.stride_1(), stride3.stride_1());
    REQUIRE_EQ(off3.stride_2(), stride3.stride_2());
    REQUIRE_EQ(off3.span(), stride3.span());

    size_t offset = 0;

    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 4; ++k, ++offset) {
          REQUIRE_EQ(off3(i, j, k), offset);
          REQUIRE_EQ(off3(i, j, k), stride3(i, j, k));
        }

    REQUIRE_EQ(off3.span(), offset);
  }

  //----------------------------------------
  // Small dimension is unpadded.
  {
    using right_s0_s0_s4 =
        flare::detail::TensorOffset<dim_s0_s0_s4, flare::LayoutRight>;

    right_s0_s0_s4 dyn_off3(std::integral_constant<unsigned, sizeof(int)>(),
                            flare::LayoutRight(2, 3, 0, 0, 0, 0, 0, 0));

    stride_s0_s0_s0 stride3(dyn_off3);

    REQUIRE_EQ(dyn_off3.m_dim.rank, 3u);
    REQUIRE_EQ(dyn_off3.m_dim.N0, 2u);
    REQUIRE_EQ(dyn_off3.m_dim.N1, 3u);
    REQUIRE_EQ(dyn_off3.m_dim.N2, 4u);
    REQUIRE_EQ(dyn_off3.m_dim.N3, 1u);
    REQUIRE_EQ(dyn_off3.size(), (size_t)2 * 3 * 4);

    REQUIRE_EQ(dyn_off3.dimension_0(), stride3.dimension_0());
    REQUIRE_EQ(dyn_off3.dimension_1(), stride3.dimension_1());
    REQUIRE_EQ(dyn_off3.dimension_2(), stride3.dimension_2());
    REQUIRE_EQ(dyn_off3.stride_0(), stride3.stride_0());
    REQUIRE_EQ(dyn_off3.stride_1(), stride3.stride_1());
    REQUIRE_EQ(dyn_off3.stride_2(), stride3.stride_2());
    REQUIRE_EQ(dyn_off3.span(), stride3.span());

    size_t offset = 0;

    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 4; ++k, ++offset) {
          REQUIRE_EQ(offset, dyn_off3(i, j, k));
          REQUIRE_EQ(dyn_off3(i, j, k), stride3(i, j, k));
        }

    REQUIRE_EQ(dyn_off3.span(), offset);
  }

  //----------------------------------------
  // Large dimension is likely padded.
  {
    constexpr size_t N0 = 2000;
    constexpr size_t N1 = 300;

    using right_s0_s0_s4 =
        flare::detail::TensorOffset<dim_s0_s0_s4, flare::LayoutRight>;

    right_s0_s0_s4 dyn_off3(std::integral_constant<unsigned, sizeof(int)>(),
                            flare::LayoutRight(N0, N1, 0, 0, 0, 0, 0, 0));

    stride_s0_s0_s0 stride3(dyn_off3);

    REQUIRE_EQ(dyn_off3.m_dim.rank, 3u);
    REQUIRE_EQ(dyn_off3.m_dim.N0, N0);
    REQUIRE_EQ(dyn_off3.m_dim.N1, N1);
    REQUIRE_EQ(dyn_off3.m_dim.N2, 4u);
    REQUIRE_EQ(dyn_off3.m_dim.N3, 1u);
    REQUIRE_EQ(dyn_off3.size(), N0 * N1 * 4);

    REQUIRE_EQ(dyn_off3.dimension_0(), stride3.dimension_0());
    REQUIRE_EQ(dyn_off3.dimension_1(), stride3.dimension_1());
    REQUIRE_EQ(dyn_off3.dimension_2(), stride3.dimension_2());
    REQUIRE_EQ(dyn_off3.stride_0(), stride3.stride_0());
    REQUIRE_EQ(dyn_off3.stride_1(), stride3.stride_1());
    REQUIRE_EQ(dyn_off3.stride_2(), stride3.stride_2());
    REQUIRE_EQ(dyn_off3.span(), stride3.span());

    size_t offset = 0;

    for (size_t i = 0; i < N0; ++i)
      for (size_t j = 0; j < N1; ++j)
        for (size_t k = 0; k < 4; ++k) {
          REQUIRE_LE(offset, dyn_off3(i, j, k));
          REQUIRE_EQ(dyn_off3(i, j, k), stride3(i, j, k));
          offset = dyn_off3(i, j, k) + 1;
        }

    REQUIRE_LE(offset, dyn_off3.span());
  }

  //----------------------------------------
  // Subtensor.
  {
    // Mapping rank 4 to rank 3
    using SubTensorExtents = flare::detail::SubTensorExtents<4, 3>;

    constexpr size_t N0 = 1000;
    constexpr size_t N1 = 2000;
    constexpr size_t N2 = 3000;
    constexpr size_t N3 = 4000;

    flare::detail::TensorDimension<N0, N1, N2, N3> dim;

    SubTensorExtents tmp(dim, N0 / 2, flare::ALL,
                       std::pair<size_t, size_t>(N2 / 4, 10 + N2 / 4),
                       flare::pair<size_t, size_t>(N3 / 4, 20 + N3 / 4));

    REQUIRE_EQ(tmp.domain_offset(0), N0 / 2);
    REQUIRE_EQ(tmp.domain_offset(1), 0u);
    REQUIRE_EQ(tmp.domain_offset(2), N2 / 4);
    REQUIRE_EQ(tmp.domain_offset(3), N3 / 4);

    REQUIRE_EQ(tmp.range_index(0), 1u);
    REQUIRE_EQ(tmp.range_index(1), 2u);
    REQUIRE_EQ(tmp.range_index(2), 3u);

    REQUIRE_EQ(tmp.range_extent(0), N1);
    REQUIRE_EQ(tmp.range_extent(1), 10u);
    REQUIRE_EQ(tmp.range_extent(2), 20u);
  }

  {
    constexpr size_t N0 = 2000;
    constexpr size_t N1 = 300;

    constexpr size_t sub_N0 = 1000;
    constexpr size_t sub_N1 = 200;
    constexpr size_t sub_N2 = 4;

    using left_s0_s0_s4 =
        flare::detail::TensorOffset<dim_s0_s0_s4, flare::LayoutLeft>;

    left_s0_s0_s4 dyn_off3(std::integral_constant<unsigned, sizeof(int)>(),
                           flare::LayoutLeft(N0, N1, 0, 0, 0, 0, 0, 0));

    flare::detail::SubTensorExtents<3, 3> sub(
        dyn_off3.m_dim, flare::pair<int, int>(0, sub_N0),
        flare::pair<int, int>(0, sub_N1), flare::pair<int, int>(0, sub_N2));

    stride_s0_s0_s0 stride3(dyn_off3, sub);

    REQUIRE_EQ(stride3.dimension_0(), sub_N0);
    REQUIRE_EQ(stride3.dimension_1(), sub_N1);
    REQUIRE_EQ(stride3.dimension_2(), sub_N2);
    REQUIRE_EQ(stride3.size(), sub_N0 * sub_N1 * sub_N2);

    REQUIRE_EQ(dyn_off3.stride_0(), stride3.stride_0());
    REQUIRE_EQ(dyn_off3.stride_1(), stride3.stride_1());
    REQUIRE_EQ(dyn_off3.stride_2(), stride3.stride_2());
    REQUIRE_GE(dyn_off3.span(), stride3.span());

    for (size_t k = 0; k < sub_N2; ++k)
      for (size_t j = 0; j < sub_N1; ++j)
        for (size_t i = 0; i < sub_N0; ++i) {
          REQUIRE_EQ(stride3(i, j, k), dyn_off3(i, j, k));
        }
  }

  {
    constexpr size_t N0 = 2000;
    constexpr size_t N1 = 300;

    constexpr size_t sub_N0 = 1000;
    constexpr size_t sub_N1 = 200;
    constexpr size_t sub_N2 = 4;

    using right_s0_s0_s4 =
        flare::detail::TensorOffset<dim_s0_s0_s4, flare::LayoutRight>;

    right_s0_s0_s4 dyn_off3(std::integral_constant<unsigned, sizeof(int)>(),
                            flare::LayoutRight(N0, N1, 0, 0, 0, 0, 0, 0));

    flare::detail::SubTensorExtents<3, 3> sub(
        dyn_off3.m_dim, flare::pair<int, int>(0, sub_N0),
        flare::pair<int, int>(0, sub_N1), flare::pair<int, int>(0, sub_N2));

    stride_s0_s0_s0 stride3(dyn_off3, sub);

    REQUIRE_EQ(stride3.dimension_0(), sub_N0);
    REQUIRE_EQ(stride3.dimension_1(), sub_N1);
    REQUIRE_EQ(stride3.dimension_2(), sub_N2);
    REQUIRE_EQ(stride3.size(), sub_N0 * sub_N1 * sub_N2);

    REQUIRE_EQ(dyn_off3.stride_0(), stride3.stride_0());
    REQUIRE_EQ(dyn_off3.stride_1(), stride3.stride_1());
    REQUIRE_EQ(dyn_off3.stride_2(), stride3.stride_2());
    REQUIRE_GE(dyn_off3.span(), stride3.span());

    for (size_t i = 0; i < sub_N0; ++i)
      for (size_t j = 0; j < sub_N1; ++j)
        for (size_t k = 0; k < sub_N2; ++k) {
          REQUIRE_EQ(stride3(i, j, k), dyn_off3(i, j, k));
        }
  }

  //----------------------------------------
  // Tensor data analysis.
  {
    using namespace flare::detail;

    static_assert(rank_dynamic<>::value == 0, "");
    static_assert(rank_dynamic<1>::value == 0, "");
    static_assert(rank_dynamic<0>::value == 1, "");
    static_assert(rank_dynamic<0, 1>::value == 1, "");
    static_assert(rank_dynamic<0, 0, 1>::value == 2, "");
  }

  {
    using namespace flare::detail;

    using a_int_r1       = TensorArrayAnalysis<int[]>;
    using a_int_r5       = TensorArrayAnalysis<int* * [4][5][6]>;
    using a_const_int_r1 = TensorArrayAnalysis<const int[]>;
    using a_const_int_r5 = TensorArrayAnalysis<const int* * [4][5][6]>;

    static_assert(a_int_r1::dimension::rank == 1, "");
    static_assert(a_int_r1::dimension::rank_dynamic == 1, "");
    static_assert(a_int_r5::dimension::ArgN0 == 0, "");
    static_assert(a_int_r5::dimension::ArgN1 == 0, "");
    static_assert(a_int_r5::dimension::ArgN2 == 4, "");
    static_assert(a_int_r5::dimension::ArgN3 == 5, "");
    static_assert(a_int_r5::dimension::ArgN4 == 6, "");
    static_assert(a_int_r5::dimension::ArgN5 == 1, "");

    static_assert(
        std::is_same<typename a_int_r1::dimension, TensorDimension<0> >::value,
        "");
    static_assert(
        std::is_same<typename a_int_r1::non_const_value_type, int>::value, "");

    static_assert(a_const_int_r1::dimension::rank == 1, "");
    static_assert(a_const_int_r1::dimension::rank_dynamic == 1, "");
    static_assert(std::is_same<typename a_const_int_r1::dimension,
                               TensorDimension<0> >::value,
                  "");
    static_assert(
        std::is_same<typename a_const_int_r1::non_const_value_type, int>::value,
        "");

    static_assert(a_const_int_r5::dimension::rank == 5, "");
    static_assert(a_const_int_r5::dimension::rank_dynamic == 2, "");

    static_assert(a_const_int_r5::dimension::ArgN0 == 0, "");
    static_assert(a_const_int_r5::dimension::ArgN1 == 0, "");
    static_assert(a_const_int_r5::dimension::ArgN2 == 4, "");
    static_assert(a_const_int_r5::dimension::ArgN3 == 5, "");
    static_assert(a_const_int_r5::dimension::ArgN4 == 6, "");
    static_assert(a_const_int_r5::dimension::ArgN5 == 1, "");

    static_assert(std::is_same<typename a_const_int_r5::dimension,
                               TensorDimension<0, 0, 4, 5, 6> >::value,
                  "");
    static_assert(
        std::is_same<typename a_const_int_r5::non_const_value_type, int>::value,
        "");

    static_assert(a_int_r5::dimension::rank == 5, "");
    static_assert(a_int_r5::dimension::rank_dynamic == 2, "");
    static_assert(std::is_same<typename a_int_r5::dimension,
                               TensorDimension<0, 0, 4, 5, 6> >::value,
                  "");
    static_assert(
        std::is_same<typename a_int_r5::non_const_value_type, int>::value, "");
  }

  {
    using namespace flare::detail;

    using t_i4 = int[4];

    // Dimensions of t_i4 are appended to the multdimensional array.
    using a_int_r5 = TensorArrayAnalysis<t_i4** * [3]>;

    static_assert(a_int_r5::dimension::rank == 5, "");
    static_assert(a_int_r5::dimension::rank_dynamic == 3, "");
    static_assert(a_int_r5::dimension::ArgN0 == 0, "");
    static_assert(a_int_r5::dimension::ArgN1 == 0, "");
    static_assert(a_int_r5::dimension::ArgN2 == 0, "");
    static_assert(a_int_r5::dimension::ArgN3 == 3, "");
    static_assert(a_int_r5::dimension::ArgN4 == 4, "");
    static_assert(
        std::is_same<typename a_int_r5::non_const_value_type, int>::value, "");
  }

  {
    using namespace flare::detail;

    using a_const_int_r1 = TensorDataAnalysis<const int[], void>;

    static_assert(std::is_void<typename a_const_int_r1::specialize>::value, "");
    static_assert(std::is_same<typename a_const_int_r1::dimension,
                               flare::detail::TensorDimension<0> >::value,
                  "");

    static_assert(
        std::is_same<typename a_const_int_r1::type, const int*>::value, "");
    static_assert(
        std::is_same<typename a_const_int_r1::value_type, const int>::value,
        "");

    static_assert(std::is_same<typename a_const_int_r1::scalar_array_type,
                               const int*>::value,
                  "");
    static_assert(
        std::is_same<typename a_const_int_r1::const_type, const int*>::value,
        "");
    static_assert(std::is_same<typename a_const_int_r1::const_value_type,
                               const int>::value,
                  "");
    static_assert(std::is_same<typename a_const_int_r1::const_scalar_array_type,
                               const int*>::value,
                  "");
    static_assert(
        std::is_same<typename a_const_int_r1::non_const_type, int*>::value, "");
    static_assert(
        std::is_same<typename a_const_int_r1::non_const_value_type, int>::value,
        "");

    using a_const_int_r3 = TensorDataAnalysis<const int* * [4], void>;

    static_assert(std::is_void<typename a_const_int_r3::specialize>::value, "");

    static_assert(std::is_same<typename a_const_int_r3::dimension,
                               flare::detail::TensorDimension<0, 0, 4> >::value,
                  "");

    static_assert(
        std::is_same<typename a_const_int_r3::type, const int* * [4]>::value,
        "");
    static_assert(
        std::is_same<typename a_const_int_r3::value_type, const int>::value,
        "");
    static_assert(std::is_same<typename a_const_int_r3::scalar_array_type,
                               const int* * [4]>::value,
                  "");
    static_assert(std::is_same<typename a_const_int_r3::const_type,
                               const int* * [4]>::value,
                  "");
    static_assert(std::is_same<typename a_const_int_r3::const_value_type,
                               const int>::value,
                  "");
    static_assert(std::is_same<typename a_const_int_r3::const_scalar_array_type,
                               const int* * [4]>::value,
                  "");
    static_assert(std::is_same<typename a_const_int_r3::non_const_type,
                               int* * [4]>::value,
                  "");
    static_assert(
        std::is_same<typename a_const_int_r3::non_const_value_type, int>::value,
        "");
    static_assert(
        std::is_same<typename a_const_int_r3::non_const_scalar_array_type,
                     int* * [4]>::value,
        "");

    // std::cout << "typeid( const int**[4] ).name() = " << typeid( const
    // int**[4] ).name() << std::endl;
  }

  //----------------------------------------

  {
    constexpr int N = 10;

    using T = flare::Tensor<int*, Space>;
    using C = flare::Tensor<const int*, Space>;

    int data[N];

    T vr1(data, N);  // Tensor of non-const.
    C cr1(vr1);      // Tensor of const from tensor of non-const.
    C cr2((const int*)data, N);

    // Generate static_assert error:
    // T tmp( cr1 );

    REQUIRE_EQ(vr1.span(), size_t(N));
    REQUIRE_EQ(cr1.span(), size_t(N));
    REQUIRE_EQ(vr1.data(), &data[0]);
    REQUIRE_EQ(cr1.data(), &data[0]);

    REQUIRE((std::is_same<typename T::data_type, int*>::value));
    REQUIRE((std::is_same<typename T::const_data_type, const int*>::value));
    REQUIRE((std::is_same<typename T::non_const_data_type, int*>::value));

    REQUIRE((std::is_same<typename T::scalar_array_type, int*>::value));
    REQUIRE(
        (std::is_same<typename T::const_scalar_array_type, const int*>::value));
    REQUIRE(
        (std::is_same<typename T::non_const_scalar_array_type, int*>::value));

    REQUIRE((std::is_same<typename T::value_type, int>::value));
    REQUIRE((std::is_same<typename T::const_value_type, const int>::value));
    REQUIRE((std::is_same<typename T::non_const_value_type, int>::value));

    REQUIRE((std::is_same<typename T::memory_space,
                              typename Space::memory_space>::value));
    REQUIRE((std::is_same<typename T::reference_type, int&>::value));

    REQUIRE_EQ(T::rank, size_t(1));

    REQUIRE((std::is_same<typename C::data_type, const int*>::value));
    REQUIRE((std::is_same<typename C::const_data_type, const int*>::value));
    REQUIRE((std::is_same<typename C::non_const_data_type, int*>::value));

    REQUIRE(
        (std::is_same<typename C::scalar_array_type, const int*>::value));
    REQUIRE(
        (std::is_same<typename C::const_scalar_array_type, const int*>::value));
    REQUIRE(
        (std::is_same<typename C::non_const_scalar_array_type, int*>::value));

    REQUIRE((std::is_same<typename C::value_type, const int>::value));
    REQUIRE((std::is_same<typename C::const_value_type, const int>::value));
    REQUIRE((std::is_same<typename C::non_const_value_type, int>::value));

    REQUIRE((std::is_same<typename C::memory_space,
                              typename Space::memory_space>::value));
    REQUIRE((std::is_same<typename C::reference_type, const int&>::value));

    REQUIRE_EQ(C::rank, size_t(1));

    REQUIRE_EQ(vr1.extent(0), size_t(N));

    if (flare::SpaceAccessibility<flare::HostSpace,
                                   typename Space::memory_space>::accessible) {
      for (int i = 0; i < N; ++i) data[i] = i + 1;
      for (int i = 0; i < N; ++i) REQUIRE_EQ(vr1[i], i + 1);
      for (int i = 0; i < N; ++i) REQUIRE_EQ(cr1[i], i + 1);

      {
        T tmp(vr1);

        for (int i = 0; i < N; ++i) REQUIRE_EQ(tmp[i], i + 1);
        for (int i = 0; i < N; ++i) vr1(i) = i + 2;
        for (int i = 0; i < N; ++i) REQUIRE_EQ(tmp[i], i + 2);
      }

      for (int i = 0; i < N; ++i) REQUIRE_EQ(vr1[i], i + 2);
    }
  }

  {
    constexpr int N = 10;
    using T         = flare::Tensor<int*, Space>;
    using C         = flare::Tensor<const int*, Space>;

    T vr1("vr1", N);
    C cr1(vr1);

    REQUIRE((std::is_same<typename T::data_type, int*>::value));
    REQUIRE((std::is_same<typename T::const_data_type, const int*>::value));
    REQUIRE((std::is_same<typename T::non_const_data_type, int*>::value));

    REQUIRE((std::is_same<typename T::scalar_array_type, int*>::value));
    REQUIRE(
        (std::is_same<typename T::const_scalar_array_type, const int*>::value));
    REQUIRE(
        (std::is_same<typename T::non_const_scalar_array_type, int*>::value));

    REQUIRE((std::is_same<typename T::value_type, int>::value));
    REQUIRE((std::is_same<typename T::const_value_type, const int>::value));
    REQUIRE((std::is_same<typename T::non_const_value_type, int>::value));

    REQUIRE((std::is_same<typename T::memory_space,
                              typename Space::memory_space>::value));
    REQUIRE((std::is_same<typename T::reference_type, int&>::value));
    REQUIRE_EQ(T::rank, size_t(1));

    REQUIRE_EQ(vr1.extent(0), size_t(N));

    if (flare::SpaceAccessibility<flare::HostSpace,
                                   typename Space::memory_space>::accessible) {
      for (int i = 0; i < N; ++i) vr1(i) = i + 1;
      for (int i = 0; i < N; ++i) REQUIRE_EQ(vr1[i], i + 1);
      for (int i = 0; i < N; ++i) REQUIRE_EQ(cr1[i], i + 1);

      {
        T tmp(vr1);
        for (int i = 0; i < N; ++i) REQUIRE_EQ(tmp[i], i + 1);
        for (int i = 0; i < N; ++i) vr1(i) = i + 2;
        for (int i = 0; i < N; ++i) REQUIRE_EQ(tmp[i], i + 2);
      }

      for (int i = 0; i < N; ++i) REQUIRE_EQ(vr1[i], i + 2);
    }
  }

  // Testing proper handling of zero-length allocations.
  {
    constexpr int N = 0;
    using T         = flare::Tensor<int*, Space>;
    using C         = flare::Tensor<const int*, Space>;

    T vr1("vr1", N);
    C cr1(vr1);

    REQUIRE_EQ(vr1.extent(0), 0u);
    REQUIRE_EQ(cr1.extent(0), 0u);
  }

  // Testing using space instance for allocation.
  // The execution space of the memory space must be available for tensor data
  // initialization.
  if (std::is_same<ExecSpace,
                   typename ExecSpace::memory_space::execution_space>::value) {
    using namespace flare;

    using memory_space = typename ExecSpace::memory_space;
    using V            = Tensor<int*, memory_space>;

    constexpr int N = 10;

    memory_space mem_space;

    V v("v", N);
    V va(tensor_alloc(), N);
    V vb(tensor_alloc("vb"), N);
    V vc(tensor_alloc("vc", AllowPadding), N);
    V vd(tensor_alloc("vd", WithoutInitializing), N);
    V ve(tensor_alloc("ve", WithoutInitializing, AllowPadding), N);
    V vf(tensor_alloc("vf", mem_space, WithoutInitializing, AllowPadding), N);
    V vg(tensor_alloc(mem_space, "vg", WithoutInitializing, AllowPadding), N);
    V vh(tensor_alloc(WithoutInitializing, AllowPadding), N);
    V vi(tensor_alloc(WithoutInitializing), N);
    V vj(tensor_alloc(std::string("vj"), AllowPadding), N);
    V vk(tensor_alloc(mem_space, std::string("vk"), AllowPadding), N);
  }

  {
    using traits_t =
        flare::TensorTraits<int***, flare::LayoutStride, ExecSpace>;
    using dims_t   = flare::detail::TensorDimension<0, 0, 0>;
    using offset_t = flare::detail::TensorOffset<dims_t, flare::LayoutStride>;

    flare::LayoutStride stride;

    stride.dimension[0] = 3;
    stride.dimension[1] = 4;
    stride.dimension[2] = 5;
    stride.stride[0]    = 4;
    stride.stride[1]    = 1;
    stride.stride[2]    = 12;

    const offset_t offset(std::integral_constant<unsigned, 0>(), stride);

    REQUIRE_EQ(offset.dimension_0(), 3u);
    REQUIRE_EQ(offset.dimension_1(), 4u);
    REQUIRE_EQ(offset.dimension_2(), 5u);

    REQUIRE_EQ(offset.stride_0(), 4u);
    REQUIRE_EQ(offset.stride_1(), 1u);
    REQUIRE_EQ(offset.stride_2(), 12u);

    REQUIRE_EQ(offset.span(), 60u);
    REQUIRE(offset.span_is_contiguous());

    flare::detail::TensorMapping<traits_t, void> v(
        flare::detail::TensorCtorProp<int*>(nullptr), stride);
  }

  {
    using V           = flare::Tensor<int**, Space>;
    using M           = typename V::HostMirror;
    using layout_type = typename flare::Tensor<int**, Space>::array_layout;

    constexpr size_t N0 = 10;
    constexpr size_t N1 = 11;

    V a("a", N0, N1);
    M b = flare::create_mirror(a);
    M c = flare::create_mirror_tensor(a);
    M d;

    for (size_t i0 = 0; i0 < N0; ++i0)
      for (size_t i1 = 0; i1 < N1; ++i1) {
        b(i0, i1) = 1 + i0 + i1 * N0;
      }

    flare::deep_copy(a, b);
    flare::deep_copy(c, a);

    for (size_t i0 = 0; i0 < N0; ++i0)
      for (size_t i1 = 0; i1 < N1; ++i1) {
        REQUIRE_EQ(b(i0, i1), c(i0, i1));
      }

    flare::resize(b, 5, 6);

    for (int i0 = 0; i0 < 5; ++i0)
      for (int i1 = 0; i1 < 6; ++i1) {
        int val = 1 + i0 + i1 * N0;
        REQUIRE_EQ(b(i0, i1), c(i0, i1));
        REQUIRE_EQ(b(i0, i1), val);
      }

    flare::realloc(c, 5, 6);
    flare::realloc(d, 5, 6);

    REQUIRE_EQ(b.extent(0), 5u);
    REQUIRE_EQ(b.extent(1), 6u);
    REQUIRE_EQ(c.extent(0), 5u);
    REQUIRE_EQ(c.extent(1), 6u);
    REQUIRE_EQ(d.extent(0), 5u);
    REQUIRE_EQ(d.extent(1), 6u);

    layout_type layout(7, 8);
    flare::resize(b, layout);
    for (int i0 = 0; i0 < 7; ++i0)
      for (int i1 = 6; i1 < 8; ++i1) {
        b(i0, i1) = 1 + i0 + i1 * N0;
      }

    for (int i0 = 5; i0 < 7; ++i0)
      for (int i1 = 0; i1 < 8; ++i1) {
        b(i0, i1) = 1 + i0 + i1 * N0;
      }

    for (int i0 = 0; i0 < 7; ++i0)
      for (int i1 = 0; i1 < 8; ++i1) {
        int val = 1 + i0 + i1 * N0;
        REQUIRE_EQ(b(i0, i1), val);
      }

    flare::realloc(c, layout);
    flare::realloc(d, layout);

    REQUIRE_EQ(b.extent(0), 7u);
    REQUIRE_EQ(b.extent(1), 8u);
    REQUIRE_EQ(c.extent(0), 7u);
    REQUIRE_EQ(c.extent(1), 8u);
    REQUIRE_EQ(d.extent(0), 7u);
    REQUIRE_EQ(d.extent(1), 8u);
  }

  {
    using V = flare::Tensor<int**, flare::LayoutStride, Space>;
    using M = typename V::HostMirror;
    using layout_type =
        typename flare::Tensor<int**, flare::LayoutStride, Space>::array_layout;

    constexpr size_t N0 = 10;
    constexpr size_t N1 = 11;

    const int dimensions[] = {N0, N1};
    const int order[]      = {1, 0};

    V a("a", flare::LayoutStride::order_dimensions(2, order, dimensions));
    M b = flare::create_mirror(a);
    M c = flare::create_mirror_tensor(a);
    M d;

    for (size_t i0 = 0; i0 < N0; ++i0)
      for (size_t i1 = 0; i1 < N1; ++i1) {
        b(i0, i1) = 1 + i0 + i1 * N0;
      }

    flare::deep_copy(a, b);
    flare::deep_copy(c, a);

    for (size_t i0 = 0; i0 < N0; ++i0)
      for (size_t i1 = 0; i1 < N1; ++i1) {
        REQUIRE_EQ(b(i0, i1), c(i0, i1));
      }

    const int dimensions2[] = {7, 8};
    const int order2[]      = {1, 0};
    layout_type layout = layout_type::order_dimensions(2, order2, dimensions2);
    flare::resize(b, layout);

    for (int i0 = 0; i0 < 7; ++i0)
      for (int i1 = 0; i1 < 8; ++i1) {
        int val = 1 + i0 + i1 * N0;
        REQUIRE_EQ(b(i0, i1), c(i0, i1));
        REQUIRE_EQ(b(i0, i1), val);
      }

    flare::realloc(c, layout);
    flare::realloc(d, layout);

    REQUIRE_EQ(b.extent(0), 7u);
    REQUIRE_EQ(b.extent(1), 8u);
    REQUIRE_EQ(c.extent(0), 7u);
    REQUIRE_EQ(c.extent(1), 8u);
    REQUIRE_EQ(d.extent(0), 7u);
    REQUIRE_EQ(d.extent(1), 8u);
  }

  {
    using V = flare::Tensor<int*, Space>;
    using U = flare::Tensor<int*, Space, flare::MemoryUnmanaged>;

    V a("a", 10);

    REQUIRE_EQ(a.use_count(), 1);

    V b = a;

    REQUIRE_EQ(a.use_count(), 2);
    REQUIRE_EQ(b.use_count(), 2);

    {
      U c = b;  // 'c' is compile-time unmanaged.

      REQUIRE_EQ(a.use_count(), 2);
      REQUIRE_EQ(b.use_count(), 2);
      REQUIRE_EQ(c.use_count(), 2);

      V d = c;  // 'd' is run-time unmanaged.

      REQUIRE_EQ(a.use_count(), 2);
      REQUIRE_EQ(b.use_count(), 2);
      REQUIRE_EQ(c.use_count(), 2);
      REQUIRE_EQ(d.use_count(), 2);
    }

    REQUIRE_EQ(a.use_count(), 2);
    REQUIRE_EQ(b.use_count(), 2);

    b = V();

    REQUIRE_EQ(a.use_count(), 1);
    REQUIRE_EQ(b.use_count(), 0);

#if !defined(FLARE_ON_CUDA_DEVICE)
    // Cannot launch host lambda when CUDA lambda is enabled.

    using host_exec_space =
        typename flare::detail::HostMirror<Space>::Space::execution_space;

    int errors = 0;
    flare::parallel_reduce(
        flare::RangePolicy<host_exec_space>(0, 10),
        FLARE_LAMBDA(int, int& e) {
          // an unmanaged copy.  When the parallel dispatch accepts a move for
          // the lambda, this count should become 1.

          if (a.use_count() != 2) ++e;
          V x = a;
          if (a.use_count() != 2) ++e;
          if (x.use_count() != 2) ++e;
        },
        errors);
    REQUIRE_EQ(errors, 0);
#endif
  }
}

TEST_CASE("TEST_CATEGORY, view_mapping") { test_tensor_mapping<TEST_EXECSPACE>(); }
/*--------------------------------------------------------------------------*/

template <class TensorType>
struct TestTensorMapOperator {
  static_assert(TensorType::reference_type_is_lvalue_reference,
                "Test only valid for lvalue reference type");

  TensorType v;

  FLARE_INLINE_FUNCTION
  void test_left(size_t i0, int64_t& error_count) const {
    typename TensorType::value_type* const base_ptr =
        &v.access(0, 0, 0, 0, 0, 0, 0, 0);
    const size_t n1 = v.extent(1);
    const size_t n2 = v.extent(2);
    const size_t n3 = v.extent(3);
    const size_t n4 = v.extent(4);
    const size_t n5 = v.extent(5);
    const size_t n6 = v.extent(6);
    const size_t n7 = v.extent(7);

    int64_t offset = 0;

    for (size_t i7 = 0; i7 < n7; ++i7)
      for (size_t i6 = 0; i6 < n6; ++i6)
        for (size_t i5 = 0; i5 < n5; ++i5)
          for (size_t i4 = 0; i4 < n4; ++i4)
            for (size_t i3 = 0; i3 < n3; ++i3)
              for (size_t i2 = 0; i2 < n2; ++i2)
                for (size_t i1 = 0; i1 < n1; ++i1) {
                  const int64_t d =
                      &v.access(i0, i1, i2, i3, i4, i5, i6, i7) - base_ptr;
                  if (d < offset) ++error_count;
                  offset = d;
                }

    if (v.span() <= size_t(offset)) ++error_count;
  }

  FLARE_INLINE_FUNCTION
  void test_right(size_t i0, int64_t& error_count) const {
    typename TensorType::value_type* const base_ptr =
        &v.access(0, 0, 0, 0, 0, 0, 0, 0);
    const size_t n1 = v.extent(1);
    const size_t n2 = v.extent(2);
    const size_t n3 = v.extent(3);
    const size_t n4 = v.extent(4);
    const size_t n5 = v.extent(5);
    const size_t n6 = v.extent(6);
    const size_t n7 = v.extent(7);

    int64_t offset = 0;

    for (size_t i1 = 0; i1 < n1; ++i1)
      for (size_t i2 = 0; i2 < n2; ++i2)
        for (size_t i3 = 0; i3 < n3; ++i3)
          for (size_t i4 = 0; i4 < n4; ++i4)
            for (size_t i5 = 0; i5 < n5; ++i5)
              for (size_t i6 = 0; i6 < n6; ++i6)
                for (size_t i7 = 0; i7 < n7; ++i7) {
                  const int64_t d =
                      &v.access(i0, i1, i2, i3, i4, i5, i6, i7) - base_ptr;
                  if (d < offset) ++error_count;
                  offset = d;
                }

    if (v.span() <= size_t(offset)) ++error_count;
  }

  FLARE_INLINE_FUNCTION
  void operator()(size_t i, int64_t& error_count) const {
    if (std::is_same<typename TensorType::array_layout,
                     flare::LayoutLeft>::value) {
      test_left(i, error_count);
    } else if (std::is_same<typename TensorType::array_layout,
                            flare::LayoutRight>::value) {
      test_right(i, error_count);
    }
  }

  enum { N0 = 10 };
  enum { N1 = 9 };
  enum { N2 = 8 };
  enum { N3 = 7 };
  enum { N4 = 6 };
  enum { N5 = 5 };
  enum { N6 = 4 };
  enum { N7 = 3 };

  TestTensorMapOperator() {
    const size_t dyn_rank = v.rank_dynamic;
    const std::string label("Test");
    switch (dyn_rank) {
      case 0: v = TensorType(label); break;
      case 1: v = TensorType(label, N0); break;
      case 2: v = TensorType(label, N0, N1); break;
      case 3: v = TensorType(label, N0, N1, N2); break;
      case 4: v = TensorType(label, N0, N1, N2, N3); break;
      case 5: v = TensorType(label, N0, N1, N2, N3, N4); break;
      case 6: v = TensorType(label, N0, N1, N2, N3, N4, N5); break;
      case 7: v = TensorType(label, N0, N1, N2, N3, N4, N5, N6); break;
      case 8:
      default: v = TensorType(label, N0, N1, N2, N3, N4, N5, N6, N7);
    }
  }

  void run() {
    REQUIRE_EQ(
        v.extent(0),
        (size_t)(0 < TensorType::rank ? TestTensorMapOperator<TensorType>::N0 : 1));
    REQUIRE_EQ(
        v.extent(1),
        (size_t)(1 < TensorType::rank ? TestTensorMapOperator<TensorType>::N1 : 1));
    REQUIRE_EQ(
        v.extent(2),
        (size_t)(2 < TensorType::rank ? TestTensorMapOperator<TensorType>::N2 : 1));
    REQUIRE_EQ(
        v.extent(3),
        (size_t)(3 < TensorType::rank ? TestTensorMapOperator<TensorType>::N3 : 1));
    REQUIRE_EQ(
        v.extent(4),
        (size_t)(4 < TensorType::rank ? TestTensorMapOperator<TensorType>::N4 : 1));
    REQUIRE_EQ(
        v.extent(5),
        (size_t)(5 < TensorType::rank ? TestTensorMapOperator<TensorType>::N5 : 1));
    REQUIRE_EQ(
        v.extent(6),
        (size_t)(6 < TensorType::rank ? TestTensorMapOperator<TensorType>::N6 : 1));
    REQUIRE_EQ(
        v.extent(7),
        (size_t)(7 < TensorType::rank ? TestTensorMapOperator<TensorType>::N7 : 1));

    REQUIRE_LE(v.extent(0) * v.extent(1) * v.extent(2) * v.extent(3) *
                  v.extent(4) * v.extent(5) * v.extent(6) * v.extent(7),
              v.span());

    int64_t error_count;
    flare::RangePolicy<typename TensorType::execution_space> range(0,
                                                                  v.extent(0));
    flare::parallel_reduce(range, *this, error_count);
    REQUIRE_EQ(0, error_count);
  }
};

template <class Space>
void test_tensor_mapping_operator() {
  using ExecSpace = typename Space::execution_space;

  {
    TestTensorMapOperator<flare::Tensor<int, flare::LayoutLeft, ExecSpace> > f;
    f.run();
  }
  {
    TestTensorMapOperator<flare::Tensor<int*, flare::LayoutLeft, ExecSpace> > f;
    f.run();
  }
  {
    TestTensorMapOperator<flare::Tensor<int**, flare::LayoutLeft, ExecSpace> > f;
    f.run();
  }
  {
    TestTensorMapOperator<flare::Tensor<int***, flare::LayoutLeft, ExecSpace> > f;
    f.run();
  }
  {
    TestTensorMapOperator<flare::Tensor<int****, flare::LayoutLeft, ExecSpace> >
        f;
    f.run();
  }
  {
    TestTensorMapOperator<flare::Tensor<int*****, flare::LayoutLeft, ExecSpace> >
        f;
    f.run();
  }
  {
    TestTensorMapOperator<flare::Tensor<int******, flare::LayoutLeft, ExecSpace> >
        f;
    f.run();
  }
  {
    TestTensorMapOperator<
        flare::Tensor<int*******, flare::LayoutLeft, ExecSpace> >
        f;
    f.run();
  }

  {
    TestTensorMapOperator<flare::Tensor<int, flare::LayoutRight, ExecSpace> > f;
    f.run();
  }
  {
    TestTensorMapOperator<flare::Tensor<int*, flare::LayoutRight, ExecSpace> > f;
    f.run();
  }
  {
    TestTensorMapOperator<flare::Tensor<int**, flare::LayoutRight, ExecSpace> > f;
    f.run();
  }
  {
    TestTensorMapOperator<flare::Tensor<int***, flare::LayoutRight, ExecSpace> >
        f;
    f.run();
  }
  {
    TestTensorMapOperator<flare::Tensor<int****, flare::LayoutRight, ExecSpace> >
        f;
    f.run();
  }
  {
    TestTensorMapOperator<flare::Tensor<int*****, flare::LayoutRight, ExecSpace> >
        f;
    f.run();
  }
  {
    TestTensorMapOperator<
        flare::Tensor<int******, flare::LayoutRight, ExecSpace> >
        f;
    f.run();
  }
  {
    TestTensorMapOperator<
        flare::Tensor<int*******, flare::LayoutRight, ExecSpace> >
        f;
    f.run();
  }
}

TEST_CASE("TEST_CATEGORY, view_mapping_operator") {
  test_tensor_mapping_operator<TEST_EXECSPACE>();
}

TEST_CASE("TEST_CATEGORY, static_extent") {
  using T = flare::Tensor<double * [2][3]>;
  REQUIRE_EQ(T::static_extent(1), 2u);
  REQUIRE_EQ(T::static_extent(2), 3u);
}

}  // namespace Test
