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

#include <vector>

#include <flare/core/static_crs_graph.h>
#include <flare/core.h>

/*--------------------------------------------------------------------------*/
namespace Test {
namespace TestStaticCrsGraph {

template <class Space>
void run_test_graph() {
  using DTensor = flare::StaticCrsGraph<unsigned, Space>;
  using hTensor = typename DTensor::HostMirror;

  const unsigned LENGTH = 1000;

  std::vector<std::vector<int> > graph(LENGTH);

  for (size_t i = 0; i < LENGTH; ++i) {
    graph[i].reserve(8);
    for (size_t j = 0; j < 8; ++j) {
      graph[i].push_back(i + j * 3);
    }
  }

  {
    DTensor d1;
    REQUIRE_FALSE(d1.is_allocated());

    d1 = flare::create_staticcrsgraph<DTensor>("d1", graph);

    DTensor d2(d1);
    DTensor d3(d1.entries, d1.row_map);

    REQUIRE(d1.is_allocated());
    REQUIRE(d2.is_allocated());
    REQUIRE(d3.is_allocated());
  }

  DTensor dx;
  hTensor hx;

  dx = flare::create_staticcrsgraph<DTensor>("dx", graph);
  hx = flare::create_mirror(dx);

  REQUIRE_EQ(hx.row_map.extent(0) - 1, LENGTH);

  for (size_t i = 0; i < LENGTH; ++i) {
    const size_t begin = hx.row_map[i];
    const size_t n     = hx.row_map[i + 1] - begin;
    REQUIRE_EQ(n, graph[i].size());
    for (size_t j = 0; j < n; ++j) {
      REQUIRE_EQ((int)hx.entries(j + begin), graph[i][j]);
    }
  }

  // Test row tensor access
  for (size_t i = 0; i < LENGTH; ++i) {
    auto rowTensor = hx.rowConst(i);
    REQUIRE_EQ(rowTensor.length, graph[i].size());
    for (size_t j = 0; j < rowTensor.length; ++j) {
      REQUIRE_EQ(rowTensor.colidx(j), (size_t)graph[i][j]);
      REQUIRE_EQ(rowTensor(j), (size_t)graph[i][j]);
    }
  }
}

template <class Space>
void run_test_graph2() {
  using DTensor = flare::StaticCrsGraph<unsigned[3], Space>;
  using hTensor = typename DTensor::HostMirror;

  const unsigned LENGTH = 10;

  std::vector<size_t> sizes(LENGTH);

  size_t total_length = 0;

  for (size_t i = 0; i < LENGTH; ++i) {
    total_length += (sizes[i] = 6 + i % 4);
  }

  DTensor dx = flare::create_staticcrsgraph<DTensor>("test", sizes);
  hTensor hx = flare::create_mirror(dx);
  hTensor mx = flare::create_mirror(dx);

  REQUIRE_EQ((size_t)dx.row_map.extent(0), (size_t)LENGTH + 1);
  REQUIRE_EQ((size_t)hx.row_map.extent(0), (size_t)LENGTH + 1);
  REQUIRE_EQ((size_t)mx.row_map.extent(0), (size_t)LENGTH + 1);

  REQUIRE_EQ((size_t)dx.entries.extent(0), (size_t)total_length);
  REQUIRE_EQ((size_t)hx.entries.extent(0), (size_t)total_length);
  REQUIRE_EQ((size_t)mx.entries.extent(0), (size_t)total_length);

  REQUIRE_EQ((size_t)dx.entries.extent(1), (size_t)3);
  REQUIRE_EQ((size_t)hx.entries.extent(1), (size_t)3);
  REQUIRE_EQ((size_t)mx.entries.extent(1), (size_t)3);

  for (size_t i = 0; i < LENGTH; ++i) {
    const size_t entry_begin = hx.row_map[i];
    const size_t entry_end   = hx.row_map[i + 1];
    for (size_t j = entry_begin; j < entry_end; ++j) {
      hx.entries(j, 0) = j + 1;
      hx.entries(j, 1) = j + 2;
      hx.entries(j, 2) = j + 3;
    }
  }

  flare::deep_copy(dx.entries, hx.entries);
  flare::deep_copy(mx.entries, dx.entries);

  REQUIRE_EQ(mx.row_map.extent(0), (size_t)LENGTH + 1);

  for (size_t i = 0; i < LENGTH; ++i) {
    const size_t entry_begin = mx.row_map[i];
    const size_t entry_end   = mx.row_map[i + 1];
    REQUIRE_EQ((entry_end - entry_begin), sizes[i]);
    for (size_t j = entry_begin; j < entry_end; ++j) {
      REQUIRE_EQ((size_t)mx.entries(j, 0), (j + 1));
      REQUIRE_EQ((size_t)mx.entries(j, 1), (j + 2));
      REQUIRE_EQ((size_t)mx.entries(j, 2), (j + 3));
    }
  }
}

template <class Space>
void run_test_graph3(size_t B, size_t N) {
  srand(10310);

  using DTensor = flare::StaticCrsGraph<int, Space>;
  using hTensor = typename DTensor::HostMirror;

  const unsigned LENGTH = 2000;

  std::vector<size_t> sizes(LENGTH);

  for (size_t i = 0; i < LENGTH; ++i) {
    sizes[i] = rand() % 1000;
  }

  sizes[1]    = N;
  sizes[1998] = N;

  int C    = 0;
  DTensor dx = flare::create_staticcrsgraph<DTensor>("test", sizes);
  dx.create_block_partitioning(B, C);
  hTensor hx = flare::create_mirror(dx);

  for (size_t i = 0; i < B; i++) {
    size_t ne = 0;
    for (auto j = hx.row_block_offsets(i); j < hx.row_block_offsets(i + 1); j++)
      ne += hx.row_map(j + 1) - hx.row_map(j) + C;

    REQUIRE_FALSE(
            ((ne > 2 * ((hx.row_map(hx.numRows()) + C * hx.numRows()) / B)) &&
        (hx.row_block_offsets(i + 1) > hx.row_block_offsets(i) + 1)));
  }
}

template <class Space>
void run_test_graph4() {
  using ordinal_type       = unsigned int;
  using layout_type        = flare::LayoutRight;
  using space_type         = Space;
  using memory_traits_type = flare::MemoryUnmanaged;
  using DTensor = flare::StaticCrsGraph<ordinal_type, layout_type, space_type,
                                       memory_traits_type>;
  using hTensor = typename DTensor::HostMirror;

  DTensor dx;

  // StaticCrsGraph with Unmanaged trait will contain row_map and entries
  // members with the Unmanaged memory trait. Use of such a StaticCrsGraph
  // requires an allocaton of memory for the unmanaged tensors to wrap.
  //
  // In this test, a graph (via raw arrays) resides on the host.
  // The pointers are wrapped by unmanaged Tensors.
  // To make use of this on the device, managed device Tensors are created
  // (allocation required), and data from the unmanaged host tensors is deep
  // copied to the device Tensors Unmanaged tensors of the appropriate type wrap the
  // device data and are assigned to their corresponding unmanaged tensor members
  // of the unmanaged StaticCrsGraph

  // Data types for raw pointers storing StaticCrsGraph info
  using ptr_row_map_type = typename DTensor::size_type;
  using ptr_entries_type = typename DTensor::data_type;

  const ordinal_type numRows = 8;
  const ordinal_type nnz     = 24;
  ptr_row_map_type ptrRaw[]  = {0, 4, 8, 10, 12, 14, 16, 20, 24};
  ptr_entries_type indRaw[]  = {0, 1, 4, 5, 0, 1, 4, 5, 2, 3, 2, 3,
                               4, 5, 4, 5, 2, 3, 6, 7, 2, 3, 6, 7};

  // Wrap pointers in unmanaged host tensors
  using local_row_map_type = typename hTensor::row_map_type;
  using local_entries_type = typename hTensor::entries_type;
  local_row_map_type unman_row_map(&(ptrRaw[0]), numRows + 1);
  local_entries_type unman_entries(&(indRaw[0]), nnz);

  hTensor hx;
  hx = hTensor(unman_entries, unman_row_map);

  // Create the device Tensors for copying the host arrays into
  // An allocation is needed on the device for the unmanaged StaticCrsGraph to
  // wrap the pointer
  using d_row_map_tensor_type =
      typename flare::Tensor<ptr_row_map_type*, layout_type, space_type>;
  using d_entries_tensor_type =
      typename flare::Tensor<ptr_entries_type*, layout_type, space_type>;

    d_row_map_tensor_type tmp_row_map("tmp_row_map", numRows + 1);
  d_entries_tensor_type tmp_entries("tmp_entries", nnz);

  flare::deep_copy(tmp_row_map, unman_row_map);
  flare::deep_copy(tmp_entries, unman_entries);

  // Wrap the pointer in unmanaged Tensor and assign to the corresponding
  // StaticCrsGraph member
  dx.row_map = typename DTensor::row_map_type(tmp_row_map.data(), numRows + 1);
  dx.entries = typename DTensor::entries_type(tmp_entries.data(), nnz);

  REQUIRE((std::is_same<typename DTensor::row_map_type::memory_traits,
                            flare::MemoryUnmanaged>::value));
  REQUIRE((std::is_same<typename DTensor::entries_type::memory_traits,
                            flare::MemoryUnmanaged>::value));
  REQUIRE((std::is_same<typename hTensor::row_map_type::memory_traits,
                            flare::MemoryUnmanaged>::value));
  REQUIRE((std::is_same<typename hTensor::entries_type::memory_traits,
                            flare::MemoryUnmanaged>::value));
}

} /* namespace TestStaticCrsGraph */

TEST_CASE("TEST_CATEGORY, staticcrsgraph") {
  TestStaticCrsGraph::run_test_graph<TEST_EXECSPACE>();
  TestStaticCrsGraph::run_test_graph2<TEST_EXECSPACE>();
  TestStaticCrsGraph::run_test_graph3<TEST_EXECSPACE>(1, 0);
  TestStaticCrsGraph::run_test_graph3<TEST_EXECSPACE>(1, 1000);
  TestStaticCrsGraph::run_test_graph3<TEST_EXECSPACE>(1, 10000);
  TestStaticCrsGraph::run_test_graph3<TEST_EXECSPACE>(1, 100000);
  TestStaticCrsGraph::run_test_graph3<TEST_EXECSPACE>(3, 0);
  TestStaticCrsGraph::run_test_graph3<TEST_EXECSPACE>(3, 1000);
  TestStaticCrsGraph::run_test_graph3<TEST_EXECSPACE>(3, 10000);
  TestStaticCrsGraph::run_test_graph3<TEST_EXECSPACE>(3, 100000);
  TestStaticCrsGraph::run_test_graph3<TEST_EXECSPACE>(75, 0);
  TestStaticCrsGraph::run_test_graph3<TEST_EXECSPACE>(75, 1000);
  TestStaticCrsGraph::run_test_graph3<TEST_EXECSPACE>(75, 10000);
  TestStaticCrsGraph::run_test_graph3<TEST_EXECSPACE>(75, 100000);
  TestStaticCrsGraph::run_test_graph4<TEST_EXECSPACE>();
}
}  // namespace Test
