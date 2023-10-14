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

#ifndef FLARE_CONTAINERS_UNORDERED_MAP_IMPL_H_
#define FLARE_CONTAINERS_UNORDERED_MAP_IMPL_H_

#include <flare/core.h>
#include <cstdint>

#include <cstdio>
#include <climits>
#include <iostream>
#include <iomanip>

namespace flare {
namespace detail {

uint32_t find_hash_size(uint32_t size);

template <typename Map>
struct UnorderedMapRehash {
  using map_type        = Map;
  using const_map_type  = typename map_type::const_map_type;
  using execution_space = typename map_type::execution_space;
  using size_type       = typename map_type::size_type;

  map_type m_dst;
  const_map_type m_src;

  UnorderedMapRehash(map_type const& dst, const_map_type const& src)
      : m_dst(dst), m_src(src) {}

  void apply() const {
    parallel_for("flare::detail::UnorderedMapRehash::apply", m_src.capacity(),
                 *this);
  }

  FLARE_INLINE_FUNCTION
  void operator()(size_type i) const {
    if constexpr (std::is_void_v<typename map_type::value_type>) {
      if (m_src.valid_at(i)) m_dst.insert(m_src.key_at(i));
    } else {
      if (m_src.valid_at(i)) m_dst.insert(m_src.key_at(i), m_src.value_at(i));
    }
  }
};

template <typename UMap>
struct UnorderedMapErase {
  using map_type        = UMap;
  using execution_space = typename map_type::execution_space;
  using size_type       = typename map_type::size_type;
  using key_type        = typename map_type::key_type;
  using value_type      = typename map_type::impl_value_type;

  map_type m_map;

  UnorderedMapErase(map_type const& map) : m_map(map) {}

  void apply() const {
    parallel_for("flare::detail::UnorderedMapErase::apply",
                 m_map.m_hash_lists.extent(0), *this);
  }

  FLARE_INLINE_FUNCTION
  void operator()(size_type i) const {
    const size_type invalid_index = map_type::invalid_index;

    size_type curr = m_map.m_hash_lists(i);
    size_type next = invalid_index;

    // remove erased head of the linked-list
    while (curr != invalid_index && !m_map.valid_at(curr)) {
      next                     = m_map.m_next_index[curr];
      m_map.m_next_index[curr] = invalid_index;
      m_map.m_keys[curr]       = key_type();
      if (m_map.is_set) m_map.m_values[curr] = value_type();
      curr                  = next;
      m_map.m_hash_lists(i) = next;
    }

    // if the list is non-empty and the head is valid
    if (curr != invalid_index && m_map.valid_at(curr)) {
      size_type prev = curr;
      curr           = m_map.m_next_index[prev];

      while (curr != invalid_index) {
        next = m_map.m_next_index[curr];
        if (m_map.valid_at(curr)) {
          prev = curr;
        } else {
          // remove curr from list
          m_map.m_next_index[prev] = next;
          m_map.m_next_index[curr] = invalid_index;
          m_map.m_keys[curr]       = key_type();
          if (map_type::is_set) m_map.m_values[curr] = value_type();
        }
        curr = next;
      }
    }
  }
};

template <typename UMap>
struct UnorderedMapHistogram {
  using map_type        = UMap;
  using execution_space = typename map_type::execution_space;
  using size_type       = typename map_type::size_type;

  using histogram_tensor      = Tensor<int[100], typename map_type::device_type>;
  using host_histogram_tensor = typename histogram_tensor::HostMirror;

  map_type m_map;
  histogram_tensor m_length;
  histogram_tensor m_distance;
  histogram_tensor m_block_distance;

  UnorderedMapHistogram(map_type const& map)
      : m_map(map),
        m_length("UnorderedMap Histogram"),
        m_distance("UnorderedMap Histogram"),
        m_block_distance("UnorderedMap Histogram") {}

  void calculate() {
    parallel_for("flare::detail::UnorderedMapHistogram::calculate",
                 m_map.m_hash_lists.extent(0), *this);
  }

  void clear() {
    flare::deep_copy(m_length, 0);
    flare::deep_copy(m_distance, 0);
    flare::deep_copy(m_block_distance, 0);
  }

  void print_length(std::ostream& out) {
    host_histogram_tensor host_copy =
        create_mirror_tensor_and_copy(flare::HostSpace{}, m_length);

    for (int i = 0, size = host_copy.extent(0); i < size; ++i) {
      out << host_copy[i] << " , ";
    }
    out << "\b\b\b   " << std::endl;
  }

  void print_distance(std::ostream& out) {
    host_histogram_tensor host_copy =
        create_mirror_tensor_and_copy(flare::HostSpace{}, m_distance);

    for (int i = 0, size = host_copy.extent(0); i < size; ++i) {
      out << host_copy[i] << " , ";
    }
    out << "\b\b\b   " << std::endl;
  }

  void print_block_distance(std::ostream& out) {
    host_histogram_tensor host_copy =
        create_mirror_tensor_and_copy(flare::HostSpace{}, m_block_distance);

    for (int i = 0, size = host_copy.extent(0); i < size; ++i) {
      out << host_copy[i] << " , ";
    }
    out << "\b\b\b   " << std::endl;
  }

  FLARE_INLINE_FUNCTION
  void operator()(size_type i) const {
    const size_type invalid_index = map_type::invalid_index;

    uint32_t length     = 0;
    size_type min_index = ~0u, max_index = 0;
    for (size_type curr = m_map.m_hash_lists(i); curr != invalid_index;
         curr           = m_map.m_next_index[curr]) {
      ++length;
      min_index = (curr < min_index) ? curr : min_index;
      max_index = (max_index < curr) ? curr : max_index;
    }

    size_type distance = (0u < length) ? max_index - min_index : 0u;
    size_type blocks   = (0u < length) ? max_index / 32u - min_index / 32u : 0u;

    // normalize data
    length   = length < 100u ? length : 99u;
    distance = distance < 100u ? distance : 99u;
    blocks   = blocks < 100u ? blocks : 99u;

    if (0u < length) {
      atomic_fetch_add(&m_length(length), 1);
      atomic_fetch_add(&m_distance(distance), 1);
      atomic_fetch_add(&m_block_distance(blocks), 1);
    }
  }
};

template <typename UMap>
struct UnorderedMapPrint {
  using map_type        = UMap;
  using execution_space = typename map_type::execution_space;
  using size_type       = typename map_type::size_type;

  map_type m_map;

  UnorderedMapPrint(map_type const& map) : m_map(map) {}

  void apply() {
    parallel_for("flare::detail::UnorderedMapPrint::apply",
                 m_map.m_hash_lists.extent(0), *this);
  }

  FLARE_INLINE_FUNCTION
  void operator()(size_type i) const {
    const size_type invalid_index = map_type::invalid_index;

    uint32_t list = m_map.m_hash_lists(i);
    for (size_type curr = list, ii = 0; curr != invalid_index;
         curr = m_map.m_next_index[curr], ++ii) {
      flare::printf("%d[%d]: %d->%d\n", list, ii, m_map.key_at(curr),
                     m_map.value_at(curr));
    }
  }
};

template <typename DKey, typename DValue, typename SKey, typename SValue>
struct UnorderedMapCanAssign : public std::false_type {};

template <typename Key, typename Value>
struct UnorderedMapCanAssign<Key, Value, Key, Value> : public std::true_type {};

template <typename Key, typename Value>
struct UnorderedMapCanAssign<const Key, Value, Key, Value>
    : public std::true_type {};

template <typename Key, typename Value>
struct UnorderedMapCanAssign<const Key, const Value, Key, Value>
    : public std::true_type {};

template <typename Key, typename Value>
struct UnorderedMapCanAssign<const Key, const Value, const Key, Value>
    : public std::true_type {};

}  // namespace detail
}  // namespace flare

#endif  // FLARE_CONTAINERS_UNORDERED_MAP_IMPL_H_
