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

#ifndef FLARE_CONTAINERS_STATIC_CRS_GRAPH_FACTORY_H_
#define FLARE_CONTAINERS_STATIC_CRS_GRAPH_FACTORY_H_

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
#include <flare/core.h>
#include <flare/core/static_crs_graph.h>

namespace flare {

template <class DataType, class Arg1Type, class Arg2Type, class Arg3Type,
          typename SizeType>
inline typename StaticCrsGraph<DataType, Arg1Type, Arg2Type, Arg3Type,
                               SizeType>::HostMirror
create_mirror_tensor(const StaticCrsGraph<DataType, Arg1Type, Arg2Type, Arg3Type,
                                        SizeType>& tensor,
                   std::enable_if_t<TensorTraits<DataType, Arg1Type, Arg2Type,
                                               Arg3Type>::is_hostspace>* = 0) {
  return tensor;
}

template <class DataType, class Arg1Type, class Arg2Type, class Arg3Type,
          typename SizeType>
inline typename StaticCrsGraph<DataType, Arg1Type, Arg2Type, Arg3Type,
                               SizeType>::HostMirror
create_mirror(const StaticCrsGraph<DataType, Arg1Type, Arg2Type, Arg3Type,
                                   SizeType>& tensor) {
  // Force copy:
  // using alloc = detail::ViewAssignment<detail::ViewDefault>; // unused
  using staticcrsgraph_type =
      StaticCrsGraph<DataType, Arg1Type, Arg2Type, Arg3Type, SizeType>;

  typename staticcrsgraph_type::HostMirror tmp;
  typename staticcrsgraph_type::row_map_type::HostMirror tmp_row_map =
      create_mirror(tensor.row_map);
  typename staticcrsgraph_type::row_block_type::HostMirror
      tmp_row_block_offsets = create_mirror(tensor.row_block_offsets);

  // Allocation to match:
  tmp.row_map = tmp_row_map;  // Assignment of 'const' from 'non-const'
  tmp.entries = create_mirror(tensor.entries);
  tmp.row_block_offsets =
      tmp_row_block_offsets;  // Assignment of 'const' from 'non-const'

  // Deep copy:
  deep_copy(tmp_row_map, tensor.row_map);
  deep_copy(tmp.entries, tensor.entries);
  deep_copy(tmp_row_block_offsets, tensor.row_block_offsets);

  return tmp;
}

template <class DataType, class Arg1Type, class Arg2Type, class Arg3Type,
          typename SizeType>
inline typename StaticCrsGraph<DataType, Arg1Type, Arg2Type, Arg3Type,
                               SizeType>::HostMirror
create_mirror_tensor(const StaticCrsGraph<DataType, Arg1Type, Arg2Type, Arg3Type,
                                        SizeType>& tensor,
                   std::enable_if_t<!TensorTraits<DataType, Arg1Type, Arg2Type,
                                                Arg3Type>::is_hostspace>* = 0) {
  return create_mirror(tensor);
}
}  // namespace flare

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace flare {

template <class StaticCrsGraphType, class InputSizeType>
inline typename StaticCrsGraphType::staticcrsgraph_type create_staticcrsgraph(
    const std::string& label, const std::vector<InputSizeType>& input) {
  using output_type  = StaticCrsGraphType;
  using entries_type = typename output_type::entries_type;
  using work_type    = Tensor<
      typename output_type::size_type[], typename output_type::array_layout,
      typename output_type::device_type, typename output_type::memory_traits>;

  output_type output;

  // Create the row map:

  const size_t length = input.size();

  {
    work_type row_work("tmp", length + 1);

    typename work_type::HostMirror row_work_host = create_mirror_tensor(row_work);

    size_t sum       = 0;
    row_work_host[0] = 0;
    for (size_t i = 0; i < length; ++i) {
      row_work_host[i + 1] = sum += input[i];
    }

    deep_copy(row_work, row_work_host);

    output.entries = entries_type(label, sum);
    output.row_map = row_work;
  }

  return output;
}

//----------------------------------------------------------------------------

template <class StaticCrsGraphType, class InputSizeType>
inline typename StaticCrsGraphType::staticcrsgraph_type create_staticcrsgraph(
    const std::string& label,
    const std::vector<std::vector<InputSizeType> >& input) {
  using output_type  = StaticCrsGraphType;
  using entries_type = typename output_type::entries_type;

  static_assert(entries_type::rank == 1, "Graph entries tensor must be rank one");

  using work_type = Tensor<
      typename output_type::size_type[], typename output_type::array_layout,
      typename output_type::device_type, typename output_type::memory_traits>;

  output_type output;

  // Create the row map:

  const size_t length = input.size();

  {
    work_type row_work("tmp", length + 1);

    typename work_type::HostMirror row_work_host = create_mirror_tensor(row_work);

    size_t sum       = 0;
    row_work_host[0] = 0;
    for (size_t i = 0; i < length; ++i) {
      row_work_host[i + 1] = sum += input[i].size();
    }

    deep_copy(row_work, row_work_host);

    output.entries = entries_type(label, sum);
    output.row_map = row_work;
  }

  // Fill in the entries:
  {
    typename entries_type::HostMirror host_entries =
        create_mirror_tensor(output.entries);

    size_t sum = 0;
    for (size_t i = 0; i < length; ++i) {
      for (size_t j = 0; j < input[i].size(); ++j, ++sum) {
        host_entries(sum) = input[i][j];
      }
    }

    deep_copy(output.entries, host_entries);
  }

  return output;
}

}  // namespace flare

#endif  // FLARE_CONTAINERS_STATIC_CRS_GRAPH_FACTORY_H_
