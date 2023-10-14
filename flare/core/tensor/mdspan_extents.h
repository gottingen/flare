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


#ifndef FLARE_CORE_TENSOR_MDSPAN_EXTENTS_H_
#define FLARE_CORE_TENSOR_MDSPAN_EXTENTS_H_

#include <flare/core/tensor/mdspan_header.h>

namespace flare::detail {

// Forward declarations from flare/core/tensor/tensor_mapping.h
// We cannot include directly since TensorMapping is used elsewhere in Tensor.
// After Tensor is fully moved to mdspan we can include it only from here.
template <class DataType>
struct TensorArrayAnalysis;

template <std::size_t... Vals>
struct TensorDimension;

template <class T, class Dim>
struct TensorDataType_;
}  // namespace flare::detail

namespace flare::experimental::detail {

// A few things to note --
// - mdspan allows for 0-rank extents similarly to Tensor, so we don't need
// special handling of this case
// - Tensor dynamic dimensions must be appear before static dimensions. This isn't
// a requirement in mdspan but won't cause an issue here
template <std::size_t N>
struct ExtentFromDimension {
  static constexpr std::size_t value = N;
};

// flare uses a dimension of '0' to denote a dynamic dimension.
template <>
struct ExtentFromDimension<std::size_t{0}> {
  static constexpr std::size_t value = dynamic_extent;
};

template <std::size_t N>
struct DimensionFromExtent {
  static constexpr std::size_t value = N;
};

template <>
struct DimensionFromExtent<dynamic_extent> {
  static constexpr std::size_t value = std::size_t{0};
};

template <class IndexType, class Dimension, class Indices>
struct ExtentsFromDimension;

template <class IndexType, class Dimension, std::size_t... Indices>
struct ExtentsFromDimension<IndexType, Dimension,
                            std::index_sequence<Indices...>> {
  using type =
      extents<IndexType,
              ExtentFromDimension<Dimension::static_extent(Indices)>::value...>;
};

template <class Extents, class Indices>
struct DimensionsFromExtent;

template <class Extents, std::size_t... Indices>
struct DimensionsFromExtent<Extents, std::index_sequence<Indices...>> {
  using type = ::flare::detail::TensorDimension<
      DimensionFromExtent<Extents::static_extent(Indices)>::value...>;
};

template <class IndexType, class DataType>
struct ExtentsFromDataType {
  using array_analysis = ::flare::detail::TensorArrayAnalysis<DataType>;
  using dimension_type = typename array_analysis::dimension;

  using type = typename ExtentsFromDimension<
      IndexType, dimension_type,
      std::make_index_sequence<dimension_type::rank>>::type;
};

template <class T, class Extents>
struct DataTypeFromExtents {
  using extents_type   = Extents;
  using dimension_type = typename DimensionsFromExtent<
      Extents, std::make_index_sequence<extents_type::rank()>>::type;

  // Will cause a compile error if it is malformed (i.e. dynamic after static)
  using type = typename ::flare::detail::TensorDataType_<T, dimension_type>::type;
};
}  // namespace flare::experimental::detail

#endif  // FLARE_CORE_TENSOR_MDSPAN_EXTENTS_H_
