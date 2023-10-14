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

#include <flare/core.h>
#include <type_traits>

namespace {

    // Helper to make static tests more succinct
    template<typename DataType, typename Extent>
    constexpr bool datatype_matches_extent =
            std::is_same_v<typename flare::experimental::detail::ExtentsFromDataType<
                    std::size_t, DataType>::type,
                    Extent>;

    template<typename DataType, typename BaseType, typename Extents>
    constexpr bool extent_matches_datatype =
            std::is_same_v<DataType, typename flare::experimental::detail::
            DataTypeFromExtents<BaseType, Extents>::type>;

    // Conversion from DataType to extents
    // 0-rank tensor
    static_assert(datatype_matches_extent<double, flare::extents<std::size_t>>);

    // Only dynamic
    static_assert(datatype_matches_extent<
            double ***,
            flare::extents<std::size_t, flare::dynamic_extent,
                    flare::dynamic_extent, flare::dynamic_extent>>);
    // Only static
    static_assert(
            datatype_matches_extent<double[2][3][17],
                    flare::extents<std::size_t, std::size_t{2},
                            std::size_t{3}, std::size_t{17}>>);

    // Both dynamic and static
    static_assert(datatype_matches_extent<
            double **[3][2][8],
            flare::extents<std::size_t, flare::dynamic_extent,
                    flare::dynamic_extent, std::size_t{3},
                    std::size_t{2}, std::size_t{8}>>);

    // Conversion from extents to DataType
    // 0-rank extents
    static_assert(
            extent_matches_datatype<double, double, flare::extents<std::size_t>>);

    // only dynamic
    static_assert(extent_matches_datatype<
            double ****, double,
            flare::extents<std::size_t, flare::dynamic_extent,
                    flare::dynamic_extent, flare::dynamic_extent,
                    flare::dynamic_extent>>);

    // only static
    static_assert(extent_matches_datatype<double[7][5][3], double,
            flare::extents<std::size_t, 7, 5, 3>>);

    // both dynamic and static
    static_assert(
            extent_matches_datatype<double ***[20][45], double,
                    flare::extents<std::size_t, flare::dynamic_extent,
                            flare::dynamic_extent,
                            flare::dynamic_extent, 20, 45>>);
}  // namespace

