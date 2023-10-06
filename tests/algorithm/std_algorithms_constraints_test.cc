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
#include <flare/core.h>
#include <flare/algorithm.h>

namespace Test {
namespace stdalgos {

TEST_CASE("std_algorithms, is_admissible_to_std_algorithms") {
  namespace KE     = flare::experimental;
  using value_type = double;

  static constexpr size_t extent0 = 13;
  static constexpr size_t extent1 = 18;
  static constexpr size_t extent2 = 18;

  //-------------
  // 1d views
  //-------------
  using static_view_1d_t = flare::View<value_type[extent0]>;
  static_view_1d_t static_view_1d{"std-algo-test-1d-contiguous-view-static"};

  using dyn_view_1d_t = flare::View<value_type*>;
  dyn_view_1d_t dynamic_view_1d{"std-algo-test-1d-contiguous-view-dynamic",
                                extent0};

  using strided_view_1d_t = flare::View<value_type*, flare::LayoutStride>;
  flare::LayoutStride layout1d{extent0, 2};
  strided_view_1d_t strided_view_1d{"std-algo-test-1d-strided-view", layout1d};
  REQUIRE_EQ(layout1d.dimension[0], 13u);
  REQUIRE_EQ(layout1d.stride[0], 2u);
  // they are admissible
  KE::detail::static_assert_is_admissible_to_flare_std_algorithms(
      static_view_1d);
  KE::detail::static_assert_is_admissible_to_flare_std_algorithms(
      dynamic_view_1d);
  KE::detail::static_assert_is_admissible_to_flare_std_algorithms(
      strided_view_1d);

  //-------------
  // 2d views
  //-------------
  using static_view_2d_t  = flare::View<value_type[extent0][extent1]>;
  using dyn_view_2d_t     = flare::View<value_type**>;
  using strided_view_2d_t = flare::View<value_type**, flare::LayoutStride>;
  // non admissible
  REQUIRE_FALSE(KE::detail::is_admissible_to_flare_std_algorithms<
               static_view_2d_t>::value);
  REQUIRE_FALSE(
      KE::detail::is_admissible_to_flare_std_algorithms<dyn_view_2d_t>::value);
  REQUIRE_FALSE(KE::detail::is_admissible_to_flare_std_algorithms<
               strided_view_2d_t>::value);

  //-------------
  // 3d views
  //-------------
  using static_view_3d_t  = flare::View<value_type[extent0][extent1][extent2]>;
  using dyn_view_3d_t     = flare::View<value_type***>;
  using strided_view_3d_t = flare::View<value_type***, flare::LayoutStride>;
  // non admissible
  REQUIRE_FALSE(KE::detail::is_admissible_to_flare_std_algorithms<
               static_view_3d_t>::value);
  REQUIRE_FALSE(
      KE::detail::is_admissible_to_flare_std_algorithms<dyn_view_3d_t>::value);
  REQUIRE_FALSE(KE::detail::is_admissible_to_flare_std_algorithms<
               strided_view_3d_t>::value);
}

}  // namespace stdalgos
}  // namespace Test
