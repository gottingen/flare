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

#ifndef FLARE_ALGORITHMS_ALGORITHM_SORT_CUSTOM_COMP_TEST_H_
#define FLARE_ALGORITHMS_ALGORITHM_SORT_CUSTOM_COMP_TEST_H_

#include <doctest.h>
#include <flare/core.h>
#include <flare/random.h>
#include <flare/sort.h>
#include <std_algorithms_common_test.h>

namespace {
namespace SortWithComp {

template <class LayoutTagType, class ValueType>
auto create_random_view_and_host_clone(
    LayoutTagType LayoutTag, std::size_t n,
    flare::pair<ValueType, ValueType> bounds, const std::string& label,
    std::size_t seedIn = 12371) {
  using namespace ::Test::stdalgos;

  // construct in memory space associated with default exespace
  auto dataView = create_view<ValueType>(LayoutTag, n, label);

  // dataView might not be deep copyable (e.g. strided layout) so to
  // randomize it, we make a new view that is for sure deep copyable,
  // modify it on the host, deep copy to device and then launch
  // a kernel to copy to dataView
  auto dataView_dc =
      create_deep_copyable_compatible_view_with_same_extent(dataView);
  auto dataView_dc_h = create_mirror_view(flare::HostSpace(), dataView_dc);

  // randomly fill the view
  flare::Random_XorShift64_Pool<flare::DefaultHostExecutionSpace> pool(
      seedIn);
  flare::fill_random(dataView_dc_h, pool, bounds.first, bounds.second);

  // copy to dataView_dc and then to dataView
  flare::deep_copy(dataView_dc, dataView_dc_h);
  // use CTAD
  CopyFunctor F1(dataView_dc, dataView);
  flare::parallel_for("copy", dataView.extent(0), F1);

  return std::make_pair(dataView, dataView_dc_h);
}

template <class T>
struct MyComp {
  FLARE_FUNCTION
  bool operator()(T a, T b) const {
    // we return a>b on purpose here, rather than doing a<b
    return a > b;
  }
};

// clang-format off
template <class ExecutionSpace, class Tag, class ValueType>
void run_all_scenarios(int api)
{
  using comp_t = MyComp<ValueType>;

  const std::vector<std::size_t> my_scenarios = {0, 1, 2, 9, 1003, 51513};
  for (std::size_t N : my_scenarios)
  {
    auto [dataView, dataViewBeforeOp_h] = create_random_view_and_host_clone(
        Tag{}, N, flare::pair<ValueType, ValueType>{-1045, 565},
        "dataView");

    namespace KE = flare::experimental;

    if (api == 0) {
      flare::sort(dataView, comp_t{});
      std::sort(KE::begin(dataViewBeforeOp_h), KE::end(dataViewBeforeOp_h),
                comp_t{});
    }

    else if (api == 1) {
      auto exespace = ExecutionSpace();
      flare::sort(exespace, dataView, comp_t{});
      std::sort(KE::begin(dataViewBeforeOp_h), KE::end(dataViewBeforeOp_h),
                comp_t{});
      exespace.fence();
    }

    auto dataView_h = Test::stdalgos::create_host_space_copy(dataView);
    Test::stdalgos::compare_views(dataViewBeforeOp_h, dataView_h);

    // To actually check that flare::sort used the custom
    // comparator MyComp, we should have a result in non-ascending order.
    // We can verify this by running std::is_sorted and if that returns
    // false, then it means everything ran as expected.
    // Note: std::is_sorted returns true for ranges of length one,
    // so this check makes sense only when N >= 2.
    if (N >= 2){
      REQUIRE_FALSE(std::is_sorted( KE::cbegin(dataView_h), KE::cend(dataView_h)));
    }
  }
}

TEST_CASE("TEST_CATEGORY, SortWithCustomComparator") {
  using ExeSpace = TEST_EXECSPACE;
  using namespace ::Test::stdalgos;
  for (int api = 0; api < 2; api++) {
    run_all_scenarios<ExeSpace, DynamicTag, int>(api);
    run_all_scenarios<ExeSpace, DynamicTag, double>(api);
    run_all_scenarios<ExeSpace, DynamicLayoutLeftTag, int>(api);
    run_all_scenarios<ExeSpace, DynamicLayoutLeftTag, double>(api);
    run_all_scenarios<ExeSpace, DynamicLayoutRightTag, int>(api);
    run_all_scenarios<ExeSpace, DynamicLayoutRightTag, double>(api);
    run_all_scenarios<ExeSpace, StridedThreeTag, int>(api);
    run_all_scenarios<ExeSpace, StridedThreeTag, double>(api);
  }
}

}  // namespace SortWithComp
}  // namespace anonym
#endif  // FLARE_ALGORITHMS_ALGORITHM_SORT_CUSTOM_COMP_TEST_H_
