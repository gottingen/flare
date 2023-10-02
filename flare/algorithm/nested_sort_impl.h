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

#ifndef FLARE_ALGORITHM_NESTED_SORT_IMPL_H_
#define FLARE_ALGORITHM_NESTED_SORT_IMPL_H_

#include <flare/core.h>
#include <flare/algorithm/swap.h>

namespace flare {
namespace experimental {
namespace detail {

// true for TeamVectorRange, false for ThreadVectorRange
template <bool teamLevel>
struct NestedRange {};

// Specialization for team-level
template <>
struct NestedRange<true> {
  template <typename TeamMember, typename SizeType>
  FLARE_FUNCTION static auto create(const TeamMember& t, SizeType len) {
    return flare::TeamVectorRange(t, len);
  }
  template <typename TeamMember>
  FLARE_FUNCTION static void barrier(const TeamMember& t) {
    t.team_barrier();
  }
};

// Specialization for thread-level
template <>
struct NestedRange<false> {
  template <typename TeamMember, typename SizeType>
  FLARE_FUNCTION static auto create(const TeamMember& t, SizeType len) {
    return flare::ThreadVectorRange(t, len);
  }
  // Barrier is no-op, as vector lanes of a thread are implicitly synchronized
  // after parallel region
  template <typename TeamMember>
  FLARE_FUNCTION static void barrier(const TeamMember&) {}
};

// When just doing sort (not sort_by_key), use nullptr_t for ValueViewType.
// This only takes the NestedRange instance for template arg deduction.
template <class TeamMember, class KeyViewType, class ValueViewType,
          class Comparator, bool useTeamLevel>
FLARE_INLINE_FUNCTION void sort_nested_impl(
    const TeamMember& t, const KeyViewType& keyView,
    [[maybe_unused]] const ValueViewType& valueView, const Comparator& comp,
    const NestedRange<useTeamLevel>) {
  using SizeType  = typename KeyViewType::size_type;
  using KeyType   = typename KeyViewType::non_const_value_type;
  using Range     = NestedRange<useTeamLevel>;
  SizeType n      = keyView.extent(0);
  SizeType npot   = 1;
  SizeType levels = 0;
  // FIXME: ceiling power-of-two is a common thing to need - make it a utility
  while (npot < n) {
    levels++;
    npot <<= 1;
  }
  for (SizeType i = 0; i < levels; i++) {
    for (SizeType j = 0; j <= i; j++) {
      // n/2 pairs of items are compared in parallel
      flare::parallel_for(Range::create(t, npot / 2), [=](const SizeType k) {
        // How big are the brown/pink boxes?
        // (Terminology comes from Wikipedia diagram)
        // https://commons.wikimedia.org/wiki/File:BitonicSort.svg#/media/File:BitonicSort.svg
        SizeType boxSize = SizeType(2) << (i - j);
        // Which box contains this thread?
        SizeType boxID     = k >> (i - j);          // k * 2 / boxSize;
        SizeType boxStart  = boxID << (1 + i - j);  // boxID * boxSize
        SizeType boxOffset = k - (boxStart >> 1);   // k - boxID * boxSize / 2;
        SizeType elem1     = boxStart + boxOffset;
        // In first phase (j == 0, brown box): within a box, compare with the
        // opposite value in the box.
        // In later phases (j > 0, pink box): within a box, compare with fixed
        // distance (boxSize / 2) apart.
        SizeType elem2 = (j == 0) ? (boxStart + boxSize - 1 - boxOffset)
                                  : (elem1 + boxSize / 2);
        if (elem2 < n) {
          KeyType key1 = keyView(elem1);
          KeyType key2 = keyView(elem2);
          if (comp(key2, key1)) {
            keyView(elem1) = key2;
            keyView(elem2) = key1;
            if constexpr (!std::is_same_v<ValueViewType, std::nullptr_t>) {
              flare::experimental::swap(valueView(elem1), valueView(elem2));
            }
          }
        }
      });
      Range::barrier(t);
    }
  }
}

}  // namespace detail
}  // namespace experimental
}  // namespace flare
#endif  // FLARE_ALGORITHM_NESTED_SORT_IMPL_H_
