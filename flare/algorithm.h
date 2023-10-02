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

#ifndef FLARE_ALGORITHMS_H_
#define FLARE_ALGORITHMS_H_

/// \file algorithms.h
/// \brief flare counterparts for Standard C++ Library algorithms

#include <flare/algorithm/constraints_impl.h>
#include <flare/algorithm/random_access_iterator_impl.h>
#include <flare/algorithm/begin_end.h>

// distance
#include <flare/algorithm/distance.h>

// note that we categorize below the headers
// following the std classification.

// modifying ops
#include <flare/algorithm/swap.h>
#include <flare/algorithm/iter_swap.h>

// non-modifying sequence
#include <flare/algorithm/adjacent_find.h>
#include <flare/algorithm/count.h>
#include <flare/algorithm/count_if.h>
#include <flare/algorithm/all_of.h>
#include <flare/algorithm/any_of.h>
#include <flare/algorithm/none_of.h>
#include <flare/algorithm/equal.h>
#include <flare/algorithm/find.h>
#include <flare/algorithm/find_if.h>
#include <flare/algorithm/find_if_not.h>
#include <flare/algorithm/find_end.h>
#include <flare/algorithm/find_first_of.h>
#include <flare/algorithm/for_each.h>
#include <flare/algorithm/for_each_n.h>
#include <flare/algorithm/lexicographical_compare.h>
#include <flare/algorithm/mismatch.h>
#include <flare/algorithm/search.h>
#include <flare/algorithm/search_n.h>

// modifying sequence
#include <flare/algorithm/fill.h>
#include <flare/algorithm/fill_n.h>
#include <flare/algorithm/replace.h>
#include <flare/algorithm/replace_if.h>
#include <flare/algorithm/replace_copy_if.h>
#include <flare/algorithm/replace_copy.h>
#include <flare/algorithm/copy.h>
#include <flare/algorithm/copy_n.h>
#include <flare/algorithm/copy_backward.h>
#include <flare/algorithm/copy_if.h>
#include <flare/algorithm/transform.h>
#include <flare/algorithm/generate.h>
#include <flare/algorithm/generate_n.h>
#include <flare/algorithm/reverse.h>
#include <flare/algorithm/reverse_copy.h>
#include <flare/algorithm/move.h>
#include <flare/algorithm/move_backward.h>
#include <flare/algorithm/swap_ranges.h>
#include <flare/algorithm/unique.h>
#include <flare/algorithm/unique_copy.h>
#include <flare/algorithm/rotate.h>
#include <flare/algorithm/rotate_copy.h>
#include <flare/algorithm/remove.h>
#include <flare/algorithm/remove_if.h>
#include <flare/algorithm/remove_copy.h>
#include <flare/algorithm/remove_copy_if.h>
#include <flare/algorithm/shift_left.h>
#include <flare/algorithm/shift_right.h>

// sorting
#include <flare/algorithm/is_sorted_until.h>
#include <flare/algorithm/is_sorted.h>

// min/max element
#include <flare/algorithm/min_element.h>
#include <flare/algorithm/max_element.h>
#include <flare/algorithm/min_max_element.h>

// partitioning
#include <flare/algorithm/is_partitioned.h>
#include <flare/algorithm/partition_copy.h>
#include <flare/algorithm/partition_point.h>

// numeric
#include <flare/algorithm/adjacent_difference.h>
#include <flare/algorithm/reduce.h>
#include <flare/algorithm/transform_reduce.h>
#include <flare/algorithm/exclusive_scan.h>
#include <flare/algorithm/transform_exclusive_scan.h>
#include <flare/algorithm/inclusive_scan.h>
#include <flare/algorithm/transform_inclusive_scan.h>

#endif  // FLARE_ALGORITHMS_H_
