// Copyright 2023 The EA Authors.
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

#pragma once

#include <fly/index.h>

namespace flare {
namespace common {
/// Creates a fly_index_t object that represents a fly_span value
fly_index_t createSpanIndex();

/// Converts a fly_seq to cononical form which is composed of positive values for
/// begin and end. The step value is not modified.
///
/// fly_seq objects represent a range of values. You can create an fly_seq object
/// with the fly::end value which is represented as -1. For example you can have
/// a sequence from 1 to end-5 which will be composed of all values in an array
/// but the first and the last five values. This function converts that value to
/// positive values taking into the account of the array size.
///
/// \param[in] s   is sequence that may have negative values
/// \param[in] len is the length of a given array along a given dimension.
///
/// \returns Returns a sequence with begin and end values in the range [0,len).
///          Step value is not modified.
///
/// \NOTE: No error checks are performed.
///
/// Sample outputs of convert2Canonical for given sequence s:
/// // Assume the array's len is 10 along dimention 0
/// s{1, end-2, 1}   will return a sequence fly_seq(1, 7, 1)
/// s{1, 2, 1};      will return the same sequence
/// s{-1, 2, -1};    will return the sequence fly_seq(9,2,-1)
fly_seq convert2Canonical(const fly_seq s, const dim_t len);
}  // namespace common
}  // namespace flare
