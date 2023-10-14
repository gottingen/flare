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

#ifndef FLARE_ALGORITHM_BIN_OPS_H_
#define FLARE_ALGORITHM_BIN_OPS_H_

#include <flare/core/defines.h>
#include <type_traits>

namespace flare {

template <class KeyTensorType>
struct BinOp1D {
  int max_bins_ = {};
  double mul_   = {};
  double min_   = {};

  BinOp1D() = delete;

  // Construct BinOp with number of bins, minimum value and maximum value
  BinOp1D(int max_bins__, typename KeyTensorType::const_value_type min,
          typename KeyTensorType::const_value_type max)
      : max_bins_(max_bins__ + 1),
        // Cast to double to avoid possible overflow when using integer
        mul_(static_cast<double>(max_bins__) /
             (static_cast<double>(max) - static_cast<double>(min))),
        min_(static_cast<double>(min)) {
    // For integral types the number of bins may be larger than the range
    // in which case we can exactly have one unique value per bin
    // and then don't need to sort bins.
    if (std::is_integral<typename KeyTensorType::const_value_type>::value &&
        (static_cast<double>(max) - static_cast<double>(min)) <=
            static_cast<double>(max_bins__)) {
      mul_ = 1.;
    }
  }

  // Determine bin index from key value
  template <class TensorType>
  FLARE_INLINE_FUNCTION int bin(TensorType& keys, const int& i) const {
    return static_cast<int>(mul_ * (static_cast<double>(keys(i)) - min_));
  }

  // Return maximum bin index + 1
  FLARE_INLINE_FUNCTION
  int max_bins() const { return max_bins_; }

  // Compare to keys within a bin if true new_val will be put before old_val
  template <class TensorType, typename iType1, typename iType2>
  FLARE_INLINE_FUNCTION bool operator()(TensorType& keys, iType1& i1,
                                         iType2& i2) const {
    return keys(i1) < keys(i2);
  }
};

template <class KeyTensorType>
struct BinOp3D {
  int max_bins_[3] = {};
  double mul_[3]   = {};
  double min_[3]   = {};

  BinOp3D() = delete;

  BinOp3D(int max_bins__[], typename KeyTensorType::const_value_type min[],
          typename KeyTensorType::const_value_type max[]) {
    max_bins_[0] = max_bins__[0];
    max_bins_[1] = max_bins__[1];
    max_bins_[2] = max_bins__[2];
    mul_[0]      = static_cast<double>(max_bins__[0]) /
              (static_cast<double>(max[0]) - static_cast<double>(min[0]));
    mul_[1] = static_cast<double>(max_bins__[1]) /
              (static_cast<double>(max[1]) - static_cast<double>(min[1]));
    mul_[2] = static_cast<double>(max_bins__[2]) /
              (static_cast<double>(max[2]) - static_cast<double>(min[2]));
    min_[0] = static_cast<double>(min[0]);
    min_[1] = static_cast<double>(min[1]);
    min_[2] = static_cast<double>(min[2]);
  }

  template <class TensorType>
  FLARE_INLINE_FUNCTION int bin(TensorType& keys, const int& i) const {
    return int((((int(mul_[0] * (keys(i, 0) - min_[0])) * max_bins_[1]) +
                 int(mul_[1] * (keys(i, 1) - min_[1]))) *
                max_bins_[2]) +
               int(mul_[2] * (keys(i, 2) - min_[2])));
  }

  FLARE_INLINE_FUNCTION
  int max_bins() const { return max_bins_[0] * max_bins_[1] * max_bins_[2]; }

  template <class TensorType, typename iType1, typename iType2>
  FLARE_INLINE_FUNCTION bool operator()(TensorType& keys, iType1& i1,
                                         iType2& i2) const {
    if (keys(i1, 0) > keys(i2, 0))
      return true;
    else if (keys(i1, 0) == keys(i2, 0)) {
      if (keys(i1, 1) > keys(i2, 1))
        return true;
      else if (keys(i1, 1) == keys(i2, 1)) {
        if (keys(i1, 2) > keys(i2, 2)) return true;
      }
    }
    return false;
  }
};

}  // namespace flare
#endif  // FLARE_ALGORITHM_BIN_OPS_H_
