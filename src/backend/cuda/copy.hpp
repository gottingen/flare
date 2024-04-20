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

#include <Array.hpp>

namespace flare {
namespace cuda {
// Copies(blocking) data from an Array<T> object to a contiguous host side
// pointer.
//
// \param dst The destination pointer on the host system.
// \param src    The source array
template<typename T>
void copyData(T *dst, const Array<T> &src);

// Create a deep copy of the \p src Array with the same size and shape. The new
// Array will not maintain the subarray metadata of the \p src array.
//
// \param   src  The source Array<T> object.
// \returns      A new Array<T> object with the same shape and data as the
//               \p src Array<T>
template<typename T>
Array<T> copyArray(const Array<T> &src);

template<typename inType, typename outType>
void copyArray(Array<outType> &out, const Array<inType> &in);

// Resize Array to target dimensions and convert type
//
// Depending on the \p outDims, the output Array can be either truncated
// or padded (towards end of respective dimensions).
//
// While resizing copying, if output dimensions are larger than input, then
// elements beyond the input dimensions are set to the \p defaultValue.
//
// \param[in] in is input Array
// \param[in] outDims is the target output dimensions
// \param[in] defaultValue is the value to which padded locations are set.
// \param[in] scale is the value by which all output elements are scaled.
//
// \returns Array<outType>
template<typename inType, typename outType>
Array<outType> reshape(const Array<inType> &in, const dim4 &outDims,
                       outType defaultValue = outType(0), double scale = 1.0);

template<typename T>
Array<T> padArrayBorders(Array<T> const &in, dim4 const &lowerBoundPadding,
                         dim4 const &upperBoundPadding,
                         const fly::borderType btype);

template<typename T>
void multiply_inplace(Array<T> &in, double val);

template<typename T>
T getScalar(const Array<T> &in);
}  // namespace cuda
}  // namespace flare