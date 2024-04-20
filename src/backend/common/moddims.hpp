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

#include <Array.hpp>
#include <fly/dim4.hpp>

namespace flare {
namespace common {

/// Modifies the shape of the Array<T> object to \p newDims
///
/// Modifies the shape of the Array<T> object to \p newDims. Depending on the
/// in Array, different operations will be performed.
///
/// * If the object is a linear array and it is an unevaluated JIT node, this
///   function will createa a JIT Node.
/// * If the object is not a JIT node but it is still linear, It will create a
///   reference to the internal array with the new shape.
/// * If the array is non-linear a moddims operation will be performed
///
/// \param in       The input array that who's shape will be modified
/// \param newDims  The new shape of the input Array<T>
///
/// \returns        a new Array<T> with the specified shape.
template<typename T>
detail::Array<T> modDims(const detail::Array<T> &in, const fly::dim4 &newDims);

/// Calls moddims where all elements are in the first dimension of the array
///
/// \param in  The input Array to be flattened
///
/// \returns A new array where all elements are in the first dimension.
template<typename T>
detail::Array<T> flat(const detail::Array<T> &in);

}  // namespace common
}  // namespace flare
