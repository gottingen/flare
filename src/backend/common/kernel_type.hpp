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

namespace flare {
namespace common {

/// \brief Maps a type between its data representation and the type used
///        during compute operations
///
/// This struct defines two types. The data type is used to reference the
/// data of an array. The compute type will be used during the computation.
/// The kernel is responsible for converting from the data type to the
/// computation type.
/// For most types these types will be the same. For fp16 type the compute
/// type will be float on platforms that don't support 16 bit floating point
/// operations.
template<typename T>
struct kernel_type {
    /// The type used to represent the data values
    using data = T;

    /// The type used when performing a computation
    using compute = T;

    /// The type defined by the compute framework for this type
    using native = compute;
};
}  // namespace common
}  // namespace flare
