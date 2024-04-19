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
#include <common/jit/ScalarNode.hpp>
#include <math.hpp>
#include <optypes.hpp>
#include <memory>

namespace flare {
namespace cuda {

template<typename T>
Array<T> createScalarNode(const dim4 &size, const T val) {
#if _MSC_VER > 1914
    // FIXME(pradeep) - Needed only in CUDA backend, didn't notice any
    // issues in other backends.
    // Either this gaurd or we need to enable extended alignment
    // by defining _ENABLE_EXTENDED_ALIGNED_STORAGE before <type_traits>
    // header is included
    using ScalarNode    = common::ScalarNode<T>;
    using ScalarNodePtr = std::shared_ptr<ScalarNode>;
    return createNodeArray<T>(size, ScalarNodePtr(new ScalarNode(val)));
#else
    return createNodeArray<T>(size,
                              std::make_shared<common::ScalarNode<T>>(val));
#endif
}

}  // namespace cuda
}  // namespace flare
