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

#ifndef FLARE_CORE_MEMORY_ANONYMOUS_SPACE_H_
#define FLARE_CORE_MEMORY_ANONYMOUS_SPACE_H_

#include <flare/core_fwd.h>
#include <flare/core/common/concepts.h>
#include <cstddef>

/*--------------------------------------------------------------------------*/

namespace flare {

    class AnonymousSpace {
    public:
        //! Tag this class as a flare memory space
        using memory_space = AnonymousSpace;
        using execution_space = flare::DefaultExecutionSpace;
        using size_type = size_t;

        //! This memory space preferred device_type
        using device_type = flare::Device<execution_space, memory_space>;

        /**\brief  Default memory space instance */
        AnonymousSpace() = default;

        AnonymousSpace(AnonymousSpace &&rhs) = default;

        AnonymousSpace(const AnonymousSpace &rhs) = default;

        AnonymousSpace &operator=(AnonymousSpace &&) = default;

        AnonymousSpace &operator=(const AnonymousSpace &) = default;

        ~AnonymousSpace() = default;

        /**\brief Return Name of the MemorySpace */
        static constexpr const char *name() { return "Anonymous"; }
    };

}  // namespace flare

namespace flare::detail {
    template<typename OtherSpace>
    struct MemorySpaceAccess<flare::AnonymousSpace, OtherSpace> {
        enum : bool {
            assignable = true
        };
        enum : bool {
            accessible = true
        };
        enum : bool {
            deepcopy = true
        };
    };

    template<typename OtherSpace>
    struct MemorySpaceAccess<OtherSpace, flare::AnonymousSpace> {
        enum : bool {
            assignable = true
        };
        enum : bool {
            accessible = true
        };
        enum : bool {
            deepcopy = true
        };
    };

    template<>
    struct MemorySpaceAccess<flare::AnonymousSpace, flare::AnonymousSpace> {
        enum : bool {
            assignable = true
        };
        enum : bool {
            accessible = true
        };
        enum : bool {
            deepcopy = true
        };
    };

}  // namespace flare::detail

#endif  // FLARE_CORE_MEMORY_ANONYMOUS_SPACE_H_
