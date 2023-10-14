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

#ifndef FLARE_BACKEND_SERIAL_SERIAL_UNIQUE_TOKEN_H_
#define FLARE_BACKEND_SERIAL_SERIAL_UNIQUE_TOKEN_H_

#include <flare/core/parallel/unique_token.h>

namespace flare {

    template<>
    class UniqueToken<Serial, UniqueTokenScope::Instance> {
    public:
        using execution_space = Serial;
        using size_type = int;

        /// \brief create object size for concurrency on the given instance
        ///
        /// This object should not be shared between instances
        UniqueToken(execution_space const & = execution_space()) noexcept {}

        /// \brief create object size for requested size on given instance
        ///
        /// It is the users responsibility to only acquire size tokens concurrently
        UniqueToken(size_type, execution_space const & = execution_space()) {}

        /// \brief upper bound for acquired values, i.e. 0 <= value < size()
        FLARE_INLINE_FUNCTION
        int size() const noexcept { return 1; }

        /// \brief acquire value such that 0 <= value < size()
        FLARE_INLINE_FUNCTION
        int acquire() const noexcept { return 0; }

        /// \brief release a value acquired by generate
        FLARE_INLINE_FUNCTION
        void release(int) const noexcept {}
    };

    template<>
    class UniqueToken<Serial, UniqueTokenScope::Global> {
    public:
        using execution_space = Serial;
        using size_type = int;

        /// \brief create object size for concurrency on the given instance
        ///
        /// This object should not be shared between instances
        UniqueToken(execution_space const & = execution_space()) noexcept {}

        /// \brief upper bound for acquired values, i.e. 0 <= value < size()
        FLARE_INLINE_FUNCTION
        int size() const noexcept { return 1; }

        /// \brief acquire value such that 0 <= value < size()
        FLARE_INLINE_FUNCTION
        int acquire() const noexcept { return 0; }

        /// \brief release a value acquired by generate
        FLARE_INLINE_FUNCTION
        void release(int) const noexcept {}
    };

}  // namespace flare

#endif  // FLARE_BACKEND_SERIAL_SERIAL_UNIQUE_TOKEN_H_
