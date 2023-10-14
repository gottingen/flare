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

#ifndef FLARE_CORE_PARALLEL_UNIQUE_TOKEN_H_
#define FLARE_CORE_PARALLEL_UNIQUE_TOKEN_H_

#include <flare/core/defines.h>
#include <flare/core/memory/memory_traits.h>
#include <flare/core_fwd.h>

namespace flare {

    enum class UniqueTokenScope : int {
        Instance, Global
    };

    /// \brief class to generate unique ids base on the required amount of
    /// concurrency
    ///
    /// This object should behave like a ref-counted object, so that when the last
    /// instance is destroy resources are free if needed
    template<typename ExecutionSpace = flare::DefaultExecutionSpace,
            UniqueTokenScope        = UniqueTokenScope::Instance>
    class UniqueToken {
    public:
        using execution_space = ExecutionSpace;
        using size_type = typename execution_space::size_type;

        /// \brief create object size for concurrency on the given instance
        ///
        /// This object should not be shared between instances
        UniqueToken(execution_space const & = execution_space());

        /// \brief upper bound for acquired values, i.e. 0 <= value < size()
        FLARE_INLINE_FUNCTION
        size_type size() const;

        /// \brief acquire value such that 0 <= value < size()
        FLARE_INLINE_FUNCTION
        size_type acquire() const;

        /// \brief release a value acquired by generate
        FLARE_INLINE_FUNCTION
        void release(size_type) const;
    };

    /// \brief Instance scope UniqueToken allows for a max size other than
    /// execution_space().concurrency()
    ///
    /// This object should behave like a ref-counted object, so that when the last
    /// instance is destroyed, resources are free if needed
    template<typename ExecutionSpace>
    class UniqueToken<ExecutionSpace, UniqueTokenScope::Instance>
            : public UniqueToken<ExecutionSpace, UniqueTokenScope::Global> {
    public:
        using execution_space = ExecutionSpace;
        using size_type = typename execution_space::size_type;

        /// \brief Create object with specified size
        ///
        /// It is required that max_size is >= the maximum number of concurrent
        /// threads that will attempt to acquire the UniqueToken. This constructor is
        /// most commonly useful when you:
        ///   1) Have a loop bound that may be smaller than
        ///   execution_space().concurrency().
        ///   2) Want a per-team unique token in the range [0,
        ///   execution_space().concurrency() / team_size)
        UniqueToken(size_type max_size, execution_space const & = execution_space());
    };

    // NOTE There was an agreement amongst developers that "AcquireUniqueToken" is a
    // bad name but at this time no one has suggested a better alternative.

    /// \brief RAII helper for per-thread unique token values.
    ///
    /// The token value will be acquired at construction and automatically
    /// released at destruction.
    template<typename ExecutionSpace,
            UniqueTokenScope TokenScope = UniqueTokenScope::Instance>
    class AcquireUniqueToken {
    public:
        using exec_space = ExecutionSpace;
        using size_type = typename exec_space::size_type;
        using token_type = UniqueToken<exec_space, TokenScope>;

    private:
        token_type my_token;
        size_type my_acquired_val;

    public:
        FLARE_FUNCTION AcquireUniqueToken(token_type t)
                : my_token(t), my_acquired_val(my_token.acquire()) {}

        FLARE_FUNCTION ~AcquireUniqueToken() { my_token.release(my_acquired_val); }

        FLARE_FUNCTION size_type value() const { return my_acquired_val; }
    };

    /// \brief RAII helper for per-team unique token values.
    ///
    /// The token value will be acquired at construction and automatically
    /// released at destruction. All threads in a team will share the same
    /// token value.
    template<typename TeamPolicy>
    class AcquireTeamUniqueToken {
    public:
        using exec_space = typename TeamPolicy::execution_space;
        using token_type = UniqueToken<exec_space>;
        using size_type = typename token_type::size_type;
        using team_member_type = typename TeamPolicy::member_type;
        using scratch_tensor =
                flare::Tensor<size_type, typename exec_space::scratch_memory_space,
                        flare::MemoryUnmanaged>;

    private:
        token_type my_token;
        size_type my_acquired_val;
        scratch_tensor my_team_acquired_val;
        team_member_type my_team;

    public:
        // NOTE The implementations of the constructor and destructor use
        // `flare::single()` which is an inline function defined in each backend.
        // This creates circular dependency issues.  Moving them to a separate header
        // is less than ideal and should be revisited later.  Having a `UniqueToken`
        // forward declaration was considered but the non-type template parameter
        // makes things complicated because it would require moving the definition of
        // `UniqueTokenScope` enumeration type and its enumerators away which would
        // hurt readability.
        FLARE_FUNCTION AcquireTeamUniqueToken(token_type t, team_member_type team);

        FLARE_FUNCTION ~AcquireTeamUniqueToken();

        FLARE_FUNCTION size_type value() const { return my_acquired_val; }

        static std::size_t shmem_size() { return scratch_tensor::shmem_size(); }
    };

}  // namespace flare

#endif  // FLARE_CORE_PARALLEL_UNIQUE_TOKEN_H_
