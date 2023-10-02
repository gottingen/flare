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

#ifndef FLARE_CORE_GRAPH_EBO_H_
#define FLARE_CORE_GRAPH_EBO_H_

#include <flare/core/defines.h>
#include <flare/core_fwd.h>
#include <utility>
#include <type_traits>

namespace flare::detail {

    template<int I>
    struct NotOnDeviceCtorDisambiguator {
    };

    template<class... Args>
    struct NoCtorsNotOnDevice : std::false_type {
    };

    template<class... Args>
    struct DefaultCtorNotOnDevice : std::false_type {
    };

    template<>
    struct DefaultCtorNotOnDevice<> : std::true_type {
    };

    template<class T, bool Empty,
            template<class...> class CtorNotOnDevice = NoCtorsNotOnDevice>
    struct EBOBaseImpl;

    template<class T, template<class...> class CtorNotOnDevice>
    struct EBOBaseImpl<T, true, CtorNotOnDevice> {
        template<class... Args, class _ignored = void,
                std::enable_if_t<std::is_void<_ignored>::value &&
                                 std::is_constructible<T, Args...>::value &&
                                 !CtorNotOnDevice<Args...>::value,
                        int> = 0>
        FLARE_FORCEINLINE_FUNCTION constexpr explicit EBOBaseImpl(
                Args &&...) noexcept {}

        template<class... Args, class _ignored = void,
                std::enable_if_t<std::is_void<_ignored>::value &&
                                 std::is_constructible<T, Args...>::value &&
                                 CtorNotOnDevice<Args...>::value,
                        long> = 0>
        inline constexpr explicit EBOBaseImpl(Args &&...) noexcept {}

        FLARE_DEFAULTED_FUNCTION
        constexpr EBOBaseImpl(EBOBaseImpl const &) = default;

        FLARE_DEFAULTED_FUNCTION
        constexpr EBOBaseImpl(EBOBaseImpl &&) = default;

        FLARE_DEFAULTED_FUNCTION
        constexpr EBOBaseImpl &operator=(EBOBaseImpl const &) = default;

        FLARE_DEFAULTED_FUNCTION
        constexpr EBOBaseImpl &operator=(EBOBaseImpl &&) = default;

        FLARE_DEFAULTED_FUNCTION
        ~EBOBaseImpl() = default;

        FLARE_INLINE_FUNCTION
        constexpr T &_ebo_data_member() &{ return *reinterpret_cast<T *>(this); }

        FLARE_INLINE_FUNCTION
        constexpr T const &_ebo_data_member() const &{
            return *reinterpret_cast<T const *>(this);
        }

        FLARE_INLINE_FUNCTION
        T volatile &_ebo_data_member() volatile &{
            return *reinterpret_cast<T volatile *>(this);
        }

        FLARE_INLINE_FUNCTION
        T const volatile &_ebo_data_member() const volatile &{
            return *reinterpret_cast<T const volatile *>(this);
        }

        FLARE_INLINE_FUNCTION
        constexpr T &&_ebo_data_member() &&{
            return std::move(*reinterpret_cast<T *>(this));
        }
    };

    template<class T, template<class...> class CTorsNotOnDevice>
    struct EBOBaseImpl<T, false, CTorsNotOnDevice> {
        T m_ebo_object;

        template<class... Args, class _ignored = void,
                std::enable_if_t<std::is_void<_ignored>::value &&
                                 !CTorsNotOnDevice<Args...>::value &&
                                 std::is_constructible<T, Args...>::value,
                        int> = 0>
        FLARE_FORCEINLINE_FUNCTION constexpr explicit EBOBaseImpl(
                Args &&... args) noexcept(noexcept(T(std::forward<Args>(args)...)))
                : m_ebo_object(std::forward<Args>(args)...) {}

        template<class... Args, class _ignored = void,
                std::enable_if_t<std::is_void<_ignored>::value &&
                                 CTorsNotOnDevice<Args...>::value &&
                                 std::is_constructible<T, Args...>::value,
                        long> = 0>
        inline constexpr explicit EBOBaseImpl(Args &&... args) noexcept(
        noexcept(T(std::forward<Args>(args)...)))
                : m_ebo_object(std::forward<Args>(args)...) {}

        // TODO @tasking @minor DSH noexcept in the right places?

        FLARE_DEFAULTED_FUNCTION
        constexpr EBOBaseImpl(EBOBaseImpl const &) = default;

        FLARE_DEFAULTED_FUNCTION
        constexpr EBOBaseImpl(EBOBaseImpl &&) noexcept = default;

        FLARE_DEFAULTED_FUNCTION
        constexpr EBOBaseImpl &operator=(EBOBaseImpl const &) = default;

        FLARE_DEFAULTED_FUNCTION
        constexpr EBOBaseImpl &operator=(EBOBaseImpl &&) = default;

        FLARE_DEFAULTED_FUNCTION
        ~EBOBaseImpl() = default;

        FLARE_INLINE_FUNCTION
        T &_ebo_data_member() &{ return m_ebo_object; }

        FLARE_INLINE_FUNCTION
        T const &_ebo_data_member() const &{ return m_ebo_object; }

        FLARE_INLINE_FUNCTION
        T volatile &_ebo_data_member() volatile &{ return m_ebo_object; }

        FLARE_INLINE_FUNCTION
        T const volatile &_ebo_data_member() const volatile &{ return m_ebo_object; }

        FLARE_INLINE_FUNCTION
        T &&_ebo_data_member() &&{ return m_ebo_object; }
    };

    /**
     *
     * @tparam T
     */
    template<class T,
            template<class...> class CtorsNotOnDevice = NoCtorsNotOnDevice>
    struct StandardLayoutNoUniqueAddressMemberEmulation
            : EBOBaseImpl<T, std::is_empty<T>::value, CtorsNotOnDevice> {
    private:
        using ebo_base_t = EBOBaseImpl<T, std::is_empty<T>::value, CtorsNotOnDevice>;

    public:
        using ebo_base_t::ebo_base_t;

        FLARE_FORCEINLINE_FUNCTION
        constexpr T &no_unique_address_data_member() &{
            return this->ebo_base_t::_ebo_data_member();
        }

        FLARE_FORCEINLINE_FUNCTION
        constexpr T const &no_unique_address_data_member() const &{
            return this->ebo_base_t::_ebo_data_member();
        }

        FLARE_FORCEINLINE_FUNCTION
        T volatile &no_unique_address_data_member() volatile &{
            return this->ebo_base_t::_ebo_data_member();
        }

        FLARE_FORCEINLINE_FUNCTION
        T const volatile &no_unique_address_data_member() const volatile &{
            return this->ebo_base_t::_ebo_data_member();
        }

        FLARE_FORCEINLINE_FUNCTION
        constexpr T &&no_unique_address_data_member() &&{
            return this->ebo_base_t::_ebo_data_member();
        }
    };

    /**
     *
     * @tparam T
     */
    template<class T,
            template<class...> class CtorsNotOnDevice = NoCtorsNotOnDevice>
    class NoUniqueAddressMemberEmulation
            : private StandardLayoutNoUniqueAddressMemberEmulation<T,
                    CtorsNotOnDevice> {
    private:
        using base_t =
                StandardLayoutNoUniqueAddressMemberEmulation<T, CtorsNotOnDevice>;

    public:
        using base_t::base_t;
        using base_t::no_unique_address_data_member;
    };

    template<class ExecutionSpace>
    class ExecutionSpaceInstanceStorage
            : private NoUniqueAddressMemberEmulation<ExecutionSpace,
                    DefaultCtorNotOnDevice> {
    private:
        using base_t =
                NoUniqueAddressMemberEmulation<ExecutionSpace, DefaultCtorNotOnDevice>;

    protected:
        constexpr explicit ExecutionSpaceInstanceStorage() : base_t() {}

        FLARE_INLINE_FUNCTION
        constexpr explicit ExecutionSpaceInstanceStorage(
                ExecutionSpace const &arg_execution_space)
                : base_t(arg_execution_space) {}

        FLARE_INLINE_FUNCTION
        constexpr explicit ExecutionSpaceInstanceStorage(
                ExecutionSpace &&arg_execution_space)
                : base_t(std::move(arg_execution_space)) {}

        FLARE_INLINE_FUNCTION ExecutionSpace &execution_space_instance() &{
            return this->no_unique_address_data_member();
        }

        FLARE_INLINE_FUNCTION ExecutionSpace const &execution_space_instance() const &{
            return this->no_unique_address_data_member();
        }

        FLARE_INLINE_FUNCTION ExecutionSpace &&execution_space_instance() &&{
            return std::move(*this).no_unique_address_data_member();
        }
    };

    template<class MemorySpace>
    class MemorySpaceInstanceStorage
            : private NoUniqueAddressMemberEmulation<MemorySpace,
                    DefaultCtorNotOnDevice> {
    private:
        using base_t =
                NoUniqueAddressMemberEmulation<MemorySpace, DefaultCtorNotOnDevice>;

    protected:
        MemorySpaceInstanceStorage() : base_t() {}

        FLARE_INLINE_FUNCTION MemorySpaceInstanceStorage(MemorySpace const &arg_memory_space)
                : base_t(arg_memory_space) {}

        FLARE_INLINE_FUNCTION constexpr explicit MemorySpaceInstanceStorage(MemorySpace &&arg_memory_space)
                : base_t(arg_memory_space) {}

        FLARE_INLINE_FUNCTION MemorySpace &memory_space_instance() &{
            return this->no_unique_address_data_member();
        }

        FLARE_INLINE_FUNCTION MemorySpace const &memory_space_instance() const &{
            return this->no_unique_address_data_member();
        }

        FLARE_INLINE_FUNCTION MemorySpace &&memory_space_instance() &&{
            return std::move(*this).no_unique_address_data_member();
        }
    };

}  // namespace flare::detail

#endif   // FLARE_CORE_GRAPH_EBO_H_
