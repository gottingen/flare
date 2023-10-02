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

#ifndef FLARE_CORE_COMMON_CORE_CONCEPTS_H_
#define FLARE_CORE_COMMON_CORE_CONCEPTS_H_

#include <type_traits>

// Needed for 'is_space<S>::host_mirror_space
#include <flare/core_fwd.h>

#include <flare/core/common/detection_idiom.h>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace flare {

// Schedules for Execution Policies
    struct Static {
    };
    struct Dynamic {
    };

// Schedule Wrapper Type
    template<class T>
    struct Schedule {
        static_assert(std::is_same<T, Static>::value ||
                      std::is_same<T, Dynamic>::value,
                      "flare: Invalid Schedule<> type.");
        using schedule_type = Schedule;
        using type = T;
    };

// Specify Iteration Index Type
    template<typename T>
    struct IndexType {
        static_assert(std::is_integral<T>::value, "flare: Invalid IndexType<>.");
        using index_type = IndexType;
        using type = T;
    };

    namespace experimental {
        struct WorkItemProperty {
            template<unsigned long Property>
            struct ImplWorkItemProperty {
                static const unsigned value = Property;
                using work_item_property = ImplWorkItemProperty<Property>;
            };

            constexpr static const ImplWorkItemProperty<0> None =
                    ImplWorkItemProperty<0>();
            constexpr static const ImplWorkItemProperty<1> HintLightWeight =
                    ImplWorkItemProperty<1>();
            constexpr static const ImplWorkItemProperty<2> HintHeavyWeight =
                    ImplWorkItemProperty<2>();
            constexpr static const ImplWorkItemProperty<4> HintRegular =
                    ImplWorkItemProperty<4>();
            constexpr static const ImplWorkItemProperty<8> HintIrregular =
                    ImplWorkItemProperty<8>();
            constexpr static const ImplWorkItemProperty<16> ImplForceGlobalLaunch =
                    ImplWorkItemProperty<16>();
            using None_t = ImplWorkItemProperty<0>;
            using HintLightWeight_t = ImplWorkItemProperty<1>;
            using HintHeavyWeight_t = ImplWorkItemProperty<2>;
            using HintRegular_t = ImplWorkItemProperty<4>;
            using HintIrregular_t = ImplWorkItemProperty<8>;
            using ImplForceGlobalLaunch_t = ImplWorkItemProperty<16>;
        };

        template<unsigned long pv1, unsigned long pv2>
        inline constexpr WorkItemProperty::ImplWorkItemProperty<pv1 | pv2> operator|(
                WorkItemProperty::ImplWorkItemProperty<pv1>,
                WorkItemProperty::ImplWorkItemProperty<pv2>) {
            return WorkItemProperty::ImplWorkItemProperty<pv1 | pv2>();
        }

        template<unsigned long pv1, unsigned long pv2>
        inline constexpr WorkItemProperty::ImplWorkItemProperty<pv1 & pv2> operator&(
                WorkItemProperty::ImplWorkItemProperty<pv1>,
                WorkItemProperty::ImplWorkItemProperty<pv2>) {
            return WorkItemProperty::ImplWorkItemProperty<pv1 & pv2>();
        }

        template<unsigned long pv1, unsigned long pv2>
        inline constexpr bool operator==(WorkItemProperty::ImplWorkItemProperty<pv1>,
                                         WorkItemProperty::ImplWorkItemProperty<pv2>) {
            return pv1 == pv2;
        }

    }  // namespace experimental

    /**\brief Specify Launch Bounds for CUDA execution.
     *
     *  If no launch bounds specified then do not set launch bounds.
     */
    template<unsigned int maxT = 0 /* Max threads per block */
            ,
            unsigned int minB = 0 /* Min blocks per SM */
    >
    struct LaunchBounds {
        using launch_bounds = LaunchBounds;
        using type = LaunchBounds<maxT, minB>;
        static constexpr unsigned int maxTperB{maxT};
        static constexpr unsigned int minBperSM{minB};
    };

}  // namespace flare

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace flare {

#define FLARE_IMPL_IS_CONCEPT(CONCEPT)                        \
  template <typename T>                                        \
  struct is_##CONCEPT {                                        \
   private:                                                    \
    template <typename U>                                      \
    using have_t = typename U::CONCEPT;                        \
    template <typename U>                                      \
    using have_type_t = typename U::CONCEPT##_type;            \
                                                               \
   public:                                                     \
    static constexpr bool value =                              \
        std::is_base_of<detected_t<have_t, T>, T>::value ||    \
        std::is_base_of<detected_t<have_type_t, T>, T>::value; \
    constexpr operator bool() const noexcept { return value; } \
  };                                                           \
  template <typename T>                                        \
  inline constexpr bool is_##CONCEPT##_v = is_##CONCEPT<T>::value;

// Public concept:

    FLARE_IMPL_IS_CONCEPT(memory_space)

    FLARE_IMPL_IS_CONCEPT(memory_traits)

    FLARE_IMPL_IS_CONCEPT(execution_space)

    FLARE_IMPL_IS_CONCEPT(execution_policy)

    FLARE_IMPL_IS_CONCEPT(array_layout)

    FLARE_IMPL_IS_CONCEPT(reducer)

    FLARE_IMPL_IS_CONCEPT(team_handle)
    namespace experimental {
        FLARE_IMPL_IS_CONCEPT(work_item_property)

        FLARE_IMPL_IS_CONCEPT(hooks_policy)
    }  // namespace experimental

    namespace detail {

// Implementation concept:

        FLARE_IMPL_IS_CONCEPT(thread_team_member)

        FLARE_IMPL_IS_CONCEPT(host_thread_team_member)

        FLARE_IMPL_IS_CONCEPT(graph_kernel)

    }  // namespace detail

#undef FLARE_IMPL_IS_CONCEPT

}  // namespace flare

namespace flare {
    namespace detail {

        template<class Object>
        class has_member_team_shmem_size {
            template<typename T>
            static int32_t test_for_member(decltype(&T::team_shmem_size)) {
                return int32_t(0);
            }

            template<typename T>
            static int64_t test_for_member(...) {
                return int64_t(0);
            }

        public:
            constexpr static bool value =
                    sizeof(test_for_member<Object>(nullptr)) == sizeof(int32_t);
        };

        template<class Object>
        class has_member_shmem_size {
            template<typename T>
            static int32_t test_for_member(decltype(&T::shmem_size_me)) {
                return int32_t(0);
            }

            template<typename T>
            static int64_t test_for_member(...) {
                return int64_t(0);
            }

        public:
            constexpr static bool value =
                    sizeof(test_for_member<Object>(0)) == sizeof(int32_t);
        };

    }  // namespace detail
}  // namespace flare
//----------------------------------------------------------------------------

namespace flare {

    template<class ExecutionSpace, class MemorySpace>
    struct Device {
        static_assert(flare::is_execution_space<ExecutionSpace>::value,
                      "Execution space is not valid");
        static_assert(flare::is_memory_space<MemorySpace>::value,
                      "Memory space is not valid");
        using execution_space = ExecutionSpace;
        using memory_space = MemorySpace;
        using device_type = Device<execution_space, memory_space>;
    };

    namespace detail {

        template<typename T>
        struct is_device_helper : std::false_type {
        };

        template<typename ExecutionSpace, typename MemorySpace>
        struct is_device_helper<Device<ExecutionSpace, MemorySpace>> : std::true_type {
        };

    }  // namespace detail

    template<typename T>
    using is_device = typename detail::is_device_helper<std::remove_cv_t<T>>::type;

    template<typename T>
    inline constexpr bool is_device_v = is_device<T>::value;

//----------------------------------------------------------------------------

    template<typename T>
    struct is_space {
    private:
        template<typename, typename = void>
        struct exe : std::false_type {
            using space = void;
        };

        template<typename, typename = void>
        struct mem : std::false_type {
            using space = void;
        };

        template<typename, typename = void>
        struct dev : std::false_type {
            using space = void;
        };

        template<typename U>
        struct exe<U, std::conditional_t<true, void, typename U::execution_space>>
                : std::is_same<U, typename U::execution_space>::type {
            using space = typename U::execution_space;
        };

        template<typename U>
        struct mem<U, std::conditional_t<true, void, typename U::memory_space>>
                : std::is_same<U, typename U::memory_space>::type {
            using space = typename U::memory_space;
        };

        template<typename U>
        struct dev<U, std::conditional_t<true, void, typename U::device_type>>
                : std::is_same<U, typename U::device_type>::type {
            using space = typename U::device_type;
        };

        using is_exe = typename is_space<T>::template exe<std::remove_cv_t<T>>;
        using is_mem = typename is_space<T>::template mem<std::remove_cv_t<T>>;
        using is_dev = typename is_space<T>::template dev<std::remove_cv_t<T>>;

    public:
        static constexpr bool value = is_exe::value || is_mem::value || is_dev::value;

        constexpr operator bool() const noexcept { return value; }

        using execution_space = typename is_exe::space;
        using memory_space = typename is_mem::space;

        // For backward compatibility, deprecated in favor of
        // flare::detail::HostMirror<S>::host_mirror_space

    private:
        // The actual definitions for host_memory_space and host_execution_spaces are
        // in do_not_use_host_memory_space and do_not_use_host_execution_space to be
        // able to use them within this class without deprecation warnings.
        using do_not_use_host_memory_space = std::conditional_t<
                std::is_same<memory_space, flare::HostSpace>::value
#if defined(FLARE_ON_CUDA_DEVICE)
                || std::is_same<memory_space, flare::CudaUVMSpace>::value ||
                std::is_same<memory_space, flare::CudaHostPinnedSpace>::value
#endif
                ,
                memory_space, flare::HostSpace>;

        using do_not_use_host_execution_space = std::conditional_t<
#if defined(FLARE_ON_CUDA_DEVICE)
                std::is_same<execution_space, flare::Cuda>::value ||
#endif
                false,
                flare::DefaultHostExecutionSpace, execution_space>;
    };

}  // namespace flare

//----------------------------------------------------------------------------

namespace flare {
    namespace detail {

/**\brief  Access relationship between DstMemorySpace and SrcMemorySpace
 *
 *  The default case can assume accessibility for the same space.
 *  Specializations must be defined for different memory spaces.
 */
        template<typename DstMemorySpace, typename SrcMemorySpace>
        struct MemorySpaceAccess {
            static_assert(flare::is_memory_space<DstMemorySpace>::value &&
                          flare::is_memory_space<SrcMemorySpace>::value,
                          "template arguments must be memory spaces");

            /**\brief  Can a View (or pointer) to memory in SrcMemorySpace
             *         be assigned to a View (or pointer) to memory marked DstMemorySpace.
             *
             *  1. DstMemorySpace::execution_space == SrcMemorySpace::execution_space
             *  2. All execution spaces that can access DstMemorySpace can also access
             *     SrcMemorySpace.
             */
            enum {
                assignable = std::is_same<DstMemorySpace, SrcMemorySpace>::value
            };

            /**\brief  For all DstExecSpace::memory_space == DstMemorySpace
             *         DstExecSpace can access SrcMemorySpace.
             */
            enum {
                accessible = assignable
            };

            /**\brief  Does a DeepCopy capability exist
             *         to DstMemorySpace from SrcMemorySpace
             */
            enum {
                deepcopy = assignable
            };
        };

    }  // namespace detail
}  // namespace flare

namespace flare {

/**\brief  Can AccessSpace access MemorySpace ?
 *
 *   Requires:
 *     flare::is_space< AccessSpace >::value
 *     flare::is_memory_space< MemorySpace >::value
 *
 *   Can AccessSpace::execution_space access MemorySpace ?
 *     enum : bool { accessible };
 *
 *   Is View<AccessSpace::memory_space> assignable from View<MemorySpace> ?
 *     enum : bool { assignable };
 *
 *   If ! accessible then through which intercessory memory space
 *   should a be used to deep copy memory for
 *     AccessSpace::execution_space
 *   to get access.
 *   When AccessSpace::memory_space == flare::HostSpace
 *   then space is the View host mirror space.
 */
    template<typename AccessSpace, typename MemorySpace>
    struct SpaceAccessibility {
    private:
        static_assert(flare::is_space<AccessSpace>::value,
                      "template argument #1 must be a flare space");

        static_assert(flare::is_memory_space<MemorySpace>::value,
                      "template argument #2 must be a flare memory space");

        // The input AccessSpace may be a Device<ExecSpace,MemSpace>
        // verify that it is a valid combination of spaces.
        static_assert(flare::detail::MemorySpaceAccess<
                              typename AccessSpace::execution_space::memory_space,
                              typename AccessSpace::memory_space>::accessible,
                      "template argument #1 is an invalid space");

        using exe_access = flare::detail::MemorySpaceAccess<
                typename AccessSpace::execution_space::memory_space, MemorySpace>;

        using mem_access =
                flare::detail::MemorySpaceAccess<typename AccessSpace::memory_space,
                        MemorySpace>;

    public:
        /**\brief  Can AccessSpace::execution_space access MemorySpace ?
         *
         *  Default based upon memory space accessibility.
         *  Specialization required for other relationships.
         */
        enum {
            accessible = exe_access::accessible
        };

        /**\brief  Can assign to AccessSpace from MemorySpace ?
         *
         *  Default based upon memory space accessibility.
         *  Specialization required for other relationships.
         */
        enum {
            assignable = is_memory_space<AccessSpace>::value && mem_access::assignable
        };

        /**\brief  Can deep copy to AccessSpace::memory_Space from MemorySpace ?  */
        enum {
            deepcopy = mem_access::deepcopy
        };

        // What intercessory space for AccessSpace::execution_space
        // to be able to access MemorySpace?
        // If same memory space or not accessible use the AccessSpace
        // else construct a device with execution space and memory space.
        using space = std::conditional_t<
                std::is_same<typename AccessSpace::memory_space, MemorySpace>::value ||
                !exe_access::accessible,
                AccessSpace,
                flare::Device<typename AccessSpace::execution_space, MemorySpace>>;
    };

}  // namespace flare

//----------------------------------------------------------------------------

#endif  // FLARE_CORE_COMMON_CORE_CONCEPTS_H_
