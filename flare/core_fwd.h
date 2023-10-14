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

#ifndef FLARE_CORE_FWD_H_
#define FLARE_CORE_FWD_H_

//----------------------------------------------------------------------------
// macros.hpp does introspection on configuration options
// and compiler environment then sets a collection of #define macros.

#include <flare/core/defines.h>
#include <flare/core/common/error.h>
#include <flare/core/common/printf.h>
#include <flare/core/common/utilities.h>


//----------------------------------------------------------------------------
// Have assumed a 64-bit build (8-byte pointers) throughout the code base.
// 32-bit build allowed but unsupported.
#ifdef FLARE_IMPL_32BIT
static_assert(sizeof(void *) == 4,
              "flare assumes 64-bit build; i.e., 4-byte pointers");
#else
static_assert(sizeof(void *) == 8,
              "flare assumes 64-bit build; i.e., 8-byte pointers");
#endif
//----------------------------------------------------------------------------

namespace flare {

    struct AUTO_t {
        FLARE_INLINE_FUNCTION
        constexpr const AUTO_t &operator()() const { return *this; }
    };

    namespace {
        /**\brief Token to indicate that a parameter's value is to be automatically
         * selected */
        constexpr AUTO_t AUTO = flare::AUTO_t();
    }  // namespace

    struct InvalidType {
    };

}  // namespace flare

//----------------------------------------------------------------------------
// Forward declarations for class interrelationships

namespace flare {

    class HostSpace;  ///< Memory space for main process and CPU execution spaces
    class AnonymousSpace;

    template<class ExecutionSpace, class MemorySpace>
    struct Device;

    class InitializationSettings;

}  // namespace flare

#include <flare/core/common/fwd.h>
//----------------------------------------------------------------------------
// Set the default execution space.

/// Define flare::DefaultExecutionSpace as per configuration option
/// or chosen from the enabled execution spaces in the following order:
/// flare::Cuda, flare::OpenMP,
/// flare::Threads, flare::Serial

#if defined(__clang_analyzer__)
#define FLARE_IMPL_DEFAULT_EXEC_SPACE_ANNOTATION \
  [[clang::annotate("DefaultExecutionSpace")]]
#define FLARE_IMPL_DEFAULT_HOST_EXEC_SPACE_ANNOTATION \
  [[clang::annotate("DefaultHostExecutionSpace")]]
#else
#define FLARE_IMPL_DEFAULT_EXEC_SPACE_ANNOTATION
#define FLARE_IMPL_DEFAULT_HOST_EXEC_SPACE_ANNOTATION
#endif

namespace flare {

#if defined(FLARE_ENABLE_DEFAULT_DEVICE_TYPE_CUDA)
    using DefaultExecutionSpace FLARE_IMPL_DEFAULT_EXEC_SPACE_ANNOTATION = Cuda;
#elif defined(FLARE_ENABLE_DEFAULT_DEVICE_TYPE_OPENMP)
    using DefaultExecutionSpace FLARE_IMPL_DEFAULT_EXEC_SPACE_ANNOTATION = OpenMP;
#elif defined(FLARE_ENABLE_DEFAULT_DEVICE_TYPE_THREADS)
    using DefaultExecutionSpace FLARE_IMPL_DEFAULT_EXEC_SPACE_ANNOTATION = Threads;
#elif defined(FLARE_ENABLE_DEFAULT_DEVICE_TYPE_SERIAL)
    using DefaultExecutionSpace FLARE_IMPL_DEFAULT_EXEC_SPACE_ANNOTATION = Serial;
#else
#error \
    "At least one of the following execution spaces must be defined in order to use flare: flare::Cuda, flare::OpenMP, flare::Threads, or flare::Serial."
#endif

#if defined(FLARE_ENABLE_DEFAULT_DEVICE_TYPE_OPENMP)
    using DefaultHostExecutionSpace FLARE_IMPL_DEFAULT_HOST_EXEC_SPACE_ANNOTATION =
        OpenMP;
#elif defined(FLARE_ENABLE_DEFAULT_DEVICE_TYPE_THREADS)
    using DefaultHostExecutionSpace FLARE_IMPL_DEFAULT_HOST_EXEC_SPACE_ANNOTATION =
        Threads;
#elif defined(FLARE_ENABLE_DEFAULT_DEVICE_TYPE_SERIAL)
    using DefaultHostExecutionSpace FLARE_IMPL_DEFAULT_HOST_EXEC_SPACE_ANNOTATION =
        Serial;
#elif defined(FLARE_ENABLE_OPENMP)
    using DefaultHostExecutionSpace FLARE_IMPL_DEFAULT_HOST_EXEC_SPACE_ANNOTATION =
        OpenMP;
#elif defined(FLARE_ENABLE_THREADS)
    using DefaultHostExecutionSpace FLARE_IMPL_DEFAULT_HOST_EXEC_SPACE_ANNOTATION =
        Threads;
#elif defined(FLARE_ENABLE_SERIAL)
    using DefaultHostExecutionSpace FLARE_IMPL_DEFAULT_HOST_EXEC_SPACE_ANNOTATION =
            Serial;
#else
#error \
    "At least one of the following execution spaces must be defined in order to use flare: flare::OpenMP, flare::Threads, or flare::Serial."
#endif

// check for devices that support sharedSpace
#if defined(FLARE_ON_CUDA_DEVICE)
    using SharedSpace = CudaUVMSpace;
#define FLARE_HAS_SHARED_SPACE
#else
    using SharedSpace               = HostSpace;
#define FLARE_HAS_SHARED_SPACE
#endif

    inline constexpr bool has_shared_space =
#if defined FLARE_HAS_SHARED_SPACE
            true;
#else
            false;
#endif

#if defined(FLARE_ON_CUDA_DEVICE)
    using SharedHostPinnedSpace = CudaHostPinnedSpace;
#define FLARE_HAS_SHARED_HOST_PINNED_SPACE
#else
    using SharedHostPinnedSpace = HostSpace;
#define FLARE_HAS_SHARED_HOST_PINNED_SPACE
#endif

    inline constexpr bool has_shared_host_pinned_space =
#if defined FLARE_HAS_SHARED_HOST_PINNED_SPACE
            true;
#else
            false;
#endif

}  // namespace flare

//----------------------------------------------------------------------------
// Detect the active execution space and define its memory space.
// This is used to verify whether a running kernel can access
// a given memory space.

namespace flare {

    template<class AccessSpace, class MemorySpace>
    struct SpaceAccessibility;

    namespace detail {

        // primary template: memory space is accessible, do nothing.
        template<class MemorySpace, class AccessSpace,
                bool = SpaceAccessibility<AccessSpace, MemorySpace>::accessible>
        struct RuntimeCheckMemoryAccessViolation {
            FLARE_FUNCTION RuntimeCheckMemoryAccessViolation(char const *const) {}
        };

        // explicit specialization: memory access violation will occur, call abort with
        // the specified error message.
        template<class MemorySpace, class AccessSpace>
        struct RuntimeCheckMemoryAccessViolation<MemorySpace, AccessSpace, false> {
            FLARE_FUNCTION RuntimeCheckMemoryAccessViolation(char const *const msg) {
                flare::abort(msg);
            }
        };

        // calls abort with default error message at runtime if memory access violation
        // will occur
        template<class MemorySpace>
        FLARE_FUNCTION void runtime_check_memory_access_violation() {
            FLARE_IF_ON_HOST((
                                     RuntimeCheckMemoryAccessViolation<MemorySpace, DefaultHostExecutionSpace>(
                                             "ERROR: attempt to access inaccessible memory space");))
            FLARE_IF_ON_DEVICE(
                    (RuntimeCheckMemoryAccessViolation<MemorySpace, DefaultExecutionSpace>(
                            "ERROR: attempt to access inaccessible memory space");))
        }

        // calls abort with specified error message at runtime if memory access
        // violation will occur
        template<class MemorySpace>
        FLARE_FUNCTION void runtime_check_memory_access_violation(
                char const *const msg) {
            FLARE_IF_ON_HOST((
                                     (void) RuntimeCheckMemoryAccessViolation<MemorySpace,
                                             DefaultHostExecutionSpace>(msg);))
            FLARE_IF_ON_DEVICE((
                                       (void)
                                               RuntimeCheckMemoryAccessViolation<MemorySpace, DefaultExecutionSpace>(
                                                       msg);))
        }

    }  // namespace detail

    namespace experimental {
        template<class, class, class, class>
        class LogicalMemorySpace;
    }

}  // namespace flare

//----------------------------------------------------------------------------

namespace flare {
#ifdef FLARE_COMPILER_INTEL
    void fence();
    void fence(const std::string &name);
#else

    void fence(const std::string &name = "flare::fence: Unnamed Global Fence");

#endif
}  // namespace flare

//----------------------------------------------------------------------------

namespace flare {

    template<class DataType, class... Properties>
    class Tensor;

    namespace detail {

        template<class DstSpace, class SrcSpace,
                class ExecutionSpace = typename DstSpace::execution_space,
                class Enable         = void>
        struct DeepCopy;

        template<class TensorType, class Layout = typename TensorType::array_layout,
                class ExecSpace = typename TensorType::execution_space,
                int Rank = TensorType::rank, typename iType = int64_t>
        struct TensorFill;

        template<class TensorTypeA, class TensorTypeB, class Layout, class ExecSpace,
                int Rank, typename iType>
        struct TensorCopy;

        template<class Functor, class Policy>
        struct FunctorPolicyExecutionSpace;

        //----------------------------------------------------------------------------
        /// \class ParallelFor
        /// \brief Implementation of the ParallelFor operator that has a
        ///   partial specialization for the device.
        ///
        /// This is an implementation detail of parallel_for.  Users should
        /// skip this and go directly to the nonmember function parallel_for.
        template<class FunctorType, class ExecPolicy,
                class ExecutionSpace = typename detail::FunctorPolicyExecutionSpace<
                        FunctorType, ExecPolicy>::execution_space>
        class ParallelFor;

        /// \class ParallelReduce
        /// \brief Implementation detail of parallel_reduce.
        ///
        /// This is an implementation detail of parallel_reduce.  Users should
        /// skip this and go directly to the nonmember function parallel_reduce.
        template<typename CombinedFunctorReducerType, typename PolicyType,
                typename ExecutionSpaceType>
        class ParallelReduce;

        template<typename FunctorType, typename FunctorAnalysisReducerType,
                typename Enable = void>
        class CombinedFunctorReducer;

        /// \class ParallelScan
        /// \brief Implementation detail of parallel_scan.
        ///
        /// This is an implementation detail of parallel_scan.  Users should
        /// skip this and go directly to the documentation of the nonmember
        /// template function flare::parallel_scan.
        template<class FunctorType, class ExecPolicy,
                class ExecutionSpace = typename detail::FunctorPolicyExecutionSpace<
                        FunctorType, ExecPolicy>::execution_space>
        class ParallelScan;

        template<class FunctorType, class ExecPolicy, class ReturnType = InvalidType,
                class ExecutionSpace = typename detail::FunctorPolicyExecutionSpace<
                        FunctorType, ExecPolicy>::execution_space>
        class ParallelScanWithTotal;

    }  // namespace detail

    template<class ScalarType, class Space = HostSpace>
    struct Sum;
    template<class ScalarType, class Space = HostSpace>
    struct Prod;
    template<class ScalarType, class Space = HostSpace>
    struct Min;
    template<class ScalarType, class Space = HostSpace>
    struct Max;
    template<class ScalarType, class Space = HostSpace>
    struct MinMax;
    template<class ScalarType, class Index, class Space = HostSpace>
    struct MinLoc;
    template<class ScalarType, class Index, class Space = HostSpace>
    struct MaxLoc;
    template<class ScalarType, class Index, class Space = HostSpace>
    struct MinMaxLoc;
    template<class ScalarType, class Space = HostSpace>
    struct BAnd;
    template<class ScalarType, class Space = HostSpace>
    struct BOr;
    template<class ScalarType, class Space = HostSpace>
    struct LAnd;
    template<class ScalarType, class Space = HostSpace>
    struct LOr;

    template<class Scalar, class Index, class Space = HostSpace>
    struct MaxFirstLoc;
    template<class Scalar, class Index, class ComparatorType,
            class Space = HostSpace>
    struct MaxFirstLocCustomComparator;

    template<class Scalar, class Index, class Space = HostSpace>
    struct MinFirstLoc;
    template<class Scalar, class Index, class ComparatorType,
            class Space = HostSpace>
    struct MinFirstLocCustomComparator;

    template<class Scalar, class Index, class Space = HostSpace>
    struct MinMaxFirstLastLoc;
    template<class Scalar, class Index, class ComparatorType,
            class Space = HostSpace>
    struct MinMaxFirstLastLocCustomComparator;

    template<class Index, class Space = HostSpace>
    struct FirstLoc;
    template<class Index, class Space = HostSpace>
    struct LastLoc;
    template<class Index, class Space = HostSpace>
    struct StdIsPartitioned;
    template<class Index, class Space = HostSpace>
    struct StdPartitionPoint;
}  // namespace flare

#endif  // FLARE_CORE_FWD_H_
