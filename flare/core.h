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

#ifndef FLARE_CORE_H_
#define FLARE_CORE_H_

//----------------------------------------------------------------------------
// In the case windows.h is included before core.hpp there might be
// errors due to the potentially defined macros with name "min" and "max" in
// windows.h. These collide with the use of "min" and "max" in names inside
// flare. The macros will be redefined at the end of core.hpp
#if defined(min)
#pragma push_macro("min")
#undef min
#define FLARE_IMPL_PUSH_MACRO_MIN
#endif
#if defined(max)
#pragma push_macro("max")
#undef max
#define FLARE_IMPL_PUSH_MACRO_MAX
#endif

//----------------------------------------------------------------------------
// Include the execution space header files for the enabled execution spaces.

#include <flare/core_fwd.h>

#if defined(FLARE_ENABLE_SERIAL)
#include <flare/backend/serial/serial.h>
#include <flare/backend/serial/serial_mdrange_policy.h>
#include <flare/backend/serial/serial_zero_memset.h>
#endif

#if defined(FLARE_ENABLE_THREADS)
#include <flare/backend/threads/threads.h>
#include <flare/backend/threads/threads_mdrange_policy.h>
#endif

#if defined(FLARE_ENABLE_OPENMP)

#include <flare/backend/openmp/openmp.h>
#include <flare/backend/openmp/openmp_mdrange_policy.h>
#include <flare/backend/openmp/openmp_unique_token.h>
#include <flare/backend/openmp/openmp_parallel_for.h>
#include <flare/backend/openmp/openmp_parallel_reduce.h>
#include <flare/backend/openmp/openmp_parallel_scan.h>

#endif

#ifdef FLARE_ENABLE_HBWSPACE
#include <flare/core/memory/hbw_space.h>
#endif

#if defined(FLARE_ON_CUDA_DEVICE)
#include <flare/backend/cuda/cuda.h>
#include <flare/backend/cuda/cuda_half_impl_type.h>
#include <flare/backend/cuda/cuda_half_conversion.h>
#include <flare/backend/cuda/cuda_parallel_mdrange.h>
#include <flare/backend/cuda/cuda_parallel_range.h>
#include <flare/backend/cuda/cuda_parallel_team.h>
#include <flare/backend/cuda/cuda_kernel_launch.h>
#include <flare/backend/cuda/cuda_instance.h>
#include <flare/backend/cuda/cuda_view.h>
#include <flare/backend/cuda/cuda_team.h>
#include <flare/backend/cuda/cuda_mdrange_policy.h>
#include <flare/backend/cuda/cuda_unique_token.h>
#include <flare/backend/cuda/cuda_zero_memset.h>
#endif

#include <flare/core/half.h>
#include <flare/core/memory/anonymous_space.h>
#include <flare/core/memory/logical_spaces.h>
#include <flare/core/pair.h>
#include <flare/core/common/min_max_clamp.h>
#include <flare/core/mathematical_constants.h>
#include <flare/core/mathematical_functions.h>
#include <flare/core/mathematical_special_functions.h>
#include <flare/core/numeric_traits.h>
#include <flare/core/bit_manipulation.h>
#include <flare/core/memory/memory_pool.h>
#include <flare/core/array.h>
#include <flare/core/tensor/view.h>
#include <flare/core/tensor/vectorization.h>
#include <flare/core/atomic.h>
#include <flare/core/memory/hwloc.h>
#include <flare/timer.h>
#include <flare/core/policy/tuners.h>
#include <flare/core/complex.h>
#include <flare/core/simd_traits.h>
#include <flare/core/tensor/copy_tensors.h>
#include <flare/core/common/team_mdpolicy.h>
#include <flare/core/common/initialization_settings.h>
#include <functional>
#include <iosfwd>
#include <memory>
#include <vector>

//----------------------------------------------------------------------------

namespace flare {

    void initialize(int &argc, char *argv[]);

    void initialize(
            InitializationSettings const &settings = InitializationSettings());

    namespace detail {

        void pre_initialize(const InitializationSettings &settings);

        void post_initialize(const InitializationSettings &settings);

        void pre_finalize();

        void post_finalize();

        void declare_configuration_metadata(const std::string &category,
                                            const std::string &key,
                                            const std::string &value);

    }  // namespace detail

    [[nodiscard]] bool is_initialized() noexcept;

    [[nodiscard]] bool is_finalized() noexcept;

    [[nodiscard]] int device_id() noexcept;

    [[nodiscard]] int num_threads() noexcept;

    bool show_warnings() noexcept;

    bool tune_internals() noexcept;

    /** \brief  Finalize the spaces that were initialized via flare::initialize */
    void finalize();

    /**
     * \brief Push a user-defined function to be called in
     *   flare::finalize, before any flare state is finalized.
     *
     * \warning Only call this after flare::initialize, but before
     *   flare::finalize.
     *
     * This function is the flare analog to std::atexit.  If you call
     * this with a function f, then your function will get called when
     * flare::finalize is called.  Specifically, it will be called BEFORE
     * flare does any finalization.  This means that all execution
     * spaces, memory spaces, etc. that were initialized will still be
     * initialized when your function is called.
     *
     * Just like std::atexit, if you call push_finalize_hook in sequence
     * with multiple functions (f, g, h), flare::finalize will call them
     * in reverse order (h, g, f), as if popping a stack.  Furthermore,
     * just like std::atexit, if any of your functions throws but does not
     * catch an exception, flare::finalize will call std::terminate.
     */
    void push_finalize_hook(std::function<void()> f);

    void fence(const std::string &name /*= "flare::fence: Unnamed Global Fence"*/);

    /** \brief Print "Bill of Materials" */
    void print_configuration(std::ostream &os, bool verbose = false);

}  // namespace flare

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace flare {

    /* Allocate memory from a memory space.
     * The allocation is tracked in flare memory tracking system, so
     * leaked memory can be identified.
     */
    template<class Space = flare::DefaultExecutionSpace::memory_space>
    inline void *flare_malloc(const std::string &arg_alloc_label,
                               const size_t arg_alloc_size) {
        using MemorySpace = typename Space::memory_space;
        return detail::SharedAllocationRecord<MemorySpace>::allocate_tracked(
                MemorySpace(), arg_alloc_label, arg_alloc_size);
    }

    template<class Space = flare::DefaultExecutionSpace::memory_space>
    inline void *flare_malloc(const size_t arg_alloc_size) {
        using MemorySpace = typename Space::memory_space;
        return detail::SharedAllocationRecord<MemorySpace>::allocate_tracked(
                MemorySpace(), "no-label", arg_alloc_size);
    }

    template<class Space = flare::DefaultExecutionSpace::memory_space>
    inline void flare_free(void *arg_alloc) {
        using MemorySpace = typename Space::memory_space;
        return detail::SharedAllocationRecord<MemorySpace>::deallocate_tracked(
                arg_alloc);
    }

    template<class Space = flare::DefaultExecutionSpace::memory_space>
    inline void *flare_realloc(void *arg_alloc, const size_t arg_alloc_size) {
        using MemorySpace = typename Space::memory_space;
        return detail::SharedAllocationRecord<MemorySpace>::reallocate_tracked(
                arg_alloc, arg_alloc_size);
    }

}  // namespace flare

namespace flare {

/** \brief  ScopeGuard
 *  Some user scope issues have been identified with some flare::finalize
 * calls; ScopeGuard aims to correct these issues.
 *
 *  Two requirements for ScopeGuard:
 *     if flare::is_initialized() in the constructor, don't call
 * flare::initialize or flare::finalize it is not copyable or assignable
 */
    namespace detail {

        inline std::string scopeguard_correct_usage() {
            return std::string(
                    "Do instead:\n"
                    "  std::unique_ptr<flare::ScopeGuard> guard =\n"
                    "    !flare::is_initialized() && !flare::is_finalized()?\n"
                    "    new ScopeGuard(argc,argv) : nullptr;\n");
        }

        inline std::string scopeguard_create_while_initialized_warning() {
            return std::string(
                    "flare Error: Creating a ScopeGuard while flare is initialized "
                    "is illegal.\n")
                    .append(scopeguard_correct_usage());
        }

        inline std::string scopeguard_create_after_finalize_warning() {
            return std::string(
                    "flare Error: Creating a ScopeGuard after flare was finalized "
                    "is illegal.\n")
                    .append(scopeguard_correct_usage());
        }

        inline std::string scopeguard_destruct_after_finalize_warning() {
            return std::string(
                    "flare Error: Destroying a ScopeGuard after flare was finalized "
                    "is illegal.\n")
                    .append(scopeguard_correct_usage());
        }

    }  // namespace detail

    class FLARE_ATTRIBUTE_NODISCARD ScopeGuard {
    public:
        template<class... Args>
#if defined(__has_cpp_attribute) && __has_cpp_attribute(nodiscard) >= 201907
        [[nodiscard]]
#endif
        ScopeGuard(Args &&... args) {
            if (is_initialized()) {
                flare::abort(
                        detail::scopeguard_create_while_initialized_warning().c_str());
            }
            if (is_finalized()) {
                flare::abort(detail::scopeguard_create_after_finalize_warning().c_str());
            }
            initialize(static_cast<Args &&>(args)...);
        }

        ~ScopeGuard() {
            if (is_finalized()) {
                flare::abort(detail::scopeguard_destruct_after_finalize_warning().c_str());
            }
            finalize();
        }

        ScopeGuard &operator=(const ScopeGuard &) = delete;

        ScopeGuard &operator=(ScopeGuard &&) = delete;

        ScopeGuard(const ScopeGuard &) = delete;

        ScopeGuard(ScopeGuard &&) = delete;
    };

}  // namespace flare

namespace flare {
    namespace experimental {
        // Partitioning an Execution Space: expects space and integer arguments for
        // relative weight
        //   Customization point for backends
        //   Default behavior is to return the passed in instance
        template<class ExecSpace, class... Args>
        std::vector<ExecSpace> partition_space(ExecSpace const &space, Args...) {
            static_assert(is_execution_space<ExecSpace>::value,
                          "flare Error: partition_space expects an Execution Space as "
                          "first argument");
            static_assert(
                    (... && std::is_arithmetic_v<Args>),
                    "flare Error: partitioning arguments must be integers or floats");
            std::vector<ExecSpace> instances(sizeof...(Args));
            for (int s = 0; s < int(sizeof...(Args)); s++) instances[s] = space;
            return instances;
        }

        template<class ExecSpace, class T>
        std::vector<ExecSpace> partition_space(ExecSpace const &space,
                                               std::vector<T> const &weights) {
            static_assert(is_execution_space<ExecSpace>::value,
                          "flare Error: partition_space expects an Execution Space as "
                          "first argument");
            static_assert(
                    std::is_arithmetic<T>::value,
                    "flare Error: partitioning arguments must be integers or floats");

            std::vector<ExecSpace> instances(weights.size());
            for (int s = 0; s < int(weights.size()); s++) instances[s] = space;
            return instances;
        }
    }  // namespace experimental
}  // namespace flare

#include <flare/core/tensor/crs.h>
#include <flare/core/graph/work_graph_policy.h>
// Including this in parallel_reduce.h led to a circular dependency
// because flare::Sum is used in combined_reducer.hpp and the default.
// The real answer is to finally break up parallel_reduce.h into
// smaller parts...
#include <flare/core/common/combined_reducer.h>
// Yet another workaround to deal with circular dependency issues because the
// implementation of the RAII wrapper is using flare::single.
#include <flare/core/parallel/acquire_unique_token_impl.h>

//----------------------------------------------------------------------------
// Redefinition of the macros min and max if we pushed them at entry of
// core.hpp
#if defined(FLARE_IMPL_PUSH_MACRO_MIN)
#pragma pop_macro("min")
#undef FLARE_IMPL_PUSH_MACRO_MIN
#endif
#if defined(FLARE_IMPL_PUSH_MACRO_MAX)
#pragma pop_macro("max")
#undef FLARE_IMPL_PUSH_MACRO_MAX
#endif


#endif  // FLARE_CORE_H_
