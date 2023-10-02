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

#ifndef FLARE_CORE_COMMON_ERROR_H_
#define FLARE_CORE_COMMON_ERROR_H_

#include <string>
#include <iosfwd>
#include <flare/core/defines.h>

#ifdef FLARE_ON_CUDA_DEVICE
#include <flare/backend/cuda/cuda_abort.h>
#endif

namespace flare {
    namespace detail {

        [[noreturn]] void host_abort(const char *const);

#if defined(FLARE_ON_CUDA_DEVICE) && defined(__CUDA_ARCH__)

#if defined(FLARE_ENABLE_DEBUG_BOUNDS_CHECK)
        // required to workaround failures in random number generator unit tests with
        // pre-volta architectures
#define FLARE_IMPL_ABORT_NORETURN
#else
        // cuda_abort aborts when building for other platforms than macOS
#define FLARE_IMPL_ABORT_NORETURN [[noreturn]]
#endif

#elif defined(FLARE_COMPILER_NVHPC)

#define FLARE_IMPL_ABORT_NORETURN
#else
        // Host aborts
#define FLARE_IMPL_ABORT_NORETURN [[noreturn]]
#endif

#if (defined(FLARE_ON_CUDA_DEVICE) && defined(FLARE_ENABLE_DEBUG_BOUNDS_CHECK))
#define FLARE_IMPL_ABORT_NORETURN_DEVICE
#else
#define FLARE_IMPL_ABORT_NORETURN_DEVICE FLARE_IMPL_ABORT_NORETURN
#endif

#if defined(FLARE_ON_CUDA_DEVICE)
        FLARE_IMPL_ABORT_NORETURN_DEVICE inline FLARE_IMPL_DEVICE_FUNCTION void
        device_abort(const char *const msg) {
            ::flare::detail::cuda_abort(msg);
        }
#endif

        [[noreturn]] void throw_runtime_exception(const std::string &msg);

        void traceback_callstack(std::ostream &);

        std::string human_memory_size(size_t arg_bytes);

    }  // namespace detail

    namespace experimental {

        class RawMemoryAllocationFailure : public std::bad_alloc {
        public:
            enum class FailureMode {
                OutOfMemoryError,
                AllocationNotAligned,
                InvalidAllocationSize,
                MaximumCudaUVMAllocationsExceeded,
                Unknown
            };
            enum class AllocationMechanism {
                StdMalloc,
                CudaMalloc,
                CudaMallocManaged,
                CudaHostAlloc,
            };

        private:
            size_t m_attempted_size;
            size_t m_attempted_alignment;
            FailureMode m_failure_mode;
            AllocationMechanism m_mechanism;

        public:
            RawMemoryAllocationFailure(
                    size_t arg_attempted_size, size_t arg_attempted_alignment,
                    FailureMode arg_failure_mode = FailureMode::OutOfMemoryError,
                    AllocationMechanism arg_mechanism =
                    AllocationMechanism::StdMalloc) noexcept
                    : m_attempted_size(arg_attempted_size),
                      m_attempted_alignment(arg_attempted_alignment),
                      m_failure_mode(arg_failure_mode),
                      m_mechanism(arg_mechanism) {}

            RawMemoryAllocationFailure() noexcept = delete;

            RawMemoryAllocationFailure(RawMemoryAllocationFailure const &) noexcept =
            default;

            RawMemoryAllocationFailure(RawMemoryAllocationFailure &&) noexcept = default;

            RawMemoryAllocationFailure &operator=(
                    RawMemoryAllocationFailure const &) noexcept = default;

            RawMemoryAllocationFailure &operator=(
                    RawMemoryAllocationFailure &&) noexcept = default;

            ~RawMemoryAllocationFailure() noexcept override = default;

            [[nodiscard]] const char *what() const noexcept override {
                if (m_failure_mode == FailureMode::OutOfMemoryError) {
                    return "Memory allocation error: out of memory";
                } else if (m_failure_mode == FailureMode::AllocationNotAligned) {
                    return "Memory allocation error: allocation result was under-aligned";
                }

                return nullptr;  // unreachable
            }

            [[nodiscard]] size_t attempted_size() const noexcept {
                return m_attempted_size;
            }

            [[nodiscard]] size_t attempted_alignment() const noexcept {
                return m_attempted_alignment;
            }

            [[nodiscard]] AllocationMechanism allocation_mechanism() const noexcept {
                return m_mechanism;
            }

            [[nodiscard]] FailureMode failure_mode() const noexcept {
                return m_failure_mode;
            }

            void print_error_message(std::ostream &o) const;

            [[nodiscard]] std::string get_error_message() const;

            virtual void append_additional_error_information(std::ostream &) const {}
        };

    }  // end namespace experimental

}  // namespace flare

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace flare {

    FLARE_IMPL_ABORT_NORETURN FLARE_INLINE_FUNCTION void abort(
            const char *const message) {
        FLARE_IF_ON_HOST(::flare::detail::host_abort(message);)
        FLARE_IF_ON_DEVICE(::flare::detail::device_abort(message);)
    }

#undef FLARE_IMPL_ABORT_NORETURN

}  // namespace flare

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#if !defined(NDEBUG) || defined(FLARE_ENFORCE_CONTRACTS) || \
    defined(FLARE_ENABLE_DEBUG)
#define FLARE_EXPECTS(...)                                                    \
  {                                                                            \
    if (!bool(__VA_ARGS__)) {                                                  \
      ::flare::abort(                                                         \
          "flare contract violation:\n  "                                     \
          "  Expected precondition `" #__VA_ARGS__                             \
          "` evaluated false.\n"                                               \
          "Error at " FLARE_IMPL_TOSTRING(__FILE__) ":" FLARE_IMPL_TOSTRING( \
              __LINE__) " \n");                                                \
    }                                                                          \
  }
#define FLARE_ENSURES(...)                                                    \
  {                                                                            \
    if (!bool(__VA_ARGS__)) {                                                  \
      ::flare::abort(                                                         \
          "flare contract violation:\n  "                                     \
          "  Ensured postcondition `" #__VA_ARGS__                             \
          "` evaluated false.\n"                                               \
          "Error at " FLARE_IMPL_TOSTRING(__FILE__) ":" FLARE_IMPL_TOSTRING( \
              __LINE__) " \n");                                                \
    }                                                                          \
  }
// some projects already define this for themselves, so don't mess
// them up
#ifndef FLARE_ASSERT
#define FLARE_ASSERT(...)                                                     \
  {                                                                            \
    if (!bool(__VA_ARGS__)) {                                                  \
      ::flare::abort(                                                         \
          "flare contract violation:\n  "                                     \
          "  Asserted condition `" #__VA_ARGS__                                \
          "` evaluated false.\n"                                               \
          "Error at " FLARE_IMPL_TOSTRING(__FILE__) ":" FLARE_IMPL_TOSTRING( \
              __LINE__) " \n");                                                \
    }                                                                          \
  }
#endif  // ifndef FLARE_ASSERT
#else   // not debug mode
#define FLARE_EXPECTS(...)
#define FLARE_ENSURES(...)
#ifndef FLARE_ASSERT
#define FLARE_ASSERT(...)
#endif  // ifndef FLARE_ASSERT
#endif  // end debug mode ifdefs

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif  // FLARE_CORE_COMMON_ERROR_H_
