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

#ifndef FLARE_BACKEND_CUDA_CUDA_ERROR_H_
#define FLARE_BACKEND_CUDA_CUDA_ERROR_H_

#include <flare/core/defines.h>
#ifdef FLARE_ON_CUDA_DEVICE

#include <flare/core/common/error.h>
#include <flare/core/profile/profiling.h>
#include <iosfwd>

namespace flare {
namespace detail {

void cuda_stream_synchronize(
    const cudaStream_t stream,
    flare::Tools::experimental::SpecialSynchronizationCases reason,
    const std::string& name);
void cuda_device_synchronize(const std::string& name);
void cuda_stream_synchronize(const cudaStream_t stream,
                             const std::string& name);

[[noreturn]] void cuda_internal_error_throw(cudaError e, const char* name,
                                            const char* file = nullptr,
                                            const int line   = 0);

#ifndef FLARE_COMPILER_NVHPC
[[noreturn]]
#endif
             void cuda_internal_error_abort(cudaError e, const char* name,
                                            const char* file = nullptr,
                                            const int line   = 0);

inline void cuda_internal_safe_call(cudaError e, const char* name,
                                    const char* file = nullptr,
                                    const int line   = 0) {
  // 1. Success -> normal continuation.
  // 2. Error codes for which, to continue using CUDA, the process must be
  //    terminated and relaunched -> call abort on the host-side.
  // 3. Any other error code -> throw a runtime error.
  switch (e) {
    case cudaSuccess: break;
    case cudaErrorIllegalAddress:
    case cudaErrorAssert:
    case cudaErrorHardwareStackError:
    case cudaErrorIllegalInstruction:
    case cudaErrorMisalignedAddress:
    case cudaErrorInvalidAddressSpace:
    case cudaErrorInvalidPc:
    case cudaErrorLaunchFailure:
      cuda_internal_error_abort(e, name, file, line);
      break;
    default: cuda_internal_error_throw(e, name, file, line); break;
  }
}

#define FLARE_IMPL_CUDA_SAFE_CALL(call) \
  flare::detail::cuda_internal_safe_call(call, #call, __FILE__, __LINE__)

}  // namespace detail

namespace experimental {

class CudaRawMemoryAllocationFailure : public RawMemoryAllocationFailure {
 private:
  using base_t = RawMemoryAllocationFailure;

  cudaError_t m_error_code = cudaSuccess;

  static FailureMode get_failure_mode(cudaError_t error_code) {
    switch (error_code) {
      case cudaErrorMemoryAllocation: return FailureMode::OutOfMemoryError;
      case cudaErrorInvalidValue: return FailureMode::InvalidAllocationSize;
      // TODO handle cudaErrorNotSupported for cudaMallocManaged
      default: return FailureMode::Unknown;
    }
  }

 public:
  // using base_t::base_t;
  // would trigger
  //
  // error: cannot determine the exception specification of the default
  // constructor due to a circular dependency
  //
  // using NVCC 9.1 and gcc 7.4
  CudaRawMemoryAllocationFailure(
      size_t arg_attempted_size, size_t arg_attempted_alignment,
      FailureMode arg_failure_mode = FailureMode::OutOfMemoryError,
      AllocationMechanism arg_mechanism =
          AllocationMechanism::StdMalloc) noexcept
      : base_t(arg_attempted_size, arg_attempted_alignment, arg_failure_mode,
               arg_mechanism) {}

  CudaRawMemoryAllocationFailure(size_t arg_attempted_size,
                                 cudaError_t arg_error_code,
                                 AllocationMechanism arg_mechanism) noexcept
      : base_t(arg_attempted_size, /* CudaSpace doesn't handle alignment? */ 1,
               get_failure_mode(arg_error_code), arg_mechanism),
        m_error_code(arg_error_code) {}

  void append_additional_error_information(std::ostream& o) const override;
};

}  // end namespace experimental

}  // namespace flare

#endif  // FLARE_ON_CUDA_DEVICE
#endif  // FLARE_BACKEND_CUDA_CUDA_ERROR_H_
