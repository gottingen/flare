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


#include <cstring>
#include <cstdlib>

#include <iostream>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <flare/core/common/error.h>
#include <flare/core/common/stacktrace.h>
#include <flare/backend/cuda/cuda_error.h>

namespace flare::detail {
    void traceback_callstack(std::ostream &msg) {
#ifdef FLARE_IMPL_ENABLE_STACKTRACE
        msg << "\nBacktrace:\n";
        save_stacktrace();
        print_demangled_saved_stacktrace(msg);
#else
        msg << "\nTraceback functionality not available\n";
#endif
    }

    void throw_runtime_exception(const std::string &msg) {
        throw std::runtime_error(msg);
    }

    void host_abort(const char *const message) {
        std::cerr << message;
        traceback_callstack(std::cerr);
        ::abort();
    }

    std::string human_memory_size(size_t arg_bytes) {
        double bytes = arg_bytes;
        const double K = 1024;
        const double M = K * 1024;
        const double G = M * 1024;

        std::ostringstream out;
        if (bytes < K) {
            out << std::setprecision(4) << bytes << " B";
        } else if (bytes < M) {
            bytes /= K;
            out << std::setprecision(4) << bytes << " K";
        } else if (bytes < G) {
            bytes /= M;
            out << std::setprecision(4) << bytes << " M";
        } else {
            bytes /= G;
            out << std::setprecision(4) << bytes << " G";
        }
        return out.str();
    }

}  // namespace flare::detail
namespace flare {
    void experimental::RawMemoryAllocationFailure::print_error_message(
            std::ostream &o) const {
        o << "Allocation of size " << detail::human_memory_size(m_attempted_size);
        o << " failed";
        switch (m_failure_mode) {
            case FailureMode::OutOfMemoryError:
                o << ", likely due to insufficient memory.";
                break;
            case FailureMode::AllocationNotAligned:
                o << " because the allocation was improperly aligned.";
                break;
            case FailureMode::InvalidAllocationSize:
                o << " because the requested allocation size is not a valid size for the"
                     " requested allocation mechanism (it's probably too large).";
                break;
                // TODO move this to the subclass for Cuda-related things
            case FailureMode::MaximumCudaUVMAllocationsExceeded:
                o << " because the maximum Cuda UVM allocations was exceeded.";
                break;
            case FailureMode::Unknown:
                o << " because of an unknown error.";
                break;
        }
        o << "  (The allocation mechanism was ";
        switch (m_mechanism) {
            case AllocationMechanism::StdMalloc:
                o << "standard malloc().";
                break;
            case AllocationMechanism::CudaMalloc:
                o << "cudaMalloc().";
                break;
            case AllocationMechanism::CudaMallocManaged:
                o << "cudaMallocManaged().";
                break;
            case AllocationMechanism::CudaHostAlloc:
                o << "cudaHostAlloc().";
                break;
            default:
                o << "unsupported.";
        }
        append_additional_error_information(o);
        o << ")" << std::endl;
    }

    std::string experimental::RawMemoryAllocationFailure::get_error_message()
    const {
        std::ostringstream out;
        print_error_message(out);
        return out.str();
    }

}  // namespace flare

