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
//
// Created by jeff on 23-10-6.
//

#ifndef FLARE_CORE_EXEC_SPACE_UTILS_H_
#define FLARE_CORE_EXEC_SPACE_UTILS_H_

#include <flare/core.h>

namespace flare::detail {

    enum ExecSpaceType {
        Exec_SERIAL,
        Exec_OMP,
        Exec_THREADS,
        Exec_CUDA
    };

    template<typename ExecutionSpace>
    FLARE_FORCEINLINE_FUNCTION ExecSpaceType flare_get_exec_space_type() {
        ExecSpaceType exec_space = Exec_SERIAL;
#if defined(FLARE_ENABLE_SERIAL)
        if (std::is_same<flare::Serial, ExecutionSpace>::value) {
            exec_space = Exec_SERIAL;
        }
#endif

#if defined(FLARE_ENABLE_THREADS)
        if (std::is_same<flare::Threads, ExecutionSpace>::value) {
            exec_space = Exec_THREADS;
        }
#endif

#if defined(FLARE_ENABLE_OPENMP)
        if (std::is_same<flare::OpenMP, ExecutionSpace>::value) {
            exec_space = Exec_OMP;
        }
#endif

#if defined(FLARE_ON_CUDA_DEVICE)
        if (std::is_same<flare::Cuda, ExecutionSpace>::value) {
            exec_space = Exec_CUDA;
        }
#endif
        return exec_space;
    }

////////////////////////////////////////////////////////////////////////////////
// GPU Exec Space Utils
////////////////////////////////////////////////////////////////////////////////

    template<typename ExecutionSpace>
    constexpr FLARE_INLINE_FUNCTION bool flare_is_gpu_exec_space() {
        return false;
    }

#ifdef FLARE_ON_CUDA_DEVICE
    template <>
    constexpr FLARE_INLINE_FUNCTION bool flare_is_gpu_exec_space<flare::Cuda>() {
        return true;
    }
#endif



    ////////////////////////////////////////////////////////////////////////////////
    // x86_64 Memory Space Utils
    ////////////////////////////////////////////////////////////////////////////////

    template<typename ExecutionSpace>
    constexpr FLARE_INLINE_FUNCTION bool flare_is_x86_64_mem_space() {
        return false;
    }

#if __x86_64__

    template<>
    constexpr FLARE_INLINE_FUNCTION bool
    flare_is_x86_64_mem_space<flare::HostSpace>() {
        return true;
    }

#endif  // x86_64 architectures

    ////////////////////////////////////////////////////////////////////////////////
    // A64FX Memory Space Utils
    ////////////////////////////////////////////////////////////////////////////////

    template<typename ExecutionSpace>
    constexpr FLARE_INLINE_FUNCTION bool flare_is_a64fx_mem_space() {
        return false;
    }

#if defined(__ARM_ARCH_ISA_A64)
    template <>
    constexpr FLARE_INLINE_FUNCTION bool flare_is_a64fx_mem_space<flare::HostSpace>() {
        return true;
    }
#endif  // a64fx architectures

        // Host function to determine free and total device memory.
        // Will throw if execution space doesn't support this.
    template<typename MemorySpace>
    inline void flare_get_free_total_memory(size_t & /* free_mem */,
                                         size_t & /* total_mem */) {
        std::ostringstream oss;
        oss << "Error: memory space " << MemorySpace::name()
            << " does not support querying free/total memory.";
        throw std::runtime_error(oss.str());
    }

    // Host function to determine free and total device memory.
    // Will throw if execution space doesn't support this.
    template<typename MemorySpace>
    inline void flare_get_free_total_memory(size_t & /* free_mem */,
                                         size_t & /* total_mem */,
                                         int /* n_streams */) {
        std::ostringstream oss;
        oss << "Error: memory space " << MemorySpace::name()
            << " does not support querying free/total memory.";
        throw std::runtime_error(oss.str());
    }

#ifdef FLARE_ON_CUDA_DEVICE
    template <>
    inline void flare_get_free_total_memory<flare::CudaSpace>(size_t& free_mem,
                                                        size_t& total_mem,
                                                        int n_streams) {
        cudaMemGetInfo(&free_mem, &total_mem);
        free_mem /= n_streams;
        total_mem /= n_streams;
    }
    template <>
    inline void flare_get_free_total_memory<flare::CudaSpace>(size_t& free_mem,
                                                        size_t& total_mem) {
        flare_get_free_total_memory<flare::CudaSpace>(free_mem, total_mem, 1);
    }
    template <>
    inline void flare_get_free_total_memory<flare::CudaUVMSpace>(size_t& free_mem,
                                                           size_t& total_mem,
                                                           int n_streams) {
        flare_get_free_total_memory<flare::CudaSpace>(free_mem, total_mem, n_streams);
    }
    template <>
    inline void flare_get_free_total_memory<flare::CudaUVMSpace>(size_t& free_mem,
                                                           size_t& total_mem) {
        flare_get_free_total_memory<flare::CudaUVMSpace>(free_mem, total_mem, 1);
    }
    template <>
    inline void flare_get_free_total_memory<flare::CudaHostPinnedSpace>(
    size_t& free_mem, size_t& total_mem, int n_streams) {
        flare_get_free_total_memory<flare::CudaSpace>(free_mem, total_mem, n_streams);
    }
    template <>
    inline void flare_get_free_total_memory<flare::CudaHostPinnedSpace>(
    size_t& free_mem, size_t& total_mem) {
        flare_get_free_total_memory<flare::CudaHostPinnedSpace>(free_mem, total_mem, 1);
    }
#endif


    template<typename ExecSpace>
    inline int flare_get_max_vector_size() {
        return flare::TeamPolicy<ExecSpace>::vector_length_max();
    }

    inline int flare_get_suggested_vector_size(const size_t nr, const size_t nnz,
                                            const ExecSpaceType exec_space) {
        int suggested_vector_size_ = 1;
        int max_vector_size = 1;
        switch (exec_space) {
            case Exec_CUDA:
                max_vector_size = 32;
                break;
            default:;
        }
        switch (exec_space) {
            default:
                break;
            case Exec_SERIAL:
            case Exec_OMP:
            case Exec_THREADS:
                break;
            case Exec_CUDA:
                if (nr > 0) suggested_vector_size_ = nnz / double(nr) + 0.5;
                if (suggested_vector_size_ < 3) {
                    suggested_vector_size_ = 2;
                } else if (suggested_vector_size_ <= 6) {
                    suggested_vector_size_ = 4;
                } else if (suggested_vector_size_ <= 12) {
                    suggested_vector_size_ = 8;
                } else if (suggested_vector_size_ <= 24) {
                    suggested_vector_size_ = 16;
                } else if (suggested_vector_size_ <= 48) {
                    suggested_vector_size_ = 32;
                } else {
                    suggested_vector_size_ = 64;
                }
                if (suggested_vector_size_ > max_vector_size)
                    suggested_vector_size_ = max_vector_size;
                break;
        }
        return suggested_vector_size_;
    }

    inline int flare_get_suggested_team_size(const int vector_size,
                                          const ExecSpaceType exec_space) {
        if (exec_space == Exec_CUDA) {
            // TODO: where this is used, tune the target value for
            // threads per block (but 256 is probably OK for CUDA and HIP)
            return 256 / vector_size;
        } else {
            return 1;
        }
    }

    namespace experimental {

        template<class ExecSpace>
        struct SpaceInstance {
            static ExecSpace create() { return ExecSpace(); }

            static void destroy(ExecSpace &) {}

            static bool overlap() { return false; }
        };

#ifdef FLARE_ON_CUDA_DEVICE
    template <>
    struct SpaceInstance<flare::Cuda> {
        static flare::Cuda create() {
            cudaStream_t stream;
            cudaStreamCreate(&stream);
            return flare::Cuda(stream);
        }
        static void destroy(flare::Cuda& space) {
            cudaStream_t stream = space.cuda_stream();
            cudaStreamDestroy(stream);
        }
        static bool overlap() {
            bool value          = true;
            auto local_rank_str = std::getenv("CUDA_LAUNCH_BLOCKING");
            if (local_rank_str) {
              value = (std::atoi(local_rank_str) == 0);
            }
            return value;
        }
    };
#endif


    }  // namespace experimental

}  // namespace flare::detail

#endif  // FLARE_CORE_EXEC_SPACE_UTILS_H_
