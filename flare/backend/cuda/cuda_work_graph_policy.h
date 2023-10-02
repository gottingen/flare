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

#ifndef FLARE_BACKEND_CUDA_CUDA_WORK_GRAPH_POLICY_H_
#define FLARE_BACKEND_CUDA_CUDA_WORK_GRAPH_POLICY_H_

#include <flare/backend/cuda/cuda.h>
#include <flare/backend/cuda/cuda_kernel_launch.h>

namespace flare::detail {

    template<class FunctorType, class... Traits>
    class ParallelFor<FunctorType, flare::WorkGraphPolicy<Traits...>,
            flare::Cuda> {
    public:
        using Policy = flare::WorkGraphPolicy<Traits...>;
        using Self = ParallelFor<FunctorType, Policy, flare::Cuda>;

    private:
        Policy m_policy;
        FunctorType m_functor;

        template<class TagType>
        __device__ inline std::enable_if_t<std::is_void<TagType>::value> exec_one(const std::int32_t w) const noexcept {
            m_functor(w);
        }

        template<class TagType>
        __device__ inline std::enable_if_t<!std::is_void<TagType>::value>
        exec_one(const std::int32_t w) const noexcept {
            const TagType t{};
            m_functor(t, w);
        }

    public:
        Policy const &get_policy() const { return m_policy; }

        __device__ inline void operator()() const noexcept {
            // The following makes most threads idle,
            // which helps significantly with throughput due to reducing conflict rates
            // on the work acquisition, updated based on perf experiments of the
            // static Fibonacci experiment on Volta
            if (0 == (threadIdx.y % 4)) {
                // Spin until COMPLETED_TOKEN.
                // END_TOKEN indicates no work is currently available.

                for (std::int32_t w = Policy::END_TOKEN;
                     Policy::COMPLETED_TOKEN != (w = m_policy.pop_work());) {
                    if (Policy::END_TOKEN != w) {
                        exec_one<typename Policy::work_tag>(w);
                        m_policy.completed_work(w);
                    }
                    // On pre-volta architectures we need a __syncwarp here to prevent
                    // infinite loops depending on the scheduling order above
#if defined(FLARE_ARCH_KEPLER) || defined(FLARE_ARCH_MAXWELL) || \
    defined(FLARE_ARCH_PASCAL)
                    __syncwarp(__activemask());
#endif
                }
            }
        }

        inline void execute() {
            const int warps_per_block = 4;
            const dim3 grid(flare::detail::cuda_internal_multiprocessor_count(), 1, 1);
            const dim3 block(1, flare::detail::CudaTraits::WarpSize, warps_per_block);
            const int shared = 0;

            flare::detail::CudaParallelLaunch<Self>(
                    *this, grid, block, shared, Cuda().impl_internal_space_instance());
        }

        inline ParallelFor(const FunctorType &arg_functor, const Policy &arg_policy)
                : m_policy(arg_policy), m_functor(arg_functor) {}
    };

}  // namespace flare::detail

#endif  // FLARE_BACKEND_CUDA_CUDA_WORK_GRAPH_POLICY_H_
