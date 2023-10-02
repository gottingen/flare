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

#ifndef FLARE_BACKEND_CUDA_CUDA_H_
#define FLARE_BACKEND_CUDA_CUDA_H_

#include <flare/core/defines.h>

#if defined(FLARE_ON_CUDA_DEVICE)

#include <flare/core_fwd.h>

#include <iosfwd>
#include <vector>

#include <flare/core/policy/analyze_policy.h>
#include <flare/backend/cuda/cuda_space.h>
#include <flare/backend/cuda/cuda_error.h>  // CUDA_SAFE_CALL

#include <flare/core/parallel/parallel.h>
#include <flare/core/memory/layout.h>
#include <flare/core/memory/scratch_space.h>
#include <flare/core/memory/memory_traits.h>
#include <flare/core/memory/host_shared_ptr.h>
#include <flare/core/common/initialization_settings.h>


namespace flare::detail {
    class CudaInternal;
}  // namespace flare::detail

namespace flare::detail {
    namespace experimental {
        enum class CudaLaunchMechanism : unsigned {
            Default = 0,
            ConstantMemory = 1,
            GlobalMemory = 2,
            LocalMemory = 4
        };

        constexpr inline CudaLaunchMechanism operator|(CudaLaunchMechanism p1,
                                                       CudaLaunchMechanism p2) {
            return static_cast<CudaLaunchMechanism>(static_cast<unsigned>(p1) |
                                                    static_cast<unsigned>(p2));
        }

        constexpr inline CudaLaunchMechanism operator&(CudaLaunchMechanism p1,
                                                       CudaLaunchMechanism p2) {
            return static_cast<CudaLaunchMechanism>(static_cast<unsigned>(p1) &
                                                    static_cast<unsigned>(p2));
        }

        template<CudaLaunchMechanism l>
        struct CudaDispatchProperties {
            CudaLaunchMechanism launch_mechanism = l;
        };
    }  // namespace experimental

    enum class ManageStream : bool {
        no, yes
    };

}  // namespace flare::detail
namespace flare {
    /// \class Cuda
    /// \brief flare Execution Space that uses CUDA to run on GPUs.
    ///
    /// An "execution space" represents a parallel execution model.  It tells flare
    /// how to parallelize the execution of kernels in a parallel_for or
    /// parallel_reduce.  For example, the Threads execution space uses
    /// C++11 threads on a CPU, the OpenMP execution space uses the OpenMP language
    /// extensions, and the Serial execution space executes "parallel" kernels
    /// sequentially.  The Cuda execution space uses NVIDIA's CUDA programming
    /// model to execute kernels in parallel on GPUs.
    class Cuda {
    public:
        //! \name Type declarations that all flare execution spaces must provide.
        //@{

        //! Tag this class as a flare execution space
        using execution_space = Cuda;

        //! This execution space's preferred memory space.
        using memory_space = CudaSpace;

        //! This execution space preferred device_type
        using device_type = flare::Device<execution_space, memory_space>;

        //! The size_type best suited for this execution space.
        using size_type = memory_space::size_type;

        //! This execution space's preferred array layout.
        using array_layout = LayoutLeft;

        //!
        using scratch_memory_space = ScratchMemorySpace<Cuda>;

        //@}
        //--------------------------------------------------
        //! \name Functions that all flare devices must implement.
        //@{

        /// \brief True if and only if this method is being called in a
        ///   thread-parallel function.
        FLARE_INLINE_FUNCTION static int in_parallel() {
#if defined(__CUDA_ARCH__)
            return true;
#else
            return false;
#endif
        }

        /** \brief  Set the device in a "sleep" state.
         *
         * This function sets the device in a "sleep" state in which it is
         * not ready for work.  This may consume less resources than if the
         * device were in an "awake" state, but it may also take time to
         * bring the device from a sleep state to be ready for work.
         *
         * \return True if the device is in the "sleep" state, else false if
         *   the device is actively working and could not enter the "sleep"
         *   state.
         */
        static bool sleep();

        /// \brief Wake the device from the 'sleep' state so it is ready for work.
        ///
        /// \return True if the device is in the "ready" state, else "false"
        ///  if the device is actively working (which also means that it's
        ///  awake).
        static bool wake();

        /// \brief Wait until all dispatched functors complete.
        ///
        /// The parallel_for or parallel_reduce dispatch of a functor may
        /// return asynchronously, before the functor completes.  This
        /// method does not return until all dispatched functors on this
        /// device have completed.
        static void impl_static_fence(const std::string &name);

        void fence(const std::string &name =
        "flare::Cuda::fence(): Unnamed Instance Fence") const;

        /** \brief  Return the maximum amount of concurrency.  */
        int concurrency() const;

        //! Print configuration information to the given output stream.
        void print_configuration(std::ostream &os, bool verbose = false) const;

        //@}
        //--------------------------------------------------
        //! \name  Cuda space instances

        Cuda();

        Cuda(cudaStream_t stream,
             detail::ManageStream manage_stream = detail::ManageStream::no);

        FLARE_DEPRECATED Cuda(cudaStream_t stream, bool manage_stream);

        //--------------------------------------------------------------------------
        //! Free any resources being consumed by the device.
        static void impl_finalize();

        //! Has been initialized
        static int impl_is_initialized();

        //! Initialize, telling the CUDA run-time library which device to use.
        static void impl_initialize(InitializationSettings const &);

        /// \brief Cuda device architecture of the selected device.
        ///
        /// This matches the __CUDA_ARCH__ specification.
        static size_type device_arch();

        //! Query device count.
        static size_type detect_device_count();

        /** \brief  Detect the available devices and their architecture
         *          as defined by the __CUDA_ARCH__ specification.
         */
        static std::vector<unsigned> detect_device_arch();

        cudaStream_t cuda_stream() const;

        int cuda_device() const;

        const cudaDeviceProp &cuda_device_prop() const;

        //@}
        //--------------------------------------------------------------------------

        static const char *name();

        inline detail::CudaInternal *impl_internal_space_instance() const {
            return m_space_instance.get();
        }

        uint32_t impl_instance_id() const noexcept;

    private:
        friend bool operator==(Cuda const &lhs, Cuda const &rhs) {
            return lhs.impl_internal_space_instance() ==
                   rhs.impl_internal_space_instance();
        }

        friend bool operator!=(Cuda const &lhs, Cuda const &rhs) {
            return !(lhs == rhs);
        }

        flare::detail::HostSharedPtr<detail::CudaInternal> m_space_instance;
    };
}  // namespace flare
namespace flare::Tools::experimental {
    template<>
    struct DeviceTypeTraits<Cuda> {
        /// \brief An ID to differentiate (for example) Serial from OpenMP in Tooling
        static constexpr DeviceType id = DeviceType::Cuda;

        static int device_id(const Cuda &exec) { return exec.cuda_device(); }
    };

}  // namespace flare::Tools::experimental

namespace flare::detail {
    template<>
    struct MemorySpaceAccess<flare::CudaSpace,
            flare::Cuda::scratch_memory_space> {
        enum : bool {
            assignable = false
        };
        enum : bool {
            accessible = true
        };
        enum : bool {
            deepcopy = false
        };
    };

}  // namespace flare::detail

#endif  // FLARE_ON_CUDA_DEVICE
#endif  // FLARE_BACKEND_CUDA_CUDA_H_
