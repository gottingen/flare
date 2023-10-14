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

#ifndef FLARE_BACKEND_CUDA_CUDA_TENSOR_H_
#define FLARE_BACKEND_CUDA_CUDA_TENSOR_H_

#include <flare/core/defines.h>

#if defined(FLARE_ON_CUDA_DEVICE)

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace flare::detail {

    template<typename ValueType, typename AliasType>
    struct CudaLDGFetch {
        const ValueType *m_ptr;

        template<typename iType>
        FLARE_FUNCTION ValueType operator[](const iType &i) const {
#if defined(FLARE_ARCH_KEPLER30) || defined(FLARE_ARCH_KEPLER32)
            return m_ptr[i];
#else
            FLARE_IF_ON_DEVICE(
                    (AliasType v = __ldg(reinterpret_cast<const AliasType *>(&m_ptr[i]));
                    return *(reinterpret_cast<ValueType *>(&v));))
            FLARE_IF_ON_HOST((return m_ptr[i];))
#endif
        }

        FLARE_FUNCTION
        operator const ValueType *() const { return m_ptr; }

        FLARE_DEFAULTED_FUNCTION
        CudaLDGFetch() = default;

        FLARE_FUNCTION
        explicit CudaLDGFetch(const ValueType *const arg_ptr) : m_ptr(arg_ptr) {}

        FLARE_FUNCTION
        CudaLDGFetch(CudaLDGFetch const rhs, size_t offset)
                : m_ptr(rhs.m_ptr + offset) {}
    };


    /** \brief  Replace Default TensorDataHandle with CudaLDGFetch
     * specialization if 'const' value type, CudaSpace and random access.
     */
    template<class Traits>
    class TensorDataHandle<
            Traits, std::enable_if_t<(
                    // Is Cuda memory space
                    (std::is_same<typename Traits::memory_space,
                            flare::CudaSpace>::value ||
                     std::is_same<typename Traits::memory_space,
                             flare::CudaUVMSpace>::value) &&
                    // Is a trivial const value of 4, 8, or 16 bytes
                    std::is_trivial<typename Traits::const_value_type>::value &&
                    std::is_same<typename Traits::const_value_type,
                            typename Traits::value_type>::value &&
                    (sizeof(typename Traits::const_value_type) == 4 ||
                     sizeof(typename Traits::const_value_type) == 8 ||
                     sizeof(typename Traits::const_value_type) == 16) &&
                    // Random access trait
                    (Traits::memory_traits::is_random_access != 0))>> {
    public:
        using track_type = flare::detail::SharedAllocationTracker;

        using value_type = typename Traits::const_value_type;
        using return_type = typename Traits::const_value_type;  // NOT a reference

        using alias_type = std::conditional_t<
                (sizeof(value_type) == 4), int,
                std::conditional_t<
                        (sizeof(value_type) == 8), ::int2,
                        std::conditional_t<(sizeof(value_type) == 16), ::int4, void>>>;

        using handle_type = flare::detail::CudaLDGFetch<value_type, alias_type>;

        FLARE_INLINE_FUNCTION
        static handle_type const &assign(handle_type const &arg_handle,
                                         track_type const & /* arg_tracker */) {
            return arg_handle;
        }

        FLARE_INLINE_FUNCTION
        static handle_type const assign(handle_type const &arg_handle,
                                        size_t offset) {
            return handle_type(arg_handle, offset);
        }

        FLARE_INLINE_FUNCTION
        static handle_type assign(value_type *arg_data_ptr,
                                  track_type const & /*arg_tracker*/) {
            if (arg_data_ptr == nullptr) return handle_type();
            return handle_type(arg_data_ptr);
        }
    };

}  // namespace flare::detail

#endif   // FLARE_ON_CUDA_DEVICE
#endif  // FLARE_BACKEND_CUDA_CUDA_TENSOR_H_
