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

#ifndef FLARE_DYNAMIC_TENSOR_H_
#define FLARE_DYNAMIC_TENSOR_H_

#include <cstdio>

#include <flare/core.h>
#include <flare/core/common/error.h>

namespace flare {
    namespace experimental {

        namespace detail {

            /// Utility class to manage memory for chunked arrays on the host and
            /// device. Allocates/deallocates memory on both the host and device along with
            /// providing utilities for creating mirrors and deep copying between them.
            template<typename MemorySpace, typename ValueType>
            struct ChunkedArrayManager {
                using value_type = ValueType;
                using pointer_type = ValueType *;
                using track_type = flare::detail::SharedAllocationTracker;

                ChunkedArrayManager() = default;

                ChunkedArrayManager(ChunkedArrayManager const &) = default;

                ChunkedArrayManager(ChunkedArrayManager &&) = default;

                ChunkedArrayManager &operator=(ChunkedArrayManager &&) = default;

                ChunkedArrayManager &operator=(const ChunkedArrayManager &) = default;

                template<typename Space, typename Value>
                friend
                struct ChunkedArrayManager;

                template<typename Space, typename Value>
                inline ChunkedArrayManager(const ChunkedArrayManager<Space, Value> &rhs)
                        : m_valid(rhs.m_valid),
                          m_chunk_max(rhs.m_chunk_max),
                          m_chunks((ValueType **) (rhs.m_chunks)),
                          m_track(rhs.m_track),
                          m_chunk_size(rhs.m_chunk_size) {
                    static_assert(
                            flare::detail::MemorySpaceAccess<MemorySpace, Space>::assignable,
                            "Incompatible ChunkedArrayManager copy construction");
                }

                ChunkedArrayManager(const unsigned arg_chunk_max,
                                    const unsigned arg_chunk_size)
                        : m_chunk_max(arg_chunk_max), m_chunk_size(arg_chunk_size) {}

            private:
                struct ACCESSIBLE_TAG {
                };
                struct INACCESSIBLE_TAG {
                };

                ChunkedArrayManager(ACCESSIBLE_TAG, pointer_type *arg_chunks,
                                    const unsigned arg_chunk_max)
                        : m_valid(true), m_chunk_max(arg_chunk_max), m_chunks(arg_chunks) {}

                ChunkedArrayManager(INACCESSIBLE_TAG, const unsigned arg_chunk_max,
                                    const unsigned arg_chunk_size)
                        : m_chunk_max(arg_chunk_max), m_chunk_size(arg_chunk_size) {}

            public:
                template<typename Space, typename Enable_ = void>
                struct IsAccessibleFrom;

                template<typename Space>
                struct IsAccessibleFrom<
                        Space, typename std::enable_if_t<flare::detail::MemorySpaceAccess<
                                MemorySpace, Space>::accessible>> : std::true_type {
                };

                template<typename Space>
                struct IsAccessibleFrom<
                        Space, typename std::enable_if_t<!flare::detail::MemorySpaceAccess<
                                MemorySpace, Space>::accessible>> : std::false_type {
                };

                template<typename Space>
                static ChunkedArrayManager<Space, ValueType> create_mirror(
                        ChunkedArrayManager<MemorySpace, ValueType> const &other,
                        std::enable_if_t<IsAccessibleFrom<Space>::value> * = nullptr) {
                    return ChunkedArrayManager<Space, ValueType>{
                            ACCESSIBLE_TAG{}, other.m_chunks, other.m_chunk_max};
                }

                template<typename Space>
                static ChunkedArrayManager<Space, ValueType> create_mirror(
                        ChunkedArrayManager<MemorySpace, ValueType> const &other,
                        std::enable_if_t<!IsAccessibleFrom<Space>::value> * = nullptr) {
                    using tag_type =
                            typename ChunkedArrayManager<Space, ValueType>::INACCESSIBLE_TAG;
                    return ChunkedArrayManager<Space, ValueType>{tag_type{}, other.m_chunk_max,
                                                                 other.m_chunk_size};
                }

            public:
                void allocate_device(const std::string &label) {
                    if (m_chunks == nullptr) {
                        m_chunks = reinterpret_cast<pointer_type *>(MemorySpace().allocate(
                                label.c_str(), (sizeof(pointer_type) * (m_chunk_max + 2))));
                    }
                }

                void initialize() {
                    for (unsigned i = 0; i < m_chunk_max + 2; i++) {
                        m_chunks[i] = nullptr;
                    }
                    m_valid = true;
                }

            private:
                /// Custom destroy functor for deallocating array chunks along with a linked
                /// allocation
                template<typename Space>
                struct Destroy {
                    Destroy() = default;

                    Destroy(Destroy &&) = default;

                    Destroy(const Destroy &) = default;

                    Destroy &operator=(Destroy &&) = default;

                    Destroy &operator=(const Destroy &) = default;

                    Destroy(std::string label, value_type **arg_chunk,
                            const unsigned arg_chunk_max, const unsigned arg_chunk_size,
                            value_type **arg_linked)
                            : m_label(label),
                              m_chunks(arg_chunk),
                              m_linked(arg_linked),
                              m_chunk_max(arg_chunk_max),
                              m_chunk_size(arg_chunk_size) {}

                    void execute() {
                        // Destroy the array of chunk pointers.
                        // Two entries beyond the max chunks are allocation counters.
                        uintptr_t const len =
                                *reinterpret_cast<uintptr_t *>(m_chunks + m_chunk_max);
                        for (unsigned i = 0; i < len; i++) {
                            Space().deallocate(m_label.c_str(), m_chunks[i],
                                               sizeof(value_type) * m_chunk_size);
                        }
                        // Destroy the linked allocation if we have one.
                        if (m_linked != nullptr) {
                            Space().deallocate(m_label.c_str(), m_linked,
                                               (sizeof(value_type *) * (m_chunk_max + 2)));
                        }
                    }

                    void destroy_shared_allocation() { execute(); }

                    std::string m_label;
                    value_type **m_chunks = nullptr;
                    value_type **m_linked = nullptr;
                    unsigned m_chunk_max;
                    unsigned m_chunk_size;
                };

            public:
                template<typename Space>
                void allocate_with_destroy(const std::string &label,
                                           pointer_type *linked_allocation = nullptr) {
                    using destroy_type = Destroy<Space>;
                    using record_type =
                            flare::detail::SharedAllocationRecord<MemorySpace, destroy_type>;

                    // Allocate + 2 extra slots so that *m_chunk[m_chunk_max] ==
                    // num_chunks_alloc and *m_chunk[m_chunk_max+1] == extent This must match in
                    // Destroy's execute(...) method
                    record_type *const record = record_type::allocate(
                            MemorySpace(), label, (sizeof(pointer_type) * (m_chunk_max + 2)));
                    m_chunks = static_cast<pointer_type *>(record->data());
                    m_track.assign_allocated_record_to_uninitialized(record);

                    record->m_destroy = destroy_type(label, m_chunks, m_chunk_max, m_chunk_size,
                                                     linked_allocation);
                }

                pointer_type *get_ptr() const { return m_chunks; }

                template<typename OtherMemorySpace, typename ExecutionSpace>
                void deep_copy_to(
                        const ExecutionSpace &exec_space,
                        ChunkedArrayManager<OtherMemorySpace, ValueType> const &other) const {
                    if (other.m_chunks != m_chunks) {
                        flare::detail::DeepCopy<OtherMemorySpace, MemorySpace, ExecutionSpace>(
                                exec_space, other.m_chunks, m_chunks,
                                sizeof(pointer_type) * (m_chunk_max + 2));
                    }
                }

                FLARE_INLINE_FUNCTION
                pointer_type *operator+(int i) const { return m_chunks + i; }

                FLARE_INLINE_FUNCTION
                pointer_type &operator[](int i) const { return m_chunks[i]; }

                track_type const &track() const { return m_track; }

                FLARE_INLINE_FUNCTION
                bool valid() const { return m_valid; }

            private:
                bool m_valid = false;
                unsigned m_chunk_max = 0;
                pointer_type *m_chunks = nullptr;
                track_type m_track;
                unsigned m_chunk_size = 0;
            };

        } /* end namespace detail */

/** \brief Dynamic tensors are restricted to rank-one and no layout.
 *         Resize only occurs on host outside of parallel_regions.
 *         Subtensors are not allowed.
 */
        template<typename DataType, typename... P>
        class DynamicTensor : public flare::TensorTraits<DataType, P...> {
        public:
            using traits = flare::TensorTraits<DataType, P...>;

            using value_type = typename traits::value_type;
            using device_space = typename traits::memory_space;
            using host_space =
                    typename flare::detail::HostMirror<device_space>::Space::memory_space;
            using device_accessor = detail::ChunkedArrayManager<device_space, value_type>;
            using host_accessor = detail::ChunkedArrayManager<host_space, value_type>;

        private:
            template<class, class...>
            friend
            class DynamicTensor;

            using track_type = flare::detail::SharedAllocationTracker;

            static_assert(traits::rank == 1 && traits::rank_dynamic == 1,
                          "DynamicTensor must be rank-one");

            // It is assumed that the value_type is trivially copyable;
            // when this is not the case, potential problems can occur.
            static_assert(std::is_void<typename traits::specialize>::value,
                          "DynamicTensor only implemented for non-specialized Tensor type");

        private:
            device_accessor m_chunks;
            host_accessor m_chunks_host;
            unsigned m_chunk_shift;  // ceil(log2(m_chunk_size))
            unsigned m_chunk_mask;   // m_chunk_size - 1
            unsigned m_chunk_max;  // number of entries in the chunk array - each pointing
            // to a chunk of extent == m_chunk_size entries
            unsigned m_chunk_size;  // 2 << (m_chunk_shift - 1)

        public:
            //----------------------------------------------------------------------

            /** \brief  Compatible tensor of array of scalar types */
            using array_type =
                    DynamicTensor<typename traits::data_type, typename traits::device_type>;

            /** \brief  Compatible tensor of const data type */
            using const_type = DynamicTensor<typename traits::const_data_type,
                    typename traits::device_type>;

            /** \brief  Compatible tensor of non-const data type */
            using non_const_type = DynamicTensor<typename traits::non_const_data_type,
                    typename traits::device_type>;

            /** \brief  Must be accessible everywhere */
            using HostMirror = DynamicTensor;

            /** \brief Unified types */
            using uniform_device =
                    flare::Device<typename traits::device_type::execution_space,
                            flare::AnonymousSpace>;
            using uniform_type = array_type;
            using uniform_const_type = const_type;
            using uniform_runtime_type = array_type;
            using uniform_runtime_const_type = const_type;
            using uniform_nomemspace_type =
                    DynamicTensor<typename traits::data_type, uniform_device>;
            using uniform_const_nomemspace_type =
                    DynamicTensor<typename traits::const_data_type, uniform_device>;
            using uniform_runtime_nomemspace_type =
                    DynamicTensor<typename traits::data_type, uniform_device>;
            using uniform_runtime_const_nomemspace_type =
                    DynamicTensor<typename traits::const_data_type, uniform_device>;

            //----------------------------------------------------------------------

            enum {
                Rank = 1
            };

            FLARE_INLINE_FUNCTION
            size_t allocation_extent() const noexcept {
                uintptr_t n =
                        *reinterpret_cast<const uintptr_t *>(m_chunks_host + m_chunk_max);
                return (n << m_chunk_shift);
            }

            FLARE_INLINE_FUNCTION
            size_t chunk_size() const noexcept { return m_chunk_size; }

            FLARE_INLINE_FUNCTION
            size_t chunk_max() const noexcept { return m_chunk_max; }

            FLARE_INLINE_FUNCTION
            size_t size() const noexcept {
                size_t extent_0 =
                        *reinterpret_cast<const size_t *>(m_chunks_host + m_chunk_max + 1);
                return extent_0;
            }

            template<typename iType>
            FLARE_INLINE_FUNCTION size_t extent(const iType &r) const {
                return r == 0 ? size() : 1;
            }

            template<typename iType>
            FLARE_INLINE_FUNCTION size_t extent_int(const iType &r) const {
                return r == 0 ? size() : 1;
            }

            FLARE_INLINE_FUNCTION constexpr size_t stride_0() const { return 0; }

            FLARE_INLINE_FUNCTION constexpr size_t stride_1() const { return 0; }

            FLARE_INLINE_FUNCTION constexpr size_t stride_2() const { return 0; }

            FLARE_INLINE_FUNCTION constexpr size_t stride_3() const { return 0; }

            FLARE_INLINE_FUNCTION constexpr size_t stride_4() const { return 0; }

            FLARE_INLINE_FUNCTION constexpr size_t stride_5() const { return 0; }

            FLARE_INLINE_FUNCTION constexpr size_t stride_6() const { return 0; }

            FLARE_INLINE_FUNCTION constexpr size_t stride_7() const { return 0; }

            template<typename iType>
            FLARE_INLINE_FUNCTION void stride(iType *const s) const {
                *s = 0;
            }

            //----------------------------------------
            // Allocation tracking properties

            FLARE_INLINE_FUNCTION
            int use_count() const { return m_chunks_host.track().use_count(); }

            inline const std::string label() const {
                return m_chunks_host.track().template get_label<host_space>();
            }

            //----------------------------------------------------------------------
            // Range span is the span which contains all members.

            using reference_type = typename traits::value_type &;
            using pointer_type = typename traits::value_type *;

            enum {
                reference_type_is_lvalue_reference =
                std::is_lvalue_reference<reference_type>::value
            };

            FLARE_INLINE_FUNCTION constexpr bool span_is_contiguous() const {
                return false;
            }

            FLARE_INLINE_FUNCTION constexpr size_t span() const { return 0; }

            FLARE_INLINE_FUNCTION constexpr pointer_type data() const { return 0; }

            //----------------------------------------

            template<typename I0, class... Args>
            FLARE_INLINE_FUNCTION reference_type
            operator()(const I0 &i0, const Args &... /*args*/) const {
                static_assert(flare::detail::are_integral<I0, Args...>::value,
                              "Indices must be integral type");

                flare::detail::runtime_check_memory_access_violation<
                        typename traits::memory_space>(
                        "flare::DynamicTensor ERROR: attempt to access inaccessible memory "
                        "space");

                // Which chunk is being indexed.
                const uintptr_t ic = uintptr_t(i0) >> m_chunk_shift;

#if defined(FLARE_ENABLE_DEBUG_BOUNDS_CHECK)
                const uintptr_t n = *reinterpret_cast<uintptr_t*>(m_chunks + m_chunk_max);
                if (n <= ic) flare::abort("flare::DynamicTensor array bounds error");
#endif

                typename traits::value_type **const ch = m_chunks + ic;
                return (*ch)[i0 & m_chunk_mask];
            }

            //----------------------------------------
            /** \brief  Resizing in serial can grow or shrink the array size
             *          up to the maximum number of chunks
             * */
            template<typename IntType>
            inline void resize_serial(IntType const &n) {
                using local_value_type = typename traits::value_type;
                using value_pointer_type = local_value_type *;

                const uintptr_t NC =
                        (n + m_chunk_mask) >>
                                           m_chunk_shift;  // New total number of chunks needed for resize

                if (m_chunk_max < NC) {
                    flare::abort("DynamicTensor::resize_serial exceeded maximum size");
                }

                // *m_chunks[m_chunk_max] stores the current number of chunks being used
                uintptr_t *const pc =
                        reinterpret_cast<uintptr_t *>(m_chunks_host + m_chunk_max);
                std::string _label = m_chunks_host.track().template get_label<host_space>();

                if (*pc < NC) {
                    while (*pc < NC) {
                        m_chunks_host[*pc] =
                                reinterpret_cast<value_pointer_type>(device_space().allocate(
                                        _label.c_str(), sizeof(local_value_type) << m_chunk_shift));
                        ++*pc;
                    }
                } else {
                    while (NC + 1 <= *pc) {
                        --*pc;
                        device_space().deallocate(_label.c_str(), m_chunks_host[*pc],
                                                  sizeof(local_value_type) << m_chunk_shift);
                        m_chunks_host[*pc] = nullptr;
                    }
                }
                // *m_chunks_host[m_chunk_max+1] stores the 'extent' requested by resize
                *(pc + 1) = n;

                typename device_space::execution_space exec{};
                m_chunks_host.deep_copy_to(exec, m_chunks);
                exec.fence(
                        "DynamicTensor::resize_serial: Fence after copying chunks to the device");
            }

            FLARE_INLINE_FUNCTION bool is_allocated() const {
                if (m_chunks_host.valid()) {
                    // *m_chunks_host[m_chunk_max] stores the current number of chunks being
                    // used
                    uintptr_t *const pc =
                            reinterpret_cast<uintptr_t *>(m_chunks_host + m_chunk_max);
                    return (*(pc + 1) > 0);
                } else {
                    return false;
                }
            }

            FLARE_FUNCTION const device_accessor &impl_get_chunks() const {
                return m_chunks;
            }

            FLARE_FUNCTION device_accessor &impl_get_chunks() { return m_chunks; }

            //----------------------------------------------------------------------

            ~DynamicTensor() = default;

            DynamicTensor() = default;

            DynamicTensor(DynamicTensor &&) = default;

            DynamicTensor(const DynamicTensor &) = default;

            DynamicTensor &operator=(DynamicTensor &&) = default;

            DynamicTensor &operator=(const DynamicTensor &) = default;

            template<class RT, class... RP>
            DynamicTensor(const DynamicTensor<RT, RP...> &rhs)
                    : m_chunks(rhs.m_chunks),
                      m_chunks_host(rhs.m_chunks_host),
                      m_chunk_shift(rhs.m_chunk_shift),
                      m_chunk_mask(rhs.m_chunk_mask),
                      m_chunk_max(rhs.m_chunk_max),
                      m_chunk_size(rhs.m_chunk_size) {
                using SrcTraits = typename DynamicTensor<RT, RP...>::traits;
                using Mapping = flare::detail::TensorMapping<traits, SrcTraits, void>;
                static_assert(Mapping::is_assignable,
                              "Incompatible DynamicTensor copy construction");
            }

            /**\brief  Allocation constructor
             *
             *  Memory is allocated in chunks
             *  A maximum size is required in order to allocate a
             *  chunk-pointer array.
             */
            template<class... Prop>
            DynamicTensor(const flare::detail::TensorCtorProp<Prop...> &arg_prop,
                          const unsigned min_chunk_size,
                          const unsigned max_extent)
                    :  // The chunk size is guaranteed to be a power of two
                    m_chunk_shift(flare::detail::integral_power_of_two_that_contains(
                            min_chunk_size))  // div ceil(log2(min_chunk_size))
                    ,
                    m_chunk_mask((1 << m_chunk_shift) - 1)  // mod
                    ,
                    m_chunk_max((max_extent + m_chunk_mask) >>
                                                            m_chunk_shift)  // max num pointers-to-chunks in array
                    ,
                    m_chunk_size(2 << (m_chunk_shift - 1)) {
                m_chunks = device_accessor(m_chunk_max, m_chunk_size);

                const std::string &label =
                        flare::detail::get_property<flare::detail::LabelTag>(arg_prop);

                if (device_accessor::template IsAccessibleFrom<host_space>::value) {
                    m_chunks.template allocate_with_destroy<device_space>(label);
                    m_chunks.initialize();
                    m_chunks_host =
                            device_accessor::template create_mirror<host_space>(m_chunks);
                } else {
                    m_chunks.allocate_device(label);
                    m_chunks_host =
                            device_accessor::template create_mirror<host_space>(m_chunks);
                    m_chunks_host.template allocate_with_destroy<device_space>(
                            label, m_chunks.get_ptr());
                    m_chunks_host.initialize();

                    using alloc_prop_input = flare::detail::TensorCtorProp<Prop...>;

                    auto arg_prop_copy = ::flare::detail::with_properties_if_unset(
                            arg_prop, typename device_space::execution_space{});

                    const auto &exec =
                            flare::detail::get_property<flare::detail::ExecutionSpaceTag>(
                                    arg_prop_copy);
                    m_chunks_host.deep_copy_to(exec, m_chunks);
                    if (!alloc_prop_input::has_execution_space)
                        exec.fence(
                                "DynamicTensor::DynamicTensor(): Fence after copying chunks to the "
                                "device");
                }
            }

            DynamicTensor(const std::string &arg_label, const unsigned min_chunk_size,
                          const unsigned max_extent)
                    : DynamicTensor(flare::tensor_alloc(arg_label), min_chunk_size, max_extent) {
            }
        };

    }  // namespace experimental

    template<class>
    struct is_dynamic_tensor : public std::false_type {
    };

    template<class D, class... P>
    struct is_dynamic_tensor<flare::experimental::DynamicTensor<D, P...>>
            : public std::true_type {
    };

    template<class T>
    inline constexpr bool is_dynamic_tensor_v = is_dynamic_tensor<T>::value;

}  // namespace flare

namespace flare {

    namespace detail {

// Deduce Mirror Types
        template<class Space, class T, class... P>
        struct MirrorDynamicTensorType {
            // The incoming tensor_type
            using src_tensor_type = typename flare::experimental::DynamicTensor<T, P...>;
            // The memory space for the mirror tensor
            using memory_space = typename Space::memory_space;
            // Check whether it is the same memory space
            enum {
                is_same_memspace =
                std::is_same<memory_space, typename src_tensor_type::memory_space>::value
            };
            // The array_layout
            using array_layout = typename src_tensor_type::array_layout;
            // The data type (we probably want it non-const since otherwise we can't even
            // deep_copy to it.)
            using data_type = typename src_tensor_type::non_const_data_type;
            // The destination tensor type if it is not the same memory space
            using dest_tensor_type =
                    flare::experimental::DynamicTensor<data_type, array_layout, Space>;
            // If it is the same memory_space return the existing tensor_type
            // This will also keep the unmanaged trait if necessary
            using tensor_type =
                    std::conditional_t<is_same_memspace, src_tensor_type, dest_tensor_type>;
        };
    }  // namespace detail

    namespace detail {
        template<class T, class... P, class... TensorCtorArgs>
        inline auto create_mirror(
                const flare::experimental::DynamicTensor<T, P...> &src,
                const detail::TensorCtorProp<TensorCtorArgs...> &arg_prop,
                std::enable_if_t<!detail::TensorCtorProp<TensorCtorArgs...>::has_memory_space> * =
                nullptr) {
            using alloc_prop_input = detail::TensorCtorProp<TensorCtorArgs...>;

            static_assert(
                    !alloc_prop_input::has_label,
                    "The tensor constructor arguments passed to flare::create_mirror "
                    "must not include a label!");
            static_assert(
                    !alloc_prop_input::has_pointer,
                    "The tensor constructor arguments passed to flare::create_mirror must "
                    "not include a pointer!");
            static_assert(
                    !alloc_prop_input::allow_padding,
                    "The tensor constructor arguments passed to flare::create_mirror must "
                    "not explicitly allow padding!");

            auto prop_copy = detail::with_properties_if_unset(
                    arg_prop, std::string(src.label()).append("_mirror"));

            auto ret = typename flare::experimental::DynamicTensor<T, P...>::HostMirror(
                    prop_copy, src.chunk_size(), src.chunk_max() * src.chunk_size());

            ret.resize_serial(src.extent(0));

            return ret;
        }

        template<class T, class... P, class... TensorCtorArgs>
        inline auto create_mirror(
                const flare::experimental::DynamicTensor<T, P...> &src,
                const detail::TensorCtorProp<TensorCtorArgs...> &arg_prop,
                std::enable_if_t<detail::TensorCtorProp<TensorCtorArgs...>::has_memory_space> * =
                nullptr) {
            using alloc_prop_input = detail::TensorCtorProp<TensorCtorArgs...>;

            static_assert(
                    !alloc_prop_input::has_label,
                    "The tensor constructor arguments passed to flare::create_mirror "
                    "must not include a label!");
            static_assert(
                    !alloc_prop_input::has_pointer,
                    "The tensor constructor arguments passed to flare::create_mirror must "
                    "not include a pointer!");
            static_assert(
                    !alloc_prop_input::allow_padding,
                    "The tensor constructor arguments passed to flare::create_mirror must "
                    "not explicitly allow padding!");

            using MemorySpace = typename alloc_prop_input::memory_space;
            auto prop_copy = detail::with_properties_if_unset(
                    arg_prop, std::string(src.label()).append("_mirror"));

            auto ret = typename flare::detail::MirrorDynamicTensorType<
                    MemorySpace, T, P...>::tensor_type(prop_copy, src.chunk_size(),
                                                       src.chunk_max() * src.chunk_size());

            ret.resize_serial(src.extent(0));

            return ret;
        }
    }  // namespace detail

// Create a mirror in host space
    template<class T, class... P>
    inline auto create_mirror(
            const flare::experimental::DynamicTensor<T, P...> &src) {
        return detail::create_mirror(src, detail::TensorCtorProp<>{});
    }

    template<class T, class... P>
    inline auto create_mirror(
            flare::detail::WithoutInitializing_t wi,
            const flare::experimental::DynamicTensor<T, P...> &src) {
        return detail::create_mirror(src, flare::tensor_alloc(wi));
    }

// Create a mirror in a new space
    template<class Space, class T, class... P>
    inline auto create_mirror(
            const Space &, const flare::experimental::DynamicTensor<T, P...> &src) {
        return detail::create_mirror(
                src, flare::tensor_alloc(typename Space::memory_space{}));
    }

    template<class Space, class T, class... P>
    typename flare::detail::MirrorDynamicTensorType<Space, T, P...>::tensor_type
    create_mirror(flare::detail::WithoutInitializing_t wi, const Space &,
                  const flare::experimental::DynamicTensor<T, P...> &src) {
        return detail::create_mirror(
                src, flare::tensor_alloc(wi, typename Space::memory_space{}));
    }

    template<class T, class... P, class... TensorCtorArgs>
    inline auto create_mirror(
            const detail::TensorCtorProp<TensorCtorArgs...> &arg_prop,
            const flare::experimental::DynamicTensor<T, P...> &src) {
        return detail::create_mirror(src, arg_prop);
    }

    namespace detail {

        template<class T, class... P, class... TensorCtorArgs>
        inline std::enable_if_t<
                !detail::TensorCtorProp<TensorCtorArgs...>::has_memory_space &&
                (std::is_same<
                        typename flare::experimental::DynamicTensor<T, P...>::memory_space,
                        typename flare::experimental::DynamicTensor<
                                T, P...>::HostMirror::memory_space>::value &&
                 std::is_same<
                         typename flare::experimental::DynamicTensor<T, P...>::data_type,
                         typename flare::experimental::DynamicTensor<
                                 T, P...>::HostMirror::data_type>::value),
                typename flare::experimental::DynamicTensor<T, P...>::HostMirror>
        create_mirror_tensor(const flare::experimental::DynamicTensor<T, P...> &src,
                             const detail::TensorCtorProp<TensorCtorArgs...> &) {
            return src;
        }

        template<class T, class... P, class... TensorCtorArgs>
        inline std::enable_if_t<
                !detail::TensorCtorProp<TensorCtorArgs...>::has_memory_space &&
                !(std::is_same<
                        typename flare::experimental::DynamicTensor<T, P...>::memory_space,
                        typename flare::experimental::DynamicTensor<
                                T, P...>::HostMirror::memory_space>::value &&
                  std::is_same<
                          typename flare::experimental::DynamicTensor<T, P...>::data_type,
                          typename flare::experimental::DynamicTensor<
                                  T, P...>::HostMirror::data_type>::value),
                typename flare::experimental::DynamicTensor<T, P...>::HostMirror>
        create_mirror_tensor(const flare::experimental::DynamicTensor<T, P...> &src,
                             const detail::TensorCtorProp<TensorCtorArgs...> &arg_prop) {
            return flare::create_mirror(arg_prop, src);
        }

        template<class T, class... P, class... TensorCtorArgs,
                class = std::enable_if_t<
                        detail::TensorCtorProp<TensorCtorArgs...>::has_memory_space>>
        std::enable_if_t<detail::MirrorDynamicTensorType<
                typename detail::TensorCtorProp<TensorCtorArgs...>::memory_space,
                T, P...>::is_same_memspace,
                typename detail::MirrorDynamicTensorType<
                        typename detail::TensorCtorProp<TensorCtorArgs...>::memory_space,
                        T, P...>::tensor_type>
        create_mirror_tensor(const flare::experimental::DynamicTensor<T, P...> &src,
                             const detail::TensorCtorProp<TensorCtorArgs...> &) {
            return src;
        }

        template<class T, class... P, class... TensorCtorArgs,
                class = std::enable_if_t<
                        detail::TensorCtorProp<TensorCtorArgs...>::has_memory_space>>
        std::enable_if_t<!detail::MirrorDynamicTensorType<
                typename detail::TensorCtorProp<TensorCtorArgs...>::memory_space,
                T, P...>::is_same_memspace,
                typename detail::MirrorDynamicTensorType<
                        typename detail::TensorCtorProp<TensorCtorArgs...>::memory_space,
                        T, P...>::tensor_type>
        create_mirror_tensor(const flare::experimental::DynamicTensor<T, P...> &src,
                             const detail::TensorCtorProp<TensorCtorArgs...> &arg_prop) {
            return flare::detail::create_mirror(src, arg_prop);
        }
    }  // namespace detail

// Create a mirror tensor in host space
    template<class T, class... P>
    inline auto create_mirror_tensor(
            const typename flare::experimental::DynamicTensor<T, P...> &src) {
        return detail::create_mirror_tensor(src, detail::TensorCtorProp<>{});
    }

    template<class T, class... P>
    inline auto create_mirror_tensor(
            flare::detail::WithoutInitializing_t wi,
            const typename flare::experimental::DynamicTensor<T, P...> &src) {
        return detail::create_mirror_tensor(src, flare::tensor_alloc(wi));
    }

// Create a mirror in a new space
    template<class Space, class T, class... P>
    inline auto create_mirror_tensor(
            const Space &, const flare::experimental::DynamicTensor<T, P...> &src) {
        return detail::create_mirror_tensor(src,
                                            tensor_alloc(typename Space::memory_space{}));
    }

    template<class Space, class T, class... P>
    inline auto create_mirror_tensor(
            flare::detail::WithoutInitializing_t wi, const Space &,
            const flare::experimental::DynamicTensor<T, P...> &src) {
        return detail::create_mirror_tensor(
                src, flare::tensor_alloc(wi, typename Space::memory_space{}));
    }

    template<class T, class... P, class... TensorCtorArgs>
    inline auto create_mirror_tensor(
            const detail::TensorCtorProp<TensorCtorArgs...> &arg_prop,
            const flare::experimental::DynamicTensor<T, P...> &src) {
        return detail::create_mirror_tensor(src, arg_prop);
    }

    template<class T, class... DP, class... SP>
    inline void deep_copy(const flare::experimental::DynamicTensor<T, DP...> &dst,
                          const flare::experimental::DynamicTensor<T, SP...> &src) {
        using dst_type = flare::experimental::DynamicTensor<T, DP...>;
        using src_type = flare::experimental::DynamicTensor<T, SP...>;

        using dst_execution_space = typename TensorTraits<T, DP...>::execution_space;
        using src_execution_space = typename TensorTraits<T, SP...>::execution_space;
        using dst_memory_space = typename TensorTraits<T, DP...>::memory_space;
        using src_memory_space = typename TensorTraits<T, SP...>::memory_space;

        constexpr bool DstExecCanAccessSrc =
                flare::SpaceAccessibility<dst_execution_space,
                        src_memory_space>::accessible;
        constexpr bool SrcExecCanAccessDst =
                flare::SpaceAccessibility<src_execution_space,
                        dst_memory_space>::accessible;

        if (DstExecCanAccessSrc)
            flare::detail::TensorRemap<dst_type, src_type, dst_execution_space>(dst, src);
        else if (SrcExecCanAccessDst)
            flare::detail::TensorRemap<dst_type, src_type, src_execution_space>(dst, src);
        else
            src.impl_get_chunks().deep_copy_to(dst_execution_space{},
                                               dst.impl_get_chunks());
        flare::fence("flare::deep_copy(DynamicTensor)");
    }

    template<class ExecutionSpace, class T, class... DP, class... SP>
    inline void deep_copy(const ExecutionSpace &exec,
                          const flare::experimental::DynamicTensor<T, DP...> &dst,
                          const flare::experimental::DynamicTensor<T, SP...> &src) {
        using dst_type = flare::experimental::DynamicTensor<T, DP...>;
        using src_type = flare::experimental::DynamicTensor<T, SP...>;

        using dst_execution_space = typename TensorTraits<T, DP...>::execution_space;
        using src_execution_space = typename TensorTraits<T, SP...>::execution_space;
        using dst_memory_space = typename TensorTraits<T, DP...>::memory_space;
        using src_memory_space = typename TensorTraits<T, SP...>::memory_space;

        constexpr bool DstExecCanAccessSrc =
                flare::SpaceAccessibility<dst_execution_space,
                        src_memory_space>::accessible;
        constexpr bool SrcExecCanAccessDst =
                flare::SpaceAccessibility<src_execution_space,
                        dst_memory_space>::accessible;

        // FIXME use execution space
        if (DstExecCanAccessSrc)
            flare::detail::TensorRemap<dst_type, src_type, dst_execution_space>(dst, src);
        else if (SrcExecCanAccessDst)
            flare::detail::TensorRemap<dst_type, src_type, src_execution_space>(dst, src);
        else
            src.impl_get_chunks().deep_copy_to(exec, dst.impl_get_chunks());
    }

    template<class T, class... DP, class... SP>
    inline void deep_copy(const Tensor<T, DP...> &dst,
                          const flare::experimental::DynamicTensor<T, SP...> &src) {
        using dst_type = Tensor<T, DP...>;
        using src_type = flare::experimental::DynamicTensor<T, SP...>;

        using dst_execution_space = typename TensorTraits<T, DP...>::execution_space;
        using src_memory_space = typename TensorTraits<T, SP...>::memory_space;

        enum {
            DstExecCanAccessSrc =
            flare::SpaceAccessibility<dst_execution_space,
                    src_memory_space>::accessible
        };

        if (DstExecCanAccessSrc) {
            // Copying data between tensors in accessible memory spaces and either
            // non-contiguous or incompatible shape.
            flare::detail::TensorRemap<dst_type, src_type>(dst, src);
            flare::fence("flare::deep_copy(DynamicTensor)");
        } else {
            flare::detail::throw_runtime_exception(
                    "deep_copy given tensors that would require a temporary allocation");
        }
    }

    template<class T, class... DP, class... SP>
    inline void deep_copy(const flare::experimental::DynamicTensor<T, DP...> &dst,
                          const Tensor<T, SP...> &src) {
        using dst_type = flare::experimental::DynamicTensor<T, DP...>;
        using src_type = Tensor<T, SP...>;

        using dst_execution_space = typename TensorTraits<T, DP...>::execution_space;
        using src_memory_space = typename TensorTraits<T, SP...>::memory_space;

        enum {
            DstExecCanAccessSrc =
            flare::SpaceAccessibility<dst_execution_space,
                    src_memory_space>::accessible
        };

        if (DstExecCanAccessSrc) {
            // Copying data between tensors in accessible memory spaces and either
            // non-contiguous or incompatible shape.
            flare::detail::TensorRemap<dst_type, src_type>(dst, src);
            flare::fence("flare::deep_copy(DynamicTensor)");
        } else {
            flare::detail::throw_runtime_exception(
                    "deep_copy given tensors that would require a temporary allocation");
        }
    }

    namespace detail {
        template<class Arg0, class... DP, class... SP>
        struct CommonSubtensor<flare::experimental::DynamicTensor<DP...>,
                flare::experimental::DynamicTensor<SP...>, 1, Arg0> {
            using DstType = flare::experimental::DynamicTensor<DP...>;
            using SrcType = flare::experimental::DynamicTensor<SP...>;
            using dst_subtensor_type = DstType;
            using src_subtensor_type = SrcType;
            dst_subtensor_type dst_sub;
            src_subtensor_type src_sub;

            CommonSubtensor(const DstType &dst, const SrcType &src, const Arg0 & /*arg0*/)
                    : dst_sub(dst), src_sub(src) {}
        };

        template<class... DP, class SrcType, class Arg0>
        struct CommonSubtensor<flare::experimental::DynamicTensor<DP...>, SrcType, 1,
                Arg0> {
            using DstType = flare::experimental::DynamicTensor<DP...>;
            using dst_subtensor_type = DstType;
            using src_subtensor_type = typename flare::Subtensor<SrcType, Arg0>;
            dst_subtensor_type dst_sub;
            src_subtensor_type src_sub;

            CommonSubtensor(const DstType &dst, const SrcType &src, const Arg0 &arg0)
                    : dst_sub(dst), src_sub(src, arg0) {}
        };

        template<class DstType, class... SP, class Arg0>
        struct CommonSubtensor<DstType, flare::experimental::DynamicTensor<SP...>, 1,
                Arg0> {
            using SrcType = flare::experimental::DynamicTensor<SP...>;
            using dst_subtensor_type = typename flare::Subtensor<DstType, Arg0>;
            using src_subtensor_type = SrcType;
            dst_subtensor_type dst_sub;
            src_subtensor_type src_sub;

            CommonSubtensor(const DstType &dst, const SrcType &src, const Arg0 &arg0)
                    : dst_sub(dst, arg0), src_sub(src) {}
        };

        template<class... DP, class TensorTypeB, class Layout, class ExecSpace,
                typename iType>
        struct TensorCopy<flare::experimental::DynamicTensor<DP...>, TensorTypeB, Layout,
                ExecSpace, 1, iType> {
            flare::experimental::DynamicTensor<DP...> a;
            TensorTypeB b;

            using policy_type = flare::RangePolicy<ExecSpace, flare::IndexType<iType>>;

            TensorCopy(const flare::experimental::DynamicTensor<DP...> &a_,
                       const TensorTypeB &b_)
                    : a(a_), b(b_) {
                flare::parallel_for("flare::TensorCopy-1D", policy_type(0, b.extent(0)),
                                    *this);
            }

            FLARE_INLINE_FUNCTION
            void operator()(const iType &i0) const { a(i0) = b(i0); };
        };

        template<class... DP, class... SP, class Layout, class ExecSpace,
                typename iType>
        struct TensorCopy<flare::experimental::DynamicTensor<DP...>,
                flare::experimental::DynamicTensor<SP...>, Layout, ExecSpace, 1,
                iType> {
            flare::experimental::DynamicTensor<DP...> a;
            flare::experimental::DynamicTensor<SP...> b;

            using policy_type = flare::RangePolicy<ExecSpace, flare::IndexType<iType>>;

            TensorCopy(const flare::experimental::DynamicTensor<DP...> &a_,
                       const flare::experimental::DynamicTensor<SP...> &b_)
                    : a(a_), b(b_) {
                const iType n = std::min(a.extent(0), b.extent(0));
                flare::parallel_for("flare::TensorCopy-1D", policy_type(0, n), *this);
            }

            FLARE_INLINE_FUNCTION
            void operator()(const iType &i0) const { a(i0) = b(i0); };
        };

    }  // namespace detail

    template<class... TensorCtorArgs, class T, class... P>
    auto create_mirror_tensor_and_copy(
            const detail::TensorCtorProp<TensorCtorArgs...> &,
            const flare::experimental::DynamicTensor<T, P...> &src,
            std::enable_if_t<
                    std::is_void<typename TensorTraits<T, P...>::specialize>::value &&
                    detail::MirrorDynamicTensorType<
                            typename detail::TensorCtorProp<TensorCtorArgs...>::memory_space, T,
                            P...>::is_same_memspace> * = nullptr) {
        using alloc_prop_input = detail::TensorCtorProp<TensorCtorArgs...>;
        static_assert(
                alloc_prop_input::has_memory_space,
                "The tensor constructor arguments passed to "
                "flare::create_mirror_tensor_and_copy must include a memory space!");
        static_assert(!alloc_prop_input::has_pointer,
                      "The tensor constructor arguments passed to "
                      "flare::create_mirror_tensor_and_copy must "
                      "not include a pointer!");
        static_assert(!alloc_prop_input::allow_padding,
                      "The tensor constructor arguments passed to "
                      "flare::create_mirror_tensor_and_copy must "
                      "not explicitly allow padding!");

        // same behavior as deep_copy(src, src)
        if (!alloc_prop_input::has_execution_space)
            fence(
                    "flare::create_mirror_tensor_and_copy: fence before returning src tensor");
        return src;
    }

    template<class... TensorCtorArgs, class T, class... P>
    auto create_mirror_tensor_and_copy(
            const detail::TensorCtorProp<TensorCtorArgs...> &arg_prop,
            const flare::experimental::DynamicTensor<T, P...> &src,
            std::enable_if_t<
                    std::is_void<typename TensorTraits<T, P...>::specialize>::value &&
                    !detail::MirrorDynamicTensorType<
                            typename detail::TensorCtorProp<TensorCtorArgs...>::memory_space, T,
                            P...>::is_same_memspace> * = nullptr) {
        using alloc_prop_input = detail::TensorCtorProp<TensorCtorArgs...>;
        static_assert(
                alloc_prop_input::has_memory_space,
                "The tensor constructor arguments passed to "
                "flare::create_mirror_tensor_and_copy must include a memory space!");
        static_assert(!alloc_prop_input::has_pointer,
                      "The tensor constructor arguments passed to "
                      "flare::create_mirror_tensor_and_copy must "
                      "not include a pointer!");
        static_assert(!alloc_prop_input::allow_padding,
                      "The tensor constructor arguments passed to "
                      "flare::create_mirror_tensor_and_copy must "
                      "not explicitly allow padding!");
        using Space = typename alloc_prop_input::memory_space;
        using Mirror =
                typename detail::MirrorDynamicTensorType<Space, T, P...>::tensor_type;

        auto arg_prop_copy = detail::with_properties_if_unset(
                arg_prop, std::string{}, WithoutInitializing,
                typename Space::execution_space{});

        std::string &label = detail::get_property<detail::LabelTag>(arg_prop_copy);
        if (label.empty()) label = src.label();
        auto mirror = typename Mirror::non_const_type(
                arg_prop_copy, src.chunk_size(), src.chunk_max() * src.chunk_size());
        mirror.resize_serial(src.extent(0));
        if constexpr (alloc_prop_input::has_execution_space) {
            deep_copy(detail::get_property<detail::ExecutionSpaceTag>(arg_prop_copy),
                      mirror, src);
        } else
            deep_copy(mirror, src);
        return mirror;
    }

    template<class Space, class T, class... P>
    auto create_mirror_tensor_and_copy(
            const Space &, const flare::experimental::DynamicTensor<T, P...> &src,
            std::string const &name = "") {
        return create_mirror_tensor_and_copy(
                flare::tensor_alloc(typename Space::memory_space{}, name), src);
    }

}  // namespace flare

#endif  // FLARE_DYNAMIC_TENSOR_H_
