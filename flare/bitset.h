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

#ifndef FLARE_BITSET_H_
#define FLARE_BITSET_H_

#include <flare/core.h>
#include <flare/functional.h>
#include <flare/containers/bitset_impl.h>

namespace flare {

    template<typename Device = flare::DefaultExecutionSpace>
    class Bitset;

    template<typename Device = flare::DefaultExecutionSpace>
    class ConstBitset;

    template<typename DstDevice, typename SrcDevice>
    void deep_copy(Bitset<DstDevice> &dst, Bitset<SrcDevice> const &src);

    template<typename DstDevice, typename SrcDevice>
    void deep_copy(Bitset<DstDevice> &dst, ConstBitset<SrcDevice> const &src);

    template<typename DstDevice, typename SrcDevice>
    void deep_copy(ConstBitset<DstDevice> &dst, ConstBitset<SrcDevice> const &src);

    /// A thread safe tensor to a bitset
    template<typename Device>
    class Bitset {
    public:
        using execution_space = typename Device::execution_space;
        using size_type = unsigned int;

        static constexpr unsigned BIT_SCAN_REVERSE = 1u;
        static constexpr unsigned MOVE_HINT_BACKWARD = 2u;

        static constexpr unsigned BIT_SCAN_FORWARD_MOVE_HINT_FORWARD = 0u;
        static constexpr unsigned BIT_SCAN_REVERSE_MOVE_HINT_FORWARD =
                BIT_SCAN_REVERSE;
        static constexpr unsigned BIT_SCAN_FORWARD_MOVE_HINT_BACKWARD =
                MOVE_HINT_BACKWARD;
        static constexpr unsigned BIT_SCAN_REVERSE_MOVE_HINT_BACKWARD =
                BIT_SCAN_REVERSE | MOVE_HINT_BACKWARD;

    private:
        enum : unsigned {
            block_size = static_cast<unsigned>(sizeof(unsigned) * CHAR_BIT)
        };
        enum : unsigned {
            block_mask = block_size - 1u
        };
        enum : unsigned {
            block_shift = flare::detail::integral_power_of_two(block_size)
        };

    public:
        /// constructor
        /// arg_size := number of bit in set
        Bitset(unsigned arg_size = 0u)
                : m_size(arg_size),
                  m_last_block_mask(0u),
                  m_blocks("Bitset", ((m_size + block_mask) >> block_shift)) {
            for (int i = 0, end = static_cast<int>(m_size & block_mask); i < end; ++i) {
                m_last_block_mask |= 1u << i;
            }
        }

        FLARE_DEFAULTED_FUNCTION
        Bitset(const Bitset<Device> &) = default;

        FLARE_DEFAULTED_FUNCTION
        Bitset &operator=(const Bitset<Device> &) = default;

        FLARE_DEFAULTED_FUNCTION
        Bitset(Bitset<Device> &&) = default;

        FLARE_DEFAULTED_FUNCTION
        Bitset &operator=(Bitset<Device> &&) = default;

        FLARE_DEFAULTED_FUNCTION
        ~Bitset() = default;

        /// number of bits in the set
        /// can be call from the host or the device
        FLARE_FORCEINLINE_FUNCTION
        unsigned size() const { return m_size; }

        /// number of bits which are set to 1
        /// can only be called from the host
        unsigned count() const {
            detail::BitsetCount<Bitset<Device> > f(*this);
            return f.apply();
        }

        /// set all bits to 1
        /// can only be called from the host
        void set() {
            flare::deep_copy(m_blocks, ~0u);

            if (m_last_block_mask) {
                // clear the unused bits in the last block
                flare::detail::DeepCopy<typename Device::memory_space, flare::HostSpace>(
                        m_blocks.data() + (m_blocks.extent(0) - 1u), &m_last_block_mask,
                        sizeof(unsigned));
                flare::fence(
                        "Bitset::set: fence after clearing unused bits copying from "
                        "HostSpace");
            }
        }

        /// set all bits to 0
        /// can only be called from the host
        void reset() { flare::deep_copy(m_blocks, 0u); }

        /// set all bits to 0
        /// can only be called from the host
        void clear() { flare::deep_copy(m_blocks, 0u); }

        /// set i'th bit to 1
        /// can only be called from the device
        FLARE_FORCEINLINE_FUNCTION
        bool set(unsigned i) const {
            if (i < m_size) {
                unsigned *block_ptr = &m_blocks[i >> block_shift];
                const unsigned mask = 1u << static_cast<int>(i & block_mask);

                return !(atomic_fetch_or(block_ptr, mask) & mask);
            }
            return false;
        }

        /// set i'th bit to 0
        /// can only be called from the device
        FLARE_FORCEINLINE_FUNCTION
        bool reset(unsigned i) const {
            if (i < m_size) {
                unsigned *block_ptr = &m_blocks[i >> block_shift];
                const unsigned mask = 1u << static_cast<int>(i & block_mask);

                return atomic_fetch_and(block_ptr, ~mask) & mask;
            }
            return false;
        }

        /// return true if the i'th bit set to 1
        /// can only be called from the device
        FLARE_FORCEINLINE_FUNCTION
        bool test(unsigned i) const {
            if (i < m_size) {
                const unsigned block = volatile_load(&m_blocks[i >> block_shift]);
                const unsigned mask = 1u << static_cast<int>(i & block_mask);
                return block & mask;
            }
            return false;
        }

        /// used with find_any_set_near or find_any_unset_near functions
        /// returns the max number of times those functions should be call
        /// when searching for an available bit
        FLARE_FORCEINLINE_FUNCTION
        unsigned max_hint() const { return m_blocks.extent(0); }

        /// find a bit set to 1 near the hint
        /// returns a pair< bool, unsigned> where if result.first is true then
        /// result.second is the bit found and if result.first is false the
        /// result.second is a new hint
        FLARE_INLINE_FUNCTION
        flare::pair<bool, unsigned> find_any_set_near(
                unsigned hint,
                unsigned scan_direction = BIT_SCAN_FORWARD_MOVE_HINT_FORWARD) const {
            const unsigned block_idx =
                    (hint >> block_shift) < m_blocks.extent(0) ? (hint >> block_shift) : 0;
            const unsigned offset = hint & block_mask;
            unsigned block = volatile_load(&m_blocks[block_idx]);
            block = !m_last_block_mask || (block_idx < (m_blocks.extent(0) - 1))
                    ? block
                    : block & m_last_block_mask;

            return find_any_helper(block_idx, offset, block, scan_direction);
        }

        /// find a bit set to 0 near the hint
        /// returns a pair< bool, unsigned> where if result.first is true then
        /// result.second is the bit found and if result.first is false the
        /// result.second is a new hint
        FLARE_INLINE_FUNCTION
        flare::pair<bool, unsigned> find_any_unset_near(
                unsigned hint,
                unsigned scan_direction = BIT_SCAN_FORWARD_MOVE_HINT_FORWARD) const {
            const unsigned block_idx = hint >> block_shift;
            const unsigned offset = hint & block_mask;
            unsigned block = volatile_load(&m_blocks[block_idx]);
            block = !m_last_block_mask || (block_idx < (m_blocks.extent(0) - 1))
                    ? ~block
                    : ~block & m_last_block_mask;

            return find_any_helper(block_idx, offset, block, scan_direction);
        }

        FLARE_INLINE_FUNCTION constexpr bool is_allocated() const {
            return m_blocks.is_allocated();
        }

    private:
        FLARE_FORCEINLINE_FUNCTION
        flare::pair<bool, unsigned> find_any_helper(unsigned block_idx,
                                                    unsigned offset, unsigned block,
                                                    unsigned scan_direction) const {
            flare::pair<bool, unsigned> result(block > 0u, 0);

            if (!result.first) {
                result.second = update_hint(block_idx, offset, scan_direction);
            } else {
                result.second =
                        scan_block((block_idx << block_shift), offset, block, scan_direction);
            }
            return result;
        }

        FLARE_FORCEINLINE_FUNCTION
        unsigned scan_block(unsigned block_start, int offset, unsigned block,
                            unsigned scan_direction) const {
            offset = !(scan_direction & BIT_SCAN_REVERSE)
                     ? offset
                     : (offset + block_mask) & block_mask;
            block = detail::rotate_right(block, offset);
            return (((!(scan_direction & BIT_SCAN_REVERSE)
                      ? detail::bit_scan_forward(block)
                      : detail::int_log2(block)) +
                     offset) &
                    block_mask) +
                   block_start;
        }

        FLARE_FORCEINLINE_FUNCTION
        unsigned update_hint(long long block_idx, unsigned offset,
                             unsigned scan_direction) const {
            block_idx += scan_direction & MOVE_HINT_BACKWARD ? -1 : 1;
            block_idx = block_idx >= 0 ? block_idx : m_blocks.extent(0) - 1;
            block_idx =
                    block_idx < static_cast<long long>(m_blocks.extent(0)) ? block_idx : 0;

            return static_cast<unsigned>(block_idx) * block_size + offset;
        }

    private:
        unsigned m_size;
        unsigned m_last_block_mask;
        Tensor<unsigned *, Device, MemoryTraits<RandomAccess> > m_blocks;

    private:
        template<typename DDevice>
        friend
        class Bitset;

        template<typename DDevice>
        friend
        class ConstBitset;

        template<typename Bitset>
        friend
        struct detail::BitsetCount;

        template<typename DstDevice, typename SrcDevice>
        friend void deep_copy(Bitset<DstDevice> &dst, Bitset<SrcDevice> const &src);

        template<typename DstDevice, typename SrcDevice>
        friend void deep_copy(Bitset<DstDevice> &dst,
                              ConstBitset<SrcDevice> const &src);
    };

/// a thread-safe tensor to a const bitset
/// i.e. can only test bits
    template<typename Device>
    class ConstBitset {
    public:
        using execution_space = typename Device::execution_space;
        using size_type = unsigned int;

    private:
        enum {
            block_size = static_cast<unsigned>(sizeof(unsigned) * CHAR_BIT)
        };
        enum {
            block_mask = block_size - 1u
        };
        enum {
            block_shift = flare::detail::integral_power_of_two(block_size)
        };

    public:
        FLARE_FUNCTION
        ConstBitset() : m_size(0) {}

        FLARE_FUNCTION
        ConstBitset(Bitset<Device> const &rhs)
                : m_size(rhs.m_size), m_blocks(rhs.m_blocks) {}

        FLARE_FUNCTION
        ConstBitset(ConstBitset<Device> const &rhs)
                : m_size(rhs.m_size), m_blocks(rhs.m_blocks) {}

        FLARE_FUNCTION
        ConstBitset<Device> &operator=(Bitset<Device> const &rhs) {
            this->m_size = rhs.m_size;
            this->m_blocks = rhs.m_blocks;

            return *this;
        }

        FLARE_FUNCTION
        ConstBitset<Device> &operator=(ConstBitset<Device> const &rhs) {
            this->m_size = rhs.m_size;
            this->m_blocks = rhs.m_blocks;

            return *this;
        }

        FLARE_FORCEINLINE_FUNCTION
        unsigned size() const { return m_size; }

        unsigned count() const {
            detail::BitsetCount<ConstBitset<Device> > f(*this);
            return f.apply();
        }

        FLARE_FORCEINLINE_FUNCTION
        bool test(unsigned i) const {
            if (i < m_size) {
                const unsigned block = m_blocks[i >> block_shift];
                const unsigned mask = 1u << static_cast<int>(i & block_mask);
                return block & mask;
            }
            return false;
        }

    private:
        unsigned m_size;
        Tensor<const unsigned *, Device, MemoryTraits<RandomAccess> > m_blocks;

    private:
        template<typename DDevice>
        friend
        class ConstBitset;

        template<typename Bitset>
        friend
        struct detail::BitsetCount;

        template<typename DstDevice, typename SrcDevice>
        friend void deep_copy(Bitset<DstDevice> &dst,
                              ConstBitset<SrcDevice> const &src);

        template<typename DstDevice, typename SrcDevice>
        friend void deep_copy(ConstBitset<DstDevice> &dst,
                              ConstBitset<SrcDevice> const &src);
    };

    template<typename DstDevice, typename SrcDevice>
    void deep_copy(Bitset<DstDevice> &dst, Bitset<SrcDevice> const &src) {
        if (dst.size() != src.size()) {
            flare::detail::throw_runtime_exception(
                    "Error: Cannot deep_copy bitsets of different sizes!");
        }

        flare::fence("Bitset::deep_copy: fence before copy operation");
        flare::detail::DeepCopy<typename DstDevice::memory_space,
                typename SrcDevice::memory_space>(
                dst.m_blocks.data(), src.m_blocks.data(),
                sizeof(unsigned) * src.m_blocks.extent(0));
        flare::fence("Bitset::deep_copy: fence after copy operation");
    }

    template<typename DstDevice, typename SrcDevice>
    void deep_copy(Bitset<DstDevice> &dst, ConstBitset<SrcDevice> const &src) {
        if (dst.size() != src.size()) {
            flare::detail::throw_runtime_exception(
                    "Error: Cannot deep_copy bitsets of different sizes!");
        }

        flare::fence("Bitset::deep_copy: fence before copy operation");
        flare::detail::DeepCopy<typename DstDevice::memory_space,
                typename SrcDevice::memory_space>(
                dst.m_blocks.data(), src.m_blocks.data(),
                sizeof(unsigned) * src.m_blocks.extent(0));
        flare::fence("Bitset::deep_copy: fence after copy operation");
    }

    template<typename DstDevice, typename SrcDevice>
    void deep_copy(ConstBitset<DstDevice> &dst, ConstBitset<SrcDevice> const &src) {
        if (dst.size() != src.size()) {
            flare::detail::throw_runtime_exception(
                    "Error: Cannot deep_copy bitsets of different sizes!");
        }

        flare::fence("Bitset::deep_copy: fence before copy operation");
        flare::detail::DeepCopy<typename DstDevice::memory_space,
                typename SrcDevice::memory_space>(
                dst.m_blocks.data(), src.m_blocks.data(),
                sizeof(unsigned) * src.m_blocks.extent(0));
        flare::fence("Bitset::deep_copy: fence after copy operation");
    }

}  // namespace flare

#endif  // FLARE_BITSET_H_
