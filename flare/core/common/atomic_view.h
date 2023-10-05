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
#ifndef FLARE_CORE_COMMON_ATOMIC_VIEW_H_
#define FLARE_CORE_COMMON_ATOMIC_VIEW_H_

#include <flare/core/defines.h>
#include <flare/core/atomic.h>

namespace flare::detail {

    // The following tag is used to prevent an implicit call of the constructor when
    // trying to assign a literal 0 int ( = 0 );
    struct AtomicViewConstTag {
    };

    template<class ViewTraits>
    class AtomicDataElement {
    public:
        using value_type = typename ViewTraits::value_type;
        using const_value_type = typename ViewTraits::const_value_type;
        using non_const_value_type = typename ViewTraits::non_const_value_type;
        value_type *const ptr;

        FLARE_INLINE_FUNCTION
        AtomicDataElement(value_type *ptr_, AtomicViewConstTag) : ptr(ptr_) {}

        FLARE_INLINE_FUNCTION
        const_value_type operator=(const_value_type &val) const {
            flare::atomic_store(ptr, val);
            return val;
        }

        FLARE_INLINE_FUNCTION
        void inc() const { flare::atomic_increment(ptr); }

        FLARE_INLINE_FUNCTION
        void dec() const { flare::atomic_decrement(ptr); }

        FLARE_INLINE_FUNCTION
        const_value_type operator++() const {
            const_value_type tmp =
                    flare::atomic_fetch_add(ptr, non_const_value_type(1));
            return tmp + 1;
        }

        FLARE_INLINE_FUNCTION
        const_value_type operator--() const {
            const_value_type tmp =
                    flare::atomic_fetch_sub(ptr, non_const_value_type(1));
            return tmp - 1;
        }

        FLARE_INLINE_FUNCTION
        const_value_type operator++(int) const {
            return flare::atomic_fetch_add(ptr, non_const_value_type(1));
        }

        FLARE_INLINE_FUNCTION
        const_value_type operator--(int) const {
            return flare::atomic_fetch_sub(ptr, non_const_value_type(1));
        }

        FLARE_INLINE_FUNCTION
        const_value_type operator+=(const_value_type &val) const {
            const_value_type tmp = flare::atomic_fetch_add(ptr, val);
            return tmp + val;
        }

        FLARE_INLINE_FUNCTION
        const_value_type operator-=(const_value_type &val) const {
            const_value_type tmp = flare::atomic_fetch_sub(ptr, val);
            return tmp - val;
        }

        FLARE_INLINE_FUNCTION
        const_value_type operator*=(const_value_type &val) const {
            return flare::atomic_mul_fetch(ptr, val);
        }

        FLARE_INLINE_FUNCTION
        const_value_type operator/=(const_value_type &val) const {
            return flare::atomic_div_fetch(ptr, val);
        }

        FLARE_INLINE_FUNCTION
        const_value_type operator%=(const_value_type &val) const {
            return flare::atomic_mod_fetch(ptr, val);
        }

        FLARE_INLINE_FUNCTION
        const_value_type operator&=(const_value_type &val) const {
            return flare::atomic_and_fetch(ptr, val);
        }

        FLARE_INLINE_FUNCTION
        const_value_type operator^=(const_value_type &val) const {
            return flare::atomic_xor_fetch(ptr, val);
        }

        FLARE_INLINE_FUNCTION
        const_value_type operator|=(const_value_type &val) const {
            return flare::atomic_or_fetch(ptr, val);
        }

        FLARE_INLINE_FUNCTION
        const_value_type operator<<=(const_value_type &val) const {
            return flare::atomic_lshift_fetch(ptr, val);
        }

        FLARE_INLINE_FUNCTION
        const_value_type operator>>=(const_value_type &val) const {
            return flare::atomic_rshift_fetch(ptr, val);
        }

        FLARE_INLINE_FUNCTION
        const_value_type operator+(const_value_type &val) const { return *ptr + val; }

        FLARE_INLINE_FUNCTION
        const_value_type operator-(const_value_type &val) const { return *ptr - val; }

        FLARE_INLINE_FUNCTION
        const_value_type operator*(const_value_type &val) const { return *ptr * val; }

        FLARE_INLINE_FUNCTION
        const_value_type operator/(const_value_type &val) const { return *ptr / val; }

        FLARE_INLINE_FUNCTION
        const_value_type operator%(const_value_type &val) const { return *ptr ^ val; }

        FLARE_INLINE_FUNCTION
        const_value_type operator!() const { return !*ptr; }

        FLARE_INLINE_FUNCTION
        const_value_type operator&&(const_value_type &val) const {
            return *ptr && val;
        }

        FLARE_INLINE_FUNCTION
        const_value_type operator||(const_value_type &val) const {
            return *ptr | val;
        }

        FLARE_INLINE_FUNCTION
        const_value_type operator&(const_value_type &val) const { return *ptr & val; }

        FLARE_INLINE_FUNCTION
        const_value_type operator|(const_value_type &val) const { return *ptr | val; }

        FLARE_INLINE_FUNCTION
        const_value_type operator^(const_value_type &val) const { return *ptr ^ val; }

        FLARE_INLINE_FUNCTION
        const_value_type operator~() const { return ~*ptr; }

        FLARE_INLINE_FUNCTION
        const_value_type operator<<(const unsigned int &val) const {
            return *ptr << val;
        }

        FLARE_INLINE_FUNCTION
        const_value_type operator>>(const unsigned int &val) const {
            return *ptr >> val;
        }

        FLARE_INLINE_FUNCTION
        bool operator==(const AtomicDataElement &val) const { return *ptr == val; }

        FLARE_INLINE_FUNCTION
        bool operator!=(const AtomicDataElement &val) const { return *ptr != val; }

        FLARE_INLINE_FUNCTION
        bool operator>=(const_value_type &val) const { return *ptr >= val; }

        FLARE_INLINE_FUNCTION
        bool operator<=(const_value_type &val) const { return *ptr <= val; }

        FLARE_INLINE_FUNCTION
        bool operator<(const_value_type &val) const { return *ptr < val; }

        FLARE_INLINE_FUNCTION
        bool operator>(const_value_type &val) const { return *ptr > val; }

        FLARE_INLINE_FUNCTION
        operator value_type() const { return flare::atomic_load(ptr); }
    };

    template<class ViewTraits>
    class AtomicViewDataHandle {
    public:
        typename ViewTraits::value_type *ptr;

        FLARE_INLINE_FUNCTION
        AtomicViewDataHandle() : ptr(nullptr) {}

        FLARE_INLINE_FUNCTION
        AtomicViewDataHandle(typename ViewTraits::value_type *ptr_) : ptr(ptr_) {}

        template<class iType>
        FLARE_INLINE_FUNCTION AtomicDataElement<ViewTraits> operator[](
                const iType &i) const {
            return AtomicDataElement<ViewTraits>(ptr + i, AtomicViewConstTag());
        }

        FLARE_INLINE_FUNCTION
        operator typename ViewTraits::value_type *() const { return ptr; }
    };

}  // namespace flare::detail

#endif  // FLARE_CORE_COMMON_ATOMIC_VIEW_H_
