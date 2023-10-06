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

#ifndef FLARE_BITSET_TEST_H_
#define FLARE_BITSET_TEST_H_

#include <doctest.h>
#include <iostream>
#include <flare/core.h>
#include <flare/bitset.h>
#include <array>

namespace Test {

    namespace detail {

        template<typename Bitset, bool Set>
        struct TestBitset {
            using bitset_type = Bitset;
            using execution_space = typename bitset_type::execution_space;
            using value_type = uint32_t;

            bitset_type m_bitset;

            TestBitset(bitset_type const &bitset) : m_bitset(bitset) {}

            unsigned testit(unsigned collisions) {
                execution_space().fence();

                unsigned count = 0;
                flare::parallel_reduce(m_bitset.size() * collisions, *this, count);
                return count;
            }

            FLARE_INLINE_FUNCTION
            void init(value_type &v) const { v = 0; }

            FLARE_INLINE_FUNCTION
            void join(value_type &dst, const value_type &src) const { dst += src; }

            FLARE_INLINE_FUNCTION
            void operator()(uint32_t i, value_type &v) const {
                i = i % m_bitset.size();
                if (Set) {
                    if (m_bitset.set(i)) {
                        if (m_bitset.test(i)) ++v;
                    }
                } else {
                    if (m_bitset.reset(i)) {
                        if (!m_bitset.test(i)) ++v;
                    }
                }
            }
        };

        template<typename Bitset>
        struct TestBitsetTest {
            using bitset_type = Bitset;
            using execution_space = typename bitset_type::execution_space;
            using value_type = uint32_t;

            bitset_type m_bitset;

            TestBitsetTest(bitset_type const &bitset) : m_bitset(bitset) {}

            unsigned testit() {
                execution_space().fence();

                unsigned count = 0;
                flare::parallel_reduce(m_bitset.size(), *this, count);
                return count;
            }

            FLARE_INLINE_FUNCTION
            void init(value_type &v) const { v = 0; }

            FLARE_INLINE_FUNCTION
            void join(value_type &dst, const value_type &src) const { dst += src; }

            FLARE_INLINE_FUNCTION
            void operator()(uint32_t i, value_type &v) const {
                if (m_bitset.test(i)) ++v;
            }
        };

        template<typename Bitset, bool Set>
        struct TestBitsetAny {
            using bitset_type = Bitset;
            using execution_space = typename bitset_type::execution_space;
            using value_type = uint32_t;

            bitset_type m_bitset;

            TestBitsetAny(bitset_type const &bitset) : m_bitset(bitset) {}

            unsigned testit() {
                execution_space().fence();

                unsigned count = 0;
                flare::parallel_reduce(m_bitset.size(), *this, count);
                return count;
            }

            FLARE_INLINE_FUNCTION
            void init(value_type &v) const { v = 0; }

            FLARE_INLINE_FUNCTION
            void join(value_type &dst, const value_type &src) const { dst += src; }

            FLARE_INLINE_FUNCTION
            void operator()(uint32_t i, value_type &v) const {
                bool result = false;
                unsigned attempts = 0;
                uint32_t hint = (i >> 4) << 4;
                while (attempts < m_bitset.max_hint()) {
                    if (Set) {
                        flare::tie(result, hint) = m_bitset.find_any_unset_near(hint, i);
                        if (result && m_bitset.set(hint)) {
                            ++v;
                            break;
                        } else if (!result) {
                            ++attempts;
                        }
                    } else {
                        flare::tie(result, hint) = m_bitset.find_any_set_near(hint, i);
                        if (result && m_bitset.reset(hint)) {
                            ++v;
                            break;
                        } else if (!result) {
                            ++attempts;
                        }
                    }
                }
            }
        };
    }  // namespace detail

    template<typename Device>
    void test_bitset() {
        using bitset_type = flare::Bitset<Device>;
        using const_bitset_type = flare::ConstBitset<Device>;

        {
            unsigned ts = 100u;
            bitset_type b1;
            REQUIRE(b1.is_allocated());

            b1 = bitset_type(ts);
            bitset_type b2(b1);
            bitset_type b3(ts);

            REQUIRE(b1.is_allocated());
            REQUIRE(b2.is_allocated());
            REQUIRE(b3.is_allocated());
        }

        std::array<unsigned, 7> test_sizes = {
                {0u, 10u, 100u, 1000u, 1u << 14, 1u << 16, 10000001}};

        for (const auto test_size: test_sizes) {
            // std::cout << "Bitset " << test_sizes[i] << std::endl;

            bitset_type bitset(test_size);

            // std::cout << "  Check initial count " << std::endl;
            // nothing should be set
            {
                detail::TestBitsetTest<bitset_type> f(bitset);
                uint32_t count = f.testit();
                REQUIRE_EQ(0u, count);
                REQUIRE_EQ(count, bitset.count());
            }

            // std::cout << "  Check set() " << std::endl;
            bitset.set();
            // everything should be set
            {
                detail::TestBitsetTest<const_bitset_type> f(bitset);
                uint32_t count = f.testit();
                REQUIRE_EQ(bitset.size(), count);
                REQUIRE_EQ(count, bitset.count());
            }

            // std::cout << "  Check reset() " << std::endl;
            bitset.reset();
            REQUIRE_EQ(0u, bitset.count());

            // std::cout << "  Check set(i) " << std::endl;
            // test setting bits
            {
                detail::TestBitset<bitset_type, true> f(bitset);
                uint32_t count = f.testit(10u);
                REQUIRE_EQ(bitset.size(), bitset.count());
                REQUIRE_EQ(bitset.size(), count);
            }

            // std::cout << "  Check reset(i) " << std::endl;
            // test resetting bits
            {
                detail::TestBitset<bitset_type, false> f(bitset);
                uint32_t count = f.testit(10u);
                REQUIRE_EQ(bitset.size(), count);
                REQUIRE_EQ(0u, bitset.count());
            }

            // std::cout << "  Check find_any_set(i) " << std::endl;
            // test setting any bits
            {
                detail::TestBitsetAny<bitset_type, true> f(bitset);
                uint32_t count = f.testit();
                REQUIRE_EQ(bitset.size(), bitset.count());
                REQUIRE_EQ(bitset.size(), count);
            }

            // std::cout << "  Check find_any_unset(i) " << std::endl;
            // test resetting any bits
            {
                detail::TestBitsetAny<bitset_type, false> f(bitset);
                uint32_t count = f.testit();
                REQUIRE_EQ(bitset.size(), count);
                REQUIRE_EQ(0u, bitset.count());
            }
        }
    }

    TEST_CASE("TEST_CATEGORY, bitset") { test_bitset<TEST_EXECSPACE>(); }
}  // namespace Test

#endif  // FLARE_BITSET_TEST_H_
