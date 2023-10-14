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

#include <doctest.h>
#include <flare/core.h>

namespace Test {
    template<class T>
    struct Greater {
        FLARE_FUNCTION constexpr bool operator()(T const &lhs, T const &rhs) {
            return lhs > rhs;
        }
    };

    struct PairIntCompareFirst {
        int first;
        int second;

    private:
        friend FLARE_FUNCTION constexpr bool operator<(
                PairIntCompareFirst const &lhs, PairIntCompareFirst const &rhs) {
            return lhs.first < rhs.first;
        }
    };
}  // namespace Test

// ----------------------------------------------------------
// test max()
// ----------------------------------------------------------
TEST_CASE("TEST_CATEGORY, max") {
    int a = 1;
    int b = 2;
    REQUIRE_EQ(flare::max(a, b), 2);

    a = 3;
    b = 1;
    REQUIRE_EQ(flare::max(a, b), 3);

    static_assert(flare::max(1, 2) == 2);
    static_assert(flare::max(1, 2, ::Test::Greater<int>{}) == 1);

    REQUIRE_EQ(flare::max({3.f, -1.f, 0.f}), 3.f);

    static_assert(flare::max({3, -1, 0}) == 3);
    static_assert(flare::max({3, -1, 0}, ::Test::Greater<int>{}) == -1);

    static_assert(flare::max({
                                     ::Test::PairIntCompareFirst{255, 0},
                                     ::Test::PairIntCompareFirst{255, 1},
                                     ::Test::PairIntCompareFirst{0, 2},
                                     ::Test::PairIntCompareFirst{0, 3},
                                     ::Test::PairIntCompareFirst{255, 4},
                                     ::Test::PairIntCompareFirst{0, 5},
                             })
                          .second == 0);  // leftmost element
}

template<class TensorType>
struct StdAlgoMinMaxOpsTestMax {
    TensorType m_tensor;

    FLARE_INLINE_FUNCTION
    void operator()(const int &ind) const {
        auto v1 = 10.;
        if (flare::max(v1, m_tensor(ind)) == 10.) {
            m_tensor(ind) = 6.;
        }
    }

    FLARE_INLINE_FUNCTION
    StdAlgoMinMaxOpsTestMax(TensorType aIn) : m_tensor(aIn) {}
};

TEST_CASE("TEST_CATEGORY, max_within_parfor") {
    namespace KE = flare::experimental;

    using tensor_t = flare::Tensor<double *>;
    tensor_t a("a", 10);

    StdAlgoMinMaxOpsTestMax<tensor_t> fnc(a);
    flare::parallel_for(a.extent(0), fnc);
    auto a_h = flare::create_mirror_tensor_and_copy(flare::HostSpace(), a);
    for (int i = 0; i < 10; ++i) {
        REQUIRE_EQ(a_h(0), 6.);
    }
}

// ----------------------------------------------------------
// test min()
// ----------------------------------------------------------
TEST_CASE("TEST_CATEGORY, min") {
    int a = 1;
    int b = 2;
    REQUIRE_EQ(flare::min(a, b), 1);

    a = 3;
    b = 2;
    REQUIRE_EQ(flare::min(a, b), 2);

    static_assert(flare::min(3.f, 2.f) == 2.f);
    static_assert(flare::min(3.f, 2.f, ::Test::Greater<int>{}) == 3.f);

    REQUIRE_EQ(flare::min({3.f, -1.f, 0.f}), -1.f);

    static_assert(flare::min({3, -1, 0}) == -1);
    static_assert(flare::min({3, -1, 0}, ::Test::Greater<int>{}) == 3);

    static_assert(flare::min({
                                     ::Test::PairIntCompareFirst{255, 0},
                                     ::Test::PairIntCompareFirst{255, 1},
                                     ::Test::PairIntCompareFirst{0, 2},
                                     ::Test::PairIntCompareFirst{0, 3},
                                     ::Test::PairIntCompareFirst{255, 4},
                                     ::Test::PairIntCompareFirst{0, 5},
                             })
                          .second == 2);  // leftmost element
}

template<class TensorType>
struct StdAlgoMinMaxOpsTestMin {
    TensorType m_tensor;

    FLARE_INLINE_FUNCTION
    void operator()(const int &ind) const {
        auto v1 = 10.;
        if (flare::min(v1, m_tensor(ind)) == 0.) {
            m_tensor(ind) = 8.;
        }
    }

    FLARE_INLINE_FUNCTION
    StdAlgoMinMaxOpsTestMin(TensorType aIn) : m_tensor(aIn) {}
};

TEST_CASE("TEST_CATEGORY, min_within_parfor") {
    namespace KE = flare::experimental;
    using tensor_t = flare::Tensor<double *>;
    tensor_t a("a", 10);

    StdAlgoMinMaxOpsTestMin<tensor_t> fnc(a);
    flare::parallel_for(a.extent(0), fnc);
    auto a_h = flare::create_mirror_tensor_and_copy(flare::HostSpace(), a);
    for (int i = 0; i < 10; ++i) {
        REQUIRE_EQ(a_h(0), 8.);
    }
}

// ----------------------------------------------------------
// test minmax()
// ----------------------------------------------------------
TEST_CASE("TEST_CATEGORY, minmax") {
    int a = 1;
    int b = 2;
    const auto &r = flare::minmax(a, b);
    REQUIRE_EQ(r.first, 1);
    REQUIRE_EQ(r.second, 2);

    a = 3;
    b = 2;
    const auto &r2 = flare::minmax(a, b);
    REQUIRE_EQ(r2.first, 2);
    REQUIRE_EQ(r2.second, 3);

    static_assert((flare::pair<float, float>(flare::minmax(3.f, 2.f)) ==
                   flare::make_pair(2.f, 3.f)));
    static_assert(
            (flare::pair<float, float>(flare::minmax(
                    3.f, 2.f, ::Test::Greater<int>{})) == flare::make_pair(3.f, 2.f)));

    REQUIRE_EQ(flare::minmax({3.f, -1.f, 0.f}), flare::make_pair(-1.f, 3.f));

    static_assert(flare::minmax({3, -1, 0}) == flare::make_pair(-1, 3));
    static_assert(flare::minmax({3, -1, 0}, ::Test::Greater<int>{}) ==
                  flare::make_pair(3, -1));

    static_assert(flare::minmax({
                                        ::Test::PairIntCompareFirst{255, 0},
                                        ::Test::PairIntCompareFirst{255, 1},
                                        ::Test::PairIntCompareFirst{0, 2},
                                        ::Test::PairIntCompareFirst{0, 3},
                                        ::Test::PairIntCompareFirst{255, 4},
                                        ::Test::PairIntCompareFirst{0, 5},
                                })
                          .first.second == 2);  // leftmost
    static_assert(flare::minmax({
                                        ::Test::PairIntCompareFirst{255, 0},
                                        ::Test::PairIntCompareFirst{255, 1},
                                        ::Test::PairIntCompareFirst{0, 2},
                                        ::Test::PairIntCompareFirst{0, 3},
                                        ::Test::PairIntCompareFirst{255, 4},
                                        ::Test::PairIntCompareFirst{0, 5},
                                })
                          .second.second == 4);  // rightmost
}

template<class TensorType>
struct StdAlgoMinMaxOpsTestMinMax {
    TensorType m_tensor;

    FLARE_INLINE_FUNCTION
    void operator()(const int &ind) const {
        auto v1 = 7.;
        const auto &r = flare::minmax(v1, m_tensor(ind));
        m_tensor(ind) = (double) (r.first - r.second);
    }

    FLARE_INLINE_FUNCTION
    StdAlgoMinMaxOpsTestMinMax(TensorType aIn) : m_tensor(aIn) {}
};

TEST_CASE("TEST_CATEGORY, minmax_within_parfor") {
    using tensor_t = flare::Tensor<double *>;
    tensor_t a("a", 10);

    StdAlgoMinMaxOpsTestMinMax<tensor_t> fnc(a);
    flare::parallel_for(a.extent(0), fnc);
    auto a_h = flare::create_mirror_tensor_and_copy(flare::HostSpace(), a);
    for (int i = 0; i < 10; ++i) {
        REQUIRE_EQ(a_h(0), -7.);
    }
}

// ----------------------------------------------------------
// test clamp()
// ----------------------------------------------------------
TEST_CASE("TEST_CATEGORY, clamp") {
    int a = 1;
    int b = 2;
    int c = 19;
    const auto &r = flare::clamp(a, b, c);
    REQUIRE_EQ(&r, &b);
    REQUIRE_EQ(r, b);

    a = 5;
    b = -2;
    c = 3;
    const auto &r2 = flare::clamp(a, b, c);
    REQUIRE_EQ(&r2, &c);
    REQUIRE_EQ(r2, c);

    a = 5;
    b = -2;
    c = 7;
    const auto &r3 = flare::clamp(a, b, c);
    REQUIRE_EQ(&r3, &a);
    REQUIRE_EQ(r3, a);
}

template<class TensorType>
struct StdAlgoMinMaxOpsTestClamp {
    TensorType m_tensor;

    FLARE_INLINE_FUNCTION
    void operator()(const int &ind) const {
        m_tensor(ind) = 10.;
        const auto b = -2.;
        const auto c = 3.;
        const auto &r = flare::clamp(m_tensor(ind), b, c);
        m_tensor(ind) = (double) (r);
    }

    FLARE_INLINE_FUNCTION
    StdAlgoMinMaxOpsTestClamp(TensorType aIn) : m_tensor(aIn) {}
};

TEST_CASE("TEST_CATEGORY, clamp_within_parfor") {
    using tensor_t = flare::Tensor<double *>;
    tensor_t a("a", 10);

    StdAlgoMinMaxOpsTestClamp<tensor_t> fnc(a);
    flare::parallel_for(a.extent(0), fnc);
    auto a_h = flare::create_mirror_tensor_and_copy(flare::HostSpace(), a);
    for (std::size_t i = 0; i < a.extent(0); ++i) {
        REQUIRE_EQ(a_h(0), 3.);
    }
}
