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

#include <sstream>
#include <iostream>
#include <limits>

#include <flare/core.h>
#include <doctest.h>

namespace Test {

    struct ReducerTag {
    };

    template<typename ScalarType, class DeviceType>
    class ReduceFunctor {
    public:
        using execution_space = DeviceType;
        using size_type = typename execution_space::size_type;

        struct value_type {
            ScalarType value[3];
        };

        const size_type nwork;

        FLARE_INLINE_FUNCTION
        ReduceFunctor(const size_type &arg_nwork) : nwork(arg_nwork) {}

        FLARE_INLINE_FUNCTION
        ReduceFunctor(const ReduceFunctor &rhs) : nwork(rhs.nwork) {}

        /*
          FLARE_INLINE_FUNCTION
          void init( value_type & dst ) const
          {
            dst.value[0] = 0;
            dst.value[1] = 0;
            dst.value[2] = 0;
          }
        */

        FLARE_INLINE_FUNCTION
        void join(value_type &dst, const value_type &src) const {
            dst.value[0] += src.value[0];
            dst.value[1] += src.value[1];
            dst.value[2] += src.value[2];
        }

        FLARE_INLINE_FUNCTION
        void operator()(size_type iwork, value_type &dst) const {
            dst.value[0] += 1;
            dst.value[1] += iwork + 1;
            dst.value[2] += nwork - iwork;
        }
    };

    template<class DeviceType>
    class ReduceFunctorFinal : public ReduceFunctor<int64_t, DeviceType> {
    public:
        using value_type = typename ReduceFunctor<int64_t, DeviceType>::value_type;

        FLARE_INLINE_FUNCTION
        ReduceFunctorFinal(const size_t n) : ReduceFunctor<int64_t, DeviceType>(n) {}

        FLARE_INLINE_FUNCTION
        void final(value_type &dst) const {
            dst.value[0] = -dst.value[0];
            dst.value[1] = -dst.value[1];
            dst.value[2] = -dst.value[2];
        }
    };

    template<class DeviceType>
    class ReduceFunctorFinalTag {
    public:
        using execution_space = DeviceType;
        using size_type = typename execution_space::size_type;
        using ScalarType = int64_t;

        struct value_type {
            ScalarType value[3];
        };

        const size_type nwork;

        FLARE_INLINE_FUNCTION
        ReduceFunctorFinalTag(const size_type arg_nwork) : nwork(arg_nwork) {}

        FLARE_INLINE_FUNCTION
        void join(const ReducerTag, value_type &dst, const value_type &src) const {
            dst.value[0] += src.value[0];
            dst.value[1] += src.value[1];
            dst.value[2] += src.value[2];
        }

        FLARE_INLINE_FUNCTION
        void operator()(const ReducerTag, size_type iwork, value_type &dst) const {
            dst.value[0] -= 1;
            dst.value[1] -= iwork + 1;
            dst.value[2] -= nwork - iwork;
        }

        FLARE_INLINE_FUNCTION
        void final(const ReducerTag, value_type &dst) const {
            ++dst.value[0];
            ++dst.value[1];
            ++dst.value[2];
        }
    };

    template<typename ScalarType, class DeviceType>
    class RuntimeReduceFunctor {
    public:
        // Required for functor:
        using execution_space = DeviceType;
        using value_type = ScalarType[];
        const unsigned value_count;

        // Unit test details:

        using size_type = typename execution_space::size_type;

        const size_type nwork;

        RuntimeReduceFunctor(const size_type arg_nwork, const size_type arg_count)
                : value_count(arg_count), nwork(arg_nwork) {}

        FLARE_INLINE_FUNCTION
        void init(ScalarType dst[]) const {
            for (unsigned i = 0; i < value_count; ++i) dst[i] = 0;
        }

        FLARE_INLINE_FUNCTION
        void join(ScalarType dst[], const ScalarType src[]) const {
            for (unsigned i = 0; i < value_count; ++i) dst[i] += src[i];
        }

        FLARE_INLINE_FUNCTION
        void operator()(size_type iwork, ScalarType dst[]) const {
            const size_type tmp[3] = {1, iwork + 1, nwork - iwork};

            for (size_type i = 0; i < static_cast<size_type>(value_count); ++i) {
                dst[i] += tmp[i % 3];
            }
        }
    };

    template<typename ScalarType, class DeviceType>
    class RuntimeReduceMinMax {
    public:
        // Required for functor:
        using execution_space = DeviceType;
        using value_type = ScalarType[];
        const unsigned value_count;

        // Unit test details:

        using size_type = typename execution_space::size_type;

        const size_type nwork;
        const ScalarType amin;
        const ScalarType amax;

        RuntimeReduceMinMax(const size_type arg_nwork, const size_type arg_count)
                : value_count(arg_count),
                  nwork(arg_nwork),
                  amin(std::numeric_limits<ScalarType>::min()),
                  amax(std::numeric_limits<ScalarType>::max()) {}

        FLARE_INLINE_FUNCTION
        void init(ScalarType dst[]) const {
            for (unsigned i = 0; i < value_count; ++i) {
                dst[i] = i % 2 ? amax : amin;
            }
        }

        FLARE_INLINE_FUNCTION
        void join(ScalarType dst[], const ScalarType src[]) const {
            for (unsigned i = 0; i < value_count; ++i) {
                dst[i] = i % 2 ? (dst[i] < src[i] ? dst[i] : src[i])   // min
                               : (dst[i] > src[i] ? dst[i] : src[i]);  // max
            }
        }

        FLARE_INLINE_FUNCTION
        void operator()(size_type iwork, ScalarType dst[]) const {
            const ScalarType tmp[2] = {ScalarType(iwork + 1),
                                       ScalarType(nwork - iwork)};

            for (size_type i = 0; i < static_cast<size_type>(value_count); ++i) {
                dst[i] = i % 2 ? (dst[i] < tmp[i % 2] ? dst[i] : tmp[i % 2])
                               : (dst[i] > tmp[i % 2] ? dst[i] : tmp[i % 2]);
            }
        }
    };

    template<class DeviceType>
    class RuntimeReduceFunctorFinal
            : public RuntimeReduceFunctor<int64_t, DeviceType> {
    public:
        using base_type = RuntimeReduceFunctor<int64_t, DeviceType>;
        using value_type = typename base_type::value_type;
        using scalar_type = int64_t;

        RuntimeReduceFunctorFinal(const size_t theNwork, const size_t count)
                : base_type(theNwork, count) {}

        FLARE_INLINE_FUNCTION
        void final(value_type dst) const {
            for (unsigned i = 0; i < base_type::value_count; ++i) {
                dst[i] = -dst[i];
            }
        }
    };

    template<class ValueType, class DeviceType>
    class CombinedReduceFunctorSameType {
    public:
        using execution_space = typename DeviceType::execution_space;
        using size_type = typename execution_space::size_type;

        const size_type nwork;

        FLARE_INLINE_FUNCTION
        constexpr explicit CombinedReduceFunctorSameType(const size_type &arg_nwork)
                : nwork(arg_nwork) {}

        FLARE_DEFAULTED_FUNCTION
        constexpr CombinedReduceFunctorSameType(
                const CombinedReduceFunctorSameType &rhs) = default;

        FLARE_INLINE_FUNCTION
        void operator()(size_type iwork, ValueType &dst1, ValueType &dst2,
                        ValueType &dst3) const {
            dst1 += 1;
            dst2 += iwork + 1;
            dst3 += nwork - iwork;
        }

        FLARE_INLINE_FUNCTION
        void operator()(size_type iwork, size_type always_zero_1,
                        size_type always_zero_2, ValueType &dst1, ValueType &dst2,
                        ValueType &dst3) const {
            dst1 += 1 + always_zero_1;
            dst2 += iwork + 1 + always_zero_2;
            dst3 += nwork - iwork;
        }
    };

    namespace {

        template<typename ScalarType, class DeviceType>
        class TestReduce {
        public:
            using execution_space = DeviceType;
            using size_type = typename execution_space::size_type;

            TestReduce(const size_type &nwork) {
                run_test(nwork);
                run_test_final(nwork);
                run_test_final_tag(nwork);
            }

            void run_test(const size_type &nwork) {
                using functor_type = Test::ReduceFunctor<ScalarType, execution_space>;
                using value_type = typename functor_type::value_type;

                enum {
                    Count = 3
                };
                enum {
                    Repeat = 100
                };

                value_type result[Repeat];

                const uint64_t nw = nwork;
                const uint64_t nsum = nw % 2 ? nw * ((nw + 1) / 2) : (nw / 2) * (nw + 1);

                for (unsigned i = 0; i < Repeat; ++i) {
                    flare::parallel_reduce(nwork, functor_type(nwork), result[i]);
                }

                for (unsigned i = 0; i < Repeat; ++i) {
                    for (unsigned j = 0; j < Count; ++j) {
                        const uint64_t correct = 0 == j % 3 ? nw : nsum;
                        REQUIRE_EQ((ScalarType) correct, result[i].value[j]);
                    }
                }
            }

            void run_test_final(const size_type &nwork) {
                using functor_type = Test::ReduceFunctorFinal<execution_space>;
                using value_type = typename functor_type::value_type;

                enum {
                    Count = 3
                };
                enum {
                    Repeat = 100
                };

                value_type result[Repeat];

                const uint64_t nw = nwork;
                const uint64_t nsum = nw % 2 ? nw * ((nw + 1) / 2) : (nw / 2) * (nw + 1);

                for (unsigned i = 0; i < Repeat; ++i) {
                    if (i % 2 == 0) {
                        flare::parallel_reduce(nwork, functor_type(nwork), result[i]);
                    } else {
                        flare::parallel_reduce("Reduce", nwork, functor_type(nwork),
                                               result[i]);
                    }
                }

                for (unsigned i = 0; i < Repeat; ++i) {
                    for (unsigned j = 0; j < Count; ++j) {
                        const uint64_t correct = 0 == j % 3 ? nw : nsum;
                        REQUIRE_EQ((ScalarType) correct, -result[i].value[j]);
                    }
                }
            }

            void run_test_final_tag(const size_type &nwork) {
                using functor_type = Test::ReduceFunctorFinalTag<execution_space>;
                using value_type = typename functor_type::value_type;

                enum {
                    Count = 3
                };
                enum {
                    Repeat = 100
                };

                value_type result[Repeat];

                const uint64_t nw = nwork;
                const uint64_t nsum = nw % 2 ? nw * ((nw + 1) / 2) : (nw / 2) * (nw + 1);

                for (unsigned i = 0; i < Repeat; ++i) {
                    if (i % 2 == 0) {
                        flare::parallel_reduce(
                                flare::RangePolicy<execution_space, ReducerTag>(0, nwork),
                                functor_type(nwork), result[i]);
                    } else {
                        flare::parallel_reduce(
                                "Reduce",
                                flare::RangePolicy<execution_space, ReducerTag>(0, nwork),
                                functor_type(nwork), result[i]);
                    }
                }

                for (unsigned i = 0; i < Repeat; ++i) {
                    for (unsigned j = 0; j < Count; ++j) {
                        const uint64_t correct = 0 == j % 3 ? nw : nsum;
                        REQUIRE_EQ((ScalarType) correct, 1 - result[i].value[j]);
                    }
                }
            }
        };

        template<typename ScalarType, class DeviceType>
        class TestReduceDynamic {
        public:
            using execution_space = DeviceType;
            using size_type = typename execution_space::size_type;

            TestReduceDynamic(const size_type nwork) {
                run_test_dynamic(nwork);
                run_test_dynamic_minmax(nwork);
                run_test_dynamic_final(nwork);
            }

            void run_test_dynamic(const size_type nwork) {
                using functor_type =
                        Test::RuntimeReduceFunctor<ScalarType, execution_space>;

                enum {
                    Count = 3
                };
                enum {
                    Repeat = 100
                };

                ScalarType result[Repeat][Count];

                const uint64_t nw = nwork;
                const uint64_t nsum = nw % 2 ? nw * ((nw + 1) / 2) : (nw / 2) * (nw + 1);

                for (unsigned i = 0; i < Repeat; ++i) {
                    if (i % 2 == 0) {
                        flare::parallel_reduce(nwork, functor_type(nwork, Count), result[i]);
                    } else {
                        flare::parallel_reduce("Reduce", nwork, functor_type(nwork, Count),
                                               result[i]);
                    }
                }

                for (unsigned i = 0; i < Repeat; ++i) {
                    for (unsigned j = 0; j < Count; ++j) {
                        const uint64_t correct = 0 == j % 3 ? nw : nsum;
                        REQUIRE_EQ((ScalarType) correct, result[i][j]);
                    }
                }
            }

            void run_test_dynamic_minmax(const size_type nwork) {
                using functor_type = Test::RuntimeReduceMinMax<ScalarType, execution_space>;

                enum {
                    Count = 2
                };
                enum {
                    Repeat = 100
                };

                ScalarType result[Repeat][Count];

                for (unsigned i = 0; i < Repeat; ++i) {
                    if (i % 2 == 0) {
                        flare::parallel_reduce(nwork, functor_type(nwork, Count), result[i]);
                    } else {
                        flare::parallel_reduce("Reduce", nwork, functor_type(nwork, Count),
                                               result[i]);
                    }
                }

                for (unsigned i = 0; i < Repeat; ++i) {
                    for (unsigned j = 0; j < Count; ++j) {
                        if (nwork == 0) {
                            ScalarType amin(std::numeric_limits<ScalarType>::min());
                            ScalarType amax(std::numeric_limits<ScalarType>::max());
                            const ScalarType correct = (j % 2) ? amax : amin;
                            REQUIRE_EQ((ScalarType) correct, result[i][j]);
                        } else {
                            const uint64_t correct = j % 2 ? 1 : nwork;
                            REQUIRE_EQ((ScalarType) correct, result[i][j]);
                        }
                    }
                }
            }

            void run_test_dynamic_final(const size_type nwork) {
                using functor_type = Test::RuntimeReduceFunctorFinal<execution_space>;

                enum {
                    Count = 3
                };
                enum {
                    Repeat = 100
                };

                typename functor_type::scalar_type result[Repeat][Count];

                const uint64_t nw = nwork;
                const uint64_t nsum = nw % 2 ? nw * ((nw + 1) / 2) : (nw / 2) * (nw + 1);

                for (unsigned i = 0; i < Repeat; ++i) {
                    if (i % 2 == 0) {
                        flare::parallel_reduce(nwork, functor_type(nwork, Count), result[i]);
                    } else {
                        flare::parallel_reduce("TestKernelReduce", nwork,
                                               functor_type(nwork, Count), result[i]);
                    }
                }

                for (unsigned i = 0; i < Repeat; ++i) {
                    for (unsigned j = 0; j < Count; ++j) {
                        const uint64_t correct = 0 == j % 3 ? nw : nsum;
                        REQUIRE_EQ((ScalarType) correct, -result[i][j]);
                    }
                }
            }
        };

        template<typename ScalarType, class DeviceType>
        class TestReduceDynamicView {
        public:
            using execution_space = DeviceType;
            using size_type = typename execution_space::size_type;

            TestReduceDynamicView(const size_type nwork) { run_test_dynamic_view(nwork); }

            void run_test_dynamic_view(const size_type nwork) {
                using functor_type =
                        Test::RuntimeReduceFunctor<ScalarType, execution_space>;

                using result_type = flare::View<ScalarType *, DeviceType>;
                using result_host_type = typename result_type::HostMirror;

                const unsigned CountLimit = 23;

                const uint64_t nw = nwork;
                const uint64_t nsum = nw % 2 ? nw * ((nw + 1) / 2) : (nw / 2) * (nw + 1);

                for (unsigned count = 0; count < CountLimit; ++count) {
                    result_type result("result", count);
                    result_host_type host_result = flare::create_mirror(result);

                    // Test result to host pointer:

                    std::string str("TestKernelReduce");
                    if (count % 2 == 0) {
                        flare::parallel_reduce(nw, functor_type(nw, count), host_result);
                    } else {
                        flare::parallel_reduce(str, nw, functor_type(nw, count), host_result);
                    }
                    flare::fence("Fence before accessing result on the host");

                    for (unsigned j = 0; j < count; ++j) {
                        const uint64_t correct = 0 == j % 3 ? nw : nsum;
                        REQUIRE_EQ(host_result(j), (ScalarType) correct);
                        host_result(j) = 0;
                    }
                }
            }
        };

    }  // namespace

    TEST_CASE("TEST_CATEGORY, int_combined_reduce") {
        using functor_type = CombinedReduceFunctorSameType<int64_t, TEST_EXECSPACE>;
        constexpr uint64_t nw = 1000;

        uint64_t nsum = (nw / 2) * (nw + 1);

        int64_t result1 = 0;
        int64_t result2 = 0;
        int64_t result3 = 0;

        flare::parallel_reduce("int_combined_reduce",
                               flare::RangePolicy<TEST_EXECSPACE>(0, nw),
                               functor_type(nw), result1, result2, result3);

        REQUIRE_EQ(nw, uint64_t(result1));
        REQUIRE_EQ(nsum, uint64_t(result2));
        REQUIRE_EQ(nsum, uint64_t(result3));
    }

    TEST_CASE("TEST_CATEGORY, mdrange_combined_reduce") {
        using functor_type = CombinedReduceFunctorSameType<int64_t, TEST_EXECSPACE>;
        constexpr uint64_t nw = 1000;

        uint64_t nsum = (nw / 2) * (nw + 1);

        int64_t result1 = 0;
        int64_t result2 = 0;
        int64_t result3 = 0;

        flare::parallel_reduce(
                "int_combined_reduce_mdrange",
                flare::MDRangePolicy<TEST_EXECSPACE, flare::Rank<3>>({{0, 0, 0}},
                                                                     {{nw, 1, 1}}),
                functor_type(nw), result1, result2, result3);

        REQUIRE_EQ(nw, uint64_t(result1));
        REQUIRE_EQ(nsum, uint64_t(result2));
        REQUIRE_EQ(nsum, uint64_t(result3));
    }

    TEST_CASE("TEST_CATEGORY, int_combined_reduce_mixed") {
        using functor_type = CombinedReduceFunctorSameType<int64_t, TEST_EXECSPACE>;

        constexpr uint64_t nw = 1000;

        uint64_t nsum = (nw / 2) * (nw + 1);
        {
            auto result1_v = flare::View<int64_t, flare::HostSpace>{"result1_v"};
            int64_t result2 = 0;
            auto result3_v = flare::View<int64_t, flare::HostSpace>{"result3_v"};
            flare::parallel_reduce("int_combined-reduce_mixed",
                                   flare::RangePolicy<TEST_EXECSPACE>(0, nw),
                                   functor_type(nw), result1_v, result2,
                                   flare::Sum<int64_t, flare::HostSpace>{result3_v});
            REQUIRE_EQ(int64_t(nw), result1_v());
            REQUIRE_EQ(int64_t(nsum), result2);
            REQUIRE_EQ(int64_t(nsum), result3_v());
        }
        {
            using MemorySpace = typename TEST_EXECSPACE::memory_space;
            auto result1_v = flare::View<int64_t, MemorySpace>{"result1_v"};
            int64_t result2 = 0;
            auto result3_v = flare::View<int64_t, MemorySpace>{"result3_v"};
            flare::parallel_reduce("int_combined-reduce_mixed",
                                   flare::RangePolicy<TEST_EXECSPACE>(0, nw),
                                   functor_type(nw), result1_v, result2,
                                   flare::Sum<int64_t, MemorySpace>{result3_v});
            int64_t result1;
            flare::deep_copy(result1, result1_v);
            REQUIRE_EQ(int64_t(nw), result1);
            REQUIRE_EQ(int64_t(nsum), result2);
            int64_t result3;
            flare::deep_copy(result3, result3_v);
            REQUIRE_EQ(int64_t(nsum), result3);
        }
    }
}  // namespace Test
