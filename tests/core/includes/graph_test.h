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

#include <flare/core.h>
#include <flare/core/graph/graph.h>
#include <doctest.h>

namespace Test {

    template<class ExecSpace>
    struct CountTestFunctor {
        using value_type = int;
        template<class T>
        using atomic_tensor =
                flare::Tensor<T, ExecSpace, flare::MemoryTraits<flare::Atomic>>;
        atomic_tensor<int> count;
        atomic_tensor<int> bugs;
        int expected_count_min;
        int expected_count_max;

        template<class... Ts>
        FLARE_FUNCTION void operator()(Ts &&...) const noexcept {
            bugs() += int(count() > expected_count_max || count() < expected_count_min);
            count()++;
        }
    };

    template<class ExecSpace, class T>
    struct SetTensorToValueFunctor {
        using value_type = T;
        using tensor_type =
                flare::Tensor<T, ExecSpace, flare::MemoryTraits<flare::Atomic>>;
        tensor_type v;
        T value;

        template<class... Ts>
        FLARE_FUNCTION void operator()(Ts &&...) const noexcept {
            v() = value;
        }
    };

    template<class ExecSpace, class T>
    struct SetResultToTensorFunctor {
        using value_type = T;
        using tensor_type =
                flare::Tensor<T, ExecSpace, flare::MemoryTraits<flare::Atomic>>;
        tensor_type v;

        template<class U>
        FLARE_FUNCTION void operator()(U &&, value_type &val) const noexcept {
            val += v();
        }
    };

    struct TEST_CATEGORY_FIXTURE(count_bugs) {
    public:
        using count_functor = CountTestFunctor<TEST_EXECSPACE>;
        using set_functor = SetTensorToValueFunctor<TEST_EXECSPACE, int>;
        using set_result_functor = SetResultToTensorFunctor<TEST_EXECSPACE, int>;
        using tensor_type = flare::Tensor<int, TEST_EXECSPACE>;
        using atomic_tensor_type = typename count_functor::template atomic_tensor<int>;
        using tensor_host = flare::Tensor<int, flare::HostSpace>;
        atomic_tensor_type count{"count"};
        atomic_tensor_type bugs{"bugs"};
        tensor_host count_host{"count_host"};
        tensor_host bugs_host{"bugs_host"};
        TEST_EXECSPACE ex{};

    protected:
        TEST_CATEGORY_FIXTURE(count_bugs)() {
            flare::deep_copy(ex, count, 0);
            flare::deep_copy(ex, bugs, 0);
            ex.fence();
        }

        ~TEST_CATEGORY_FIXTURE(count_bugs)() {}
    };

    TEST_CASE_FIXTURE(TEST_CATEGORY_FIXTURE(count_bugs), "launch_one") {
        auto graph =
                flare::experimental::create_graph<TEST_EXECSPACE>([&](auto root) {
                    root.then_parallel_for(1, count_functor{count, bugs, 0, 0});
                });
        graph.submit();
        flare::deep_copy(graph.get_execution_space(), count_host, count);
        flare::deep_copy(graph.get_execution_space(), bugs_host, bugs);
        graph.get_execution_space().fence();
        REQUIRE_EQ(1, count_host());
        REQUIRE_EQ(0, bugs_host());
    }

    TEST_CASE_FIXTURE(TEST_CATEGORY_FIXTURE(count_bugs), "launch_one_rvalue") {
        flare::experimental::create_graph(ex, [&](auto root) {
            root.then_parallel_for(1, count_functor{count, bugs, 0, 0});
        }).submit();
        flare::deep_copy(ex, count_host, count);
        flare::deep_copy(ex, bugs_host, bugs);
        ex.fence();
        REQUIRE_EQ(1, count_host());
        REQUIRE_EQ(0, bugs_host());
    }

    TEST_CASE_FIXTURE(TEST_CATEGORY_FIXTURE(count_bugs), "launch_six") {
        auto graph = flare::experimental::create_graph(ex, [&](auto root) {
            auto f_setup_count = root.then_parallel_for(1, set_functor{count, 0});
            auto f_setup_bugs = root.then_parallel_for(1, set_functor{bugs, 0});

            //----------------------------------------
            auto ready = flare::experimental::when_all(f_setup_count, f_setup_bugs);

            //----------------------------------------
            ready.then_parallel_for(1, count_functor{count, bugs, 0, 6});
            //----------------------------------------
            ready.then_parallel_for(flare::RangePolicy<TEST_EXECSPACE>{0, 1},
                                    count_functor{count, bugs, 0, 6});
            //----------------------------------------
            ready.then_parallel_for(
                    flare::MDRangePolicy<TEST_EXECSPACE, flare::Rank<2>>{{0, 0},
                                                                         {1, 1}},
                    count_functor{count, bugs, 0, 6});
            //----------------------------------------
            ready.then_parallel_for(flare::TeamPolicy<TEST_EXECSPACE>{1, 1},
                                    count_functor{count, bugs, 0, 6});
            //----------------------------------------
            ready.then_parallel_for(2, count_functor{count, bugs, 0, 6});
            //----------------------------------------
        });
        graph.submit();
        flare::deep_copy(ex, count_host, count);
        flare::deep_copy(ex, bugs_host, bugs);
        ex.fence();

        REQUIRE_EQ(6, count_host());
        REQUIRE_EQ(0, bugs_host());
    }

    TEST_CASE_FIXTURE(TEST_CATEGORY_FIXTURE(count_bugs), "when_all_cycle") {
        tensor_type reduction_out{"reduction_out"};
        tensor_host reduction_host{"reduction_host"};
        flare::experimental::create_graph(ex, [&](auto root) {
            //----------------------------------------
            // Test when_all when redundant dependencies are given
            auto f1 = root.then_parallel_for(1, set_functor{count, 0});
            auto f2 = f1.then_parallel_for(1, count_functor{count, bugs, 0, 0});
            auto f3 = f2.then_parallel_for(5, count_functor{count, bugs, 1, 5});
            auto f4 = flare::experimental::when_all(f2, f3).then_parallel_for(
                    1, count_functor{count, bugs, 6, 6});
            flare::experimental::when_all(f1, f4, f3)
                    .then_parallel_reduce(6, set_result_functor{count}, reduction_out);
            //----------------------------------------
        }).submit();
        flare::deep_copy(ex, bugs_host, bugs);
        flare::deep_copy(ex, count_host, count);
        flare::deep_copy(ex, reduction_host, reduction_out);
        ex.fence();
        REQUIRE_EQ(0, bugs_host());
        REQUIRE_EQ(7, count_host());
        REQUIRE_EQ(42, reduction_host());
        //----------------------------------------
    }

    // This test is disabled because we don't currently support copying to host,
    // even asynchronously. We _may_ want to do that eventually?
    TEST_CASE_FIXTURE(TEST_CATEGORY_FIXTURE(count_bugs), "DISABLED_repeat_chain") {
        auto graph = flare::experimental::create_graph(
                ex, [&, count_host = count_host](auto root) {
                    //----------------------------------------
                    root.then_parallel_for(1, set_functor{count, 0})
                            .then_parallel_for(1, count_functor{count, bugs, 0, 0})
                            .then_parallel_for(1, count_functor{count, bugs, 1, 1})
                            .then_parallel_reduce(1, set_result_functor{count}, count_host)
                            .then_parallel_reduce(
                                    1, set_result_functor{bugs},
                                    flare::Sum<int, flare::HostSpace>{bugs_host});
                    //----------------------------------------
                });

        //----------------------------------------
        constexpr int repeats = 10;

        for (int i = 0; i < repeats; ++i) {
            graph.submit();
            ex.fence();
            REQUIRE_EQ(2, count_host());
            REQUIRE_EQ(0, bugs_host());
        }
        //----------------------------------------
    }

    TEST_CASE_FIXTURE(TEST_CATEGORY_FIXTURE(count_bugs), "zero_work_reduce") {
        auto graph = flare::experimental::create_graph(ex, [&](auto root) {
            root.then_parallel_reduce(0, set_result_functor{bugs}, count);
        });
// These fences are only necessary because of the weirdness of how CUDA
// UVM works on pre pascal cards.
#if defined(FLARE_ON_CUDA_DEVICE) && \
    (defined(FLARE_ARCH_KEPLER) || defined(FLARE_ARCH_MAXWELL))
        flare::fence();
#endif
        graph.submit();
        flare::deep_copy(ex, count, 1);
// These fences are only necessary because of the weirdness of how CUDA
// UVM works on pre pascal cards.
#if defined(FLARE_ON_CUDA_DEVICE) && \
    (defined(FLARE_ARCH_KEPLER) || defined(FLARE_ARCH_MAXWELL))
        flare::fence();
#endif
        graph.submit();  // should reset to 0, but doesn't
        flare::deep_copy(ex, count_host, count);
        ex.fence();
        REQUIRE_EQ(count_host(), 0);
    }

}  // end namespace Test
