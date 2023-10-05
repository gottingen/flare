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
#include <doctest.h>

namespace TestCXX11 {

    template<class DeviceType>
    struct FunctorAddTest {
        using view_type = flare::View<double **, DeviceType>;
        using execution_space = DeviceType;
        using team_member = typename flare::TeamPolicy<execution_space>::member_type;

        view_type a_, b_;

        FunctorAddTest(view_type &a, view_type &b) : a_(a), b_(b) {}

        FLARE_INLINE_FUNCTION
        void operator()(const int &i) const {
            b_(i, 0) = a_(i, 1) + a_(i, 2);
            b_(i, 1) = a_(i, 0) - a_(i, 3);
            b_(i, 2) = a_(i, 4) + a_(i, 0);
            b_(i, 3) = a_(i, 2) - a_(i, 1);
            b_(i, 4) = a_(i, 3) + a_(i, 4);
        }

        FLARE_INLINE_FUNCTION
        void operator()(const team_member &dev) const {
            const int begin = dev.league_rank() * 4;
            const int end = begin + 4;
            for (int i = begin + dev.team_rank(); i < end; i += dev.team_size()) {
                b_(i, 0) = a_(i, 1) + a_(i, 2);
                b_(i, 1) = a_(i, 0) - a_(i, 3);
                b_(i, 2) = a_(i, 4) + a_(i, 0);
                b_(i, 3) = a_(i, 2) - a_(i, 1);
                b_(i, 4) = a_(i, 3) + a_(i, 4);
            }
        }
    };

    template<class DeviceType, bool PWRTest>
    double AddTestFunctor() {
        using policy_type = flare::TeamPolicy<DeviceType>;

        flare::View<double **, DeviceType> a("A", 100, 5);
        flare::View<double **, DeviceType> b("B", 100, 5);
        typename flare::View<double **, DeviceType>::HostMirror h_a =
                flare::create_mirror_view(a);
        typename flare::View<double **, DeviceType>::HostMirror h_b =
                flare::create_mirror_view(b);

        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < 5; j++) {
                h_a(i, j) = 0.1 * i / (1.1 * j + 1.0) + 0.5 * j;
            }
        }
        flare::deep_copy(a, h_a);

        if (PWRTest == false) {
            flare::parallel_for(100, FunctorAddTest<DeviceType>(a, b));
        } else {
            flare::parallel_for(policy_type(25, flare::AUTO),
                                FunctorAddTest<DeviceType>(a, b));
        }
        flare::deep_copy(h_b, b);

        double result = 0;
        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < 5; j++) {
                result += h_b(i, j);
            }
        }

        return result;
    }

#if defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)

    template<class DeviceType, bool PWRTest>
    double AddTestLambda() {
        flare::View<double **, DeviceType> a("A", 100, 5);
        flare::View<double **, DeviceType> b("B", 100, 5);
        typename flare::View<double **, DeviceType>::HostMirror h_a =
                flare::create_mirror_view(a);
        typename flare::View<double **, DeviceType>::HostMirror h_b =
                flare::create_mirror_view(b);

        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < 5; j++) {
                h_a(i, j) = 0.1 * i / (1.1 * j + 1.0) + 0.5 * j;
            }
        }
        flare::deep_copy(a, h_a);

        if (PWRTest == false) {
            flare::parallel_for(
                    100, FLARE_LAMBDA(const int &i) {
                        b(i, 0) = a(i, 1) + a(i, 2);
                        b(i, 1) = a(i, 0) - a(i, 3);
                        b(i, 2) = a(i, 4) + a(i, 0);
                        b(i, 3) = a(i, 2) - a(i, 1);
                        b(i, 4) = a(i, 3) + a(i, 4);
                    });
        } else {
            using policy_type = flare::TeamPolicy<DeviceType>;
            using team_member = typename policy_type::member_type;

            policy_type policy(25, flare::AUTO);

            flare::parallel_for(
                    policy, FLARE_LAMBDA(const team_member &dev) {
                        const unsigned int begin = dev.league_rank() * 4;
                        const unsigned int end = begin + 4;
                        for (unsigned int i = begin + dev.team_rank(); i < end;
                             i += dev.team_size()) {
                            b(i, 0) = a(i, 1) + a(i, 2);
                            b(i, 1) = a(i, 0) - a(i, 3);
                            b(i, 2) = a(i, 4) + a(i, 0);
                            b(i, 3) = a(i, 2) - a(i, 1);
                            b(i, 4) = a(i, 3) + a(i, 4);
                        }
                    });
        }
        flare::deep_copy(h_b, b);

        double result = 0;
        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < 5; j++) {
                result += h_b(i, j);
            }
        }

        return result;
    }

#else
    template <class DeviceType, bool PWRTest>
    double AddTestLambda() {
      return AddTestFunctor<DeviceType, PWRTest>();
    }
#endif

    template<class DeviceType>
    struct FunctorReduceTest {
        using view_type = flare::View<double **, DeviceType>;
        using execution_space = DeviceType;
        using value_type = double;
        using team_member = typename flare::TeamPolicy<execution_space>::member_type;

        view_type a_;

        FunctorReduceTest(view_type &a) : a_(a) {}

        FLARE_INLINE_FUNCTION
        void operator()(const int &i, value_type &sum) const {
            sum += a_(i, 1) + a_(i, 2);
            sum += a_(i, 0) - a_(i, 3);
            sum += a_(i, 4) + a_(i, 0);
            sum += a_(i, 2) - a_(i, 1);
            sum += a_(i, 3) + a_(i, 4);
        }

        FLARE_INLINE_FUNCTION
        void operator()(const team_member &dev, value_type &sum) const {
            const int begin = dev.league_rank() * 4;
            const int end = begin + 4;
            for (int i = begin + dev.team_rank(); i < end; i += dev.team_size()) {
                sum += a_(i, 1) + a_(i, 2);
                sum += a_(i, 0) - a_(i, 3);
                sum += a_(i, 4) + a_(i, 0);
                sum += a_(i, 2) - a_(i, 1);
                sum += a_(i, 3) + a_(i, 4);
            }
        }

        FLARE_INLINE_FUNCTION
        void init(value_type &update) const { update = 0.0; }

        FLARE_INLINE_FUNCTION
        void join(value_type &update, value_type const &input) const {
            update += input;
        }
    };

    template<class DeviceType, bool PWRTest>
    double ReduceTestFunctor() {
        using policy_type = flare::TeamPolicy<DeviceType>;
        using view_type = flare::View<double **, DeviceType>;
        using unmanaged_result =
                flare::View<double, flare::HostSpace, flare::MemoryUnmanaged>;

        view_type a("A", 100, 5);
        typename view_type::HostMirror h_a = flare::create_mirror_view(a);

        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < 5; j++) {
                h_a(i, j) = 0.1 * i / (1.1 * j + 1.0) + 0.5 * j;
            }
        }
        flare::deep_copy(a, h_a);

        double result = 0.0;
        if (PWRTest == false) {
            flare::parallel_reduce(100, FunctorReduceTest<DeviceType>(a),
                                   unmanaged_result(&result));
        } else {
            flare::parallel_reduce(policy_type(25, flare::AUTO),
                                   FunctorReduceTest<DeviceType>(a),
                                   unmanaged_result(&result));
        }
        flare::fence();

        return result;
    }

#if defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)

    template<class DeviceType, bool PWRTest>
    double ReduceTestLambda() {
        using policy_type = flare::TeamPolicy<DeviceType>;
        using view_type = flare::View<double **, DeviceType>;
        using unmanaged_result =
                flare::View<double, flare::HostSpace, flare::MemoryUnmanaged>;

        view_type a("A", 100, 5);
        typename view_type::HostMirror h_a = flare::create_mirror_view(a);

        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < 5; j++) {
                h_a(i, j) = 0.1 * i / (1.1 * j + 1.0) + 0.5 * j;
            }
        }
        flare::deep_copy(a, h_a);

        double result = 0.0;

        if (PWRTest == false) {
            flare::parallel_reduce(
                    100,
                    FLARE_LAMBDA(const int &i, double &sum) {
                        sum += a(i, 1) + a(i, 2);
                        sum += a(i, 0) - a(i, 3);
                        sum += a(i, 4) + a(i, 0);
                        sum += a(i, 2) - a(i, 1);
                        sum += a(i, 3) + a(i, 4);
                    },
                    unmanaged_result(&result));
        } else {
            using team_member = typename policy_type::member_type;
            flare::parallel_reduce(
                    policy_type(25, flare::AUTO),
                    FLARE_LAMBDA(const team_member &dev, double &sum) {
                        const unsigned int begin = dev.league_rank() * 4;
                        const unsigned int end = begin + 4;
                        for (unsigned int i = begin + dev.team_rank(); i < end;
                             i += dev.team_size()) {
                            sum += a(i, 1) + a(i, 2);
                            sum += a(i, 0) - a(i, 3);
                            sum += a(i, 4) + a(i, 0);
                            sum += a(i, 2) - a(i, 1);
                            sum += a(i, 3) + a(i, 4);
                        }
                    },
                    unmanaged_result(&result));
        }
        flare::fence();

        return result;
    }

#else
    template <class DeviceType, bool PWRTest>
    double ReduceTestLambda() {
      return ReduceTestFunctor<DeviceType, PWRTest>();
    }
#endif

    template<class DeviceType>
    double TestVariantLambda(int test) {
        switch (test) {
            case 1:
                return AddTestLambda<DeviceType, false>();
            case 2:
                return AddTestLambda<DeviceType, true>();
            case 3:
                return ReduceTestLambda<DeviceType, false>();
            case 4:
                return ReduceTestLambda<DeviceType, true>();
        }

        return 0;
    }

    template<class DeviceType>
    double TestVariantFunctor(int test) {
        switch (test) {
            case 1:
                return AddTestFunctor<DeviceType, false>();
            case 2:
                return AddTestFunctor<DeviceType, true>();
            case 3:
                return ReduceTestFunctor<DeviceType, false>();
            case 4:
                return ReduceTestFunctor<DeviceType, true>();
        }

        return 0;
    }

    template<class DeviceType>
    bool Test(int test) {
#ifdef FLARE_ENABLE_CXX11_DISPATCH_LAMBDA
        double res_functor = TestVariantFunctor<DeviceType>(test);
        double res_lambda = TestVariantLambda<DeviceType>(test);

        char testnames[5][256] = {" ", "AddTest", "AddTest TeamPolicy", "ReduceTest",
                                  "ReduceTest TeamPolicy"};
        bool passed = true;

        auto a = res_functor;
        auto b = res_lambda;
        // use a tolerant comparison because functors and lambdas vectorize
        // differently https://github.com/trilinos/Trilinos/issues/3233
        auto rel_err = (std::abs(b - a) / std::max(std::abs(a), std::abs(b)));
        auto tol = 1e-14;
        if (rel_err > tol) {
            passed = false;

            std::cout << "CXX11 ( test = '" << testnames[test]
                      << "' FAILED : relative error " << rel_err << " > tolerance "
                      << tol << std::endl;
        }

        return passed;
#else
        (void)test;
        return true;
#endif
    }

}  // namespace TestCXX11

namespace Test {
    TEST_CASE("TEST_CATEGORY, cxx11") {
        if (std::is_same<flare::DefaultExecutionSpace, TEST_EXECSPACE>::value) {
            REQUIRE((TestCXX11::Test<TEST_EXECSPACE>(1)));
            REQUIRE((TestCXX11::Test<TEST_EXECSPACE>(2)));
            REQUIRE((TestCXX11::Test<TEST_EXECSPACE>(3)));
            REQUIRE((TestCXX11::Test<TEST_EXECSPACE>(4)));
        }
    }

}  // namespace Test
