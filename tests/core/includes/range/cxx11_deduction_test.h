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
#ifndef TEST_CXX11_DEDUCTION_H_
#define TEST_CXX11_DEDUCTION_H_

#include <flare/core.h>
#include <doctest.h>


namespace TestCXX11 {

    struct TestReductionDeductionTagA {
    };
    struct TestReductionDeductionTagB {
    };

    template<class ExecSpace>
    struct TestReductionDeductionFunctor {
        // FLARE_INLINE_FUNCTION
        // void operator()( long i, long & value ) const
        // { value += i + 1; }

        FLARE_INLINE_FUNCTION
        void operator()(TestReductionDeductionTagA, long i, long &value) const {
            value += (2 * i + 1) + (2 * i + 2);
        }

        FLARE_INLINE_FUNCTION
        void operator()(const TestReductionDeductionTagB &, const long i,
                        long &value) const {
            value += (3 * i + 1) + (3 * i + 2) + (3 * i + 3);
        }
    };

    template<class ExecSpace>
    void test_reduction_deduction() {
        using Functor = TestReductionDeductionFunctor<ExecSpace>;

        const long N = 50;
        // const long answer  = N % 2 ? ( N * ( ( N + 1 ) / 2 ) ) : ( ( N / 2 ) * ( N
        // + 1 ) );
        const long answerA =
                N % 2 ? ((2 * N) * (((2 * N) + 1) / 2)) : (((2 * N) / 2) * ((2 * N) + 1));
        const long answerB =
                N % 2 ? ((3 * N) * (((3 * N) + 1) / 2)) : (((3 * N) / 2) * ((3 * N) + 1));
        long result = 0;

        // flare::parallel_reduce( flare::RangePolicy< ExecSpace >( 0, N ),
        // Functor(), result ); REQUIRE_EQ( answer, result );

        flare::parallel_reduce(
                flare::RangePolicy<ExecSpace, TestReductionDeductionTagA>(0, N),
                Functor(), result);
        REQUIRE_EQ(answerA, result);

        flare::parallel_reduce(
                flare::RangePolicy<ExecSpace, TestReductionDeductionTagB>(0, N),
                Functor(), result);
        REQUIRE_EQ(answerB, result);
    }

}  // namespace TestCXX11

namespace Test {

    TEST_CASE("TEST_CATEGORY, reduction_deduction") {
        TestCXX11::test_reduction_deduction<TEST_EXECSPACE>();
    }
}  // namespace Test
#endif  // TEST_CXX11_DEDUCTION_H_
