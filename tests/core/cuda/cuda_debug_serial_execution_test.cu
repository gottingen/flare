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
#include <cuda_category_test.h>
#include <doctest.h>

namespace Test {

    using ViewType = flare::View<double *>;

    struct TestForFunctor {
        ViewType a;
        ViewType b;

        TestForFunctor(int N) : a(ViewType("A", N)), b(ViewType("B", N)) {}

        FLARE_INLINE_FUNCTION
        void operator()(int i) const { a(i) = b(i); }

        double time_par_for() {
            flare::Timer timer;
            flare::parallel_for("CudaDebugSerialExecution::par_for", a.extent(0),
                                *this);
            flare::fence();
            return timer.seconds();
        }
    };

    struct TestRedFunctor {
        ViewType a;
        ViewType b;

        TestRedFunctor(int N) : a(ViewType("A", N)), b(ViewType("B", N)) {}

        FLARE_INLINE_FUNCTION
        void operator()(int i, double &val) const { val += a(i) * b(i); }

        double time_par_red() {
            flare::Timer timer;
            double dot;
            flare::parallel_reduce("CudaDebugSerialExecution::par_red", a.extent(0),
                                   *this, dot);
            flare::fence();
            return timer.seconds();
        }
    };

    struct TestScanFunctor {
        ViewType a;
        ViewType b;

        TestScanFunctor(int N) : a(ViewType("A", N)), b(ViewType("B", N)) {}

        FLARE_INLINE_FUNCTION
        void operator()(int i, double &val, bool final) const {
            val += b(i);
            if (final) a(i) = val;
        }

        double time_par_scan() {
            flare::Timer timer;
            double dot;
            flare::parallel_scan("CudaDebugSerialExecution::par_scan", a.extent(0),
                                 *this, dot);
            flare::fence();
            return timer.seconds();
        }
    };

    TEST_CASE("cuda debug_serial_execution") {
        double time_par_for_1, time_par_for_2, time_par_for_serial;
        double time_par_red_1, time_par_red_2, time_par_red_serial;
        double time_par_scan_1, time_par_scan_2, time_par_scan_serial;

        int N = 10000000;
        {
            TestForFunctor f(N);
            f.time_par_for();
            time_par_for_1 = f.time_par_for();
#ifdef FLARE_IMPL_DEBUG_CUDA_SERIAL_EXECUTION
            flare_impl_cuda_set_serial_execution(true);
#endif
            time_par_for_serial = f.time_par_for();
#ifdef FLARE_IMPL_DEBUG_CUDA_SERIAL_EXECUTION
            flare_impl_cuda_set_serial_execution(false);
#endif
            time_par_for_2 = f.time_par_for();

            bool passed_par_for =
#ifdef FLARE_IMPL_DEBUG_CUDA_SERIAL_EXECUTION
                    (time_par_for_serial > time_par_for_1 * 20.0) &&
                    (time_par_for_serial > time_par_for_2 * 20.0);
#else
                    (time_par_for_serial < time_par_for_1 * 2.0) &&
                    (time_par_for_serial < time_par_for_2 * 2.0);
#endif
            if (!passed_par_for)
                printf("Time For1: %lf For2: %lf ForSerial: %lf\n", time_par_for_1,
                       time_par_for_2, time_par_for_serial);
            REQUIRE(passed_par_for);
        }
        {
            TestRedFunctor f(N);
            f.time_par_red();
            time_par_red_1 = f.time_par_red();
#ifdef FLARE_IMPL_DEBUG_CUDA_SERIAL_EXECUTION
            flare_impl_cuda_set_serial_execution(true);
#endif
            time_par_red_serial = f.time_par_red();
#ifdef FLARE_IMPL_DEBUG_CUDA_SERIAL_EXECUTION
            flare_impl_cuda_set_serial_execution(false);
#endif
            time_par_red_2 = f.time_par_red();

            bool passed_par_red =
#ifdef FLARE_IMPL_DEBUG_CUDA_SERIAL_EXECUTION
                    (time_par_red_serial > time_par_red_1 * 2.0) &&
                    (time_par_red_serial > time_par_red_2 * 2.0);
#else
                    (time_par_red_serial < time_par_red_1 * 2.0) &&
                    (time_par_red_serial < time_par_red_2 * 2.0);
#endif
            if (!passed_par_red)
                printf("Time Red1: %lf Red2: %lf RedSerial: %lf\n", time_par_red_1,
                       time_par_red_2, time_par_red_serial);
            REQUIRE(passed_par_red);
        }
        {
            TestScanFunctor f(N);
            f.time_par_scan();
            time_par_scan_1 = f.time_par_scan();
#ifdef FLARE_IMPL_DEBUG_CUDA_SERIAL_EXECUTION
            flare_impl_cuda_set_serial_execution(true);
#endif
            time_par_scan_serial = f.time_par_scan();
#ifdef FLARE_IMPL_DEBUG_CUDA_SERIAL_EXECUTION
            flare_impl_cuda_set_serial_execution(false);
#endif
            time_par_scan_2 = f.time_par_scan();

            bool passed_par_scan =
#ifdef FLARE_IMPL_DEBUG_CUDA_SERIAL_EXECUTION
                    (time_par_scan_serial > time_par_scan_1 * 2.0) &&
                    (time_par_scan_serial > time_par_scan_2 * 2.0);
#else
                    (time_par_scan_serial < time_par_scan_1 * 2.0) &&
                    (time_par_scan_serial < time_par_scan_2 * 2.0);
#endif
            if (!passed_par_scan)
                printf("Time Scan1: %lf Scan2: %lf ScanSerial: %lf\n", time_par_scan_1,
                       time_par_scan_2, time_par_scan_serial);
            REQUIRE(passed_par_scan);
        }
        printf("pass\n");
    }

}  // namespace Test
