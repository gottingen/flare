// Copyright 2023 The EA Authors.
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

#include <flare.h>
#include <math.h>
#include <stdio.h>
#include <fly/util.h>
#include <iostream>

using namespace fly;
template<class ty>
dtype get_dtype();

template<>
dtype get_dtype<float>() {
    return f32;
}
template<>
dtype get_dtype<double>() {
    return f64;
}

template<class ty, bool use_barrier>
static ty monte_carlo_barrier(int N, ty K, ty t, ty vol, ty r, ty strike,
                              int steps, ty B) {
    dtype pres   = get_dtype<ty>();
    array payoff = constant(0, N, 1, pres);

    ty dt   = t / (ty)(steps - 1);
    array s = constant(strike, N, 1, pres);

    array randmat = randn(N, steps - 1, pres);
    randmat = exp((r - (vol * vol * 0.5)) * dt + vol * sqrt(dt) * randmat);

    array S = product(join(1, s, randmat), 1);

    if (use_barrier) { S = S * allTrue(S < B, 1); }

    payoff = max(0.0, S - K);
    ty P   = mean<ty>(payoff) * exp(-r * t);
    return P;
}

template<class ty, bool use_barrier>
double monte_carlo_bench(int N) {
    int steps      = 180;
    ty stock_price = 100.0;
    ty maturity    = 0.5;
    ty volatility  = .30;
    ty rate        = .01;
    ty strike      = 100;
    ty barrier     = 115.0;

    timer::start();
    for (int i = 0; i < 10; i++) {
        monte_carlo_barrier<ty, use_barrier>(
            N, stock_price, maturity, volatility, rate, strike, steps, barrier);
    }
    return timer::stop() / 10;
}

int main() {
    try {
        // Warm up and caching
        monte_carlo_bench<float, false>(1000);
        monte_carlo_bench<float, true>(1000);

        for (int n = 10000; n <= 100000; n += 10000) {
            printf(
                "Time for %7d paths - "
                "vanilla method: %4.3f ms,  "
                "barrier method: %4.3f ms\n",
                n, 1000 * monte_carlo_bench<float, false>(n),
                1000 * monte_carlo_bench<float, true>(n));
        }
    } catch (fly::exception &ae) { std::cerr << ae.what() << std::endl; }

    return 0;
}
