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

#ifndef FLARE_ALGORITHMS_UNITTESTS_TEST_RANDOM_HPP
#define FLARE_ALGORITHMS_UNITTESTS_TEST_RANDOM_HPP

#include <doctest.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <flare/dyn_rank_view.h>
#include <flare/timer.h>
#include <flare/core.h>
#include <flare/random.h>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>

namespace Test {
namespace AlgoRandomImpl {

// This test runs the random number generators and uses some statistic tests to
// check the 'goodness' of the random numbers:
//    (i)   mean:         the mean is expected to be 0.5*RAND_MAX
//    (ii)  variance:     the variance is 1/3*mean*mean
//    (iii) covariance:   the covariance is 0
//    (iv)  1-tupledistr: the mean, variance and covariance of a 1D Histrogram
//    of random numbers (v)   3-tupledistr: the mean, variance and covariance of
//    a 3D Histrogram of random numbers

#define HIST_DIM3D 24
#define HIST_DIM1D (HIST_DIM3D * HIST_DIM3D * HIST_DIM3D)

struct RandomProperties {
  uint64_t count;
  double mean;
  double variance;
  double covariance;
  double min;
  double max;

  FLARE_INLINE_FUNCTION
  RandomProperties() {
    count      = 0;
    mean       = 0.0;
    variance   = 0.0;
    covariance = 0.0;
    min        = 1e64;
    max        = -1e64;
  }

  FLARE_INLINE_FUNCTION
  RandomProperties& operator+=(const RandomProperties& add) {
    count += add.count;
    mean += add.mean;
    variance += add.variance;
    covariance += add.covariance;
    min = add.min < min ? add.min : min;
    max = add.max > max ? add.max : max;
    return *this;
  }
};

FLARE_INLINE_FUNCTION
RandomProperties operator+(const RandomProperties& org,
                           const RandomProperties& add) {
  RandomProperties val = org;
  val += add;
  return val;
}

template <class GeneratorPool, class Scalar>
struct test_random_functor {
  using rnd_type = typename GeneratorPool::generator_type;

  using value_type  = RandomProperties;
  using device_type = typename GeneratorPool::device_type;

  GeneratorPool rand_pool;
  const double mean;

  // NOTE (mfh 03 Nov 2014): flare::rand::max() is supposed to define
  // an exclusive upper bound on the range of random numbers that
  // draw() can generate.  However, for the float specialization, some
  // implementations might violate this upper bound, due to rounding
  // error.  Just in case, we leave an extra space at the end of each
  // dimension, in the View types below.
  using type_1d =
      flare::View<int[HIST_DIM1D + 1], typename GeneratorPool::device_type>;
  type_1d density_1d;
  using type_3d =
      flare::View<int[HIST_DIM3D + 1][HIST_DIM3D + 1][HIST_DIM3D + 1],
                   typename GeneratorPool::device_type>;
  type_3d density_3d;

  test_random_functor(GeneratorPool rand_pool_, type_1d d1d, type_3d d3d)
      : rand_pool(rand_pool_),
        mean(0.5 * flare::rand<rnd_type, Scalar>::max()),
        density_1d(d1d),
        density_3d(d3d) {}

  FLARE_INLINE_FUNCTION
  void operator()(int /*i*/, RandomProperties& prop) const {
    using flare::atomic_fetch_add;

    rnd_type rand_gen = rand_pool.get_state();
    for (int k = 0; k < 1024; ++k) {
      const Scalar tmp = flare::rand<rnd_type, Scalar>::draw(rand_gen);
      prop.count++;
      prop.mean += tmp;
      prop.variance += (tmp - mean) * (tmp - mean);
      const Scalar tmp2 = flare::rand<rnd_type, Scalar>::draw(rand_gen);
      prop.count++;
      prop.mean += tmp2;
      prop.variance += (tmp2 - mean) * (tmp2 - mean);
      prop.covariance += (tmp - mean) * (tmp2 - mean);
      const Scalar tmp3 = flare::rand<rnd_type, Scalar>::draw(rand_gen);
      prop.count++;
      prop.mean += tmp3;
      prop.variance += (tmp3 - mean) * (tmp3 - mean);
      prop.covariance += (tmp2 - mean) * (tmp3 - mean);

      // NOTE (mfh 03 Nov 2014): flare::rand::max() is supposed to
      // define an exclusive upper bound on the range of random
      // numbers that draw() can generate.  However, for the float
      // specialization, some implementations might violate this upper
      // bound, due to rounding error.  Just in case, we have left an
      // extra space at the end of each dimension of density_1d and
      // density_3d.
      //
      // Please note that those extra entries might not get counted in
      // the histograms.  However, if flare::rand is broken and only
      // returns values of max(), the histograms will still catch this
      // indirectly, since none of the other values will be filled in.

      const Scalar theMax = flare::rand<rnd_type, Scalar>::max();

      const uint64_t ind1_1d =
          static_cast<uint64_t>(1.0 * HIST_DIM1D * tmp / theMax);
      const uint64_t ind2_1d =
          static_cast<uint64_t>(1.0 * HIST_DIM1D * tmp2 / theMax);
      const uint64_t ind3_1d =
          static_cast<uint64_t>(1.0 * HIST_DIM1D * tmp3 / theMax);

      const uint64_t ind1_3d =
          static_cast<uint64_t>(1.0 * HIST_DIM3D * tmp / theMax);
      const uint64_t ind2_3d =
          static_cast<uint64_t>(1.0 * HIST_DIM3D * tmp2 / theMax);
      const uint64_t ind3_3d =
          static_cast<uint64_t>(1.0 * HIST_DIM3D * tmp3 / theMax);
      atomic_fetch_add(&density_1d(ind1_1d), 1);
      atomic_fetch_add(&density_1d(ind2_1d), 1);
      atomic_fetch_add(&density_1d(ind3_1d), 1);
      atomic_fetch_add(&density_3d(ind1_3d, ind2_3d, ind3_3d), 1);
    }
    rand_pool.free_state(rand_gen);
  }
};

template <class DeviceType>
struct test_histogram1d_functor {
  using value_type      = RandomProperties;
  using execution_space = typename DeviceType::execution_space;
  using memory_space    = typename DeviceType::memory_space;

  // NOTE (mfh 03 Nov 2014): flare::rand::max() is supposed to define
  // an exclusive upper bound on the range of random numbers that
  // draw() can generate.  However, for the float specialization, some
  // implementations might violate this upper bound, due to rounding
  // error.  Just in case, we leave an extra space at the end of each
  // dimension, in the View type below.
  using type_1d = flare::View<int[HIST_DIM1D + 1], memory_space>;
  type_1d density_1d;
  double mean;

  test_histogram1d_functor(type_1d d1d, int num_draws)
      : density_1d(d1d), mean(1.0 * num_draws / HIST_DIM1D * 3) {}

  FLARE_INLINE_FUNCTION void operator()(
      const typename memory_space::size_type i, RandomProperties& prop) const {
    using size_type    = typename memory_space::size_type;
    const double count = density_1d(i);
    prop.mean += count;
    prop.variance += 1.0 * (count - mean) * (count - mean);
    // prop.covariance += 1.0*count*count;
    prop.min = count < prop.min ? count : prop.min;
    prop.max = count > prop.max ? count : prop.max;
    if (i < static_cast<size_type>(HIST_DIM1D - 1)) {
      prop.covariance += (count - mean) * (density_1d(i + 1) - mean);
    }
  }
};

template <class DeviceType>
struct test_histogram3d_functor {
  using value_type      = RandomProperties;
  using execution_space = typename DeviceType::execution_space;
  using memory_space    = typename DeviceType::memory_space;

  // NOTE (mfh 03 Nov 2014): flare::rand::max() is supposed to define
  // an exclusive upper bound on the range of random numbers that
  // draw() can generate.  However, for the float specialization, some
  // implementations might violate this upper bound, due to rounding
  // error.  Just in case, we leave an extra space at the end of each
  // dimension, in the View type below.
  using type_3d =
      flare::View<int[HIST_DIM3D + 1][HIST_DIM3D + 1][HIST_DIM3D + 1],
                   memory_space>;
  type_3d density_3d;
  double mean;

  test_histogram3d_functor(type_3d d3d, int num_draws)
      : density_3d(d3d), mean(1.0 * num_draws / HIST_DIM1D) {}

  FLARE_INLINE_FUNCTION void operator()(
      const typename memory_space::size_type i, RandomProperties& prop) const {
    using size_type    = typename memory_space::size_type;
    const double count = density_3d(
        i / (HIST_DIM3D * HIST_DIM3D),
        (i % (HIST_DIM3D * HIST_DIM3D)) / HIST_DIM3D, i % HIST_DIM3D);
    prop.mean += count;
    prop.variance += (count - mean) * (count - mean);
    if (i < static_cast<size_type>(HIST_DIM1D - 1)) {
      const double count_next =
          density_3d((i + 1) / (HIST_DIM3D * HIST_DIM3D),
                     ((i + 1) % (HIST_DIM3D * HIST_DIM3D)) / HIST_DIM3D,
                     (i + 1) % HIST_DIM3D);
      prop.covariance += (count - mean) * (count_next - mean);
    }
  }
};

//
// Templated test that uses the above functors.
//
template <class RandomGenerator, class Scalar>
struct test_random_scalar {
  using rnd_type = typename RandomGenerator::generator_type;

  test_random_scalar(
      typename test_random_functor<RandomGenerator, int>::type_1d& density_1d,
      typename test_random_functor<RandomGenerator, int>::type_3d& density_3d,
      RandomGenerator& pool, unsigned int num_draws) {
    using flare::parallel_reduce;
    using std::cout;
    using std::endl;

    {
      cout << " -- Testing randomness properties" << endl;

      RandomProperties result;
      using functor_type = test_random_functor<RandomGenerator, Scalar>;
      parallel_reduce(num_draws / 1024,
                      functor_type(pool, density_1d, density_3d), result);

      // printf("Result: %lf %lf
      // %lf\n",result.mean/num_draws/3,result.variance/num_draws/3,result.covariance/num_draws/2);
      double tolerance       = 1.6 * std::sqrt(1.0 / num_draws);
      double mean_expect     = 0.5 * flare::rand<rnd_type, Scalar>::max();
      double variance_expect = 1.0 / 3.0 * mean_expect * mean_expect;
      double mean_eps = mean_expect / (result.mean / num_draws / 3) - 1.0;
      double variance_eps =
          variance_expect / (result.variance / num_draws / 3) - 1.0;
      double covariance_eps =
          result.covariance / num_draws / 2 / variance_expect;
#if defined(FLARE_BHALF_T_IS_FLOAT) && !FLARE_BHALF_T_IS_FLOAT
      if (!std::is_same<Scalar, flare::experimental::bhalf_t>::value) {
#endif
        REQUIRE_LT(std::abs(mean_eps), tolerance);
        REQUIRE_LT(std::abs(variance_eps), 1.5 * tolerance);
        REQUIRE_LT(std::abs(covariance_eps), 2.0 * tolerance);
#if defined(FLARE_BHALF_T_IS_FLOAT) && !FLARE_BHALF_T_IS_FLOAT
      }
#endif
    }
    {
      cout << " -- Testing 1-D histogram" << endl;

      RandomProperties result;
      using functor_type =
          test_histogram1d_functor<typename RandomGenerator::device_type>;
      parallel_reduce(HIST_DIM1D, functor_type(density_1d, num_draws), result);
      double mean_eps_expect       = 0.0001;
      double variance_eps_expect   = 0.07;
      double covariance_eps_expect = 0.06;
      double tolerance             = 6 * std::sqrt(1.0 / HIST_DIM1D);
      double mean_expect           = 1.0 * num_draws * 3 / HIST_DIM1D;
      double variance_expect =
          1.0 * num_draws * 3 / HIST_DIM1D * (1.0 - 1.0 / HIST_DIM1D);
      double covariance_expect = -1.0 * num_draws * 3 / HIST_DIM1D / HIST_DIM1D;
      double mean_eps          = mean_expect / (result.mean / HIST_DIM1D) - 1.0;
      double variance_eps =
          variance_expect / (result.variance / HIST_DIM1D) - 1.0;
      double covariance_eps =
          (result.covariance / HIST_DIM1D - covariance_expect) / mean_expect;

#if defined(FLARE_HALF_T_IS_FLOAT) && !FLARE_HALF_T_IS_FLOAT
      if (std::is_same<Scalar, flare::experimental::half_t>::value) {
        mean_eps_expect       = 0.0003;
        variance_eps_expect   = 1.0;
        covariance_eps_expect = 5.0e4;
      }
#endif

#if defined(FLARE_BHALF_T_IS_FLOAT) && !FLARE_BHALF_T_IS_FLOAT
      if (!std::is_same<Scalar, flare::experimental::bhalf_t>::value) {
#endif
        REQUIRE_LT(std::abs(mean_eps), mean_eps_expect);
        REQUIRE_LT(std::abs(variance_eps), variance_eps_expect);
        REQUIRE_LT(std::abs(covariance_eps), covariance_eps_expect);
#if defined(FLARE_BHALF_T_IS_FLOAT) && !FLARE_BHALF_T_IS_FLOAT
      }
#endif

      cout << "Density 1D: " << mean_eps << " " << variance_eps << " "
           << (result.covariance / HIST_DIM1D / HIST_DIM1D) << " || "
           << tolerance << " " << result.min << " " << result.max << " || "
           << result.variance / HIST_DIM1D << " "
           << 1.0 * num_draws * 3 / HIST_DIM1D * (1.0 - 1.0 / HIST_DIM1D)
           << " || " << result.covariance / HIST_DIM1D << " "
           << -1.0 * num_draws * 3 / HIST_DIM1D / HIST_DIM1D << endl;
    }
    {
      cout << " -- Testing 3-D histogram" << endl;

      RandomProperties result;
      using functor_type =
          test_histogram3d_functor<typename RandomGenerator::device_type>;
      parallel_reduce(HIST_DIM1D, functor_type(density_3d, num_draws), result);

      double variance_factor = 1.2;
      double tolerance       = 6 * std::sqrt(1.0 / HIST_DIM1D);
      double mean_expect     = 1.0 * num_draws / HIST_DIM1D;
      double variance_expect =
          1.0 * num_draws / HIST_DIM1D * (1.0 - 1.0 / HIST_DIM1D);
      double covariance_expect = -1.0 * num_draws / HIST_DIM1D / HIST_DIM1D;
      double mean_eps          = mean_expect / (result.mean / HIST_DIM1D) - 1.0;
      double variance_eps =
          variance_expect / (result.variance / HIST_DIM1D) - 1.0;
      double covariance_eps =
          (result.covariance / HIST_DIM1D - covariance_expect) / mean_expect;

#if defined(FLARE_HALF_T_IS_FLOAT) && !FLARE_HALF_T_IS_FLOAT
      if (std::is_same<Scalar, flare::experimental::half_t>::value) {
        variance_factor = 7;
      }
#endif

#if defined(FLARE_BHALF_T_IS_FLOAT) && !FLARE_BHALF_T_IS_FLOAT
      if (!std::is_same<Scalar, flare::experimental::bhalf_t>::value) {
#endif
        REQUIRE_LT(std::abs(mean_eps), tolerance);
        REQUIRE_LT(std::abs(variance_eps), variance_factor);
        REQUIRE_LT(std::abs(covariance_eps), variance_factor);
#if defined(FLARE_BHALF_T_IS_FLOAT) && !FLARE_BHALF_T_IS_FLOAT
      }
#endif

      cout << "Density 3D: " << mean_eps << " " << variance_eps << " "
           << result.covariance / HIST_DIM1D / HIST_DIM1D << " || " << tolerance
           << " " << result.min << " " << result.max << endl;
    }
  }
};

template <class RandomGenerator>
void test_random(unsigned int num_draws) {
  using std::cout;
  using std::endl;
  typename test_random_functor<RandomGenerator, int>::type_1d density_1d("D1d");
  typename test_random_functor<RandomGenerator, int>::type_3d density_3d("D3d");

  uint64_t ticks =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  cout << "Test Seed:" << ticks << endl;

  RandomGenerator pool(ticks);

  cout << "Test Scalar=int" << endl;
  test_random_scalar<RandomGenerator, int> test_int(density_1d, density_3d,
                                                    pool, num_draws);
  deep_copy(density_1d, 0);
  deep_copy(density_3d, 0);

  cout << "Test Scalar=unsigned int" << endl;
  test_random_scalar<RandomGenerator, unsigned int> test_uint(
      density_1d, density_3d, pool, num_draws);
  deep_copy(density_1d, 0);
  deep_copy(density_3d, 0);

  cout << "Test Scalar=int64_t" << endl;
  test_random_scalar<RandomGenerator, int64_t> test_int64(
      density_1d, density_3d, pool, num_draws);
  deep_copy(density_1d, 0);
  deep_copy(density_3d, 0);

  cout << "Test Scalar=uint64_t" << endl;
  test_random_scalar<RandomGenerator, uint64_t> test_uint64(
      density_1d, density_3d, pool, num_draws);
  deep_copy(density_1d, 0);
  deep_copy(density_3d, 0);

  cout << "Test Scalar=half" << endl;
  test_random_scalar<RandomGenerator, flare::experimental::half_t> test_half(
      density_1d, density_3d, pool, num_draws);
  deep_copy(density_1d, 0);
  deep_copy(density_3d, 0);

  cout << "Test Scalar=bhalf" << endl;
  test_random_scalar<RandomGenerator, flare::experimental::bhalf_t> test_bhalf(
      density_1d, density_3d, pool, num_draws);
  deep_copy(density_1d, 0);
  deep_copy(density_3d, 0);

  cout << "Test Scalar=float" << endl;
  test_random_scalar<RandomGenerator, float> test_float(density_1d, density_3d,
                                                        pool, num_draws);
  deep_copy(density_1d, 0);
  deep_copy(density_3d, 0);

  cout << "Test Scalar=double" << endl;
  test_random_scalar<RandomGenerator, double> test_double(
      density_1d, density_3d, pool, num_draws);
}

template <class ExecutionSpace, class Pool>
struct TestDynRankView {
  using ReducerType      = flare::MinMax<double, flare::HostSpace>;
  using ReducerValueType = typename ReducerType::value_type;

  flare::DynRankView<double, ExecutionSpace> A;

  TestDynRankView(int n) : A("a", n) {}

  FLARE_FUNCTION void operator()(int i, ReducerValueType& update) const {
    if (A(i) < update.min_val) update.min_val = A(i);
    if (A(i) > update.max_val) update.max_val = A(i);
  }

  void run() {
    Pool random(13);
    double min = 10.;
    double max = 100.;
    ExecutionSpace exec;
    flare::fill_random(exec, A, random, min, max);

    ReducerValueType val;
    flare::parallel_reduce(
        flare::RangePolicy<ExecutionSpace>(exec, 0, A.size()), *this,
        ReducerType(val));

    exec.fence();
    REQUIRE_GE(val.min_val, min);
    REQUIRE_LE(val.max_val, max);
  }
};

template <class ExecutionSpace, class GeneratorPool>
struct generate_random_stream {
  using ViewType = flare::View<uint64_t**, ExecutionSpace>;

  ViewType vals;
  GeneratorPool rand_pool;
  int samples;

  generate_random_stream(ViewType vals_, GeneratorPool rand_pool_, int samples_)
      : vals(vals_), rand_pool(rand_pool_), samples(samples_) {}

  FLARE_INLINE_FUNCTION
  void operator()(int i) const {
    typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

    for (int k = 0; k < samples; k++) vals(i, k) = rand_gen.urand64();

    rand_pool.free_state(rand_gen);
  }
};

// NOTE: this doesn't test the statistical independence of multiple streams
// generated by a Random pool, it only tests for complete duplicates.
template <class ExecutionSpace, class Pool>
void test_duplicate_stream() {
  using ViewType = flare::View<uint64_t**, ExecutionSpace>;

  // Heuristic to create a "large enough" number of streams.
  int n_streams = ExecutionSpace{}.concurrency() * 4;
  int samples   = 8;

  Pool rand_pool(42);
  ViewType vals_d("Vals", n_streams, samples);

  flare::parallel_for(
      flare::RangePolicy<ExecutionSpace>(0, n_streams),
      generate_random_stream<ExecutionSpace, Pool>(vals_d, rand_pool, samples));

  auto vals_h =
      flare::create_mirror_view_and_copy(flare::HostSpace{}, vals_d);

  /*
  To quickly find streams that are identical, we sort them by the first number,
  if that's equal then the second and so on. We then test each neighbor pair
  for duplicates.
  */
  std::vector<size_t> indices(n_streams);
  std::iota(indices.begin(), indices.end(), 0);

  auto comparator = [&](int i, int j) {
    for (int k = 0; k < samples; k++) {
      if (vals_h(i, k) != vals_h(j, k)) return vals_h(i, k) < vals_h(j, k);
    }
    return false;
  };
  std::sort(indices.begin(), indices.end(), comparator);

  for (int i = 0; i < n_streams - 1; i++) {
    int idx1 = indices[i];
    int idx2 = indices[i + 1];

    int k = 0;
    while (k < samples && vals_h(idx1, k) == vals_h(idx2, k)) k++;
    REQUIRE_LT(k, samples);
  }
}

}  // namespace AlgoRandomImpl

TEST_CASE("TEST_CATEGORY, Random_XorShift64") {
  using ExecutionSpace = TEST_EXECSPACE;

#if defined(FLARE_ENABLE_CUDA)
  const int num_draws = 132141141;
#else  // SERIAL, OPENMP
  const int num_draws = 10240000;
#endif
  AlgoRandomImpl::test_random<flare::Random_XorShift64_Pool<ExecutionSpace>>(
      num_draws);
  AlgoRandomImpl::test_random<flare::Random_XorShift64_Pool<
      flare::Device<ExecutionSpace, typename ExecutionSpace::memory_space>>>(
      num_draws);
  AlgoRandomImpl::TestDynRankView<
      ExecutionSpace, flare::Random_XorShift64_Pool<ExecutionSpace>>(10000)
      .run();
}

TEST_CASE("TEST_CATEGORY, Random_XorShift1024_0") {
  using ExecutionSpace = TEST_EXECSPACE;

#if defined(FLARE_ENABLE_CUDA)
  const int num_draws = 52428813;
#else  // SERIAL, OPENMP
  const int num_draws = 10130144;
#endif
  AlgoRandomImpl::test_random<flare::Random_XorShift1024_Pool<ExecutionSpace>>(
      num_draws);
  AlgoRandomImpl::test_random<flare::Random_XorShift1024_Pool<
      flare::Device<ExecutionSpace, typename ExecutionSpace::memory_space>>>(
      num_draws);
  AlgoRandomImpl::TestDynRankView<
      ExecutionSpace, flare::Random_XorShift1024_Pool<ExecutionSpace>>(10000)
      .run();
}

TEST_CASE("TEST_CATEGORY, Multi_streams") {
  using ExecutionSpace = TEST_EXECSPACE;

  using Pool64   = flare::Random_XorShift64_Pool<ExecutionSpace>;
  using Pool1024 = flare::Random_XorShift1024_Pool<ExecutionSpace>;

  AlgoRandomImpl::test_duplicate_stream<ExecutionSpace, Pool64>();
  AlgoRandomImpl::test_duplicate_stream<ExecutionSpace, Pool1024>();
}

}  // namespace Test
#endif
