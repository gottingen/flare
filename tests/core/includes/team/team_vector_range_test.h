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

#include <flare/timer.h>
#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <cinttypes>

namespace TestTeamVectorRange {

struct my_complex {
  double re, im;
  int dummy;

  FLARE_INLINE_FUNCTION
  my_complex() {
    re    = 0.0;
    im    = 0.0;
    dummy = 0;
  }

  FLARE_INLINE_FUNCTION
  my_complex(const my_complex& src) {
    re    = src.re;
    im    = src.im;
    dummy = src.dummy;
  }

  FLARE_INLINE_FUNCTION
  my_complex& operator=(const my_complex& src) {
    re    = src.re;
    im    = src.im;
    dummy = src.dummy;
    return *this;
  }

  FLARE_INLINE_FUNCTION
  my_complex(const double& val) {
    re    = val;
    im    = 0.0;
    dummy = 0;
  }

  FLARE_INLINE_FUNCTION
  my_complex& operator+=(const my_complex& src) {
    re += src.re;
    im += src.im;
    dummy += src.dummy;
    return *this;
  }

  FLARE_INLINE_FUNCTION
  my_complex operator+(const my_complex& src) {
    my_complex tmp = *this;
    tmp.re += src.re;
    tmp.im += src.im;
    tmp.dummy += src.dummy;
    return tmp;
  }

  FLARE_INLINE_FUNCTION
  my_complex& operator*=(const my_complex& src) {
    double re_tmp = re * src.re - im * src.im;
    double im_tmp = re * src.im + im * src.re;
    re            = re_tmp;
    im            = im_tmp;
    dummy *= src.dummy;
    return *this;
  }

  FLARE_INLINE_FUNCTION
  bool operator==(const my_complex& src) const {
    return (re == src.re) && (im == src.im) && (dummy == src.dummy);
  }

  FLARE_INLINE_FUNCTION
  bool operator!=(const my_complex& src) const {
    return (re != src.re) || (im != src.im) || (dummy != src.dummy);
  }

  FLARE_INLINE_FUNCTION
  bool operator!=(const double& val) const {
    return (re != val) || (im != 0) || (dummy != 0);
  }

  FLARE_INLINE_FUNCTION
  my_complex& operator=(const int& val) {
    re    = val;
    im    = 0.0;
    dummy = 0;
    return *this;
  }

  FLARE_INLINE_FUNCTION
  my_complex& operator=(const double& val) {
    re    = val;
    im    = 0.0;
    dummy = 0;
    return *this;
  }

  FLARE_INLINE_FUNCTION
  operator double() { return re; }
};
}  // namespace TestTeamVectorRange

namespace flare {
template <>
struct reduction_identity<TestTeamVectorRange::my_complex> {
  using t_red_ident = reduction_identity<double>;
  FLARE_FORCEINLINE_FUNCTION static TestTeamVectorRange::my_complex sum() {
    return TestTeamVectorRange::my_complex(t_red_ident::sum());
  }
  FLARE_FORCEINLINE_FUNCTION static TestTeamVectorRange::my_complex prod() {
    return TestTeamVectorRange::my_complex(t_red_ident::prod());
  }
};
}  // namespace flare

namespace TestTeamVectorRange {

template <typename Scalar, class ExecutionSpace>
struct functor_teamvector_for {
  using policy_type     = flare::TeamPolicy<ExecutionSpace>;
  using execution_space = ExecutionSpace;

  flare::View<int, flare::LayoutLeft, ExecutionSpace> flag;

  functor_teamvector_for(
      flare::View<int, flare::LayoutLeft, ExecutionSpace> flag_)
      : flag(flag_) {}

  using shmem_space = typename ExecutionSpace::scratch_memory_space;
  using shared_int =
      flare::View<Scalar*, shmem_space, flare::MemoryUnmanaged>;
  unsigned team_shmem_size(int /*team_size*/) const {
    return shared_int::shmem_size(131);
  }

  FLARE_INLINE_FUNCTION
  void operator()(typename policy_type::member_type team) const {
    using size_type           = typename shmem_space::size_type;
    const size_type shmemSize = 131;
    shared_int values         = shared_int(team.team_shmem(), shmemSize);

    if (values.data() == nullptr || values.extent(0) < shmemSize) {
      flare::printf("FAILED to allocate shared memory of size %u\n",
                     static_cast<unsigned int>(shmemSize));
    } else {
      // Initialize shared memory.
      flare::parallel_for(flare::TeamVectorRange(team, 131),
                           [&](int i) { values(i) = 0; });
      // Wait for all memory to be written.
      team.team_barrier();

      // Accumulate value into per thread shared memory.
      // This is non blocking.
      flare::parallel_for(flare::TeamVectorRange(team, 131), [&](int i) {
        values(i) +=
            i - team.league_rank() + team.league_size() + team.team_size();
      });

      // Wait for all memory to be written.
      team.team_barrier();

      // One thread per team executes the comparison.
      flare::single(flare::PerTeam(team), [&]() {
        Scalar test  = 0;
        Scalar value = 0;

        for (int i = 0; i < 131; ++i) {
          test +=
              i - team.league_rank() + team.league_size() + team.team_size();
        }

        for (int i = 0; i < 131; ++i) {
          value += values(i);
        }

        if (test != value) {
          flare::printf("FAILED teamvector_parallel_for %i %i %lf %lf\n",
                         team.league_rank(), team.team_rank(),
                         static_cast<double>(test), static_cast<double>(value));
          flag() = 1;
        }
      });
    }
  }
};

template <typename Scalar, class ExecutionSpace>
struct functor_teamvector_reduce {
  using policy_type     = flare::TeamPolicy<ExecutionSpace>;
  using execution_space = ExecutionSpace;

  flare::View<int, flare::LayoutLeft, ExecutionSpace> flag;

  functor_teamvector_reduce(
      flare::View<int, flare::LayoutLeft, ExecutionSpace> flag_)
      : flag(flag_) {}

  using shmem_space = typename ExecutionSpace::scratch_memory_space;
  using shared_scalar_t =
      flare::View<Scalar*, shmem_space, flare::MemoryUnmanaged>;
  unsigned team_shmem_size(int team_size) const {
    return shared_scalar_t::shmem_size(team_size * 13);
  }

  FLARE_INLINE_FUNCTION
  void operator()(typename policy_type::member_type team) const {
    Scalar value = Scalar();
    shared_scalar_t shared_value(team.team_scratch(0), 1);

    flare::parallel_reduce(
        flare::TeamVectorRange(team, 131),
        [&](int i, Scalar& val) {
          val += i - team.league_rank() + team.league_size() + team.team_size();
        },
        shared_value(0));

    team.team_barrier();
    flare::parallel_reduce(
        flare::TeamVectorRange(team, 131),
        [&](int i, Scalar& val) {
          val += i - team.league_rank() + team.league_size() + team.team_size();
        },
        value);

    //    flare::parallel_reduce( flare::TeamVectorRange( team, 131 ), [&] (
    //    int i, Scalar & val )
    //    {
    //      val += i - team.league_rank() + team.league_size() +
    //      team.team_size();
    //    }, shared_value(0) );

    team.team_barrier();

    flare::single(flare::PerTeam(team), [&]() {
      Scalar test = 0;

      for (int i = 0; i < 131; ++i) {
        test += i - team.league_rank() + team.league_size() + team.team_size();
      }

      if (test != value) {
        if (team.league_rank() == 0) {
          flare::printf(
              "FAILED teamvector_parallel_reduce %i %i %lf %lf %lu\n",
              (int)team.league_rank(), (int)team.team_rank(),
              static_cast<double>(test), static_cast<double>(value),
              static_cast<unsigned long>(sizeof(Scalar)));
        }

        flag() = 1;
      }
      if (test != shared_value(0)) {
        if (team.league_rank() == 0) {
          flare::printf(
              "FAILED teamvector_parallel_reduce with shared result %i %i %lf "
              "%lf %lu\n",
              static_cast<int>(team.league_rank()),
              static_cast<int>(team.team_rank()), static_cast<double>(test),
              static_cast<double>(shared_value(0)),
              static_cast<unsigned long>(sizeof(Scalar)));
        }

        flag() = 1;
      }
    });
  }
};

template <typename Scalar, class ExecutionSpace>
struct functor_teamvector_reduce_reducer {
  using policy_type     = flare::TeamPolicy<ExecutionSpace>;
  using execution_space = ExecutionSpace;

  flare::View<int, flare::LayoutLeft, ExecutionSpace> flag;

  functor_teamvector_reduce_reducer(
      flare::View<int, flare::LayoutLeft, ExecutionSpace> flag_)
      : flag(flag_) {}

  using shmem_space = typename ExecutionSpace::scratch_memory_space;
  using shared_scalar_t =
      flare::View<Scalar*, shmem_space, flare::MemoryUnmanaged>;
  unsigned team_shmem_size(int team_size) const {
    return shared_scalar_t::shmem_size(team_size * 13);
  }

  FLARE_INLINE_FUNCTION
  void operator()(typename policy_type::member_type team) const {
    Scalar value = 0;
    shared_scalar_t shared_value(team.team_scratch(0), 1);

    flare::parallel_reduce(
        flare::TeamVectorRange(team, 131),
        [&](int i, Scalar& val) {
          val += i - team.league_rank() + team.league_size() + team.team_size();
        },
        flare::Sum<Scalar>(value));

    flare::parallel_reduce(
        flare::TeamVectorRange(team, 131),
        [&](int i, Scalar& val) {
          val += i - team.league_rank() + team.league_size() + team.team_size();
        },
        flare::Sum<Scalar>(shared_value(0)));

    team.team_barrier();

    flare::single(flare::PerTeam(team), [&]() {
      Scalar test = 0;

      for (int i = 0; i < 131; ++i) {
        test += i - team.league_rank() + team.league_size() + team.team_size();
      }

      if (test != value) {
        flare::printf(
            "FAILED teamvector_parallel_reduce_reducer %i %i %lf %lf\n",
            team.league_rank(), team.team_rank(), static_cast<double>(test),
            static_cast<double>(value));

        flag() = 1;
      }
      if (test != shared_value(0)) {
        flare::printf(
            "FAILED teamvector_parallel_reduce_reducer shared value %i %i %lf "
            "%lf\n",
            team.league_rank(), team.team_rank(), static_cast<double>(test),
            static_cast<double>(shared_value(0)));

        flag() = 1;
      }
    });
  }
};

template <typename Scalar, class ExecutionSpace>
bool test_scalar(int nteams, int team_size, int test) {
  flare::View<int, flare::LayoutLeft, ExecutionSpace> d_flag("flag");
  typename flare::View<int, flare::LayoutLeft, ExecutionSpace>::HostMirror
      h_flag("h_flag");
  h_flag() = 0;
  flare::deep_copy(d_flag, h_flag);

  flare::TeamPolicy<ExecutionSpace> policy(nteams, team_size, 8);


  if (test == 0) {
    flare::parallel_for(
        "Test::TeamVectorFor", policy,
        functor_teamvector_for<Scalar, ExecutionSpace>(d_flag));
  } else if (test == 1) {
    flare::parallel_for(
        "Test::TeamVectorReduce", policy,
        functor_teamvector_reduce<Scalar, ExecutionSpace>(d_flag));
  } else if (test == 2) {
    flare::parallel_for(
        "Test::TeamVectorReduceReducer",
        flare::TeamPolicy<ExecutionSpace>(nteams, team_size, 8),
        functor_teamvector_reduce_reducer<Scalar, ExecutionSpace>(d_flag));
  }

  flare::deep_copy(h_flag, d_flag);

  return (h_flag() == 0);
}

template <class ExecutionSpace>
bool Test(int test) {
  bool passed = true;

  int team_size = 33;
  int const concurrency = ExecutionSpace().concurrency();
  if (team_size > concurrency) team_size = concurrency;
  passed = passed && test_scalar<int, ExecutionSpace>(317, team_size, test);
  passed = passed &&
           test_scalar<long long int, ExecutionSpace>(317, team_size, test);
  passed = passed && test_scalar<float, ExecutionSpace>(317, team_size, test);
  passed = passed && test_scalar<double, ExecutionSpace>(317, team_size, test);
  passed =
      passed && test_scalar<my_complex, ExecutionSpace>(317, team_size, test);

  return passed;
}

}  // namespace TestTeamVectorRange

namespace Test {

TEST(TEST_CATEGORY, team_teamvector_range) {
  ASSERT_TRUE((TestTeamVectorRange::Test<TEST_EXECSPACE>(0)));
#if defined(FLARE_ENABLE_CUDA) && \
    defined(FLARE_COMPILER_NVHPC)  // FIXME_NVHPC 23.7
  if constexpr (std::is_same_v<TEST_EXECSPACE, flare::Cuda>) {
    GTEST_SKIP() << "Disabling 2/3rd of the test for now";
  }
#endif
  ASSERT_TRUE((TestTeamVectorRange::Test<TEST_EXECSPACE>(1)));
  ASSERT_TRUE((TestTeamVectorRange::Test<TEST_EXECSPACE>(2)));
}
}  // namespace Test
