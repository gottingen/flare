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
#include <non_trivial_scalar_types_test.h>

namespace TestTeamVector {

template <typename Scalar, class ExecutionSpace>
struct functor_team_for {
  using policy_type     = flare::TeamPolicy<ExecutionSpace>;
  using execution_space = ExecutionSpace;

  flare::Tensor<int, flare::LayoutLeft, ExecutionSpace> flag;

  functor_team_for(flare::Tensor<int, flare::LayoutLeft, ExecutionSpace> flag_)
      : flag(flag_) {}

  using shmem_space = typename ExecutionSpace::scratch_memory_space;
  using shared_int =
      flare::Tensor<Scalar *, shmem_space, flare::MemoryUnmanaged>;
  unsigned team_shmem_size(int team_size) const {
    return shared_int::shmem_size(team_size * 13);
  }

  FLARE_INLINE_FUNCTION
  void operator()(typename policy_type::member_type team) const {
    using size_type           = typename shmem_space::size_type;
    const size_type shmemSize = team.team_size() * 13;
    shared_int values         = shared_int(team.team_shmem(), shmemSize);

    if (values.data() == nullptr ||
        static_cast<size_type>(values.extent(0)) < shmemSize) {
      flare::printf("FAILED to allocate shared memory of size %u\n",
                     static_cast<unsigned int>(shmemSize));
    } else {
      // Initialize shared memory.
      values(team.team_rank()) = 0;

      // Accumulate value into per thread shared memory.
      // This is non blocking.
      flare::parallel_for(flare::TeamThreadRange(team, 131), [&](int i) {
        values(team.team_rank()) +=
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

        for (int i = 0; i < team.team_size(); ++i) {
          value += values(i);
        }

        if (test != value) {
          flare::printf("FAILED team_parallel_for %i %i %lf %lf\n",
                         team.league_rank(), team.team_rank(),
                         static_cast<double>(test), static_cast<double>(value));
          flag() = 1;
        }
      });
    }
  }
};

template <typename Scalar, class ExecutionSpace>
struct functor_team_reduce {
  using policy_type     = flare::TeamPolicy<ExecutionSpace>;
  using execution_space = ExecutionSpace;

  flare::Tensor<int, flare::LayoutLeft, ExecutionSpace> flag;

  functor_team_reduce(
      flare::Tensor<int, flare::LayoutLeft, ExecutionSpace> flag_)
      : flag(flag_) {}

  using shmem_space = typename ExecutionSpace::scratch_memory_space;
  using shared_scalar_t =
      flare::Tensor<Scalar *, shmem_space, flare::MemoryUnmanaged>;
  unsigned team_shmem_size(int team_size) const {
    return shared_scalar_t::shmem_size(team_size * 13);
  }

  FLARE_INLINE_FUNCTION
  void operator()(typename policy_type::member_type team) const {
    Scalar value = Scalar();
    shared_scalar_t shared_value(team.team_scratch(0), 1);

    flare::parallel_reduce(
        flare::TeamThreadRange(team, 131),
        [&](int i, Scalar &val) {
          val += i - team.league_rank() + team.league_size() + team.team_size();
        },
        value);

    flare::parallel_reduce(
        flare::TeamThreadRange(team, 131),
        [&](int i, Scalar &val) {
          val += i - team.league_rank() + team.league_size() + team.team_size();
        },
        shared_value(0));

    team.team_barrier();

    flare::single(flare::PerTeam(team), [&]() {
      Scalar test = 0;

      for (int i = 0; i < 131; ++i) {
        test += i - team.league_rank() + team.league_size() + team.team_size();
      }

      if (test != value) {
        if (team.league_rank() == 0) {
          flare::printf("FAILED team_parallel_reduce %i %i %lf %lf %lu\n",
                         team.league_rank(), team.team_rank(),
                         static_cast<double>(test), static_cast<double>(value),
                         static_cast<unsigned long>(sizeof(Scalar)));
        }

        flag() = 1;
      }
      if (test != shared_value(0)) {
        if (team.league_rank() == 0) {
          flare::printf(
              "FAILED team_parallel_reduce with shared result %i %i %lf %lf "
              "%lu\n",
              team.league_rank(), team.team_rank(), static_cast<double>(test),
              static_cast<double>(shared_value(0)),
              static_cast<unsigned long>(sizeof(Scalar)));
        }

        flag() = 1;
      }
    });
  }
};

template <typename Scalar, class ExecutionSpace>
struct functor_team_reduce_reducer {
  using policy_type     = flare::TeamPolicy<ExecutionSpace>;
  using execution_space = ExecutionSpace;

  flare::Tensor<int, flare::LayoutLeft, ExecutionSpace> flag;

  functor_team_reduce_reducer(
      flare::Tensor<int, flare::LayoutLeft, ExecutionSpace> flag_)
      : flag(flag_) {}

  using shmem_space = typename ExecutionSpace::scratch_memory_space;
  using shared_scalar_t =
      flare::Tensor<Scalar *, shmem_space, flare::MemoryUnmanaged>;
  unsigned team_shmem_size(int team_size) const {
    return shared_scalar_t::shmem_size(team_size * 13);
  }

  FLARE_INLINE_FUNCTION
  void operator()(typename policy_type::member_type team) const {
    Scalar value = 0;
    shared_scalar_t shared_value(team.team_scratch(0), 1);

    flare::parallel_reduce(
        flare::TeamThreadRange(team, 131),
        [&](int i, Scalar &val) {
          val += i - team.league_rank() + team.league_size() + team.team_size();
        },
        flare::Sum<Scalar>(value));

    flare::parallel_reduce(
        flare::TeamThreadRange(team, 131),
        [&](int i, Scalar &val) {
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
            "FAILED team_vector_parallel_reduce_reducer %i %i %lf %lf\n",
            team.league_rank(), team.team_rank(), static_cast<double>(test),
            static_cast<double>(value));

        flag() = 1;
      }
      if (test != shared_value(0)) {
        flare::printf(
            "FAILED team_vector_parallel_reduce_reducer shared value %i %i %lf "
            "%lf\n",
            team.league_rank(), team.team_rank(), static_cast<double>(test),
            static_cast<double>(shared_value(0)));

        flag() = 1;
      }
    });
  }
};

template <typename Scalar, class ExecutionSpace>
struct functor_team_vector_for {
  using policy_type     = flare::TeamPolicy<ExecutionSpace>;
  using execution_space = ExecutionSpace;

  flare::Tensor<int, flare::LayoutLeft, ExecutionSpace> flag;

  functor_team_vector_for(
      flare::Tensor<int, flare::LayoutLeft, ExecutionSpace> flag_)
      : flag(flag_) {}

  using shmem_space = typename ExecutionSpace::scratch_memory_space;
  using shared_int =
      flare::Tensor<Scalar *, shmem_space, flare::MemoryUnmanaged>;
  unsigned team_shmem_size(int team_size) const {
    return shared_int::shmem_size(team_size * 13);
  }

  FLARE_INLINE_FUNCTION
  void operator()(typename policy_type::member_type team) const {
    using size_type = typename shared_int::size_type;

    const size_type shmemSize = team.team_size() * 13;
    shared_int values         = shared_int(team.team_shmem(), shmemSize);

    if (values.data() == nullptr ||
        static_cast<size_type>(values.extent(0)) < shmemSize) {
      flare::printf("FAILED to allocate shared memory of size %u\n",
                     static_cast<unsigned int>(shmemSize));
    } else {
      team.team_barrier();

      flare::single(flare::PerThread(team),
                     [&]() { values(team.team_rank()) = 0; });

      flare::parallel_for(flare::TeamThreadRange(team, 131), [&](int i) {
        flare::single(flare::PerThread(team), [&]() {
          values(team.team_rank()) +=
              i - team.league_rank() + team.league_size() + team.team_size();
        });
      });

      team.team_barrier();

      flare::single(flare::PerTeam(team), [&]() {
        Scalar test  = 0;
        Scalar value = 0;

        for (int i = 0; i < 131; ++i) {
          test +=
              i - team.league_rank() + team.league_size() + team.team_size();
        }

        for (int i = 0; i < team.team_size(); ++i) {
          value += values(i);
        }

        if (test != value) {
          flare::printf("FAILED team_vector_parallel_for %i %i %lf %lf\n",
                         team.league_rank(), team.team_rank(),
                         static_cast<double>(test), static_cast<double>(value));
          flag() = 1;
        }
      });
    }
  }
};

template <typename Scalar, class ExecutionSpace>
struct functor_team_vector_reduce {
  using policy_type     = flare::TeamPolicy<ExecutionSpace>;
  using execution_space = ExecutionSpace;

  flare::Tensor<int, flare::LayoutLeft, ExecutionSpace> flag;
  functor_team_vector_reduce(
      flare::Tensor<int, flare::LayoutLeft, ExecutionSpace> flag_)
      : flag(flag_) {}

  using shmem_space = typename ExecutionSpace::scratch_memory_space;
  using shared_int =
      flare::Tensor<Scalar *, shmem_space, flare::MemoryUnmanaged>;
  unsigned team_shmem_size(int team_size) const {
    return shared_int::shmem_size(team_size * 13);
  }

  FLARE_INLINE_FUNCTION
  void operator()(typename policy_type::member_type team) const {
    Scalar value = Scalar();

    flare::parallel_reduce(
        flare::TeamThreadRange(team, 131),
        [&](int i, Scalar &val) {
          val += i - team.league_rank() + team.league_size() + team.team_size();
        },
        value);

    team.team_barrier();

    flare::single(flare::PerTeam(team), [&]() {
      Scalar test = 0;

      for (int i = 0; i < 131; ++i) {
        test += i - team.league_rank() + team.league_size() + team.team_size();
      }

      if (test != value) {
        if (team.league_rank() == 0) {
          flare::printf(
              "FAILED team_vector_parallel_reduce %i %i %lf %lf %lu\n",
              team.league_rank(), team.team_rank(), static_cast<double>(test),
              static_cast<double>(value),
              static_cast<unsigned long>(sizeof(Scalar)));
        }

        flag() = 1;
      }
    });
  }
};

template <typename Scalar, class ExecutionSpace>
struct functor_team_vector_reduce_reducer {
  using policy_type     = flare::TeamPolicy<ExecutionSpace>;
  using execution_space = ExecutionSpace;

  flare::Tensor<int, flare::LayoutLeft, ExecutionSpace> flag;

  functor_team_vector_reduce_reducer(
      flare::Tensor<int, flare::LayoutLeft, ExecutionSpace> flag_)
      : flag(flag_) {}

  using shmem_space = typename ExecutionSpace::scratch_memory_space;
  using shared_int =
      flare::Tensor<Scalar *, shmem_space, flare::MemoryUnmanaged>;
  unsigned team_shmem_size(int team_size) const {
    return shared_int::shmem_size(team_size * 13);
  }

  FLARE_INLINE_FUNCTION
  void operator()(typename policy_type::member_type team) const {
    Scalar value = 0;

    flare::parallel_reduce(
        flare::TeamThreadRange(team, 131),
        [&](int i, Scalar &val) {
          val += i - team.league_rank() + team.league_size() + team.team_size();
        },
        flare::Sum<Scalar>(value));

    team.team_barrier();

    flare::single(flare::PerTeam(team), [&]() {
      Scalar test = 0;

      for (int i = 0; i < 131; ++i) {
        test += i - team.league_rank() + team.league_size() + team.team_size();
      }

      if (test != value) {
        flare::printf(
            "FAILED team_vector_parallel_reduce_reducer %i %i %lf %lf\n",
            team.league_rank(), team.team_rank(), static_cast<double>(test),
            static_cast<double>(value));

        flag() = 1;
      }
    });
  }
};

template <typename Scalar, class ExecutionSpace>
struct functor_vec_single {
  using policy_type     = flare::TeamPolicy<ExecutionSpace>;
  using execution_space = ExecutionSpace;

  flare::Tensor<int, flare::LayoutLeft, ExecutionSpace> flag;
  int nStart;
  int nEnd;

  functor_vec_single(
      flare::Tensor<int, flare::LayoutLeft, ExecutionSpace> flag_,
      const int start_, const int end_)
      : flag(flag_), nStart(start_), nEnd(end_) {}

  FLARE_INLINE_FUNCTION
  void operator()(typename policy_type::member_type team) const {
    // Warning: this test case intentionally violates permissible semantics.
    // It is not valid to get references to members of the enclosing region
    // inside a parallel_for and write to it.
    Scalar value = 0;

    flare::parallel_for(flare::ThreadVectorRange(team, nStart, nEnd),
                         [&](int i) {
                           value = i;  // This write is violating flare
                                       // semantics for nested parallelism.
                         });

    flare::single(
        flare::PerThread(team), [&](Scalar &val) { val = 1; }, value);

    Scalar value2 = 0;
    flare::parallel_reduce(
        flare::ThreadVectorRange(team, nStart, nEnd),
        [&](int /*i*/, Scalar &val) { val += value; }, value2);

    if (value2 != (value * Scalar(nEnd - nStart))) {
      flare::printf("FAILED vector_single broadcast %i %i %lf %lf\n",
                     team.league_rank(), team.team_rank(),
                     static_cast<double>(value2), static_cast<double>(value));

      flag() = 1;
    }
  }
};

template <typename Scalar, class ExecutionSpace>
struct functor_vec_for {
  using policy_type     = flare::TeamPolicy<ExecutionSpace>;
  using execution_space = ExecutionSpace;

  flare::Tensor<int, flare::LayoutLeft, ExecutionSpace> flag;

  functor_vec_for(flare::Tensor<int, flare::LayoutLeft, ExecutionSpace> flag_)
      : flag(flag_) {}

  using shmem_space = typename ExecutionSpace::scratch_memory_space;
  using shared_int =
      flare::Tensor<Scalar *, shmem_space, flare::MemoryUnmanaged>;
  unsigned team_shmem_size(int team_size) const {
    return shared_int::shmem_size(team_size * 13);
  }

  FLARE_INLINE_FUNCTION
  void operator()(typename policy_type::member_type team) const {
    shared_int values = shared_int(team.team_shmem(), team.team_size() * 13);

    if (values.data() == nullptr ||
        values.extent(0) < (unsigned)team.team_size() * 13) {
      flare::printf("FAILED to allocate memory of size %i\n",
                     static_cast<int>(team.team_size() * 13));
      flag() = 1;
    } else {
      flare::parallel_for(flare::ThreadVectorRange(team, 13), [&](int i) {
        values(13 * team.team_rank() + i) =
            i - team.team_rank() - team.league_rank() + team.league_size() +
            team.team_size();
      });

      flare::single(flare::PerThread(team), [&]() {
        Scalar test  = 0;
        Scalar value = 0;

        for (int i = 0; i < 13; ++i) {
          test += i - team.team_rank() - team.league_rank() +
                  team.league_size() + team.team_size();
          value += values(13 * team.team_rank() + i);
        }

        if (test != value) {
          flare::printf("FAILED vector_par_for %i %i %lf %lf\n",
                         team.league_rank(), team.team_rank(),
                         static_cast<double>(test), static_cast<double>(value));

          flag() = 1;
        }
      });
    }
  }
};

template <typename Scalar, class ExecutionSpace>
struct functor_vec_red {
  using policy_type     = flare::TeamPolicy<ExecutionSpace>;
  using execution_space = ExecutionSpace;

  flare::Tensor<int, flare::LayoutLeft, ExecutionSpace> flag;

  functor_vec_red(flare::Tensor<int, flare::LayoutLeft, ExecutionSpace> flag_)
      : flag(flag_) {}

  FLARE_INLINE_FUNCTION
  void operator()(typename policy_type::member_type team) const {
    Scalar value = 0;

    // When no reducer is given the default is summation.
    flare::parallel_reduce(
        flare::ThreadVectorRange(team, 13),
        [&](int i, Scalar &val) { val += i; }, value);

    flare::single(flare::PerThread(team), [&]() {
      Scalar test = 0;

      for (int i = 0; i < 13; i++) test += i;

      if (test != value) {
        flare::printf("FAILED vector_par_reduce %i %i %lf %lf\n",
                       team.league_rank(), team.team_rank(), (double)test,
                       (double)value);
        flag() = 1;
      }
    });
  }
};

template <typename Scalar, class ExecutionSpace>
struct functor_vec_red_reducer {
  using policy_type     = flare::TeamPolicy<ExecutionSpace>;
  using execution_space = ExecutionSpace;

  flare::Tensor<int, flare::LayoutLeft, ExecutionSpace> flag;

  functor_vec_red_reducer(
      flare::Tensor<int, flare::LayoutLeft, ExecutionSpace> flag_)
      : flag(flag_) {}

  FLARE_INLINE_FUNCTION
  void operator()(typename policy_type::member_type team) const {
    // Must initialize to the identity value for the reduce operation
    // for this test:
    //   ( identity, operation ) = ( 1 , *= )
    Scalar value = 1;

    flare::parallel_reduce(
        flare::ThreadVectorRange(team, 13),
        [&](int i, Scalar &val) { val *= (i % 5 + 1); },
        flare::Prod<Scalar>(value));

    flare::single(flare::PerThread(team), [&]() {
      Scalar test = 1;

      for (int i = 0; i < 13; i++) test *= (i % 5 + 1);

      if (test != value) {
        flare::printf("FAILED vector_par_reduce_reducer %i %i %lf %lf\n",
                       team.league_rank(), team.team_rank(), (double)test,
                       (double)value);

        flag() = 1;
      }
    });
  }
};

template <typename Scalar, class ExecutionSpace>
struct functor_vec_scan {
  using policy_type     = flare::TeamPolicy<ExecutionSpace>;
  using execution_space = ExecutionSpace;

  flare::Tensor<int, flare::LayoutLeft, ExecutionSpace> flag;
  functor_vec_scan(flare::Tensor<int, flare::LayoutLeft, ExecutionSpace> flag_)
      : flag(flag_) {}

  FLARE_INLINE_FUNCTION
  void operator()(typename policy_type::member_type team) const {
    flare::parallel_scan(flare::ThreadVectorRange(team, 13), [&](int i,
                                                                   Scalar &val,
                                                                   bool final) {
      val += i;

      if (final) {
        Scalar test = 0;
        for (int k = 0; k <= i; k++) test += k;

        if (test != val) {
          flare::printf("FAILED vector_par_scan %i %i %lf %lf\n",
                         team.league_rank(), team.team_rank(),
                         static_cast<double>(test), static_cast<double>(val));

          flag() = 1;
        }
      }
    });
  }
};

template <typename Scalar, class ExecutionSpace>
struct functor_reduce {
  using value_type      = double;
  using policy_type     = flare::TeamPolicy<ExecutionSpace>;
  using execution_space = ExecutionSpace;

  flare::Tensor<int, flare::LayoutLeft, ExecutionSpace> flag;
  functor_reduce(flare::Tensor<int, flare::LayoutLeft, ExecutionSpace> flag_)
      : flag(flag_) {}

  FLARE_INLINE_FUNCTION
  void operator()(typename policy_type::member_type team, double &sum) const {
    sum += team.league_rank() * 100 + team.thread_rank();
  }
};

template <typename Scalar, class ExecutionSpace>
bool test_scalar(int nteams, int team_size, int test) {
  flare::Tensor<int, flare::LayoutLeft, ExecutionSpace> d_flag("flag");
  typename flare::Tensor<int, flare::LayoutLeft, ExecutionSpace>::HostMirror
      h_flag("h_flag");
  h_flag() = 0;
  flare::deep_copy(d_flag, h_flag);

  if (test == 0) {
    flare::parallel_for(
        std::string("A"),
        flare::TeamPolicy<ExecutionSpace>(nteams, team_size, 8),
        functor_vec_red<Scalar, ExecutionSpace>(d_flag));
  } else if (test == 1) {
    flare::parallel_for(
        flare::TeamPolicy<ExecutionSpace>(nteams, team_size, 8),
        functor_vec_red_reducer<Scalar, ExecutionSpace>(d_flag));
  } else if (test == 2) {
    flare::parallel_for(
        flare::TeamPolicy<ExecutionSpace>(nteams, team_size, 8),
        functor_vec_scan<Scalar, ExecutionSpace>(d_flag));
  } else if (test == 3) {
    flare::parallel_for(
        flare::TeamPolicy<ExecutionSpace>(nteams, team_size, 8),
        functor_vec_for<Scalar, ExecutionSpace>(d_flag));
  } else if (test == 4) {
    flare::parallel_for(
        "B", flare::TeamPolicy<ExecutionSpace>(nteams, team_size, 8),
        functor_vec_single<Scalar, ExecutionSpace>(d_flag, 0, 13));
  } else if (test == 5) {
    flare::parallel_for(flare::TeamPolicy<ExecutionSpace>(nteams, team_size),
                         functor_team_for<Scalar, ExecutionSpace>(d_flag));
  } else if (test == 6) {
    flare::parallel_for(flare::TeamPolicy<ExecutionSpace>(nteams, team_size),
                         functor_team_reduce<Scalar, ExecutionSpace>(d_flag));
  } else if (test == 7) {
    flare::parallel_for(
        flare::TeamPolicy<ExecutionSpace>(nteams, team_size),
        functor_team_reduce_reducer<Scalar, ExecutionSpace>(d_flag));
  } else if (test == 8) {
    flare::parallel_for(
        flare::TeamPolicy<ExecutionSpace>(nteams, team_size, 8),
        functor_team_vector_for<Scalar, ExecutionSpace>(d_flag));
  } else if (test == 9) {
    flare::parallel_for(
        flare::TeamPolicy<ExecutionSpace>(nteams, team_size, 8),
        functor_team_vector_reduce<Scalar, ExecutionSpace>(d_flag));
  } else if (test == 10) {
    flare::parallel_for(
        flare::TeamPolicy<ExecutionSpace>(nteams, team_size, 8),
        functor_team_vector_reduce_reducer<Scalar, ExecutionSpace>(d_flag));
  } else if (test == 11) {
    flare::parallel_for(
        "B", flare::TeamPolicy<ExecutionSpace>(nteams, team_size, 8),
        functor_vec_single<Scalar, ExecutionSpace>(d_flag, 4, 13));
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
  passed = passed &&
           test_scalar<Test::my_complex, ExecutionSpace>(317, team_size, test);
  passed = passed && test_scalar<Test::array_reduce<double, 1>, ExecutionSpace>(
                         317, team_size, test);
  passed = passed && test_scalar<Test::array_reduce<float, 1>, ExecutionSpace>(
                         317, team_size, test);
  passed = passed && test_scalar<Test::array_reduce<double, 3>, ExecutionSpace>(
                         317, team_size, test);

  return passed;
}

}  // namespace TestTeamVector

namespace Test {


template <typename ScalarType, class DeviceType>
class TestTripleNestedReduce {
 public:
  using execution_space = DeviceType;
  using size_type       = typename execution_space::size_type;

  TestTripleNestedReduce(const size_type &nrows, const size_type &ncols,
                         const size_type &team_size,
                         const size_type &vector_length) {
    run_test(nrows, ncols, team_size, vector_length);
  }

  void run_test(const size_type &nrows, const size_type &ncols,
                size_type team_size, const size_type &vector_length) {
    auto const concurrency =
        static_cast<size_type>(execution_space().concurrency());
    if (team_size > concurrency) team_size = concurrency;

    // using Layout = flare::LayoutLeft;
    using Layout = flare::LayoutRight;

    using TensorVector = flare::Tensor<ScalarType *, DeviceType>;
    using TensorMatrix = flare::Tensor<ScalarType **, Layout, DeviceType>;

    TensorVector y("y", nrows);
    TensorVector x("x", ncols);
    TensorMatrix A("A", nrows, ncols);

    using range_policy = flare::RangePolicy<DeviceType>;

    // Initialize y vector.
    flare::parallel_for(
        range_policy(0, nrows), FLARE_LAMBDA(const int i) { y(i) = 1; });

    // Initialize x vector.
    flare::parallel_for(
        range_policy(0, ncols), FLARE_LAMBDA(const int i) { x(i) = 1; });
    flare::fence();

    using team_policy = flare::TeamPolicy<DeviceType>;
    using member_type = typename flare::TeamPolicy<DeviceType>::member_type;

    // Initialize A matrix, note 2D indexing computation.
    flare::parallel_for(
        team_policy(nrows, flare::AUTO),
        FLARE_LAMBDA(const member_type &teamMember) {
          const int j = teamMember.league_rank();
          flare::parallel_for(flare::TeamThreadRange(teamMember, ncols),
                               [&](const int i) { A(j, i) = 1; });
        });
    flare::fence();

    // Three level parallelism kernel to force caching of vector x.
    ScalarType result = 0.0;
    int chunk_size    = 128;
    flare::parallel_reduce(
        team_policy(nrows / chunk_size, team_size, vector_length),
        FLARE_LAMBDA(const member_type &teamMember, double &update) {
          const int row_start = teamMember.league_rank() * chunk_size;
          const int row_end   = row_start + chunk_size;
          flare::parallel_for(
              flare::TeamThreadRange(teamMember, row_start, row_end),
              [&](const int i) {
                ScalarType sum_i = 0.0;
                flare::parallel_reduce(
                    flare::ThreadVectorRange(teamMember, ncols),
                    [&](const int j, ScalarType &innerUpdate) {
                      innerUpdate += A(i, j) * x(j);
                    },
                    sum_i);
                flare::single(flare::PerThread(teamMember),
                               [&]() { update += y(i) * sum_i; });
              });
        },
        result);
    flare::fence();

    const ScalarType solution = (ScalarType)nrows * (ScalarType)ncols;
    if (int64_t(solution) != int64_t(result)) {
      printf("  TestTripleNestedReduce failed solution(%" PRId64
             ") != result(%" PRId64
             "),"
             " nrows(%" PRId32 ") ncols(%" PRId32 ") league_size(%" PRId32
             ") team_size(%" PRId32 ")\n",
             int64_t(solution), int64_t(result), int32_t(nrows), int32_t(ncols),
             int32_t(nrows / chunk_size), int32_t(team_size));
    }

    ASSERT_EQ(solution, result);
  }
};

namespace VectorScanReducer {
enum class ScanType : bool { Inclusive, Exclusive };

template <typename ExecutionSpace, ScanType scan_type, int n,
          int n_vector_range, class Reducer>
struct checkScan {
  const int n_team_thread_range = 1000;
  const int n_per_team          = n_team_thread_range * n_vector_range;

  using size_type  = typename ExecutionSpace::size_type;
  using value_type = typename Reducer::value_type;
  using tensor_type  = flare::Tensor<value_type[n], ExecutionSpace>;

  tensor_type inputs  = tensor_type{"inputs"};
  tensor_type outputs = tensor_type{"outputs"};

  value_type result;
  Reducer reducer = {result};

  struct ThreadVectorFunctor {
    FLARE_FUNCTION void operator()(const size_type j, value_type &update,
                                    const bool final) const {
      const size_type element = j + m_team_offset + m_thread_offset;
      const auto tmp          = m_inputs(element);
      if (scan_type == ScanType::Inclusive) {
        m_reducer.join(update, tmp);
        if (final) {
          m_outputs(element) = update;
        }
      } else {
        if (final) {
          m_outputs(element) = update;
        }
        m_reducer.join(update, tmp);
      }
    }

    const Reducer &m_reducer;
    const size_type &m_team_offset;
    const size_type &m_thread_offset;
    const tensor_type &m_outputs;
    const tensor_type &m_inputs;
  };

  struct TeamThreadRangeFunctor {
    FLARE_FUNCTION void operator()(const size_type i) const {
      const size_type thread_offset = i * n_vector_range;
      flare::parallel_scan(
          flare::ThreadVectorRange(m_team, n_vector_range),
          ThreadVectorFunctor{m_reducer, m_team_offset, thread_offset,
                              m_outputs, m_inputs},
          m_reducer);
    }

    const typename flare::TeamPolicy<ExecutionSpace>::member_type &m_team;
    const Reducer &m_reducer;
    const size_type &m_team_offset;
    const tensor_type &m_outputs;
    const tensor_type &m_inputs;
  };

  FLARE_FUNCTION void operator()(
      const typename flare::TeamPolicy<ExecutionSpace>::member_type &team)
      const {
    const size_type iTeam       = team.league_rank();
    const size_type iTeamOffset = iTeam * n_per_team;
    flare::parallel_for(
        flare::TeamThreadRange(team, n_team_thread_range),
        TeamThreadRangeFunctor{team, reducer, iTeamOffset, outputs, inputs});
  }

  FLARE_FUNCTION void operator()(size_type i) const { inputs(i) = i * 1. / n; }

  void run() {
    const int n_teams = n / n_per_team;

    flare::parallel_for(flare::RangePolicy<ExecutionSpace>(0, n), *this);

    // run ThreadVectorRange parallel_scan
    flare::TeamPolicy<ExecutionSpace> policy(n_teams, flare::AUTO,
                                              flare::AUTO);
    const std::string label =
        (scan_type == ScanType::Inclusive ? std::string("inclusive")
                                          : std::string("exclusive")) +
        "Scan" + typeid(Reducer).name();
    flare::parallel_for(label, policy, *this);
    flare::fence();

    auto host_outputs =
        flare::create_mirror_tensor_and_copy(flare::HostSpace{}, outputs);
    auto host_inputs =
        flare::create_mirror_tensor_and_copy(flare::HostSpace{}, inputs);

    flare::Tensor<value_type[n], flare::HostSpace> expected("expected");
    {
      value_type identity;
      reducer.init(identity);
      for (int i = 0; i < expected.extent_int(0); ++i) {
        const int vector       = i % n_vector_range;
        const value_type accum = vector == 0 ? identity : expected(i - 1);
        const value_type val =
            scan_type == ScanType::Inclusive
                ? host_inputs(i)
                : (vector == 0 ? identity : host_inputs(i - 1));
        expected(i) = accum;
        reducer.join(expected(i), val);
      }
    }
    for (int i = 0; i < host_outputs.extent_int(0); ++i)
      ASSERT_EQ(host_outputs(i), expected(i)) << "differ at index " << i;
  }
};
}  // namespace VectorScanReducer

#if !defined(FLARE_IMPL_CUDA_CLANG_WORKAROUND)
TEST(TEST_CATEGORY, team_vector) {
  ASSERT_TRUE((TestTeamVector::Test<TEST_EXECSPACE>(0)));
  ASSERT_TRUE((TestTeamVector::Test<TEST_EXECSPACE>(1)));
  ASSERT_TRUE((TestTeamVector::Test<TEST_EXECSPACE>(2)));
  ASSERT_TRUE((TestTeamVector::Test<TEST_EXECSPACE>(3)));
  ASSERT_TRUE((TestTeamVector::Test<TEST_EXECSPACE>(4)));
  ASSERT_TRUE((TestTeamVector::Test<TEST_EXECSPACE>(5)));
  ASSERT_TRUE((TestTeamVector::Test<TEST_EXECSPACE>(6)));
  ASSERT_TRUE((TestTeamVector::Test<TEST_EXECSPACE>(7)));
  ASSERT_TRUE((TestTeamVector::Test<TEST_EXECSPACE>(8)));
  ASSERT_TRUE((TestTeamVector::Test<TEST_EXECSPACE>(9)));
  ASSERT_TRUE((TestTeamVector::Test<TEST_EXECSPACE>(10)));
  ASSERT_TRUE((TestTeamVector::Test<TEST_EXECSPACE>(11)));
}
#endif

#if !defined(FLARE_IMPL_CUDA_CLANG_WORKAROUND)
TEST(TEST_CATEGORY, triple_nested_parallelism) {
// With FLARE_ENABLE_DEBUG enabled, the functor uses too many registers to run
// with a team size of 32 on GPUs, 16 is the max possible (at least on a K80
// GPU)
#if defined(FLARE_ENABLE_DEBUG) && defined(FLARE_ON_CUDA_DEVICE)
  if (!std::is_same<TEST_EXECSPACE, flare::Cuda>::value)
#endif
  {
    TestTripleNestedReduce<double, TEST_EXECSPACE>(8192, 2048, 32, 32);
    TestTripleNestedReduce<double, TEST_EXECSPACE>(8192, 2048, 32, 16);
  }
  {
    TestTripleNestedReduce<double, TEST_EXECSPACE>(8192, 2048, 16, 33);
    TestTripleNestedReduce<double, TEST_EXECSPACE>(8192, 2048, 16, 19);
  }
  TestTripleNestedReduce<double, TEST_EXECSPACE>(8192, 2048, 16, 16);
  TestTripleNestedReduce<double, TEST_EXECSPACE>(8192, 2048, 7, 16);
}
#endif

TEST(TEST_CATEGORY, parallel_scan_with_reducers) {
  using T = double;
  using namespace VectorScanReducer;

  constexpr int n              = 1000000;
  constexpr int n_vector_range = 100;

#if defined(FLARE_ON_CUDA_DEVICE) && \
    defined(FLARE_COMPILER_NVHPC)  // FIXME_NVHPC 23.7
  if constexpr (std::is_same_v<TEST_EXECSPACE, flare::Cuda>) {
    GTEST_SKIP() << "All but max inclusive scan differ at index 101";
  }
#endif

  checkScan<TEST_EXECSPACE, ScanType::Exclusive, n, n_vector_range,
            flare::Prod<T, TEST_EXECSPACE>>()
      .run();
  checkScan<TEST_EXECSPACE, ScanType::Inclusive, n, n_vector_range,
            flare::Prod<T, TEST_EXECSPACE>>()
      .run();

  checkScan<TEST_EXECSPACE, ScanType::Exclusive, n, n_vector_range,
            flare::Max<T, TEST_EXECSPACE>>()
      .run();
  checkScan<TEST_EXECSPACE, ScanType::Inclusive, n, n_vector_range,
            flare::Max<T, TEST_EXECSPACE>>()
      .run();

  checkScan<TEST_EXECSPACE, ScanType::Exclusive, n, n_vector_range,
            flare::Min<T, TEST_EXECSPACE>>()
      .run();
  checkScan<TEST_EXECSPACE, ScanType::Inclusive, n, n_vector_range,
            flare::Min<T, TEST_EXECSPACE>>()
      .run();

  (void)n;
  (void)n_vector_range;
}

}  // namespace Test
