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

#include <cstdio>
#include <sstream>
#include <iostream>

#include <flare/core.h>

namespace Test {

namespace {

template <class ExecSpace, class ScheduleType>
struct TestTeamPolicy {
  using team_member =
      typename flare::TeamPolicy<ScheduleType, ExecSpace>::member_type;
  using tensor_type = flare::Tensor<int **, ExecSpace>;

  tensor_type m_flags;

  TestTeamPolicy(const size_t league_size)
      : m_flags(flare::tensor_alloc(flare::WithoutInitializing, "flags"),
                flare::TeamPolicy<ScheduleType, ExecSpace>(1, 1).team_size_max(
                    *this, flare::ParallelReduceTag()),
                league_size) {
  }

  struct VerifyInitTag {};

  FLARE_INLINE_FUNCTION
  void operator()(const team_member &member) const {
    const int tid =
        member.team_rank() + member.team_size() * member.league_rank();

    m_flags(member.team_rank(), member.league_rank()) = tid;
    static_assert(
        (std::is_same<typename team_member::execution_space, ExecSpace>::value),
        "TeamMember::execution_space is not the same as "
        "TeamPolicy<>::execution_space");
  }

  FLARE_INLINE_FUNCTION
  void operator()(const VerifyInitTag &, const team_member &member) const {
    const int tid =
        member.team_rank() + member.team_size() * member.league_rank();

    if (tid != m_flags(member.team_rank(), member.league_rank())) {
      flare::printf("TestTeamPolicy member(%d,%d) error %d != %d\n",
                     member.league_rank(), member.team_rank(), tid,
                     m_flags(member.team_rank(), member.league_rank()));
    }
  }

  // Included for test_small_league_size.
  TestTeamPolicy() : m_flags() {}

  // Included for test_small_league_size.
  struct NoOpTag {};

  FLARE_INLINE_FUNCTION
  void operator()(const NoOpTag &, const team_member & /*member*/) const {}

  static void test_small_league_size() {
    int bs = 8;   // batch size (number of elements per batch)
    int ns = 16;  // total number of "problems" to process

    // Calculate total scratch memory space size.
    const int level     = 0;
    int mem_size        = 960;
    const int num_teams = ns / bs;
    flare::TeamPolicy<ExecSpace, NoOpTag> policy(num_teams, flare::AUTO());

    flare::parallel_for(
        policy.set_scratch_size(level, flare::PerTeam(mem_size),
                                flare::PerThread(0)),
        TestTeamPolicy());
  }

  static void test_constructors() {
    constexpr const int smallest_work = 1;

    flare::TeamPolicy<ExecSpace, NoOpTag> none_auto(
        smallest_work, smallest_work, smallest_work);
    (void)none_auto;
    flare::TeamPolicy<ExecSpace, NoOpTag> both_auto(
        smallest_work, flare::AUTO(), flare::AUTO());
    (void)both_auto;
    flare::TeamPolicy<ExecSpace, NoOpTag> auto_vector(
        smallest_work, smallest_work, flare::AUTO());
    (void)auto_vector;
    flare::TeamPolicy<ExecSpace, NoOpTag> auto_team(
        smallest_work, flare::AUTO(), smallest_work);
    (void)auto_team;
  }

  static void test_for(const size_t league_size) {
    {
      TestTeamPolicy functor(league_size);
      using policy_type = flare::TeamPolicy<ScheduleType, ExecSpace>;
      using policy_type_init =
          flare::TeamPolicy<ScheduleType, ExecSpace, VerifyInitTag>;
      const int team_size =
          policy_type(league_size, 1)
              .team_size_max(functor, flare::ParallelForTag());
      const int team_size_init =
          policy_type_init(league_size, 1)
              .team_size_max(functor, flare::ParallelForTag());

      flare::parallel_for(policy_type(league_size, team_size), functor);
      flare::parallel_for(policy_type_init(league_size, team_size_init),
                           functor);
    }

    test_small_league_size();
    test_constructors();
  }

  struct ReduceTag {};

  using value_type = int64_t;

  FLARE_INLINE_FUNCTION
  void operator()(const team_member &member, value_type &update) const {
    update += member.team_rank() + member.team_size() * member.league_rank();
  }

  FLARE_INLINE_FUNCTION
  void operator()(const ReduceTag &, const team_member &member,
                  value_type &update) const {
    update +=
        1 + member.team_rank() + member.team_size() * member.league_rank();
  }

  static void test_reduce(const size_t league_size) {
    TestTeamPolicy functor(league_size);

    using policy_type = flare::TeamPolicy<ScheduleType, ExecSpace>;
    using policy_type_reduce =
        flare::TeamPolicy<ScheduleType, ExecSpace, ReduceTag>;

    const int team_size =
        policy_type_reduce(league_size, 1)
            .team_size_max(functor, flare::ParallelReduceTag());

    const int64_t N = team_size * league_size;

    int64_t total = 0;

    flare::parallel_reduce(policy_type(league_size, team_size), functor,
                            total);
    ASSERT_EQ(size_t((N - 1) * (N)) / 2, size_t(total));

    flare::parallel_reduce(policy_type_reduce(league_size, team_size), functor,
                            total);
    ASSERT_EQ((size_t(N) * size_t(N + 1)) / 2, size_t(total));
  }
};

}  // namespace

}  // namespace Test

/*--------------------------------------------------------------------------*/

namespace Test {

template <typename ScalarType, class DeviceType, class ScheduleType>
class ReduceTeamFunctor {
 public:
  using execution_space = DeviceType;
  using policy_type     = flare::TeamPolicy<ScheduleType, execution_space>;
  using size_type       = typename execution_space::size_type;

  struct value_type {
    ScalarType value[3];
  };

  const size_type nwork;

  FLARE_INLINE_FUNCTION
  ReduceTeamFunctor(const size_type &arg_nwork) : nwork(arg_nwork) {}

  FLARE_INLINE_FUNCTION
  ReduceTeamFunctor(const ReduceTeamFunctor &rhs) : nwork(rhs.nwork) {}

  FLARE_INLINE_FUNCTION
  void init(value_type &dst) const {
    dst.value[0] = 0;
    dst.value[1] = 0;
    dst.value[2] = 0;
  }

  FLARE_INLINE_FUNCTION
  void join(value_type &dst, const value_type &src) const {
    dst.value[0] += src.value[0];
    dst.value[1] += src.value[1];
    dst.value[2] += src.value[2];
  }

  FLARE_INLINE_FUNCTION
  void operator()(const typename policy_type::member_type ind,
                  value_type &dst) const {
    const int thread_rank =
        ind.team_rank() + ind.team_size() * ind.league_rank();
    const int thread_size = ind.team_size() * ind.league_size();
    const int chunk       = (nwork + thread_size - 1) / thread_size;

    size_type iwork           = chunk * thread_rank;
    const size_type iwork_end = iwork + chunk < nwork ? iwork + chunk : nwork;

    for (; iwork < iwork_end; ++iwork) {
      dst.value[0] += 1;
      dst.value[1] += iwork + 1;
      dst.value[2] += nwork - iwork;
    }
  }
};

}  // namespace Test

namespace {

template <typename ScalarType, class DeviceType, class ScheduleType>
class TestReduceTeam {
 public:
  using execution_space = DeviceType;
  using policy_type     = flare::TeamPolicy<ScheduleType, execution_space>;
  using size_type       = typename execution_space::size_type;

  TestReduceTeam(const size_type &nwork) { run_test(nwork); }

  void run_test(const size_type &nwork) {
    using functor_type =
        Test::ReduceTeamFunctor<ScalarType, execution_space, ScheduleType>;
    using value_type = typename functor_type::value_type;
    using result_type =
        flare::Tensor<value_type, flare::HostSpace, flare::MemoryUnmanaged>;

    enum { Count = 3 };
    enum { Repeat = 100 };

    value_type result[Repeat];

    const uint64_t nw   = nwork;
    const uint64_t nsum = nw % 2 ? nw * ((nw + 1) / 2) : (nw / 2) * (nw + 1);

    policy_type team_exec(nw, 1);

    const unsigned team_size = team_exec.team_size_recommended(
        functor_type(nwork), flare::ParallelReduceTag());
    const unsigned league_size = (nwork + team_size - 1) / team_size;

    team_exec = policy_type(league_size, team_size);

    for (unsigned i = 0; i < Repeat; ++i) {
      result_type tmp(&result[i]);
      flare::parallel_reduce(team_exec, functor_type(nwork), tmp);
    }

    execution_space().fence();

    for (unsigned i = 0; i < Repeat; ++i) {
      for (unsigned j = 0; j < Count; ++j) {
        const uint64_t correct = 0 == j % 3 ? nw : nsum;
        ASSERT_EQ((ScalarType)correct, result[i].value[j]);
      }
    }
  }
};

}  // namespace

/*--------------------------------------------------------------------------*/

namespace Test {

template <class DeviceType, class ScheduleType>
class ScanTeamFunctor {
 public:
  using execution_space = DeviceType;
  using policy_type     = flare::TeamPolicy<ScheduleType, execution_space>;
  using value_type      = int64_t;

  flare::Tensor<value_type, execution_space> accum;
  flare::Tensor<value_type, execution_space> total;

  ScanTeamFunctor() : accum("accum"), total("total") {}

  FLARE_INLINE_FUNCTION
  void init(value_type &error) const { error = 0; }

  FLARE_INLINE_FUNCTION
  void join(value_type &error, value_type const &input) const {
    if (input) error = 1;
  }

  struct JoinMax {
    using value_type = int64_t;

    FLARE_INLINE_FUNCTION
    void join(value_type &dst, value_type const &input) const {
      if (dst < input) dst = input;
    }
  };

  FLARE_INLINE_FUNCTION
  void operator()(const typename policy_type::member_type ind,
                  value_type &error) const {
    if (0 == ind.league_rank() && 0 == ind.team_rank()) {
      const int64_t thread_count = ind.league_size() * ind.team_size();
      total()                    = (thread_count * (thread_count + 1)) / 2;
    }

    // Team max:
    int64_t m = (int64_t)(ind.league_rank() + ind.team_rank());
    ind.team_reduce(flare::Max<int64_t>(m));

    if (m != ind.league_rank() + (ind.team_size() - 1)) {
      flare::printf(
          "ScanTeamFunctor[%i.%i of %i.%i] reduce_max_answer(%li) != "
          "reduce_max(%li)\n",
          static_cast<int>(ind.league_rank()),
          static_cast<int>(ind.team_rank()),
          static_cast<int>(ind.league_size()),
          static_cast<int>(ind.team_size()),
          static_cast<long>(ind.league_rank() + (ind.team_size() - 1)),
          static_cast<long>(m));
    }

    // Scan:
    const int64_t answer = (ind.league_rank() + 1) * ind.team_rank() +
                           (ind.team_rank() * (ind.team_rank() + 1)) / 2;

    const int64_t result =
        ind.team_scan(ind.league_rank() + 1 + ind.team_rank() + 1);

    const int64_t result2 =
        ind.team_scan(ind.league_rank() + 1 + ind.team_rank() + 1);

    if (answer != result || answer != result2) {
      flare::printf(
          "ScanTeamFunctor[%i.%i of %i.%i] answer(%li) != scan_first(%li) or "
          "scan_second(%li)\n",
          static_cast<int>(ind.league_rank()),
          static_cast<int>(ind.team_rank()),
          static_cast<int>(ind.league_size()),
          static_cast<int>(ind.team_size()), static_cast<long>(answer),
          static_cast<long>(result), static_cast<long>(result2));

      error = 1;
    }

    const int64_t thread_rank =
        ind.team_rank() + ind.team_size() * ind.league_rank();
    ind.team_scan(1 + thread_rank, accum.data());
  }
};

template <class DeviceType, class ScheduleType>
class TestScanTeam {
 public:
  using execution_space = DeviceType;
  using value_type      = int64_t;
  using policy_type     = flare::TeamPolicy<ScheduleType, execution_space>;
  using functor_type    = Test::ScanTeamFunctor<DeviceType, ScheduleType>;

  TestScanTeam(const size_t nteam) { run_test(nteam); }

  void run_test(const size_t nteam) {
    using result_type =
        flare::Tensor<int64_t, flare::HostSpace, flare::MemoryUnmanaged>;

    const unsigned REPEAT = 100000;
    unsigned Repeat;

    if (nteam == 0) {
      Repeat = 1;
    } else {
      Repeat = (REPEAT + nteam - 1) / nteam;  // Error here.
    }

    functor_type functor;

    policy_type team_exec(nteam, 1);
    const auto team_size =
        team_exec.team_size_max(functor, flare::ParallelReduceTag());
    team_exec = policy_type(nteam, team_size);

    for (unsigned i = 0; i < Repeat; ++i) {
      int64_t accum = 0;
      int64_t total = 0;
      int64_t error = 0;
      flare::deep_copy(functor.accum, total);

      flare::parallel_reduce(team_exec, functor, result_type(&error));
      DeviceType().fence();

      flare::deep_copy(accum, functor.accum);
      flare::deep_copy(total, functor.total);

      ASSERT_EQ(error, 0);
      ASSERT_EQ(total, accum);
    }

    execution_space().fence();
  }
};

}  // namespace Test

/*--------------------------------------------------------------------------*/

namespace Test {

template <class ExecSpace, class ScheduleType>
struct SharedTeamFunctor {
  using execution_space = ExecSpace;
  using value_type      = int;
  using policy_type     = flare::TeamPolicy<ScheduleType, execution_space>;

  enum { SHARED_COUNT = 1000 };

  using shmem_space = typename ExecSpace::scratch_memory_space;

  // TBD: MemoryUnmanaged should be the default for shared memory space.
  using shared_int_array_type =
      flare::Tensor<int *, shmem_space, flare::MemoryUnmanaged>;

  // Tell how much shared memory will be required by this functor.
  inline unsigned team_shmem_size(int /*team_size*/) const {
    return shared_int_array_type::shmem_size(SHARED_COUNT) +
           shared_int_array_type::shmem_size(SHARED_COUNT);
  }

  FLARE_INLINE_FUNCTION
  void operator()(const typename policy_type::member_type &ind,
                  value_type &update) const {
    const shared_int_array_type shared_A(ind.team_shmem(), SHARED_COUNT);
    const shared_int_array_type shared_B(ind.team_shmem(), SHARED_COUNT);

    if ((shared_A.data() == nullptr && SHARED_COUNT > 0) ||
        (shared_B.data() == nullptr && SHARED_COUNT > 0)) {
      flare::printf(
          "member( %i/%i , %i/%i ) Failed to allocate shared memory of size "
          "%lu\n",
          static_cast<int>(ind.league_rank()),
          static_cast<int>(ind.league_size()),
          static_cast<int>(ind.team_rank()), static_cast<int>(ind.team_size()),
          static_cast<unsigned long>(SHARED_COUNT));

      ++update;  // Failure to allocate is an error.
    } else {
      for (int i = ind.team_rank(); i < SHARED_COUNT; i += ind.team_size()) {
        shared_A[i] = i + ind.league_rank();
        shared_B[i] = 2 * i + ind.league_rank();
      }

      ind.team_barrier();

      if (ind.team_rank() + 1 == ind.team_size()) {
        for (int i = 0; i < SHARED_COUNT; ++i) {
          if (shared_A[i] != i + ind.league_rank()) {
            ++update;
          }

          if (shared_B[i] != 2 * i + ind.league_rank()) {
            ++update;
          }
        }
      }
    }
  }
};

}  // namespace Test

namespace {

template <class ExecSpace, class ScheduleType>
struct TestSharedTeam {
  TestSharedTeam() { run(); }

  void run() {
    using Functor = Test::SharedTeamFunctor<ExecSpace, ScheduleType>;
    using result_type =
        flare::Tensor<typename Functor::value_type, flare::HostSpace,
                     flare::MemoryUnmanaged>;
    const size_t team_size =
        flare::TeamPolicy<ScheduleType, ExecSpace>(8192, 1).team_size_max(
            Functor(), flare::ParallelReduceTag());

    flare::TeamPolicy<ScheduleType, ExecSpace> team_exec(8192 / team_size,
                                                          team_size);

    typename Functor::value_type error_count = 0;

    flare::parallel_reduce(team_exec, Functor(), result_type(&error_count));
    flare::fence();

    ASSERT_EQ(error_count, 0);
  }
};

}  // namespace

namespace Test {

#if defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)
template <class MemorySpace, class ExecSpace, class ScheduleType>
struct TestLambdaSharedTeam {
  TestLambdaSharedTeam() { run(); }

  void run() {
    using Functor     = Test::SharedTeamFunctor<ExecSpace, ScheduleType>;
    using result_type = flare::Tensor<typename Functor::value_type, MemorySpace,
                                     flare::MemoryUnmanaged>;

    using shmem_space = typename ExecSpace::scratch_memory_space;

    // TBD: MemoryUnmanaged should be the default for shared memory space.
    using shared_int_array_type =
        flare::Tensor<int *, shmem_space, flare::MemoryUnmanaged>;

    const int SHARED_COUNT = 1000;
    int team_size = 1;

#ifdef FLARE_ON_CUDA_DEVICE
    if (std::is_same<ExecSpace, flare::Cuda>::value) team_size = 128;
#endif

    flare::TeamPolicy<ScheduleType, ExecSpace> team_exec(8192 / team_size,
                                                          team_size);

    int scratch_size = shared_int_array_type::shmem_size(SHARED_COUNT) * 2;
    team_exec = team_exec.set_scratch_size(0, flare::PerTeam(scratch_size));

    typename Functor::value_type error_count = 0;

    flare::parallel_reduce(
        team_exec,
        FLARE_LAMBDA(
            const typename flare::TeamPolicy<ScheduleType,
                                              ExecSpace>::member_type &ind,
            int &update) {
          const shared_int_array_type shared_A(ind.team_shmem(), SHARED_COUNT);
          const shared_int_array_type shared_B(ind.team_shmem(), SHARED_COUNT);

          if ((shared_A.data() == nullptr && SHARED_COUNT > 0) ||
              (shared_B.data() == nullptr && SHARED_COUNT > 0)) {
            flare::printf("Failed to allocate shared memory of size %lu\n",
                           static_cast<unsigned long>(SHARED_COUNT));

            ++update;  // Failure to allocate is an error.
          } else {
            for (int i = ind.team_rank(); i < SHARED_COUNT;
                 i += ind.team_size()) {
              shared_A[i] = i + ind.league_rank();
              shared_B[i] = 2 * i + ind.league_rank();
            }

            ind.team_barrier();

            if (ind.team_rank() + 1 == ind.team_size()) {
              for (int i = 0; i < SHARED_COUNT; ++i) {
                if (shared_A[i] != i + ind.league_rank()) {
                  ++update;
                }

                if (shared_B[i] != 2 * i + ind.league_rank()) {
                  ++update;
                }
              }
            }
          }
        },
        result_type(&error_count));

    flare::fence();

    ASSERT_EQ(error_count, 0);
  }
};
#endif

}  // namespace Test

namespace Test {

template <class ExecSpace, class ScheduleType>
struct ScratchTeamFunctor {
  using execution_space = ExecSpace;
  using value_type      = int;
  using policy_type     = flare::TeamPolicy<ScheduleType, execution_space>;

  enum { SHARED_TEAM_COUNT = 100 };
  enum { SHARED_THREAD_COUNT = 10 };

  using shmem_space = typename ExecSpace::scratch_memory_space;

  // TBD: MemoryUnmanaged should be the default for shared memory space.
  using shared_int_array_type =
      flare::Tensor<size_t *, shmem_space, flare::MemoryUnmanaged>;

  FLARE_INLINE_FUNCTION
  void operator()(const typename policy_type::member_type &ind,
                  value_type &update) const {
    const shared_int_array_type scratch_ptr(ind.team_scratch(1),
                                            3 * ind.team_size());
    const shared_int_array_type scratch_A(ind.team_scratch(1),
                                          SHARED_TEAM_COUNT);
    const shared_int_array_type scratch_B(ind.thread_scratch(1),
                                          SHARED_THREAD_COUNT);

    if ((scratch_ptr.data() == nullptr) ||
        (scratch_A.data() == nullptr && SHARED_TEAM_COUNT > 0) ||
        (scratch_B.data() == nullptr && SHARED_THREAD_COUNT > 0)) {
      flare::printf("Failed to allocate shared memory of size %lu\n",
                     static_cast<unsigned long>(SHARED_TEAM_COUNT));

      ++update;  // Failure to allocate is an error.
    } else {
      flare::parallel_for(
          flare::TeamThreadRange(ind, 0, (int)SHARED_TEAM_COUNT),
          [&](const int &i) { scratch_A[i] = i + ind.league_rank(); });

      for (int i = 0; i < SHARED_THREAD_COUNT; i++) {
        scratch_B[i] = 10000 * ind.league_rank() + 100 * ind.team_rank() + i;
      }

      scratch_ptr[ind.team_rank()]                   = (size_t)scratch_A.data();
      scratch_ptr[ind.team_rank() + ind.team_size()] = (size_t)scratch_B.data();

      ind.team_barrier();

      for (int i = 0; i < SHARED_TEAM_COUNT; i++) {
        if (scratch_A[i] != size_t(i + ind.league_rank())) ++update;
      }

      for (int i = 0; i < ind.team_size(); i++) {
        if (scratch_ptr[0] != scratch_ptr[i]) ++update;
      }

      if (scratch_ptr[1 + ind.team_size()] - scratch_ptr[0 + ind.team_size()] <
          SHARED_THREAD_COUNT * sizeof(size_t)) {
        ++update;
      }

      for (int i = 1; i < ind.team_size(); i++) {
        if ((scratch_ptr[i + ind.team_size()] -
             scratch_ptr[i - 1 + ind.team_size()]) !=
            (scratch_ptr[1 + ind.team_size()] -
             scratch_ptr[0 + ind.team_size()])) {
          ++update;
        }
      }
    }
  }
};

}  // namespace Test

namespace {

template <class ExecSpace, class ScheduleType>
struct TestScratchTeam {
  TestScratchTeam() { run(); }

  void run() {
    using Functor = Test::ScratchTeamFunctor<ExecSpace, ScheduleType>;
    using result_type =
        flare::Tensor<typename Functor::value_type, flare::HostSpace,
                     flare::MemoryUnmanaged>;
    using p_type = flare::TeamPolicy<ScheduleType, ExecSpace>;

    typename Functor::value_type error_count = 0;

    int thread_scratch_size = Functor::shared_int_array_type::shmem_size(
        Functor::SHARED_THREAD_COUNT);

    p_type team_exec = p_type(8192, 1).set_scratch_size(
        1,
        flare::PerTeam(Functor::shared_int_array_type::shmem_size(
            Functor::SHARED_TEAM_COUNT)),
        flare::PerThread(thread_scratch_size + 3 * sizeof(int)));

    const size_t team_size =
        team_exec.team_size_max(Functor(), flare::ParallelReduceTag());

    int team_scratch_size =
        Functor::shared_int_array_type::shmem_size(Functor::SHARED_TEAM_COUNT) +
        Functor::shared_int_array_type::shmem_size(3 * team_size);

    team_exec          = p_type(8192 / team_size, team_size);

    flare::parallel_reduce(
        team_exec.set_scratch_size(1, flare::PerTeam(team_scratch_size),
                                   flare::PerThread(thread_scratch_size)),
        Functor(), result_type(&error_count));
    flare::fence();
    ASSERT_EQ(error_count, 0);

    flare::parallel_reduce(
        team_exec.set_scratch_size(1, flare::PerTeam(team_scratch_size),
                                   flare::PerThread(thread_scratch_size)),
        Functor(), flare::Sum<typename Functor::value_type>(error_count));
    flare::fence();
    ASSERT_EQ(error_count, 0);
  }
};

}  // namespace

namespace Test {

template <class ExecSpace>
FLARE_INLINE_FUNCTION int test_team_mulit_level_scratch_loop_body(
    const typename flare::TeamPolicy<ExecSpace>::member_type &team) {
  flare::Tensor<double *, ExecSpace, flare::MemoryTraits<flare::Unmanaged>>
      a_team1(team.team_scratch(0), 128);
  flare::Tensor<double *, ExecSpace, flare::MemoryTraits<flare::Unmanaged>>
      a_thread1(team.thread_scratch(0), 16);
  flare::Tensor<double *, ExecSpace, flare::MemoryTraits<flare::Unmanaged>>
      a_team2(team.team_scratch(0), 128);
  flare::Tensor<double *, ExecSpace, flare::MemoryTraits<flare::Unmanaged>>
      a_thread2(team.thread_scratch(0), 16);

  flare::Tensor<double *, ExecSpace, flare::MemoryTraits<flare::Unmanaged>>
      b_team1(team.team_scratch(1), 12800);
  flare::Tensor<double *, ExecSpace, flare::MemoryTraits<flare::Unmanaged>>
      b_thread1(team.thread_scratch(1), 1600);
  flare::Tensor<double *, ExecSpace, flare::MemoryTraits<flare::Unmanaged>>
      b_team2(team.team_scratch(1), 12800);
  flare::Tensor<double *, ExecSpace, flare::MemoryTraits<flare::Unmanaged>>
      b_thread2(team.thread_scratch(1), 1600);

  flare::Tensor<double *, ExecSpace, flare::MemoryTraits<flare::Unmanaged>>
      a_team3(team.team_scratch(0), 128);
  flare::Tensor<double *, ExecSpace, flare::MemoryTraits<flare::Unmanaged>>
      a_thread3(team.thread_scratch(0), 16);
  flare::Tensor<double *, ExecSpace, flare::MemoryTraits<flare::Unmanaged>>
      b_team3(team.team_scratch(1), 12800);
  flare::Tensor<double *, ExecSpace, flare::MemoryTraits<flare::Unmanaged>>
      b_thread3(team.thread_scratch(1), 1600);

  // The explicit types for 0 and 128 are here to test TeamThreadRange accepting
  // different types for begin and end.
  flare::parallel_for(flare::TeamThreadRange(team, int(0), unsigned(128)),
                       [&](const int &i) {
                         a_team1(i) = 1000000 + i + team.league_rank() * 100000;
                         a_team2(i) = 2000000 + i + team.league_rank() * 100000;
                         a_team3(i) = 3000000 + i + team.league_rank() * 100000;
                       });
  team.team_barrier();

  flare::parallel_for(flare::ThreadVectorRange(team, int(0), unsigned(16)),
                       [&](const int &i) {
                         a_thread1(i) = 1000000 + 100000 * team.team_rank() +
                                        16 - i + team.league_rank() * 100000;
                         a_thread2(i) = 2000000 + 100000 * team.team_rank() +
                                        16 - i + team.league_rank() * 100000;
                         a_thread3(i) = 3000000 + 100000 * team.team_rank() +
                                        16 - i + team.league_rank() * 100000;
                       });

  flare::parallel_for(flare::TeamThreadRange(team, int(0), unsigned(12800)),
                       [&](const int &i) {
                         b_team1(i) = 1000000 + i + team.league_rank() * 100000;
                         b_team2(i) = 2000000 + i + team.league_rank() * 100000;
                         b_team3(i) = 3000000 + i + team.league_rank() * 100000;
                       });
  team.team_barrier();

  flare::parallel_for(flare::ThreadVectorRange(team, 1600),
                       [&](const int &i) {
                         b_thread1(i) = 1000000 + 100000 * team.team_rank() +
                                        16 - i + team.league_rank() * 100000;
                         b_thread2(i) = 2000000 + 100000 * team.team_rank() +
                                        16 - i + team.league_rank() * 100000;
                         b_thread3(i) = 3000000 + 100000 * team.team_rank() +
                                        16 - i + team.league_rank() * 100000;
                       });

  team.team_barrier();

  int error = 0;
  flare::parallel_for(
      flare::TeamThreadRange(team, 0, 128), [&](const int &i) {
        if (a_team1(i) != 1000000 + i + team.league_rank() * 100000) error++;
        if (a_team2(i) != 2000000 + i + team.league_rank() * 100000) error++;
        if (a_team3(i) != 3000000 + i + team.league_rank() * 100000) error++;
      });
  team.team_barrier();

  flare::parallel_for(flare::ThreadVectorRange(team, 16), [&](const int &i) {
    if (a_thread1(i) != 1000000 + 100000 * team.team_rank() + 16 - i +
                            team.league_rank() * 100000)
      error++;
    if (a_thread2(i) != 2000000 + 100000 * team.team_rank() + 16 - i +
                            team.league_rank() * 100000)
      error++;
    if (a_thread3(i) != 3000000 + 100000 * team.team_rank() + 16 - i +
                            team.league_rank() * 100000)
      error++;
  });

  flare::parallel_for(
      flare::TeamThreadRange(team, 0, 12800), [&](const int &i) {
        if (b_team1(i) != 1000000 + i + team.league_rank() * 100000) error++;
        if (b_team2(i) != 2000000 + i + team.league_rank() * 100000) error++;
        if (b_team3(i) != 3000000 + i + team.league_rank() * 100000) error++;
      });
  team.team_barrier();

  flare::parallel_for(
      flare::ThreadVectorRange(team, 1600), [&](const int &i) {
        if (b_thread1(i) != 1000000 + 100000 * team.team_rank() + 16 - i +
                                team.league_rank() * 100000)
          error++;
        if (b_thread2(i) != 2000000 + 100000 * team.team_rank() + 16 - i +
                                team.league_rank() * 100000)
          error++;
        if (b_thread3(i) != 3000000 + 100000 * team.team_rank() + 16 - i +
                                team.league_rank() * 100000)
          error++;
      });

  return error;
}

struct TagReduce {};
struct TagFor {};

template <class ExecSpace, class ScheduleType>
struct ClassNoShmemSizeFunction {
  using member_type =
      typename flare::TeamPolicy<ExecSpace, ScheduleType>::member_type;

  flare::Tensor<int, ExecSpace, flare::MemoryTraits<flare::Atomic>> errors;

  FLARE_INLINE_FUNCTION
  void operator()(const TagFor &, const member_type &team) const {
    int error = test_team_mulit_level_scratch_loop_body<ExecSpace>(team);
    errors() += error;
  }

  FLARE_INLINE_FUNCTION
  void operator()(const TagReduce &, const member_type &team,
                  int &error) const {
    error += test_team_mulit_level_scratch_loop_body<ExecSpace>(team);
  }

  void run() {
    flare::Tensor<int, ExecSpace> d_errors =
        flare::Tensor<int, ExecSpace>("Errors");
    errors = d_errors;

    const int per_team0 =
        3 *
        flare::Tensor<double *, ExecSpace,
                     flare::MemoryTraits<flare::Unmanaged>>::shmem_size(128);
    const int per_thread0 =
        3 *
        flare::Tensor<double *, ExecSpace,
                     flare::MemoryTraits<flare::Unmanaged>>::shmem_size(16);

    const int per_team1 =
        3 * flare::Tensor<
                double *, ExecSpace,
                flare::MemoryTraits<flare::Unmanaged>>::shmem_size(12800);
    const int per_thread1 =
        3 *
        flare::Tensor<double *, ExecSpace,
                     flare::MemoryTraits<flare::Unmanaged>>::shmem_size(1600);

    int team_size      = 8;
    int const concurrency = ExecSpace().concurrency();
    if (team_size > concurrency) team_size = concurrency;
    {
      flare::TeamPolicy<TagFor, ExecSpace, ScheduleType> policy(10, team_size,
                                                                 16);

      flare::parallel_for(
          policy
              .set_scratch_size(0, flare::PerTeam(per_team0),
                                flare::PerThread(per_thread0))
              .set_scratch_size(1, flare::PerTeam(per_team1),
                                flare::PerThread(per_thread1)),
          *this);
      flare::fence();

      typename flare::Tensor<int, ExecSpace>::HostMirror h_errors =
          flare::create_mirror_tensor(d_errors);
      flare::deep_copy(h_errors, d_errors);
      ASSERT_EQ(h_errors(), 0);
    }

    {
      int error = 0;
      flare::TeamPolicy<TagReduce, ExecSpace, ScheduleType> policy(
          10, team_size, 16);

      flare::parallel_reduce(
          policy
              .set_scratch_size(0, flare::PerTeam(per_team0),
                                flare::PerThread(per_thread0))
              .set_scratch_size(1, flare::PerTeam(per_team1),
                                flare::PerThread(per_thread1)),
          *this, error);

      ASSERT_EQ(error, 0);
    }
  };
};

template <class ExecSpace, class ScheduleType>
struct ClassWithShmemSizeFunction {
  using member_type =
      typename flare::TeamPolicy<ExecSpace, ScheduleType>::member_type;

  flare::Tensor<int, ExecSpace, flare::MemoryTraits<flare::Atomic>> errors;

  FLARE_INLINE_FUNCTION
  void operator()(const TagFor &, const member_type &team) const {
    int error = test_team_mulit_level_scratch_loop_body<ExecSpace>(team);
    errors() += error;
  }

  FLARE_INLINE_FUNCTION
  void operator()(const TagReduce &, const member_type &team,
                  int &error) const {
    error += test_team_mulit_level_scratch_loop_body<ExecSpace>(team);
  }

  void run() {
    flare::Tensor<int, ExecSpace> d_errors =
        flare::Tensor<int, ExecSpace>("Errors");
    errors = d_errors;

    const int per_team1 =
        3 * flare::Tensor<
                double *, ExecSpace,
                flare::MemoryTraits<flare::Unmanaged>>::shmem_size(12800);
    const int per_thread1 =
        3 *
        flare::Tensor<double *, ExecSpace,
                     flare::MemoryTraits<flare::Unmanaged>>::shmem_size(1600);

    int team_size = 8;

    int const concurrency = ExecSpace().concurrency();
    if (team_size > concurrency) team_size = concurrency;

    {
      flare::TeamPolicy<TagFor, ExecSpace, ScheduleType> policy(10, team_size,
                                                                 16);

      flare::parallel_for(
          policy.set_scratch_size(1, flare::PerTeam(per_team1),
                                  flare::PerThread(per_thread1)),
          *this);
      flare::fence();

      typename flare::Tensor<int, ExecSpace>::HostMirror h_errors =
          flare::create_mirror_tensor(d_errors);
      flare::deep_copy(h_errors, d_errors);
      ASSERT_EQ(h_errors(), 0);
    }

    {
      int error = 0;
      flare::TeamPolicy<TagReduce, ExecSpace, ScheduleType> policy(
          10, team_size, 16);

      flare::parallel_reduce(
          policy.set_scratch_size(1, flare::PerTeam(per_team1),
                                  flare::PerThread(per_thread1)),
          *this, error);

      ASSERT_EQ(error, 0);
    }
  };

  unsigned team_shmem_size(int team_size) const {
    const int per_team0 =
        3 *
        flare::Tensor<double *, ExecSpace,
                     flare::MemoryTraits<flare::Unmanaged>>::shmem_size(128);
    const int per_thread0 =
        3 *
        flare::Tensor<double *, ExecSpace,
                     flare::MemoryTraits<flare::Unmanaged>>::shmem_size(16);
    return per_team0 + team_size * per_thread0;
  }
};

template <class ExecSpace, class ScheduleType>
void test_team_mulit_level_scratch_test_lambda() {
#ifdef FLARE_ENABLE_CXX11_DISPATCH_LAMBDA
  flare::Tensor<int, ExecSpace, flare::MemoryTraits<flare::Atomic>> errors;
  flare::Tensor<int, ExecSpace> d_errors("Errors");
  errors = d_errors;

  const int per_team0 =
      3 *
      flare::Tensor<double *, ExecSpace,
                   flare::MemoryTraits<flare::Unmanaged>>::shmem_size(128);
  const int per_thread0 =
      3 * flare::Tensor<double *, ExecSpace,
                       flare::MemoryTraits<flare::Unmanaged>>::shmem_size(16);

  const int per_team1 =
      3 *
      flare::Tensor<double *, ExecSpace,
                   flare::MemoryTraits<flare::Unmanaged>>::shmem_size(12800);
  const int per_thread1 =
      3 *
      flare::Tensor<double *, ExecSpace,
                   flare::MemoryTraits<flare::Unmanaged>>::shmem_size(1600);

  int team_size = 8;
  int const concurrency = ExecSpace().concurrency();
  if (team_size > concurrency) team_size = concurrency;

  flare::TeamPolicy<ExecSpace, ScheduleType> policy(10, team_size, 16);

  flare::parallel_for(
      policy
          .set_scratch_size(0, flare::PerTeam(per_team0),
                            flare::PerThread(per_thread0))
          .set_scratch_size(1, flare::PerTeam(per_team1),
                            flare::PerThread(per_thread1)),
      FLARE_LAMBDA(
          const typename flare::TeamPolicy<ExecSpace>::member_type &team) {
        int error = test_team_mulit_level_scratch_loop_body<ExecSpace>(team);
        errors() += error;
      });
  flare::fence();

  typename flare::Tensor<int, ExecSpace>::HostMirror h_errors =
      flare::create_mirror_tensor(errors);
  flare::deep_copy(h_errors, d_errors);
  ASSERT_EQ(h_errors(), 0);

  int error = 0;
  flare::parallel_reduce(
      policy
          .set_scratch_size(0, flare::PerTeam(per_team0),
                            flare::PerThread(per_thread0))
          .set_scratch_size(1, flare::PerTeam(per_team1),
                            flare::PerThread(per_thread1)),
      FLARE_LAMBDA(
          const typename flare::TeamPolicy<ExecSpace>::member_type &team,
          int &count) {
        count += test_team_mulit_level_scratch_loop_body<ExecSpace>(team);
      },
      error);
  ASSERT_EQ(error, 0);
#endif
}

}  // namespace Test

namespace {

template <class ExecSpace, class ScheduleType>
struct TestMultiLevelScratchTeam {
  TestMultiLevelScratchTeam() { run(); }

  void run() {
#ifdef FLARE_ENABLE_CXX11_DISPATCH_LAMBDA
    Test::test_team_mulit_level_scratch_test_lambda<ExecSpace, ScheduleType>();
#endif
    Test::ClassNoShmemSizeFunction<ExecSpace, ScheduleType> c1;
    c1.run();

    Test::ClassWithShmemSizeFunction<ExecSpace, ScheduleType> c2;
    c2.run();
  }
};

}  // namespace

namespace Test {

template <class ExecSpace>
struct TestShmemSize {
  TestShmemSize() { run(); }

  void run() {
    using tensor_type = flare::Tensor<int64_t ***, ExecSpace>;

    size_t d1 = 5;
    size_t d2 = 6;
    size_t d3 = 7;

    size_t size = tensor_type::shmem_size(d1, d2, d3);

    ASSERT_EQ(size, (d1 * d2 * d3 + 1) * sizeof(int64_t));

    test_layout_stride();
  }

  void test_layout_stride() {
    int rank       = 3;
    int order[3]   = {2, 0, 1};
    int extents[3] = {100, 10, 3};
    auto s1 =
        flare::Tensor<double ***, flare::LayoutStride, ExecSpace>::shmem_size(
            flare::LayoutStride::order_dimensions(rank, order, extents));
    auto s2 =
        flare::Tensor<double ***, flare::LayoutRight, ExecSpace>::shmem_size(
            extents[0], extents[1], extents[2]);
    ASSERT_EQ(s1, s2);
  }
};

}  // namespace Test

/*--------------------------------------------------------------------------*/

namespace Test {

namespace {

template <class ExecSpace, class ScheduleType, class T, class Enabled = void>
struct TestTeamBroadcast;

template <class ExecSpace, class ScheduleType, class T>
struct TestTeamBroadcast<ExecSpace, ScheduleType, T,
                         std::enable_if_t<(sizeof(T) == sizeof(char)), void>> {
  using team_member =
      typename flare::TeamPolicy<ScheduleType, ExecSpace>::member_type;
  using memory_space = typename ExecSpace::memory_space;
  using value_type   = T;

  const value_type offset;

  TestTeamBroadcast(const size_t /*league_size*/, const value_type os_)
      : offset(os_) {}

  struct BroadcastTag {};

  FLARE_INLINE_FUNCTION
  void operator()(const team_member &teamMember, value_type &update) const {
    int lid = teamMember.league_rank();
    int tid = teamMember.team_rank();
    int ts  = teamMember.team_size();

    value_type parUpdate = 0;
    value_type value     = (value_type)(tid % 0xFF) + offset;

    // broadcast boolean and value to team from source thread
    teamMember.team_broadcast(value, lid % ts);

    flare::parallel_reduce(
        flare::TeamThreadRange(teamMember, ts),
        [&](const int /*j*/, value_type &teamUpdate) { teamUpdate |= value; },
        flare::BOr<value_type, memory_space>(parUpdate));

    if (teamMember.team_rank() == 0) update |= parUpdate;
  }

  FLARE_INLINE_FUNCTION
  void operator()(const BroadcastTag &, const team_member &teamMember,
                  value_type &update) const {
    int lid = teamMember.league_rank();
    int tid = teamMember.team_rank();
    int ts  = teamMember.team_size();

    value_type parUpdate = 0;
    value_type value     = (value_type)(tid % 0xFF) + offset;

    teamMember.team_broadcast([&](value_type &var) { var -= offset; }, value,
                              lid % ts);

    flare::parallel_reduce(
        flare::TeamThreadRange(teamMember, ts),
        [&](const int /*j*/, value_type &teamUpdate) { teamUpdate |= value; },
        flare::BOr<value_type, memory_space>(parUpdate));

    if (teamMember.team_rank() == 0) update |= parUpdate;
  }

  static void test_teambroadcast(const size_t league_size,
                                 const value_type off) {
    TestTeamBroadcast functor(league_size, off);

    using policy_type = flare::TeamPolicy<ScheduleType, ExecSpace>;
    using policy_type_f =
        flare::TeamPolicy<ScheduleType, ExecSpace, BroadcastTag>;

    int fake_team_size = 1;
    const int team_size =
        policy_type_f(league_size, fake_team_size)
            .team_size_max(
                functor,
                flare::
                    ParallelReduceTag());  // printf("team_size=%d\n",team_size);

    // team_broadcast with value
    value_type total = 0;

    flare::parallel_reduce(policy_type(league_size, team_size), functor,
                            flare::BOr<value_type, flare::HostSpace>(total));

    value_type expected_result = 0;
    for (unsigned int i = 0; i < league_size; i++) {
      value_type val = (value_type((i % team_size % 0xFF)) + off);
      expected_result |= val;
    }
    ASSERT_EQ(expected_result, total);
    // printf("team_broadcast with value --"
    //"expected_result=%x,"
    //"total=%x\n",expected_result, total);

    // team_broadcast with function object
    total = 0;

    flare::parallel_reduce(policy_type_f(league_size, team_size), functor,
                            flare::BOr<value_type, flare::HostSpace>(total));

    expected_result = 0;
    for (unsigned int i = 0; i < league_size; i++) {
      value_type val = ((value_type)((i % team_size % 0xFF)));
      expected_result |= val;
    }
    ASSERT_EQ(expected_result, total);
    // printf("team_broadcast with function object --"
    // "expected_result=%x,"
    // "total=%x\n",expected_result, total);
  }
};

template <class ExecSpace, class ScheduleType, class T>
struct TestTeamBroadcast<ExecSpace, ScheduleType, T,
                         std::enable_if_t<(sizeof(T) > sizeof(char)), void>> {
  using team_member =
      typename flare::TeamPolicy<ScheduleType, ExecSpace>::member_type;
  using value_type = T;

  const value_type offset;

  TestTeamBroadcast(const size_t /*league_size*/, const value_type os_)
      : offset(os_) {}

  struct BroadcastTag {};

  FLARE_INLINE_FUNCTION
  void operator()(const team_member &teamMember, value_type &update) const {
    int lid = teamMember.league_rank();
    int tid = teamMember.team_rank();
    int ts  = teamMember.team_size();

    value_type parUpdate = 0;
    value_type value     = (value_type)(tid * 3) + offset;

    // setValue is used to determine if the update should be
    // performed at the bottom.  The thread id must match the
    // thread id used to broadcast the value.  It is the
    // thread id that matches the league rank mod team size
    // this way each league rank will use a different thread id
    // which is likely not 0
    bool setValue = ((lid % ts) == tid);

    // broadcast boolean and value to team from source thread
    teamMember.team_broadcast(value, lid % ts);
    teamMember.team_broadcast(setValue, lid % ts);

    flare::parallel_reduce(
        flare::TeamThreadRange(teamMember, ts),
        [&](const int /*j*/, value_type &teamUpdate) { teamUpdate += value; },
        parUpdate);

    if (teamMember.team_rank() == 0 && setValue) update += parUpdate;
  }

  FLARE_INLINE_FUNCTION
  void operator()(const BroadcastTag &, const team_member &teamMember,
                  value_type &update) const {
    int lid = teamMember.league_rank();
    int tid = teamMember.team_rank();
    int ts  = teamMember.team_size();

    value_type parUpdate = 0;
    value_type value     = (value_type)(tid * 3) + offset;

    // setValue is used to determine if the update should be
    // performed at the bottom.  The thread id must match the
    // thread id used to broadcast the value.  It is the
    // thread id that matches the league rank mod team size
    // this way each league rank will use a different thread id
    // which is likely not 0. Note the logic is switched from
    // above because the functor switches it back.
    bool setValue = ((lid % ts) != tid);

    teamMember.team_broadcast([&](value_type &var) { var += var; }, value,
                              lid % ts);
    teamMember.team_broadcast([&](bool &bVar) { bVar = !bVar; }, setValue,
                              lid % ts);

    flare::parallel_reduce(
        flare::TeamThreadRange(teamMember, ts),
        [&](const int /*j*/, value_type &teamUpdate) { teamUpdate += value; },
        parUpdate);

    if (teamMember.team_rank() == 0 && setValue) update += parUpdate;
  }

  template <class ScalarType>
  static inline std::enable_if_t<!std::is_integral<ScalarType>::value, void>
  compare_test(ScalarType A, ScalarType B, double epsilon_factor) {
    if (std::is_same<ScalarType, double>::value ||
        std::is_same<ScalarType, float>::value) {
      ASSERT_NEAR((double)A, (double)B,
                  epsilon_factor * std::abs(A) *
                      std::numeric_limits<ScalarType>::epsilon());
    } else {
      ASSERT_EQ(A, B);
    }
  }

  template <class ScalarType>
  static inline std::enable_if_t<std::is_integral<ScalarType>::value, void>
  compare_test(ScalarType A, ScalarType B, double) {
    ASSERT_EQ(A, B);
  }

  static void test_teambroadcast(const size_t league_size,
                                 const value_type off) {
    TestTeamBroadcast functor(league_size, off);

    using policy_type = flare::TeamPolicy<ScheduleType, ExecSpace>;
    using policy_type_f =
        flare::TeamPolicy<ScheduleType, ExecSpace, BroadcastTag>;

    int fake_team_size = 1;
    const int team_size =
        policy_type_f(league_size, fake_team_size)
            .team_size_max(
                functor,
                flare::
                    ParallelReduceTag());  // printf("team_size=%d\n",team_size);
    // team_broadcast with value
    value_type total = 0;

    flare::parallel_reduce(policy_type(league_size, team_size), functor,
                            total);

    value_type expected_result = 0;
    for (unsigned int i = 0; i < league_size; i++) {
      value_type val =
          (value_type((i % team_size) * 3) + off) * value_type(team_size);
      expected_result += val;
    }
    // For comparison purposes treat the reduction as a random walk in the
    // least significant digit, which gives a typical walk distance
    // sqrt(league_size) Add 4x for larger sigma
    compare_test(expected_result, total, 4.0 * std::sqrt(league_size));

    // team_broadcast with function object
    total = 0;

    flare::parallel_reduce(policy_type_f(league_size, team_size), functor,
                            total);

    expected_result = 0;
    for (unsigned int i = 0; i < league_size; i++) {
      value_type val = ((value_type)((i % team_size) * 3) + off) *
                       (value_type)(2 * team_size);
      expected_result += val;
    }
    // For comparison purposes treat the reduction as a random walk in the
    // least significant digit, which gives a typical walk distance
    // sqrt(league_size) Add 4x for larger sigma
    compare_test(expected_result, total, 4.0 * std::sqrt(league_size));
  }
};

template <class ExecSpace>
struct TestScratchAlignment {
  struct TestScalar {
    double x, y, z;
  };
  TestScratchAlignment() {
    test_tensor(true);
    test_tensor(false);
    test_minimal();
    test_raw();
  }
  using ScratchTensor =
      flare::Tensor<TestScalar *, typename ExecSpace::scratch_memory_space>;
  using ScratchTensorInt =
      flare::Tensor<int *, typename ExecSpace::scratch_memory_space>;
  void test_tensor(bool allocate_small) {
    int shmem_size = ScratchTensor::shmem_size(11);
    int team_size      = 1;
    if (allocate_small) shmem_size += ScratchTensorInt::shmem_size(1);
    flare::parallel_for(
        flare::TeamPolicy<ExecSpace>(1, team_size)
            .set_scratch_size(0, flare::PerTeam(shmem_size)),
        FLARE_LAMBDA(
            const typename flare::TeamPolicy<ExecSpace>::member_type &team) {
          if (allocate_small) ScratchTensorInt(team.team_scratch(0), 1);
          ScratchTensor a(team.team_scratch(0), 11);
          if (ptrdiff_t(a.data()) % sizeof(TestScalar) != 0)
            flare::abort("Error: invalid scratch tensor alignment\n");
        });
    flare::fence();
  }

  // test really small size of scratch space, produced error before
  void test_minimal() {
    using member_type = typename flare::TeamPolicy<ExecSpace>::member_type;
    int team_size      = 1;
    flare::TeamPolicy<ExecSpace> policy(1, team_size);
    size_t scratch_size = sizeof(int);
    flare::Tensor<int, ExecSpace> flag("Flag");

    flare::parallel_for(
        policy.set_scratch_size(0, flare::PerTeam(scratch_size)),
        FLARE_LAMBDA(const member_type &team) {
          int *scratch_ptr = (int *)team.team_shmem().get_shmem(scratch_size);
          if (scratch_ptr == nullptr) flag() = 1;
        });
    flare::fence();
    int minimal_scratch_allocation_failed = 0;
    flare::deep_copy(minimal_scratch_allocation_failed, flag);
    ASSERT_EQ(minimal_scratch_allocation_failed, 0);
  }

  // test alignment of successive allocations
  void test_raw() {
    using member_type = typename flare::TeamPolicy<ExecSpace>::member_type;
    int team_size      = 1;
    flare::TeamPolicy<ExecSpace> policy(1, team_size);
    flare::Tensor<int, ExecSpace> flag("Flag");

    flare::parallel_for(
        policy.set_scratch_size(0, flare::PerTeam(1024)),
        FLARE_LAMBDA(const member_type &team) {
          // first get some unaligned allocations, should give back
          // exactly the requested number of bytes
          auto scratch_ptr1 =
              reinterpret_cast<intptr_t>(team.team_shmem().get_shmem(24));
          auto scratch_ptr2 =
              reinterpret_cast<intptr_t>(team.team_shmem().get_shmem(32));
          auto scratch_ptr3 =
              reinterpret_cast<intptr_t>(team.team_shmem().get_shmem(12));

          if (((scratch_ptr2 - scratch_ptr1) != 24) ||
              ((scratch_ptr3 - scratch_ptr2) != 32))
            flag() = 1;

          // Now request aligned memory such that the allocation after
          // scratch_ptr2 would be unaligned if it doesn't pad correctly.
          // Depending on scratch_ptr3 being 4 or 8 byte aligned
          // we need to request a different amount of memory.
          if ((scratch_ptr3 + 12) % 8 == 4)
            scratch_ptr1 = reinterpret_cast<intptr_t>(
                team.team_shmem().get_shmem_aligned(24, 4));
          else {
            scratch_ptr1 = reinterpret_cast<intptr_t>(
                team.team_shmem().get_shmem_aligned(12, 4));
          }
          scratch_ptr2 = reinterpret_cast<intptr_t>(
              team.team_shmem().get_shmem_aligned(32, 8));
          scratch_ptr3 = reinterpret_cast<intptr_t>(
              team.team_shmem().get_shmem_aligned(8, 4));

          // The difference between scratch_ptr2 and scratch_ptr1 should be 4
          // bytes larger than what we requested in either case.
          if (((scratch_ptr2 - scratch_ptr1) != 28) &&
              ((scratch_ptr2 - scratch_ptr1) != 16))
            flag() = 1;
          // Check that there wasn't unneccessary padding happening. Since
          // scratch_ptr2 was allocated with a 32 byte request and scratch_ptr3
          // is then already aligned, its difference should match 32 bytes.
          if ((scratch_ptr3 - scratch_ptr2) != 32) flag() = 1;

          // check actually alignment of ptrs is as requested
          // cast to int here to avoid failure with icpx in mixed integer type
          // comparison
          if ((int(scratch_ptr1 % 4) != 0) || (int(scratch_ptr2 % 8) != 0) ||
              (int(scratch_ptr3 % 4) != 0))
            flag() = 1;
        });
    flare::fence();
    int raw_get_shmem_alignment_failed = 0;
    flare::deep_copy(raw_get_shmem_alignment_failed, flag);
    ASSERT_EQ(raw_get_shmem_alignment_failed, 0);
  }
};

}  // namespace

namespace {
template <class ExecSpace>
struct TestTeamPolicyHandleByValue {
  using scalar     = double;
  using exec_space = ExecSpace;
  using mem_space  = typename ExecSpace::memory_space;

  TestTeamPolicyHandleByValue() { test(); }

  void test() {
#if defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)
    const int M = 1, N = 1;
    flare::Tensor<scalar **, mem_space> a("a", M, N);
    flare::Tensor<scalar **, mem_space> b("b", M, N);
    flare::deep_copy(a, 0.0);
    flare::deep_copy(b, 1.0);
    flare::parallel_for(
        "test_tphandle_by_value",
        flare::TeamPolicy<exec_space>(M, flare::AUTO(), 1),
        FLARE_LAMBDA(
            const typename flare::TeamPolicy<exec_space>::member_type team) {
          const int i = team.league_rank();
          flare::parallel_for(flare::TeamThreadRange(team, 0, N),
                               [&](const int j) { a(i, j) += b(i, j); });
        });
#endif
  }
};

}  // namespace

namespace {
template <typename ExecutionSpace>
struct TestRepeatedTeamReduce {
  static constexpr int ncol = 1500;  // nothing special, just some work

  FLARE_FUNCTION void operator()(
      const typename flare::TeamPolicy<ExecutionSpace>::member_type &team)
      const {
    // non-divisible by power of two to make triggering problems easier
    constexpr int nlev = 129;
    constexpr auto pi  = flare::numbers::pi;
    double b           = 0.;
    for (int ri = 0; ri < 10; ++ri) {
      // The contributions here must be sufficiently complex, simply adding ones
      // wasn't enough to trigger the bug.
      const auto g1 = [&](const int k, double &acc) {
        acc += flare::cos(pi * double(k) / nlev);
      };
      const auto g2 = [&](const int k, double &acc) {
        acc += flare::sin(pi * double(k) / nlev);
      };
      double a1, a2;
      flare::parallel_reduce(flare::TeamThreadRange(team, nlev), g1, a1);
      flare::parallel_reduce(flare::TeamThreadRange(team, nlev), g2, a2);
      b += a1;
      b += a2;
    }
    const auto h = [&]() {
      const auto col = team.league_rank();
      v(col)         = b + col;
    };
    flare::single(flare::PerTeam(team), h);
  }

  FLARE_FUNCTION void operator()(const int i, int &bad) const {
    if (v(i) != v(0) + i) {
      ++bad;
      flare::printf("Failing at %d!\n", i);
    }
  }

  TestRepeatedTeamReduce() : v("v", ncol) { test(); }

  void test() {
    int team_size_recommended =
        flare::TeamPolicy<ExecutionSpace>(1, 1).team_size_recommended(
            *this, flare::ParallelForTag());
    // Choose a non-recommened (non-power of two for GPUs) team size
    int team_size = team_size_recommended > 1 ? team_size_recommended - 1 : 1;

    // The failure was non-deterministic so run the test a bunch of times
    for (int it = 0; it < 100; ++it) {
      flare::parallel_for(
          flare::TeamPolicy<ExecutionSpace>(ncol, team_size, 1), *this);

      int bad = 0;
      flare::parallel_reduce(flare::RangePolicy<ExecutionSpace>(0, ncol),
                              *this, bad);
      ASSERT_EQ(bad, 0) << " Failing in iteration " << it;
    }
  }

  flare::Tensor<double *, ExecutionSpace> v;
};

}  // namespace

}  // namespace Test

/*--------------------------------------------------------------------------*/
