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
#include <doctest.h>
#include <flare/core.h>

namespace Test {

namespace {

template <class ExecSpace, class ScheduleType>
struct TestRange {
  using value_type = int;  ///< alias required for the parallel_reduce

  using tensor_type = flare::Tensor<value_type *, ExecSpace>;

  tensor_type m_flags;
  tensor_type result_tensor;

  struct VerifyInitTag {};
  struct ResetTag {};
  struct VerifyResetTag {};
  struct OffsetTag {};
  struct VerifyOffsetTag {};

  int N;
  static const int offset = 13;
  TestRange(const size_t N_)
      : m_flags(flare::tensor_alloc(flare::WithoutInitializing, "flags"), N_),
        result_tensor(flare::tensor_alloc(flare::WithoutInitializing, "results"),
                    N_),
        N(N_) {
  }

  void test_for() {
    typename tensor_type::HostMirror host_flags =
        flare::create_mirror_tensor(m_flags);

    flare::parallel_for(flare::RangePolicy<ExecSpace, ScheduleType>(0, N),
                         *this);

    {
      using ThisType = TestRange<ExecSpace, ScheduleType>;
      std::string label("parallel_for");
      flare::detail::ParallelConstructName<ThisType, void> pcn(label);
      REQUIRE_EQ(pcn.get(), label);
      std::string empty_label("");
      flare::detail::ParallelConstructName<ThisType, void> empty_pcn(
          empty_label);
      REQUIRE_EQ(empty_pcn.get(), typeid(ThisType).name());
    }

    flare::parallel_for(
        flare::RangePolicy<ExecSpace, ScheduleType, VerifyInitTag>(0, N),
        *this);

    {
      using ThisType = TestRange<ExecSpace, ScheduleType>;
      std::string label("parallel_for");
      flare::detail::ParallelConstructName<ThisType, VerifyInitTag> pcn(label);
      REQUIRE_EQ(pcn.get(), label);
      std::string empty_label("");
      flare::detail::ParallelConstructName<ThisType, VerifyInitTag> empty_pcn(
          empty_label);
      REQUIRE_EQ(empty_pcn.get(), std::string(typeid(ThisType).name()) + "/" +
                                     typeid(VerifyInitTag).name());
    }

    flare::deep_copy(host_flags, m_flags);

    int error_count = 0;
    for (int i = 0; i < N; ++i) {
      if (int(i) != host_flags(i)) ++error_count;
    }
    REQUIRE_EQ(error_count, int(0));

    flare::parallel_for(
        flare::RangePolicy<ExecSpace, ScheduleType, ResetTag>(0, N), *this);
    flare::parallel_for(
        std::string("TestKernelFor"),
        flare::RangePolicy<ExecSpace, ScheduleType, VerifyResetTag>(0, N),
        *this);

    flare::deep_copy(host_flags, m_flags);

    error_count = 0;
    for (int i = 0; i < N; ++i) {
      if (int(2 * i) != host_flags(i)) ++error_count;
    }
    REQUIRE_EQ(error_count, int(0));

    flare::parallel_for(
        flare::RangePolicy<ExecSpace, ScheduleType, OffsetTag>(offset,
                                                                N + offset),
        *this);
    flare::parallel_for(
        std::string("TestKernelFor"),
        flare::RangePolicy<ExecSpace, ScheduleType, VerifyOffsetTag>(0, N),
        *this);

    flare::deep_copy(host_flags, m_flags);

    error_count = 0;
    for (int i = 0; i < N; ++i) {
      if (i + offset != host_flags(i)) ++error_count;
    }
    REQUIRE_EQ(error_count, int(0));
  }

  FLARE_INLINE_FUNCTION
  void operator()(const int i) const { m_flags(i) = i; }

  FLARE_INLINE_FUNCTION
  void operator()(const VerifyInitTag &, const int i) const {
    if (i != m_flags(i)) {
      flare::printf("TestRange::test_for_error at %d != %d\n", i, m_flags(i));
    }
  }

  FLARE_INLINE_FUNCTION
  void operator()(const ResetTag &, const int i) const {
    m_flags(i) = 2 * m_flags(i);
  }

  FLARE_INLINE_FUNCTION
  void operator()(const VerifyResetTag &, const int i) const {
    if (2 * i != m_flags(i)) {
      flare::printf("TestRange::test_for_error at %d != %d\n", i, m_flags(i));
    }
  }

  FLARE_INLINE_FUNCTION
  void operator()(const OffsetTag &, const int i) const {
    m_flags(i - offset) = i;
  }

  FLARE_INLINE_FUNCTION
  void operator()(const VerifyOffsetTag &, const int i) const {
    if (i + offset != m_flags(i)) {
      flare::printf("TestRange::test_for_error at %d != %d\n", i + offset,
                     m_flags(i));
    }
  }

  //----------------------------------------

  void test_reduce() {
    value_type total = 0;

    flare::parallel_for(flare::RangePolicy<ExecSpace, ScheduleType>(0, N),
                         *this);

    flare::parallel_reduce("TestKernelReduce",
                            flare::RangePolicy<ExecSpace, ScheduleType>(0, N),
                            *this, total);
    // sum( 0 .. N-1 )
    REQUIRE_EQ(size_t((N - 1) * (N) / 2), size_t(total));

    flare::parallel_reduce(
        "TestKernelReduce_long",
        flare::RangePolicy<ExecSpace, ScheduleType, long>(0, N), *this, total);
    // sum( 0 .. N-1 )
    REQUIRE_EQ(size_t((N - 1) * (N) / 2), size_t(total));

    flare::parallel_reduce(
        flare::RangePolicy<ExecSpace, ScheduleType, OffsetTag>(offset,
                                                                N + offset),
        *this, total);
    // sum( 1 .. N )
    REQUIRE_EQ(size_t((N) * (N + 1) / 2), size_t(total));
  }

  FLARE_INLINE_FUNCTION
  void operator()(const int i, value_type &update) const {
    update += m_flags(i);
  }

  FLARE_INLINE_FUNCTION
  void operator()(const OffsetTag &, const int i, value_type &update) const {
    update += 1 + m_flags(i - offset);
  }

  void test_dynamic_policy() {
#if defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)
    auto const N_no_implicit_capture = N;
    using policy_t =
        flare::RangePolicy<ExecSpace, flare::Schedule<flare::Dynamic> >;
    int const concurrency = ExecSpace().concurrency();

    {
      flare::Tensor<size_t *, ExecSpace, flare::MemoryTraits<flare::Atomic> >
          count("Count", concurrency);
      flare::Tensor<int *, ExecSpace> a("A", N);

      flare::parallel_for(
          policy_t(0, N), FLARE_LAMBDA(const int &i) {
            for (int k = 0; k < (i < N_no_implicit_capture / 2 ? 1 : 10000);
                 k++) {
              a(i)++;
            }
            count(ExecSpace::impl_hardware_thread_id())++;
          });

      int error = 0;
      flare::parallel_reduce(
          flare::RangePolicy<ExecSpace>(0, N),
          FLARE_LAMBDA(const int &i, value_type &lsum) {
            lsum += (a(i) != (i < N_no_implicit_capture / 2 ? 1 : 10000));
          },
          error);
      REQUIRE_EQ(error, 0);

      if ((concurrency > 1) && (N > 4 * concurrency)) {
        size_t min = N;
        size_t max = 0;
        for (int t = 0; t < concurrency; t++) {
          if (count(t) < min) min = count(t);
          if (count(t) > max) max = count(t);
        }
        REQUIRE_LT(min, max);

        // if ( concurrency > 2 ) {
        //  REQUIRE_LT( 2 * min, max );
        //}
      }
    }

    {
      flare::Tensor<size_t *, ExecSpace, flare::MemoryTraits<flare::Atomic> >
          count("Count", concurrency);
      flare::Tensor<int *, ExecSpace> a("A", N);

      value_type sum = 0;
      flare::parallel_reduce(
          policy_t(0, N),
          FLARE_LAMBDA(const int &i, value_type &lsum) {
            for (int k = 0; k < (i < N_no_implicit_capture / 2 ? 1 : 10000);
                 k++) {
              a(i)++;
            }
            count(ExecSpace::impl_hardware_thread_id())++;
            lsum++;
          },
          sum);
      REQUIRE_EQ(sum, N);

      int error = 0;
      flare::parallel_reduce(
          flare::RangePolicy<ExecSpace>(0, N),
          FLARE_LAMBDA(const int &i, value_type &lsum) {
            lsum += (a(i) != (i < N_no_implicit_capture / 2 ? 1 : 10000));
          },
          error);
      REQUIRE_EQ(error, 0);

      if ((concurrency > 1) && (N > 4 * concurrency)) {
        size_t min = N;
        size_t max = 0;
        for (int t = 0; t < concurrency; t++) {
          if (count(t) < min) min = count(t);
          if (count(t) > max) max = count(t);
        }
        REQUIRE_LT(min, max);

        // if ( concurrency > 2 ) {
        //  REQUIRE_LT( 2 * min, max );
        //}
      }
    }
#endif
  }
};

}  // namespace

TEST_CASE("TEST_CATEGORY, range_for") {
  {
    TestRange<TEST_EXECSPACE, flare::Schedule<flare::Static> > f(0);
    f.test_for();
  }
  {
    TestRange<TEST_EXECSPACE, flare::Schedule<flare::Dynamic> > f(0);
    f.test_for();
  }

  {
    TestRange<TEST_EXECSPACE, flare::Schedule<flare::Static> > f(2);
    f.test_for();
  }
  {
    TestRange<TEST_EXECSPACE, flare::Schedule<flare::Dynamic> > f(3);
    f.test_for();
  }

  {
    TestRange<TEST_EXECSPACE, flare::Schedule<flare::Static> > f(1000);
    f.test_for();
  }
  {
    TestRange<TEST_EXECSPACE, flare::Schedule<flare::Dynamic> > f(1001);
    f.test_for();
  }
}

TEST_CASE("TEST_CATEGORY, range_reduce") {
  {
    TestRange<TEST_EXECSPACE, flare::Schedule<flare::Static> > f(0);
    f.test_reduce();
  }
  {
    TestRange<TEST_EXECSPACE, flare::Schedule<flare::Dynamic> > f(0);
    f.test_reduce();
  }

  {
    TestRange<TEST_EXECSPACE, flare::Schedule<flare::Static> > f(2);
    f.test_reduce();
  }
  {
    TestRange<TEST_EXECSPACE, flare::Schedule<flare::Dynamic> > f(3);
    f.test_reduce();
  }

  {
    TestRange<TEST_EXECSPACE, flare::Schedule<flare::Static> > f(1000);
    f.test_reduce();
  }
  {
    TestRange<TEST_EXECSPACE, flare::Schedule<flare::Dynamic> > f(1001);
    f.test_reduce();
  }
}

TEST_CASE("TEST_CATEGORY, range_dynamic_policy") {
#if !defined(FLARE_ON_CUDA_DEVICE)
  {
    TestRange<TEST_EXECSPACE, flare::Schedule<flare::Dynamic> > f(0);
    f.test_dynamic_policy();
  }
  {
    TestRange<TEST_EXECSPACE, flare::Schedule<flare::Dynamic> > f(3);
    f.test_dynamic_policy();
  }
  {
    TestRange<TEST_EXECSPACE, flare::Schedule<flare::Dynamic> > f(1001);
    f.test_dynamic_policy();
  }
#endif
}

}  // namespace Test
