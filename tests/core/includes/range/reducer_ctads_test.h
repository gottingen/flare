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

namespace {

struct TestReducerCTADS {
  using execspace   = TEST_EXECSPACE;
  using scalar_type = double;
  using index_type  = int;
  using memspace    = execspace::memory_space;

  struct CustomComparator {
    bool operator()(scalar_type, scalar_type) const;
  };
  static CustomComparator comparator;

  struct TestSum {
    static flare::View<scalar_type, memspace> view;
    static flare::View<scalar_type, memspace,
                        flare::MemoryTraits<flare::Unmanaged>>
        unmanaged;
    static flare::Sum<scalar_type, memspace> rt;

    static_assert(std::is_same_v<decltype(rt), decltype(flare::Sum(view))>);
    static_assert(std::is_same_v<decltype(rt), decltype(flare::Sum(rt))>);
    static_assert(
        std::is_same_v<decltype(rt), decltype(flare::Sum(std::move(rt)))>);
    static_assert(
        std::is_same_v<decltype(rt), decltype(flare::Sum(unmanaged))>);
  };

  struct TestProd {
    static flare::View<scalar_type, memspace> view;
    static flare::View<scalar_type, memspace,
                        flare::MemoryTraits<flare::Unmanaged>>
        unmanaged;
    static flare::Prod<scalar_type, memspace> rt;

    static_assert(std::is_same_v<decltype(rt), decltype(flare::Prod(view))>);
    static_assert(std::is_same_v<decltype(rt), decltype(flare::Prod(rt))>);
    static_assert(
        std::is_same_v<decltype(rt), decltype(flare::Prod(std::move(rt)))>);
    static_assert(
        std::is_same_v<decltype(rt), decltype(flare::Prod(unmanaged))>);
  };

  struct TestMin {
    static flare::View<scalar_type, memspace> view;
    static flare::View<scalar_type, memspace,
                        flare::MemoryTraits<flare::Unmanaged>>
        unmanaged;
    static flare::Min<scalar_type, memspace> rt;

    static_assert(std::is_same_v<decltype(rt), decltype(flare::Min(view))>);
    static_assert(std::is_same_v<decltype(rt), decltype(flare::Min(rt))>);
    static_assert(
        std::is_same_v<decltype(rt), decltype(flare::Min(std::move(rt)))>);
    static_assert(
        std::is_same_v<decltype(rt), decltype(flare::Min(unmanaged))>);
  };

  struct TestMax {
    static flare::View<scalar_type, memspace> view;
    static flare::View<scalar_type, memspace,
                        flare::MemoryTraits<flare::Unmanaged>>
        unmanaged;
    static flare::Max<scalar_type, memspace> rt;

    static_assert(std::is_same_v<decltype(rt), decltype(flare::Max(view))>);
    static_assert(std::is_same_v<decltype(rt), decltype(flare::Max(rt))>);
    static_assert(
        std::is_same_v<decltype(rt), decltype(flare::Max(std::move(rt)))>);
    static_assert(
        std::is_same_v<decltype(rt), decltype(flare::Max(unmanaged))>);
  };

  struct TestLAnd {
    static flare::View<scalar_type, memspace> view;
    static flare::View<scalar_type, memspace,
                        flare::MemoryTraits<flare::Unmanaged>>
        unmanaged;
    static flare::LAnd<scalar_type, memspace> rt;

    static_assert(std::is_same_v<decltype(rt), decltype(flare::LAnd(view))>);
    static_assert(std::is_same_v<decltype(rt), decltype(flare::LAnd(rt))>);
    static_assert(
        std::is_same_v<decltype(rt), decltype(flare::LAnd(std::move(rt)))>);
    static_assert(
        std::is_same_v<decltype(rt), decltype(flare::LAnd(unmanaged))>);
  };

  struct TestLOr {
    static flare::View<scalar_type, memspace> view;
    static flare::View<scalar_type, memspace,
                        flare::MemoryTraits<flare::Unmanaged>>
        unmanaged;
    static flare::LOr<scalar_type, memspace> rt;

    static_assert(std::is_same_v<decltype(rt), decltype(flare::LOr(view))>);
    static_assert(std::is_same_v<decltype(rt), decltype(flare::LOr(rt))>);
    static_assert(
        std::is_same_v<decltype(rt), decltype(flare::LOr(std::move(rt)))>);
    static_assert(
        std::is_same_v<decltype(rt), decltype(flare::LOr(unmanaged))>);
  };

  struct TestBAnd {
    static flare::View<scalar_type, memspace> view;
    static flare::View<scalar_type, memspace,
                        flare::MemoryTraits<flare::Unmanaged>>
        unmanaged;
    static flare::BAnd<scalar_type, memspace> rt;

    static_assert(std::is_same_v<decltype(rt), decltype(flare::BAnd(view))>);
    static_assert(std::is_same_v<decltype(rt), decltype(flare::BAnd(rt))>);
    static_assert(
        std::is_same_v<decltype(rt), decltype(flare::BAnd(std::move(rt)))>);
    static_assert(
        std::is_same_v<decltype(rt), decltype(flare::BAnd(unmanaged))>);
  };

  struct TestBOr {
    static flare::View<scalar_type, memspace> view;
    static flare::View<scalar_type, memspace,
                        flare::MemoryTraits<flare::Unmanaged>>
        unmanaged;
    static flare::BOr<scalar_type, memspace> rt;

    static_assert(std::is_same_v<decltype(rt), decltype(flare::BOr(view))>);
    static_assert(std::is_same_v<decltype(rt), decltype(flare::BOr(rt))>);
    static_assert(
        std::is_same_v<decltype(rt), decltype(flare::BOr(std::move(rt)))>);
    static_assert(
        std::is_same_v<decltype(rt), decltype(flare::BOr(unmanaged))>);
  };

  struct TestMinLoc {
    static flare::View<flare::ValLocScalar<scalar_type, index_type>, memspace>
        view;
    static flare::View<flare::ValLocScalar<scalar_type, index_type>, memspace,
                        flare::MemoryTraits<flare::Unmanaged>>
        unmanaged;
    static flare::MinLoc<scalar_type, index_type, memspace> rt;

    static_assert(std::is_same_v<decltype(rt), decltype(flare::MinLoc(view))>);
    static_assert(std::is_same_v<decltype(rt), decltype(flare::MinLoc(rt))>);
    static_assert(
        std::is_same_v<decltype(rt), decltype(flare::MinLoc(std::move(rt)))>);
    static_assert(
        std::is_same_v<decltype(rt), decltype(flare::MinLoc(unmanaged))>);
  };

  struct TestMaxLoc {
    static flare::View<flare::ValLocScalar<scalar_type, index_type>, memspace>
        view;
    static flare::View<flare::ValLocScalar<scalar_type, index_type>, memspace,
                        flare::MemoryTraits<flare::Unmanaged>>
        unmanaged;
    static flare::MaxLoc<scalar_type, index_type, memspace> rt;

    static_assert(std::is_same_v<decltype(rt), decltype(flare::MaxLoc(view))>);
    static_assert(std::is_same_v<decltype(rt), decltype(flare::MaxLoc(rt))>);
    static_assert(
        std::is_same_v<decltype(rt), decltype(flare::MaxLoc(std::move(rt)))>);
    static_assert(
        std::is_same_v<decltype(rt), decltype(flare::MaxLoc(unmanaged))>);
  };

  struct TestMinMax {
    static flare::View<flare::MinMaxScalar<scalar_type>, memspace> view;
    static flare::View<flare::MinMaxScalar<scalar_type>, memspace,
                        flare::MemoryTraits<flare::Unmanaged>>
        unmanaged;
    static flare::MinMax<scalar_type, memspace> rt;

    static_assert(std::is_same_v<decltype(rt), decltype(flare::MinMax(view))>);
    static_assert(std::is_same_v<decltype(rt), decltype(flare::MinMax(rt))>);
    static_assert(
        std::is_same_v<decltype(rt), decltype(flare::MinMax(std::move(rt)))>);
    static_assert(
        std::is_same_v<decltype(rt), decltype(flare::MinMax(unmanaged))>);
  };

  struct TestMinMaxLoc {
    static flare::View<flare::MinMaxLocScalar<scalar_type, index_type>,
                        memspace>
        view;
    static flare::View<flare::MinMaxLocScalar<scalar_type, index_type>,
                        memspace, flare::MemoryTraits<flare::Unmanaged>>
        unmanaged;
    static flare::MinMaxLoc<scalar_type, index_type, memspace> rt;

    static_assert(
        std::is_same_v<decltype(rt), decltype(flare::MinMaxLoc(view))>);
    static_assert(
        std::is_same_v<decltype(rt), decltype(flare::MinMaxLoc(rt))>);
    static_assert(std::is_same_v<decltype(rt),
                                 decltype(flare::MinMaxLoc(std::move(rt)))>);
    static_assert(
        std::is_same_v<decltype(rt), decltype(flare::MinMaxLoc(unmanaged))>);
  };

  struct TestMaxFirstLoc {
    static flare::View<flare::ValLocScalar<scalar_type, index_type>, memspace>
        view;
    static flare::View<flare::ValLocScalar<scalar_type, index_type>, memspace,
                        flare::MemoryTraits<flare::Unmanaged>>
        unmanaged;
    static flare::MaxFirstLoc<scalar_type, index_type, memspace> rt;

    static_assert(
        std::is_same_v<decltype(rt), decltype(flare::MaxFirstLoc(view))>);
    static_assert(
        std::is_same_v<decltype(rt), decltype(flare::MaxFirstLoc(rt))>);
    static_assert(std::is_same_v<decltype(rt),
                                 decltype(flare::MaxFirstLoc(std::move(rt)))>);
    static_assert(
        std::is_same_v<decltype(rt), decltype(flare::MaxFirstLoc(unmanaged))>);
  };

  struct TestMaxFirstLocCustomComparator {
    static flare::View<flare::ValLocScalar<scalar_type, index_type>, memspace>
        view;
    static flare::View<flare::ValLocScalar<scalar_type, index_type>, memspace,
                        flare::MemoryTraits<flare::Unmanaged>>
        unmanaged;
    static flare::MaxFirstLocCustomComparator<scalar_type, index_type,
                                               CustomComparator, memspace>
        rt;

    static_assert(std::is_same_v<decltype(rt),
                                 decltype(flare::MaxFirstLocCustomComparator(
                                     view, comparator))>);
    static_assert(
        std::is_same_v<decltype(rt),
                       decltype(flare::MaxFirstLocCustomComparator(rt))>);
    static_assert(std::is_same_v<decltype(rt),
                                 decltype(flare::MaxFirstLocCustomComparator(
                                     std::move(rt)))>);
    static_assert(std::is_same_v<decltype(rt),
                                 decltype(flare::MaxFirstLocCustomComparator(
                                     unmanaged, comparator))>);
  };

  struct TestMinFirstLoc {
    static flare::View<flare::ValLocScalar<scalar_type, index_type>, memspace>
        view;
    static flare::View<flare::ValLocScalar<scalar_type, index_type>, memspace,
                        flare::MemoryTraits<flare::Unmanaged>>
        unmanaged;
    static flare::MinFirstLoc<scalar_type, index_type, memspace> rt;

    static_assert(
        std::is_same_v<decltype(rt), decltype(flare::MinFirstLoc(view))>);
    static_assert(
        std::is_same_v<decltype(rt), decltype(flare::MinFirstLoc(rt))>);
    static_assert(std::is_same_v<decltype(rt),
                                 decltype(flare::MinFirstLoc(std::move(rt)))>);
    static_assert(
        std::is_same_v<decltype(rt), decltype(flare::MinFirstLoc(unmanaged))>);
  };

  struct TestMinFirstLocCustomComparator {
    static flare::View<flare::ValLocScalar<scalar_type, index_type>, memspace>
        view;
    static flare::View<flare::ValLocScalar<scalar_type, index_type>, memspace,
                        flare::MemoryTraits<flare::Unmanaged>>
        unmanaged;
    static flare::MinFirstLocCustomComparator<scalar_type, index_type,
                                               CustomComparator, memspace>
        rt;

    static_assert(std::is_same_v<decltype(rt),
                                 decltype(flare::MinFirstLocCustomComparator(
                                     view, comparator))>);
    static_assert(
        std::is_same_v<decltype(rt),
                       decltype(flare::MinFirstLocCustomComparator(rt))>);
    static_assert(std::is_same_v<decltype(rt),
                                 decltype(flare::MinFirstLocCustomComparator(
                                     std::move(rt)))>);
    static_assert(std::is_same_v<decltype(rt),
                                 decltype(flare::MinFirstLocCustomComparator(
                                     unmanaged, comparator))>);
  };

  struct TestMinMaxFirstLastLoc {
    static flare::View<flare::MinMaxLocScalar<scalar_type, index_type>,
                        memspace>
        view;
    static flare::View<flare::MinMaxLocScalar<scalar_type, index_type>,
                        memspace, flare::MemoryTraits<flare::Unmanaged>>
        unmanaged;
    static flare::MinMaxFirstLastLoc<scalar_type, index_type, memspace> rt;

    static_assert(std::is_same_v<decltype(rt),
                                 decltype(flare::MinMaxFirstLastLoc(view))>);
    static_assert(
        std::is_same_v<decltype(rt), decltype(flare::MinMaxFirstLastLoc(rt))>);
    static_assert(
        std::is_same_v<decltype(rt),
                       decltype(flare::MinMaxFirstLastLoc(std::move(rt)))>);
    static_assert(
        std::is_same_v<decltype(rt),
                       decltype(flare::MinMaxFirstLastLoc(unmanaged))>);
  };

  struct TestMinMaxFirstLastLocCustomComparator {
    static flare::View<flare::MinMaxLocScalar<scalar_type, index_type>,
                        memspace>
        view;
    static flare::View<flare::MinMaxLocScalar<scalar_type, index_type>,
                        memspace, flare::MemoryTraits<flare::Unmanaged>>
        unmanaged;
    static flare::MinMaxFirstLastLocCustomComparator<
        scalar_type, index_type, CustomComparator, memspace>
        rt;

    static_assert(
        std::is_same_v<decltype(rt),
                       decltype(flare::MinMaxFirstLastLocCustomComparator(
                           view, comparator))>);
    static_assert(std::is_same_v<
                  decltype(rt),
                  decltype(flare::MinMaxFirstLastLocCustomComparator(rt))>);
    static_assert(
        std::is_same_v<decltype(rt),
                       decltype(flare::MinMaxFirstLastLocCustomComparator(
                           std::move(rt)))>);
    static_assert(
        std::is_same_v<decltype(rt),
                       decltype(flare::MinMaxFirstLastLocCustomComparator(
                           unmanaged, comparator))>);
  };

  struct TestFirstLoc {
    static flare::View<flare::FirstLocScalar<index_type>, memspace> view;
    static flare::View<flare::FirstLocScalar<index_type>, memspace,
                        flare::MemoryTraits<flare::Unmanaged>>
        unmanaged;
    static flare::FirstLoc<index_type, memspace> rt;

    static_assert(
        std::is_same_v<decltype(rt), decltype(flare::FirstLoc(view))>);
    static_assert(std::is_same_v<decltype(rt), decltype(flare::FirstLoc(rt))>);
    static_assert(std::is_same_v<decltype(rt),
                                 decltype(flare::FirstLoc(std::move(rt)))>);
    static_assert(
        std::is_same_v<decltype(rt), decltype(flare::FirstLoc(unmanaged))>);
  };

  struct TestLastLoc {
    static flare::View<flare::LastLocScalar<index_type>, memspace> view;
    static flare::View<flare::LastLocScalar<index_type>, memspace,
                        flare::MemoryTraits<flare::Unmanaged>>
        unmanaged;
    static flare::LastLoc<index_type, memspace> rt;

    static_assert(
        std::is_same_v<decltype(rt), decltype(flare::LastLoc(view))>);
    static_assert(std::is_same_v<decltype(rt), decltype(flare::LastLoc(rt))>);
    static_assert(
        std::is_same_v<decltype(rt), decltype(flare::LastLoc(std::move(rt)))>);
    static_assert(
        std::is_same_v<decltype(rt), decltype(flare::LastLoc(unmanaged))>);
  };

  struct TestStdIsPartitioned {
    static flare::View<flare::StdIsPartScalar<index_type>, memspace> view;
    static flare::View<flare::StdIsPartScalar<index_type>, memspace,
                        flare::MemoryTraits<flare::Unmanaged>>
        unmanaged;
    static flare::StdIsPartitioned<index_type, memspace> rt;

    static_assert(
        std::is_same_v<decltype(rt), decltype(flare::StdIsPartitioned(view))>);
    static_assert(
        std::is_same_v<decltype(rt), decltype(flare::StdIsPartitioned(rt))>);
    static_assert(
        std::is_same_v<decltype(rt),
                       decltype(flare::StdIsPartitioned(std::move(rt)))>);
    static_assert(std::is_same_v<
                  decltype(rt), decltype(flare::StdIsPartitioned(unmanaged))>);
  };

  struct TestStdPartitionPoint {
    static flare::View<flare::StdPartPointScalar<index_type>, memspace> view;
    static flare::View<flare::StdPartPointScalar<index_type>, memspace,
                        flare::MemoryTraits<flare::Unmanaged>>
        unmanaged;
    static flare::StdPartitionPoint<index_type, memspace> rt;

    static_assert(std::is_same_v<decltype(rt),
                                 decltype(flare::StdPartitionPoint(view))>);
    static_assert(
        std::is_same_v<decltype(rt), decltype(flare::StdPartitionPoint(rt))>);
    static_assert(
        std::is_same_v<decltype(rt),
                       decltype(flare::StdPartitionPoint(std::move(rt)))>);
    static_assert(
        std::is_same_v<decltype(rt),
                       decltype(flare::StdPartitionPoint(unmanaged))>);
  };
};

}  // namespace
