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
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <flare/runtime/taskflow.h>
#include <flare/runtime/algorithm/transform.h>

// ----------------------------------------------------------------------------
// Parallel Transform 1
// ----------------------------------------------------------------------------

template<typename T, typename P>
void parallel_transform(size_t W) {

  std::srand(static_cast<unsigned int>(time(NULL)));

  flare::rt::Taskflow taskflow;
  flare::rt::Executor executor(W);

  for(size_t N=0; N<1000; N=(N+1)*2) {
    for(size_t c : {0, 1, 3, 7, 99}) {

      taskflow.clear();

      typename T::const_iterator src_beg;
      typename T::const_iterator src_end;
      std::list<std::string>::iterator tgt_beg;

      T src;
      std::list<std::string> tgt;

      taskflow.clear();

      auto from = taskflow.emplace([&](){
        src.resize(N);
        for(auto& d : src) {
          d = ::rand() % 10;
          tgt.emplace_back("hi");
        }
        src_beg = src.begin();
        src_end = src.end();
        tgt_beg = tgt.begin();
      });

      auto to = taskflow.transform(
        std::ref(src_beg), std::ref(src_end), std::ref(tgt_beg),
        [] (const auto& in) {
          return std::to_string(in+10);
        },
        P(c)
      );

      from.precede(to);

      executor.run(taskflow).wait();

      auto s_itr = src.begin();
      auto d_itr = tgt.begin();
      while(s_itr != src.end()) {
        REQUIRE(*d_itr++ == std::to_string(*s_itr++ + 10));
      }
    }
  }
}

// guided
TEST_CASE("ParallelTransform.Guided.1thread") {
  parallel_transform<std::vector<int>, flare::rt::GuidedPartitioner>(1);
  parallel_transform<std::list<int>, flare::rt::GuidedPartitioner>(1);
}

TEST_CASE("ParallelTransform.Guided.2threads") {
  parallel_transform<std::vector<int>, flare::rt::GuidedPartitioner>(2);
  parallel_transform<std::list<int>, flare::rt::GuidedPartitioner>(2);
}

TEST_CASE("ParallelTransform.Guided.3threads") {
  parallel_transform<std::vector<int>, flare::rt::GuidedPartitioner>(3);
  parallel_transform<std::list<int>, flare::rt::GuidedPartitioner>(3);
}

TEST_CASE("ParallelTransform.Guided.4threads") {
  parallel_transform<std::vector<int>, flare::rt::GuidedPartitioner>(4);
  parallel_transform<std::list<int>, flare::rt::GuidedPartitioner>(4);
}

// random
TEST_CASE("ParallelTransform.Random.1thread") {
  parallel_transform<std::vector<int>, flare::rt::RandomPartitioner>(1);
  parallel_transform<std::list<int>, flare::rt::RandomPartitioner>(1);
}

TEST_CASE("ParallelTransform.Random.2threads") {
  parallel_transform<std::vector<int>, flare::rt::RandomPartitioner>(2);
  parallel_transform<std::list<int>, flare::rt::RandomPartitioner>(2);
}

TEST_CASE("ParallelTransform.Random.3threads") {
  parallel_transform<std::vector<int>, flare::rt::RandomPartitioner>(3);
  parallel_transform<std::list<int>, flare::rt::RandomPartitioner>(3);
}

TEST_CASE("ParallelTransform.Random.4threads") {
  parallel_transform<std::vector<int>, flare::rt::RandomPartitioner>(4);
  parallel_transform<std::list<int>, flare::rt::RandomPartitioner>(4);
}

// static
TEST_CASE("ParallelTransform.Static.1thread") {
  parallel_transform<std::vector<int>, flare::rt::StaticPartitioner>(1);
  parallel_transform<std::list<int>, flare::rt::StaticPartitioner>(1);
}

TEST_CASE("ParallelTransform.Static.2threads") {
  parallel_transform<std::vector<int>, flare::rt::StaticPartitioner>(2);
  parallel_transform<std::list<int>, flare::rt::StaticPartitioner>(2);
}

TEST_CASE("ParallelTransform.Static.3threads") {
  parallel_transform<std::vector<int>, flare::rt::StaticPartitioner>(3);
  parallel_transform<std::list<int>, flare::rt::StaticPartitioner>(3);
}

TEST_CASE("ParallelTransform.Static.4threads") {
  parallel_transform<std::vector<int>, flare::rt::StaticPartitioner>(4);
  parallel_transform<std::list<int>, flare::rt::StaticPartitioner>(4);
}

// ----------------------------------------------------------------------------
// Parallel Transform 2
// ----------------------------------------------------------------------------

template<typename T, typename P>
void parallel_transform2(size_t W) {

  std::srand(static_cast<unsigned int>(time(NULL)));

  flare::rt::Taskflow taskflow;
  flare::rt::Executor executor(W);

  for(size_t N=0; N<1000; N=(N+1)*2) {
    for(size_t c : {0, 1, 3, 7, 99}) {

      taskflow.clear();

      typename T::const_iterator src_beg;
      typename T::const_iterator src_end;
      std::list<std::string>::iterator tgt_beg;

      T src;
      std::list<std::string> tgt;

      taskflow.clear();

      auto from = taskflow.emplace([&](){
        src.resize(N);
        for(auto& d : src) {
          d = ::rand() % 10;
          tgt.emplace_back("hi");
        }
        src_beg = src.begin();
        src_end = src.end();
        tgt_beg = tgt.begin();
      });

      auto to = taskflow.transform(
        std::ref(src_beg), std::ref(src_end), std::ref(src_beg), std::ref(tgt_beg),
        [] (const auto& in1, const auto& in2) {
          return std::to_string(in1 + in2 + 10);
        },
        P(c)
      );

      from.precede(to);

      executor.run(taskflow).wait();

      auto s_itr = src.begin();
      auto d_itr = tgt.begin();
      while(s_itr != src.end()) {
        REQUIRE(*d_itr++ == std::to_string(2 * *s_itr++ + 10));
      }
    }
  }
}

// guided
TEST_CASE("ParallelTransform2.Guided.1thread") {
  parallel_transform2<std::vector<int>, flare::rt::GuidedPartitioner>(1);
  parallel_transform2<std::list<int>, flare::rt::GuidedPartitioner>(1);
}

TEST_CASE("ParallelTransform2.Guided.2threads") {
  parallel_transform2<std::vector<int>, flare::rt::GuidedPartitioner>(2);
  parallel_transform2<std::list<int>, flare::rt::GuidedPartitioner>(2);
}

TEST_CASE("ParallelTransform2.Guided.3threads") {
  parallel_transform2<std::vector<int>, flare::rt::GuidedPartitioner>(3);
  parallel_transform2<std::list<int>, flare::rt::GuidedPartitioner>(3);
}

TEST_CASE("ParallelTransform2.Guided.4threads") {
  parallel_transform2<std::vector<int>, flare::rt::GuidedPartitioner>(4);
  parallel_transform2<std::list<int>, flare::rt::GuidedPartitioner>(4);
}

// dynamic
TEST_CASE("ParallelTransform2.Dynamic.1thread") {
  parallel_transform2<std::vector<int>, flare::rt::DynamicPartitioner>(1);
  parallel_transform2<std::list<int>, flare::rt::DynamicPartitioner>(1);
}

TEST_CASE("ParallelTransform2.Dynamic.2threads") {
  parallel_transform2<std::vector<int>, flare::rt::DynamicPartitioner>(2);
  parallel_transform2<std::list<int>, flare::rt::DynamicPartitioner>(2);
}

TEST_CASE("ParallelTransform2.Dynamic.3threads") {
  parallel_transform2<std::vector<int>, flare::rt::DynamicPartitioner>(3);
  parallel_transform2<std::list<int>, flare::rt::DynamicPartitioner>(3);
}

TEST_CASE("ParallelTransform2.Dynamic.4threads") {
  parallel_transform2<std::vector<int>, flare::rt::DynamicPartitioner>(4);
  parallel_transform2<std::list<int>, flare::rt::DynamicPartitioner>(4);
}

// static
TEST_CASE("ParallelTransform2.Static.1thread") {
  parallel_transform2<std::vector<int>, flare::rt::StaticPartitioner>(1);
  parallel_transform2<std::list<int>, flare::rt::StaticPartitioner>(1);
}

TEST_CASE("ParallelTransform2.Static.2threads") {
  parallel_transform2<std::vector<int>, flare::rt::StaticPartitioner>(2);
  parallel_transform2<std::list<int>, flare::rt::StaticPartitioner>(2);
}

TEST_CASE("ParallelTransform2.Static.3threads") {
  parallel_transform2<std::vector<int>, flare::rt::StaticPartitioner>(3);
  parallel_transform2<std::list<int>, flare::rt::StaticPartitioner>(3);
}

TEST_CASE("ParallelTransform2.Static.4threads") {
  parallel_transform2<std::vector<int>, flare::rt::StaticPartitioner>(4);
  parallel_transform2<std::list<int>, flare::rt::StaticPartitioner>(4);
}

// random
TEST_CASE("ParallelTransform2.Random.1thread") {
  parallel_transform2<std::vector<int>, flare::rt::RandomPartitioner>(1);
  parallel_transform2<std::list<int>, flare::rt::RandomPartitioner>(1);
}

TEST_CASE("ParallelTransform2.Random.2threads") {
  parallel_transform2<std::vector<int>, flare::rt::RandomPartitioner>(2);
  parallel_transform2<std::list<int>, flare::rt::RandomPartitioner>(2);
}

TEST_CASE("ParallelTransform2.Random.3threads") {
  parallel_transform2<std::vector<int>, flare::rt::RandomPartitioner>(3);
  parallel_transform2<std::list<int>, flare::rt::RandomPartitioner>(3);
}

TEST_CASE("ParallelTransform2.Random.4threads") {
  parallel_transform2<std::vector<int>, flare::rt::RandomPartitioner>(4);
  parallel_transform2<std::list<int>, flare::rt::RandomPartitioner>(4);
}


// ----------------------------------------------------------------------------
// Parallel Transform 3
// ----------------------------------------------------------------------------

template <typename P>
void parallel_transform3(size_t W) {

  std::srand(static_cast<unsigned int>(time(NULL)));

  flare::rt::Taskflow taskflow;
  flare::rt::Executor executor(W);

  using std::string;
  using std::size_t;

  for(size_t N=0; N<1000; N=(N+1)*2) {

    std::multimap<int, size_t> src;

    /** Reference implementation with std::transform */
    std::vector<string> ref;

    /** Target implementation with Subflow::transform */
    std::vector<string> tgt;

    typename std::vector<string>::iterator tgt_beg;

    /** A generic function to cast integers to string */
    const auto myFunction = [](const size_t x) -> string {
      return "id_" + std::to_string(x);
    };

    taskflow.clear();

    /** Group integers 0..(N-1) into ten groups,
     * each having an unique key `d`.
     */
    auto from = taskflow.emplace([&, N](){
      for(size_t i = 0; i < N; i++) {
        const int d = ::rand() % 10;
        src.emplace(d, i);
      }

      ref.resize(N);

      tgt.resize(N);
      tgt_beg = tgt.begin();
    });

    auto to_ref = taskflow.emplace([&]() {

      // Find entries matching key = 0.
      // This can return empty results.
      const auto [src_beg, src_end] = src.equal_range(0);
      const size_t n_matching = std::distance(src_beg, src_end);
      ref.resize(n_matching);

      // Extract all values having matching key value.
      std::transform(src_beg, src_end, ref.begin(),
        [&](const auto& x) -> string {
          return myFunction(x.second);
        }
      );
    });

    /** Dynamic scheduling with Subflow::transform */
    auto to_tgt = taskflow.emplace([&](flare::rt::Subflow& subflow) {

      // Find entries matching key = 0
      const auto [src_beg, src_end] = src.equal_range(0);
      const size_t n_matching = std::distance(src_beg, src_end);
      tgt.resize(n_matching);

      subflow.transform(
        std::ref(src_beg), std::ref(src_end), std::ref(tgt_beg),
        [&] (const auto& x) -> string {
          return myFunction(x.second);
      }, P());

      subflow.join();
    });

    from.precede(to_ref);
    from.precede(to_tgt);

    executor.run(taskflow).wait();

    /** Target entries much match. */
    REQUIRE(std::equal(tgt.begin(), tgt.end(), ref.begin()));
  }
}

// guided
TEST_CASE("ParallelTransform3.Guided.1thread") {
  parallel_transform3<flare::rt::GuidedPartitioner>(1);
  parallel_transform3<flare::rt::GuidedPartitioner>(1);
}

TEST_CASE("ParallelTransform3.Guided.2threads") {
  parallel_transform3<flare::rt::GuidedPartitioner>(2);
  parallel_transform3<flare::rt::GuidedPartitioner>(2);
}

TEST_CASE("ParallelTransform3.Guided.3threads") {
  parallel_transform3<flare::rt::GuidedPartitioner>(3);
  parallel_transform3<flare::rt::GuidedPartitioner>(3);
}

TEST_CASE("ParallelTransform3.Guided.4threads") {
  parallel_transform3<flare::rt::GuidedPartitioner>(4);
  parallel_transform3<flare::rt::GuidedPartitioner>(4);
}

// dynamic
TEST_CASE("ParallelTransform3.Dynamic.1thread") {
  parallel_transform3<flare::rt::DynamicPartitioner>(1);
  parallel_transform3<flare::rt::DynamicPartitioner>(1);
}

TEST_CASE("ParallelTransform3.Dynamic.2threads") {
  parallel_transform3<flare::rt::DynamicPartitioner>(2);
  parallel_transform3<flare::rt::DynamicPartitioner>(2);
}

TEST_CASE("ParallelTransform3.Dynamic.3threads") {
  parallel_transform3<flare::rt::DynamicPartitioner>(3);
  parallel_transform3<flare::rt::DynamicPartitioner>(3);
}

TEST_CASE("ParallelTransform3.Dynamic.4threads") {
  parallel_transform3<flare::rt::DynamicPartitioner>(4);
  parallel_transform3<flare::rt::DynamicPartitioner>(4);
}

// static
TEST_CASE("ParallelTransform3.Static.1thread") {
  parallel_transform3<flare::rt::StaticPartitioner>(1);
  parallel_transform3<flare::rt::StaticPartitioner>(1);
}

TEST_CASE("ParallelTransform3.Static.2threads") {
  parallel_transform3<flare::rt::StaticPartitioner>(2);
  parallel_transform3<flare::rt::StaticPartitioner>(2);
}

TEST_CASE("ParallelTransform3.Static.3threads") {
  parallel_transform3<flare::rt::StaticPartitioner>(3);
  parallel_transform3<flare::rt::StaticPartitioner>(3);
}

TEST_CASE("ParallelTransform3.Static.4threads") {
  parallel_transform3<flare::rt::StaticPartitioner>(4);
  parallel_transform3<flare::rt::StaticPartitioner>(4);
}

// random
TEST_CASE("ParallelTransform3.Random.1thread") {
  parallel_transform3<flare::rt::RandomPartitioner>(1);
  parallel_transform3<flare::rt::RandomPartitioner>(1);
}

TEST_CASE("ParallelTransform3.Random.2threads") {
  parallel_transform3<flare::rt::RandomPartitioner>(2);
  parallel_transform3<flare::rt::RandomPartitioner>(2);
}

TEST_CASE("ParallelTransform3.Random.3threads") {
  parallel_transform3<flare::rt::RandomPartitioner>(3);
  parallel_transform3<flare::rt::RandomPartitioner>(3);
}

TEST_CASE("ParallelTransform3.Random.4threads") {
  parallel_transform3<flare::rt::RandomPartitioner>(4);
  parallel_transform3<flare::rt::RandomPartitioner>(4);
}

//// ----------------------------------------------------------------------------
//// ParallelTransform Exception
//// ----------------------------------------------------------------------------
//
//void parallel_transform_exception(unsigned W) {
//  flare::rt::Taskflow taskflow;
//  flare::rt::Executor executor(W);
//
//  std::vector<int> src(1000000, 0);
//  std::vector<int> tgt(1000000, 0);
//
//  taskflow.transform(src.begin(), src.end(), tgt.begin(), [](int&) {
//    throw std::runtime_error("x");
//    return 1;
//  }); 
//  REQUIRE_THROWS_WITH_AS(executor.run(taskflow).get(), "x", std::runtime_error);
//}
//
//TEST_CASE("ParallelTransform.Exception.1thread") {
//  parallel_transform_exception(1);
//}
//
//TEST_CASE("ParallelTransform.Exception.2threads") {
//  parallel_transform_exception(2);
//}
//
//TEST_CASE("ParallelTransform.Exception.3threads") {
//  parallel_transform_exception(3);
//}
//
//TEST_CASE("ParallelTransform.Exception.4threads") {
//  parallel_transform_exception(4);
//}



