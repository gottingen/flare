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

// --------------------------------------------------------
// Testcase: Composition
// --------------------------------------------------------
TEST_CASE("Composition-1" * doctest::timeout(300)) {

  for(unsigned w=1; w<=8; ++w) {

    flare::rt::Executor executor(w);

    flare::rt::Taskflow f0;

    int cnt {0};

    auto A = f0.emplace([&cnt](){ ++cnt; });
    auto B = f0.emplace([&cnt](){ ++cnt; });
    auto C = f0.emplace([&cnt](){ ++cnt; });
    auto D = f0.emplace([&cnt](){ ++cnt; });
    auto E = f0.emplace([&cnt](){ ++cnt; });

    A.precede(B);
    B.precede(C);
    C.precede(D);
    D.precede(E);

    flare::rt::Taskflow f1;

    // module 1
    std::tie(A, B, C, D, E) = f1.emplace(
      [&cnt] () { ++cnt; },
      [&cnt] () { ++cnt; },
      [&cnt] () { ++cnt; },
      [&cnt] () { ++cnt; },
      [&cnt] () { ++cnt; }
    );
    A.precede(B);
    B.precede(C);
    C.precede(D);
    D.precede(E);
    auto m1_1 = f1.composed_of(f0);
    E.precede(m1_1);

    executor.run(f1).get();
    REQUIRE(cnt == 10);

    cnt = 0;
    executor.run_n(f1, 100).get();
    REQUIRE(cnt == 10 * 100);

    auto m1_2 = f1.composed_of(f0);
    m1_1.precede(m1_2);

    for(int n=0; n<100; n++) {
      cnt = 0;
      executor.run_n(f1, n).get();
      REQUIRE(cnt == 15*n);
    }

    cnt = 0;
    for(int n=0; n<100; n++) {
      executor.run(f1);
    }

    executor.wait_for_all();

    REQUIRE(cnt == 1500);
  }
}

// TESTCASE: composition-2
TEST_CASE("Composition-2" * doctest::timeout(300)) {

  for(unsigned w=1; w<=8; ++w) {

    flare::rt::Executor executor(w);

    int cnt {0};

    // level 0 (+5)
    flare::rt::Taskflow f0;

    auto A = f0.emplace([&cnt](){ ++cnt; }).name("f0A");
    auto B = f0.emplace([&cnt](){ ++cnt; }).name("f0B");
    auto C = f0.emplace([&cnt](){ ++cnt; }).name("f0C");
    auto D = f0.emplace([&cnt](){ ++cnt; }).name("f0D");
    auto E = f0.emplace([&cnt](){ ++cnt; }).name("f0E");

    A.precede(B);
    B.precede(C);
    C.precede(D);
    D.precede(E);

    // level 1 (+10)
    flare::rt::Taskflow f1;
    auto m1_1 = f1.composed_of(f0).name("m1_1");
    auto m1_2 = f1.composed_of(f0).name("m1_2");
    m1_1.precede(m1_2);

    // level 2 (+20)
    flare::rt::Taskflow f2;
    auto m2_1 = f2.composed_of(f1).name("m2_1");
    auto m2_2 = f2.composed_of(f1).name("m2_2");
    m2_1.precede(m2_2);

    //f2.dump(std::cout);

    // synchronous run
    for(int n=0; n<100; n++) {
      cnt = 0;
      executor.run_n(f2, n).get();
      REQUIRE(cnt == 20*n);
    }

    // asynchronous run
    cnt = 0;
    for(int n=0; n<100; n++) {
      executor.run(f2);
    }
    executor.wait_for_all();
    REQUIRE(cnt == 100*20);
  }
}

// TESTCASE: composition-3
TEST_CASE("Composition-3" * doctest::timeout(300)) {

  for(unsigned w=1; w<=8; ++w) {

    flare::rt::Executor executor(w);

    int cnt {0};

    // level 0 (+2)
    flare::rt::Taskflow f0;

    auto A = f0.emplace([&cnt](){ ++cnt; });
    auto B = f0.emplace([&cnt](){ ++cnt; });

    A.precede(B);

    // level 1 (+4)
    flare::rt::Taskflow f1;
    auto m1_1 = f1.composed_of(f0);
    auto m1_2 = f1.composed_of(f0);
    m1_1.precede(m1_2);

    // level 2 (+8)
    flare::rt::Taskflow f2;
    auto m2_1 = f2.composed_of(f1);
    auto m2_2 = f2.composed_of(f1);
    m2_1.precede(m2_2);

    // level 3 (+16)
    flare::rt::Taskflow f3;
    auto m3_1 = f3.composed_of(f2);
    auto m3_2 = f3.composed_of(f2);
    m3_1.precede(m3_2);

    // synchronous run
    for(int n=0; n<100; n++) {
      cnt = 0;
      executor.run_n(f3, n).get();
      REQUIRE(cnt == 16*n);
    }

    // asynchronous run
    cnt = 0;
    for(int n=0; n<100; n++) {
      executor.run(f3);
    }
    executor.wait_for_all();
    REQUIRE(cnt == 16*100);
  }
}

// ----------------------------------------------------------------------------
// ParallelCompositions
// ----------------------------------------------------------------------------
TEST_CASE("ParallelCompositions") {

  std::vector<flare::rt::Taskflow> taskflows(100);

  flare::rt::Executor executor(4);
  flare::rt::Taskflow taskflow;

  std::atomic<int> counter{0};

  for(auto& tf : taskflows) {
    for(size_t n=0; n<100; n++) {
      auto [A, B, C, D, E, F, G, H] = tf.emplace(
        [&](){ counter.fetch_add(1, std::memory_order_relaxed); },
        [&](){ counter.fetch_add(1, std::memory_order_relaxed); },
        [&](){ counter.fetch_add(1, std::memory_order_relaxed); },
        [&](){ counter.fetch_add(1, std::memory_order_relaxed); },
        [&](){ counter.fetch_add(1, std::memory_order_relaxed); },
        [&](){ counter.fetch_add(1, std::memory_order_relaxed); },
        [&](){ counter.fetch_add(1, std::memory_order_relaxed); },
        [&](){ counter.fetch_add(1, std::memory_order_relaxed); }
      );
      A.precede(B);
      A.precede(C);
      D.precede(E);
      D.precede(F);
    }
    taskflow.composed_of(tf);
  }

  executor.run(taskflow).wait();

  REQUIRE(counter == 80000);
}





