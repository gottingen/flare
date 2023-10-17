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

#include <flare/runtime/utility/traits.h>
#include <flare/runtime/utility/object_pool.h>
#include <flare/core/common/small_vector.h>
#include <flare/runtime/utility/uuid.h>
#include <flare/runtime/utility/iterator.h>
#include <flare/runtime/utility/math.h>

// --------------------------------------------------------
// Testcase: SmallVector
// --------------------------------------------------------
TEST_CASE("SmallVector" * doctest::timeout(300)) {

  //SUBCASE("constructor")
  {
    flare::detail::SmallVector<int> vec1;
    REQUIRE(vec1.size() == 0);
    REQUIRE(vec1.empty() == true);

    flare::detail::SmallVector<int, 4> vec2;
    REQUIRE(vec2.data() != nullptr);
    REQUIRE(vec2.size() == 0);
    REQUIRE(vec2.empty() == true);
    REQUIRE(vec2.capacity() == 4);
  }

  //SUBCASE("constructor_n")
  {
    for(int N=0; N<=65536; N = (N ? N << 1 : 1)) {
      flare::detail::SmallVector<int> vec(N);
      REQUIRE(vec.size() == N);
      REQUIRE(vec.empty() == (N == 0));
      REQUIRE(vec.max_size() >= vec.size());
      REQUIRE(vec.capacity() >= vec.size());
    }
  }

  //SUBCASE("copy_constructor")
  {
    for(int N=0; N<=65536; N = (N ? N << 1 : 1)) {
      flare::detail::SmallVector<int> vec1(N);
      for(auto& item : vec1) {
        item = N;
      }

      flare::detail::SmallVector<int> vec2(vec1);
      REQUIRE(vec1.size() == N);
      REQUIRE(vec2.size() == N);
      for(size_t i=0; i<vec1.size(); ++i) {
        REQUIRE(vec1[i] == vec2[i]);
        REQUIRE(vec1[i] == N);
      }
    }
  }

  //SUBCASE("move_constructor")
  {
    for(int N=0; N<=65536; N = (N ? N << 1 : 1)) {
      flare::detail::SmallVector<int> vec1(N);
      for(auto& item : vec1) {
        item = N;
      }

      flare::detail::SmallVector<int> vec2(std::move(vec1));
      REQUIRE(vec1.size() == 0);
      REQUIRE(vec1.empty() == true);
      REQUIRE(vec2.size() == N);

      for(size_t i=0; i<vec2.size(); ++i) {
        REQUIRE(vec2[i] == N);
      }
    }
  }

  //SUBCASE("push_back")
  {
    for(int N=0; N<=65536; N = (N ? N << 1 : 1)) {
      flare::detail::SmallVector<int> vec;
      size_t pcap {0};
      size_t ncap {0};
      for(int n=0; n<N; ++n) {
        vec.push_back(n);
        REQUIRE(vec.size() == n+1);
        ncap = vec.capacity();
        REQUIRE(ncap >= pcap);
        pcap = ncap;
      }
      for(int n=0; n<N; ++n) {
        REQUIRE(vec[n] == n);
      }
      REQUIRE(vec.empty() == (N == 0));
    }
  }

  //SUBCASE("pop_back")
  {
    size_t size {0};
    size_t pcap {0};
    size_t ncap {0};
    flare::detail::SmallVector<int> vec;
    for(int N=0; N<=65536; N = (N ? N << 1 : N + 1)) {
      vec.push_back(N);
      ++size;
      REQUIRE(vec.size() == size);
      if(N % 4 == 0) {
        vec.pop_back();
        --size;
        REQUIRE(vec.size() == size);
      }
      ncap = vec.capacity();
      REQUIRE(ncap >= pcap);
      pcap = ncap;
    }
    REQUIRE(vec.size() == size);
    for(size_t i=0; i<vec.size(); ++i) {
      REQUIRE(vec[i] % 4 != 0);
    }
  }

  //SUBCASE("iterator")
  {
    for(int N=0; N<=65536; N = (N ? N << 1 : 1)) {
      flare::detail::SmallVector<int> vec;
      for(int n=0; n<N; ++n) {
        vec.push_back(n);
        REQUIRE(vec.size() == n+1);
      }

      // non-constant iterator
      {
        int val {0};
        for(auto item : vec) {
          REQUIRE(item == val);
          ++val;
        }
      }

      // constant iterator
      {
        int val {0};
        for(const auto& item : vec) {
          REQUIRE(item == val);
          ++val;
        }
      }

      // change the value
      {
        for(auto& item : vec) {
          item = 1234;
        }
        for(auto& item : vec) {
          REQUIRE(item == 1234);
        }
      }
    }
  }

  //SUBCASE("clear")
  {
    for(int N=0; N<=65536; N = (N ? N << 1 : 1)) {
      flare::detail::SmallVector<int> vec(N);
      auto cap = vec.capacity();
      REQUIRE(vec.size() == N);
      vec.clear();
      REQUIRE(vec.size() == 0);
      REQUIRE(vec.capacity() == cap);
    }
  }

  //SUBCASE("comparison")
  {
    for(int N=0; N<=65536; N = (N ? N << 1 : 1)) {
      flare::detail::SmallVector<int> vec1;
      for(int i=0; i<N; ++i) {
        vec1.push_back(i);
      }
      flare::detail::SmallVector<int> vec2(vec1);
      REQUIRE(vec1 == vec2);
    }
  }
}

// --------------------------------------------------------
// Testcase: distance
// --------------------------------------------------------
TEST_CASE("distance.integral" * doctest::timeout(300)) {

  auto count = [] (int beg, int end, int step) {
    size_t c = 0;
    for(int i=beg; step > 0 ? i < end : i > end; i += step) {
      ++c;
    }
    return c;
  };

  for(int beg=-50; beg<=50; ++beg) {
    for(int end=-50; end<=50; ++end) {
      if(beg < end) {   // positive step
        for(int s=1; s<=50; s++) {
          REQUIRE((flare::rt::distance(beg, end, s) == count(beg, end, s)));
        }
      }
      else {            // negative step
        for(int s=-1; s>=-50; s--) {
          REQUIRE((flare::rt::distance(beg, end, s) == count(beg, end, s)));
        }
      }
    }
  }

}

// --------------------------------------------------------
// Testcase: ObjectPool.Sequential
// --------------------------------------------------------
void test_threaded_uuid(size_t N) {

  std::vector<flare::rt::UUID> uuids(65536);

  // threaded
  std::mutex mutex;
  std::vector<std::thread> threads;

  for(size_t i=0; i<N; ++i) {
    threads.emplace_back([&](){
      for(int i=0; i<1000; ++i) {
        std::lock_guard<std::mutex> lock(mutex);
        uuids.push_back(flare::rt::UUID());
      }
    });
  }

  for(auto& t : threads) {
    t.join();
  }

  auto size = uuids.size();
  std::sort(uuids.begin(), uuids.end());
  std::unique(uuids.begin(), uuids.end());
  REQUIRE(uuids.size() == size);
}

TEST_CASE("uuid") {

  flare::rt::UUID u1, u2, u3, u4;

  // Comparator.
  REQUIRE(u1 == u1);

  // Copy
  u2 = u1;
  REQUIRE(u1 == u2);

  // Move
  u3 = std::move(u1);
  REQUIRE(u2 == u3);

  // Copy constructor
  flare::rt::UUID u5(u4);
  REQUIRE(u5 == u4);

  // Move constructor.
  flare::rt::UUID u6(std::move(u4));
  REQUIRE(u5 == u6);

  // Uniqueness
  std::vector<flare::rt::UUID> uuids(65536);
  std::sort(uuids.begin(), uuids.end());
  std::unique(uuids.begin(), uuids.end());
  REQUIRE(uuids.size() == 65536);

}

TEST_CASE("uuid.10threads") {
  test_threaded_uuid(10);
}

TEST_CASE("uuid.100threads") {
  test_threaded_uuid(100);
}


// --------------------------------------------------------
// Testcase: ObjectPool.Sequential
// --------------------------------------------------------
struct Poolable {
  std::string str;
  std::vector<int> vec;
  int a;
  char b;

  TF_ENABLE_POOLABLE_ON_THIS;
};

TEST_CASE("ObjectPool.Sequential" * doctest::timeout(300)) {

  for(unsigned w=1; w<=4; w++) {

    flare::rt::ObjectPool<Poolable> pool(w);

    REQUIRE(pool.num_heaps() > 0);
    REQUIRE(pool.num_local_heaps() > 0);
    REQUIRE(pool.num_global_heaps() > 0);
    REQUIRE(pool.num_bins_per_local_heap() > 0);
    REQUIRE(pool.num_objects_per_bin() > 0);
    REQUIRE(pool.num_objects_per_block() > 0);
    REQUIRE(pool.emptiness_threshold() > 0);

    // fill out all objects
    size_t N = 100*pool.num_objects_per_block();

    std::set<Poolable*> set;

    for(size_t i=0; i<N; ++i) {
      auto item = pool.animate();
      REQUIRE(set.find(item) == set.end());
      set.insert(item);
    }

    REQUIRE(set.size() == N);

    for(auto s : set) {
      pool.recycle(s);
    }

    REQUIRE(N == pool.capacity());
    REQUIRE(N == pool.num_available_objects());
    REQUIRE(0 == pool.num_allocated_objects());

    for(size_t i=0; i<N; ++i) {
      auto item = pool.animate();
      REQUIRE(set.find(item) != set.end());
    }

    REQUIRE(pool.num_available_objects() == 0);
    REQUIRE(pool.num_allocated_objects() == N);
  }
}

// --------------------------------------------------------
// Testcase: ObjectPool.Threaded
// --------------------------------------------------------

template <typename T>
void threaded_objectpool(unsigned W) {

  flare::rt::ObjectPool<T> pool;

  std::vector<std::thread> threads;

  for(unsigned w=0; w<W; ++w) {
    threads.emplace_back([&pool](){
      std::vector<T*> items;
      for(int i=0; i<65536; ++i) {
        auto item = pool.animate();
        items.push_back(item);
      }
      for(auto item : items) {
        pool.recycle(item);
      }
    });
  }

  for(auto& thread : threads) {
    thread.join();
  }

  REQUIRE(pool.num_allocated_objects() == 0);
  REQUIRE(pool.num_available_objects() == pool.capacity());
}

TEST_CASE("ObjectPool.1thread" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(1);
}

TEST_CASE("ObjectPool.2threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(2);
}

TEST_CASE("ObjectPool.3threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(3);
}

TEST_CASE("ObjectPool.4threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(4);
}

TEST_CASE("ObjectPool.5threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(5);
}

TEST_CASE("ObjectPool.6threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(6);
}

TEST_CASE("ObjectPool.7threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(7);
}

TEST_CASE("ObjectPool.8threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(8);
}

TEST_CASE("ObjectPool.9threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(9);
}

TEST_CASE("ObjectPool.10threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(10);
}

TEST_CASE("ObjectPool.11threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(11);
}

TEST_CASE("ObjectPool.12threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(12);
}

TEST_CASE("ObjectPool.13threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(13);
}

TEST_CASE("ObjectPool.14threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(14);
}

TEST_CASE("ObjectPool.15threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(15);
}

TEST_CASE("ObjectPool.16threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(16);
}

// --------------------------------------------------------
// Testcase: Reference Wrapper
// --------------------------------------------------------

TEST_CASE("RefWrapper" * doctest::timeout(300)) {

  static_assert(std::is_same<
    flare::rt::unwrap_reference_t<int>, int
  >::value, "");

  static_assert(std::is_same<
    flare::rt::unwrap_reference_t<int&>, int&
  >::value, "");

  static_assert(std::is_same<
    flare::rt::unwrap_reference_t<int&&>, int&&
  >::value, "");

  static_assert(std::is_same<
    flare::rt::unwrap_reference_t<std::reference_wrapper<int>>, int&
  >::value, "");

  static_assert(std::is_same<
    flare::rt::unwrap_reference_t<std::reference_wrapper<std::reference_wrapper<int>>>,
    std::reference_wrapper<int>&
  >::value, "");

  static_assert(std::is_same<
    flare::rt::unwrap_ref_decay_t<int>, int
  >::value, "");

  static_assert(std::is_same<
    flare::rt::unwrap_ref_decay_t<int&>, int
  >::value, "");

  static_assert(std::is_same<
    flare::rt::unwrap_ref_decay_t<int&&>, int
  >::value, "");

  static_assert(std::is_same<
    flare::rt::unwrap_ref_decay_t<std::reference_wrapper<int>>, int&
  >::value, "");

  static_assert(std::is_same<
    flare::rt::unwrap_ref_decay_t<std::reference_wrapper<std::reference_wrapper<int>>>,
    std::reference_wrapper<int>&
  >::value, "");

}

//// --------------------------------------------------------
//// Testcase: FunctionTraits
//// --------------------------------------------------------
//void func1() {
//}
//
//int func2(int, double, float, char) {
//  return 0;
//}
//
//TEST_CASE("FunctionTraits" * doctest::timeout(300)) {
//
//  SUBCASE("func1") {
//    using func1_traits = flare::rt::function_traits<decltype(func1)>;
//    static_assert(std::is_same<func1_traits::return_type, void>::value, "");
//    static_assert(func1_traits::arity == 0, "");
//  }
//
//  SUBCASE("func2") {
//    using func2_traits = flare::rt::function_traits<decltype(func2)>;
//    static_assert(std::is_same<func2_traits::return_type, int>::value, "");
//    static_assert(func2_traits::arity == 4, "");
//    static_assert(std::is_same<func2_traits::argument_t<0>, int>::value,   "");
//    static_assert(std::is_same<func2_traits::argument_t<1>, double>::value,"");
//    static_assert(std::is_same<func2_traits::argument_t<2>, float>::value, "");
//    static_assert(std::is_same<func2_traits::argument_t<3>, char>::value,  "");
//  }
//
//  SUBCASE("lambda1") {
//    auto lambda1 = [] () mutable {
//      return 1;
//    };
//    using lambda1_traits = flare::rt::function_traits<decltype(lambda1)>;
//    static_assert(std::is_same<lambda1_traits::return_type, int>::value, "");
//    static_assert(lambda1_traits::arity == 0, "");
//  }
//
//  SUBCASE("lambda2") {
//    auto lambda2 = [] (int, double, char&) {
//    };
//    using lambda2_traits = flare::rt::function_traits<decltype(lambda2)>;
//    static_assert(std::is_same<lambda2_traits::return_type, void>::value, "");
//    static_assert(lambda2_traits::arity == 3, "");
//    static_assert(std::is_same<lambda2_traits::argument_t<0>, int>::value, "");
//    static_assert(std::is_same<lambda2_traits::argument_t<1>, double>::value, "");
//    static_assert(std::is_same<lambda2_traits::argument_t<2>, char&>::value, "");
//  }
//
//  SUBCASE("class") {
//    struct foo {
//      int operator ()(int, float) const;
//    };
//    using foo_traits = flare::rt::function_traits<foo>;
//    static_assert(std::is_same<foo_traits::return_type, int>::value, "");
//    static_assert(foo_traits::arity == 2, "");
//    static_assert(std::is_same<foo_traits::argument_t<0>, int>::value, "");
//    static_assert(std::is_same<foo_traits::argument_t<1>, float>::value, "");
//  }
//
//  SUBCASE("std-function") {
//    using ft1 = flare::rt::function_traits<std::function<void()>>;
//    static_assert(std::is_same<ft1::return_type, void>::value, "");
//    static_assert(ft1::arity == 0, "");
//
//    using ft2 = flare::rt::function_traits<std::function<int(int&, double&&)>&>;
//    static_assert(std::is_same<ft2::return_type, int>::value, "");
//    static_assert(ft2::arity == 2, "");
//    static_assert(std::is_same<ft2::argument_t<0>, int&>::value, "");
//    static_assert(std::is_same<ft2::argument_t<1>, double&&>::value, "");
//
//    using ft3 = flare::rt::function_traits<std::function<int(int&, double&&)>&&>;
//    static_assert(std::is_same<ft3::return_type, int>::value, "");
//    static_assert(ft3::arity == 2, "");
//    static_assert(std::is_same<ft3::argument_t<0>, int&>::value, "");
//    static_assert(std::is_same<ft3::argument_t<1>, double&&>::value, "");
//
//    using ft4 = flare::rt::function_traits<const std::function<void(int)>&>;
//    static_assert(std::is_same<ft4::return_type, void>::value, "");
//    static_assert(ft4::arity == 1, "");
//    static_assert(std::is_same<ft4::argument_t<0>, int>::value, "");
//  }
//}

// --------------------------------------------------------
// Math utilities
// --------------------------------------------------------
TEST_CASE("NextPow2") {

  static_assert(flare::rt::next_pow2(0u) == 1);
  static_assert(flare::rt::next_pow2(1u) == 1);
  static_assert(flare::rt::next_pow2(100u) == 128u);
  static_assert(flare::rt::next_pow2(245u) == 256u);
  static_assert(flare::rt::next_pow2(512u) == 512u);
  static_assert(flare::rt::next_pow2(513u) == 1024u);

  REQUIRE(flare::rt::next_pow2(0u) == 1u);
  REQUIRE(flare::rt::next_pow2(2u) == 2u);
  REQUIRE(flare::rt::next_pow2(1u) == 1u);
  REQUIRE(flare::rt::next_pow2(33u) == 64u);
  REQUIRE(flare::rt::next_pow2(100u) == 128u);
  REQUIRE(flare::rt::next_pow2(211u) == 256u);
  REQUIRE(flare::rt::next_pow2(23u) == 32u);
  REQUIRE(flare::rt::next_pow2(54u) == 64u);

  uint64_t z = 0;
  uint64_t a = 1;
  REQUIRE(flare::rt::next_pow2(z) == 1);
  REQUIRE(flare::rt::next_pow2(a) == a);
  REQUIRE(flare::rt::next_pow2((a<<5)  + 0) == (a<<5));
  REQUIRE(flare::rt::next_pow2((a<<5)  + 1) == (a<<6));
  REQUIRE(flare::rt::next_pow2((a<<32) + 0) == (a<<32));
  REQUIRE(flare::rt::next_pow2((a<<32) + 1) == (a<<33));
  REQUIRE(flare::rt::next_pow2((a<<41) + 0) == (a<<41));
  REQUIRE(flare::rt::next_pow2((a<<41) + 1) == (a<<42));

  REQUIRE(flare::rt::is_pow2(0) == false);
  REQUIRE(flare::rt::is_pow2(1) == true);
  REQUIRE(flare::rt::is_pow2(2) == true);
  REQUIRE(flare::rt::is_pow2(3) == false);
  REQUIRE(flare::rt::is_pow2(0u) == false);
  REQUIRE(flare::rt::is_pow2(1u) == true);
  REQUIRE(flare::rt::is_pow2(54u) == false);
  REQUIRE(flare::rt::is_pow2(64u) == true);
}





