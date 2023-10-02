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
#include <cstdio>

//
// First reduction (parallel_reduce) example:
//   1. Start up flare
//   2. Execute a parallel_reduce loop in the default execution space,
//      using a functor to define the loop body
//   3. Shut down flare
//
struct collision {
  // Reduction functor
  // For each i, we generate 10 hashes, look for and count collisions
  // We use parallel_reduce to count the total collisions
  // Note that we're just counting collisions within the 10 generated
  // one i.
  // This function was chosen as one that very simply can increase the
  // register count.
  using value_type = int;

  FLARE_INLINE_FUNCTION
  int hash(int q) const {
    // A simple hash by Justin Sobel
    // Thanks to Arash Partow (partow.net)
    char* fourchars = (char*)&q;
    int hash        = 1315423911;
    for (int i = 0; i < 4; fourchars++, i++) {
      hash ^= ((hash << 5) + *fourchars + (hash >> 2));
    }
    return hash;
  }

  FLARE_INLINE_FUNCTION
  void operator()(const int i, int& lsum) const {
    // This is a silly function which generates 10 hashes
    // then checks for collisions
    int a = hash(i) % 64;
    int b = hash(i * 3) % 64;
    int c = hash(i * 5) % 64;
    int d = hash(i * 7) % 64;
    int e = hash(i * 11) % 64;
    int f = hash(i * 17) % 64;
    int g = hash(i * 23) % 64;
    int h = hash(i * 29) % 64;
    int j = hash(i * 31) % 64;
    int k = hash(i * 37) % 64;

    if (a == b) lsum++;
    if (a == c) lsum++;
    if (a == d) lsum++;
    if (a == e) lsum++;
    if (a == f) lsum++;
    if (a == g) lsum++;
    if (a == h) lsum++;
    if (a == j) lsum++;
    if (a == k) lsum++;
    if (b == c) lsum++;
    if (b == d) lsum++;
    if (b == e) lsum++;
    if (b == f) lsum++;
    if (b == g) lsum++;
    if (b == h) lsum++;
    if (b == j) lsum++;
    if (b == k) lsum++;
    if (c == d) lsum++;
    if (c == e) lsum++;
    if (c == f) lsum++;
    if (c == g) lsum++;
    if (c == h) lsum++;
    if (c == j) lsum++;
    if (c == k) lsum++;
    if (d == e) lsum++;
    if (d == f) lsum++;
    if (d == g) lsum++;
    if (d == h) lsum++;
    if (d == j) lsum++;
    if (d == k) lsum++;
    if (e == f) lsum++;
    if (e == g) lsum++;
    if (e == h) lsum++;
    if (e == j) lsum++;
    if (e == k) lsum++;
    if (f == g) lsum++;
    if (f == h) lsum++;
    if (f == j) lsum++;
    if (f == k) lsum++;
    if (g == h) lsum++;
    if (g == j) lsum++;
    if (g == k) lsum++;
    if (h == j) lsum++;
    if (h == k) lsum++;
    if (j == k) lsum++;
  }
};

int main(int argc, char* argv[]) {
  flare::initialize(argc, argv);
  const int n = 10000;

  // Compute and count hash collisions in
  // parallel, using flare.
  // This is not really a useful algorithm, but it demonstrates the
  // LaunchBounds functionality
  int sum1 = 0;
  int sum2 = 0;

  // Without LaunchBounds, the kernel uses 56 registers
  flare::parallel_reduce(n, collision(), sum1);

  // With LaunchBounds, we can reduce the register usage to 32
  flare::parallel_reduce(
      flare::RangePolicy<flare::LaunchBounds<512, 4>>(0, n), collision(),
      sum2);

  printf(
      "Number of collisions, "
      "computed in parallel, is %i\n",
      sum1);

  if (sum1 != sum2) {
    printf("Uh-oh! Results do not match\n");
    return -1;
  }

  flare::finalize();

  return 0;
}
