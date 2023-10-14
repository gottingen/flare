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
#include <cstdlib>
#include <cmath>

// Type of a one-dimensional length-N array of int.
using tensor_type      = flare::Tensor<int*>;
using host_tensor_type = tensor_type::HostMirror;
// This is a "zero-dimensional" Tensor, that is, a Tensor of a single
// value (an int, in this case).  Access the value using operator()
// with no arguments: e.g., 'count()'.
//
// Zero-dimensional Tensors are useful for reduction results that stay
// resident in device memory, as well as for irregularly updated
// shared state.  We use it for the latter in this example.
using count_type      = flare::Tensor<int>;
using host_count_type = count_type::HostMirror;

// Functor for finding a list of primes in a given set of numbers.  If
// run in parallel, the order of results is nondeterministic, because
// hardware atomic updates do not guarantee an order of execution.
struct findprimes {
  tensor_type data;
  tensor_type result;
  count_type count;

  findprimes(tensor_type data_, tensor_type result_, count_type count_)
      : data(data_), result(result_), count(count_) {}

  // Test if data(i) is prime.  If it is, increment the count of
  // primes (stored in the zero-dimensional Tensor 'count') and add the
  // value to the current list of primes 'result'.
  FLARE_INLINE_FUNCTION
  void operator()(const int i) const {
    const int number = data(i);  // the current number

    // Test all numbers from 3 to ceiling(sqrt(data(i))), to see if
    // they are factors of data(i).  It's not the most efficient prime
    // test, but it works.
    const int upper_bound = std::sqrt(1.0 * number) + 1;
    bool is_prime         = !(number % 2 == 0);
    int k                 = 3;
    while (k < upper_bound && is_prime) {
      is_prime = !(number % k == 0);
      k += 2;  // don't have to test even numbers
    }

    if (is_prime) {
      // Use an atomic update both to update the current count of
      // primes, and to find a place in the current list of primes for
      // the new result.
      //
      // atomic_fetch_add results the _current_ count, but increments
      // it (by 1 in this case).  The current count of primes indexes
      // into the first unoccupied position of the 'result' array.
      const int idx = flare::atomic_fetch_add(&count(), 1);
      result(idx)   = number;
    }
  }
};

int main() {
  flare::initialize();

  {
    srand(61391);  // Set the random seed

    int nnumbers = 100000;
    tensor_type data("RND", nnumbers);
    tensor_type result("Prime", nnumbers);
    count_type count("Count");

    host_tensor_type h_data   = flare::create_mirror_tensor(data);
    host_tensor_type h_result = flare::create_mirror_tensor(result);
    host_count_type h_count = flare::create_mirror_tensor(count);

    using size_type = tensor_type::size_type;
    // Fill the 'data' array on the host with random numbers.  We assume
    // that they come from some process which is only implemented on the
    // host, via some library.  (That's true in this case.)
    for (size_type i = 0; i < static_cast<size_type>(data.extent(0)); ++i) {
      h_data(i) = rand() % nnumbers;
    }
    flare::deep_copy(data, h_data);  // copy from host to device

    flare::parallel_for(data.extent(0), findprimes(data, result, count));
    flare::deep_copy(h_count, count);  // copy from device to host

    printf("Found %i prime numbers in %i random numbers\n", h_count(),
           nnumbers);
  }
  flare::finalize();
}
