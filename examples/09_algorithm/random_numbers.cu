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
#include <flare/random.h>
#include <flare/dual_tensor.h>
#include <flare/timer.h>
#include <cstdlib>

using DefaultHostType = flare::HostSpace::execution_space;

// flare provides two different random number generators with a 64 bit and a
// 1024 bit state. These generators are based on Vigna, Sebastiano (2014). "An
// experimental exploration of Marsaglia's xorshift generators, scrambled" See:
// http://arxiv.org/abs/1402.6246 The generators can be used fully independently
// on each thread and have been tested to produce good statistics for both inter
// and intra thread numbers. Note that within a kernel NO random number
// operations are (team) collective operations. Everything can be called within
// branches. This is a difference to the curand library where certain operations
// are required to be called by all threads in a block.
//
// In flare you are required to create a pool of generator states, so that
// threads can grep their own. On CPU architectures the pool size is equal to
// the thread number, on CUDA about 128k states are generated (enough to give
// every potentially simultaneously running thread its own state). With a kernel
// a thread is required to acquire a state from the pool and later return it. On
// CPUs the Random number generator is deterministic if using the same number of
// threads. On GPUs (i.e. using the CUDA backend it is not deterministic because
// threads acquire states via atomics.

// A Functor for generating uint64_t random numbers templated on the
// GeneratorPool type
template <class GeneratorPool>
struct generate_random {
  // Output Tensor for the random numbers
  flare::Tensor<uint64_t**> vals;

  // The GeneratorPool
  GeneratorPool rand_pool;

  int samples;

  // Initialize all members
  generate_random(flare::Tensor<uint64_t**> vals_, GeneratorPool rand_pool_,
                  int samples_)
      : vals(vals_), rand_pool(rand_pool_), samples(samples_) {}

  FLARE_INLINE_FUNCTION
  void operator()(int i) const {
    // Get a random number state from the pool for the active thread
    typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

    // Draw samples numbers from the pool as urand64 between 0 and
    // rand_pool.MAX_URAND64 Note there are function calls to get other type of
    // scalars, and also to specify Ranges or get a normal distributed float.
    for (int k = 0; k < samples; k++) vals(i, k) = rand_gen.urand64();

    // Give the state back, which will allow another thread to acquire it
    rand_pool.free_state(rand_gen);
  }
};

int main(int argc, char* args[]) {
  flare::initialize(argc, args);
  if (argc != 3) {
    printf("Please pass two integers on the command line\n");
  } else {
    // Initialize flare
    int size    = std::stoi(args[1]);
    int samples = std::stoi(args[2]);

    // Create two random number generator pools one for 64bit states and one for
    // 1024 bit states Both take an 64 bit unsigned integer seed to initialize a
    // Random_XorShift64 generator which is used to fill the generators of the
    // pool.
    flare::Random_XorShift64_Pool<> rand_pool64(5374857);
    flare::Random_XorShift1024_Pool<> rand_pool1024(5374857);
    flare::DualTensor<uint64_t**> vals("Vals", size, samples);

    // Run some performance comparisons
    flare::Timer timer;
    flare::parallel_for(size,
                         generate_random<flare::Random_XorShift64_Pool<> >(
                             vals.d_tensor, rand_pool64, samples));
    flare::fence();

    timer.reset();
    flare::parallel_for(size,
                         generate_random<flare::Random_XorShift64_Pool<> >(
                             vals.d_tensor, rand_pool64, samples));
    flare::fence();
    double time_64 = timer.seconds();

    flare::parallel_for(size,
                         generate_random<flare::Random_XorShift1024_Pool<> >(
                             vals.d_tensor, rand_pool1024, samples));
    flare::fence();

    timer.reset();
    flare::parallel_for(size,
                         generate_random<flare::Random_XorShift1024_Pool<> >(
                             vals.d_tensor, rand_pool1024, samples));
    flare::fence();
    double time_1024 = timer.seconds();

    printf("#Time XorShift64*:   %e %e\n", time_64,
           1.0e-9 * samples * size / time_64);
    printf("#Time XorShift1024*: %e %e\n", time_1024,
           1.0e-9 * samples * size / time_1024);

    flare::deep_copy(vals.h_tensor, vals.d_tensor);
  }
  flare::finalize();
  return 0;
}
