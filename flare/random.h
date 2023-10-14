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

#ifndef FLARE_RANDOM_H_
#define FLARE_RANDOM_H_

#include <flare/core.h>
#include <flare/core/complex.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

/// \file random.h
/// \brief Pseudorandom number generators
///
/// These generators are based on Vigna, Sebastiano (2014). "An
/// experimental exploration of Marsaglia's xorshift generators,
/// scrambled."  See: http://arxiv.org/abs/1402.6246

namespace flare {

// clang-format off
  /*Template functions to get equidistributed random numbers from a generator for a specific Scalar type

       template<class Generator,Scalar>
       struct rand{

         //Max value returned by draw(Generator& gen)
         FLARE_INLINE_FUNCTION
         static Scalar max();

         //Returns a value between zero and max()
         FLARE_INLINE_FUNCTION
         static Scalar draw(Generator& gen);

         //Returns a value between zero and range()
         //Note: for floating point values range can be larger than max()
         FLARE_INLINE_FUNCTION
         static Scalar draw(Generator& gen, const Scalar& range){}

         //Return value between start and end
         FLARE_INLINE_FUNCTION
         static Scalar draw(Generator& gen, const Scalar& start, const Scalar& end);
      };

    The Random number generators themselves have two components a state-pool and the actual generator
    A state-pool manages a number of generators, so that each active thread is able to grep its own.
    This allows the generation of random numbers which are independent between threads. Note that
    in contrast to CuRand none of the functions of the pool (or the generator) are collectives,
    i.e. all functions can be called inside conditionals.

    template<class Device>
    class Pool {
     public:
      //The flare device type
      using device_type = Device;
      //The actual generator type
      using generator_type = Generator<Device>;

      //Default constructor: does not initialize a pool
      Pool();

      //Initializing constructor: calls init(seed,Device_Specific_Number);
      Pool(unsigned int seed);

      //Initialize Pool with seed as a starting seed with a pool_size of num_states
      //The Random_XorShift64 generator is used in serial to initialize all states,
      //thus the initialization process is platform independent and deterministic.
      void init(unsigned int seed, int num_states);

      //Get a generator. This will lock one of the states, guaranteeing that each thread
      //will have its private generator. Note: on Cuda getting a state involves atomics,
      //and is thus not deterministic!
      generator_type get_state();

      //Give a state back to the pool. This unlocks the state, and writes the modified
      //state of the generator back to the pool.
      void free_state(generator_type gen);

    }

    template<class Device>
    class Generator {
     public:
     //The flare device type
    using device_type = DeviceType;

    //Max return values of respective [X]rand[S]() functions
    enum {MAX_URAND = 0xffffffffU};
    enum {MAX_URAND64 = 0xffffffffffffffffULL-1};
    enum {MAX_RAND = static_cast<int>(0xffffffffU/2)};
    enum {MAX_RAND64 = static_cast<int64_t>(0xffffffffffffffffULL/2-1)};


    //Init with a state and the idx with respect to pool. Note: in serial the
    //Generator can be used by just giving it the necessary state arguments
    FLARE_INLINE_FUNCTION
    Generator (STATE_ARGUMENTS, int state_idx = 0);

    //Draw a equidistributed uint32_t in the range [0,MAX_URAND)
    FLARE_INLINE_FUNCTION
    uint32_t urand();

    //Draw a equidistributed uint64_t in the range [0,MAX_URAND64)
    FLARE_INLINE_FUNCTION
    uint64_t urand64();

    //Draw a equidistributed uint32_t in the range [0,range)
    FLARE_INLINE_FUNCTION
    uint32_t urand(const uint32_t& range);

    //Draw a equidistributed uint32_t in the range [start,end)
    FLARE_INLINE_FUNCTION
    uint32_t urand(const uint32_t& start, const uint32_t& end );

    //Draw a equidistributed uint64_t in the range [0,range)
    FLARE_INLINE_FUNCTION
    uint64_t urand64(const uint64_t& range);

    //Draw a equidistributed uint64_t in the range [start,end)
    FLARE_INLINE_FUNCTION
    uint64_t urand64(const uint64_t& start, const uint64_t& end );

    //Draw a equidistributed int in the range [0,MAX_RAND)
    FLARE_INLINE_FUNCTION
    int rand();

    //Draw a equidistributed int in the range [0,range)
    FLARE_INLINE_FUNCTION
    int rand(const int& range);

    //Draw a equidistributed int in the range [start,end)
    FLARE_INLINE_FUNCTION
    int rand(const int& start, const int& end );

    //Draw a equidistributed int64_t in the range [0,MAX_RAND64)
    FLARE_INLINE_FUNCTION
    int64_t rand64();

    //Draw a equidistributed int64_t in the range [0,range)
    FLARE_INLINE_FUNCTION
    int64_t rand64(const int64_t& range);

    //Draw a equidistributed int64_t in the range [start,end)
    FLARE_INLINE_FUNCTION
    int64_t rand64(const int64_t& start, const int64_t& end );

    //Draw a equidistributed float in the range [0,1.0)
    FLARE_INLINE_FUNCTION
    float frand();

    //Draw a equidistributed float in the range [0,range)
    FLARE_INLINE_FUNCTION
    float frand(const float& range);

    //Draw a equidistributed float in the range [start,end)
    FLARE_INLINE_FUNCTION
    float frand(const float& start, const float& end );

    //Draw a equidistributed double in the range [0,1.0)
    FLARE_INLINE_FUNCTION
    double drand();

    //Draw a equidistributed double in the range [0,range)
    FLARE_INLINE_FUNCTION
    double drand(const double& range);

    //Draw a equidistributed double in the range [start,end)
    FLARE_INLINE_FUNCTION
    double drand(const double& start, const double& end );

    //Draw a standard normal distributed double
    FLARE_INLINE_FUNCTION
    double normal() ;

    //Draw a normal distributed double with given mean and standard deviation
    FLARE_INLINE_FUNCTION
    double normal(const double& mean, const double& std_dev=1.0);
    }

    //Additional Functions:

    //Fills tensor with random numbers in the range [0,range)
    template<class TensorType, class PoolType>
    void fill_random(TensorType tensor, PoolType pool, TensorType::value_type range);

    //Fills tensor with random numbers in the range [start,end)
    template<class TensorType, class PoolType>
    void fill_random(TensorType tensor, PoolType pool,
                     TensorType::value_type start, TensorType::value_type end);

*/
// clang-format on

template <class Generator, class Scalar>
struct rand;

template <class Generator>
struct rand<Generator, char> {
  FLARE_INLINE_FUNCTION
  static short max() { return 127; }
  FLARE_INLINE_FUNCTION
  static short draw(Generator& gen) {
    return short((gen.rand() & 0xff + 256) % 256);
  }
  FLARE_INLINE_FUNCTION
  static short draw(Generator& gen, const char& range) {
    return char(gen.rand(range));
  }
  FLARE_INLINE_FUNCTION
  static short draw(Generator& gen, const char& start, const char& end) {
    return char(gen.rand(start, end));
  }
};

template <class Generator>
struct rand<Generator, short> {
  FLARE_INLINE_FUNCTION
  static short max() { return 32767; }
  FLARE_INLINE_FUNCTION
  static short draw(Generator& gen) {
    return short((gen.rand() & 0xffff + 65536) % 32768);
  }
  FLARE_INLINE_FUNCTION
  static short draw(Generator& gen, const short& range) {
    return short(gen.rand(range));
  }
  FLARE_INLINE_FUNCTION
  static short draw(Generator& gen, const short& start, const short& end) {
    return short(gen.rand(start, end));
  }
};

template <class Generator>
struct rand<Generator, int> {
  FLARE_INLINE_FUNCTION
  static int max() { return Generator::MAX_RAND; }
  FLARE_INLINE_FUNCTION
  static int draw(Generator& gen) { return gen.rand(); }
  FLARE_INLINE_FUNCTION
  static int draw(Generator& gen, const int& range) { return gen.rand(range); }
  FLARE_INLINE_FUNCTION
  static int draw(Generator& gen, const int& start, const int& end) {
    return gen.rand(start, end);
  }
};

template <class Generator>
struct rand<Generator, unsigned int> {
  FLARE_INLINE_FUNCTION
  static unsigned int max() { return Generator::MAX_URAND; }
  FLARE_INLINE_FUNCTION
  static unsigned int draw(Generator& gen) { return gen.urand(); }
  FLARE_INLINE_FUNCTION
  static unsigned int draw(Generator& gen, const unsigned int& range) {
    return gen.urand(range);
  }
  FLARE_INLINE_FUNCTION
  static unsigned int draw(Generator& gen, const unsigned int& start,
                           const unsigned int& end) {
    return gen.urand(start, end);
  }
};

template <class Generator>
struct rand<Generator, long> {
  FLARE_INLINE_FUNCTION
  static long max() {
    // FIXME (mfh 26 Oct 2014) It would be better to select the
    // return value at compile time, using something like enable_if.
    return sizeof(long) == 4 ? static_cast<long>(Generator::MAX_RAND)
                             : static_cast<long>(Generator::MAX_RAND64);
  }
  FLARE_INLINE_FUNCTION
  static long draw(Generator& gen) {
    // FIXME (mfh 26 Oct 2014) It would be better to select the
    // return value at compile time, using something like enable_if.
    return sizeof(long) == 4 ? static_cast<long>(gen.rand())
                             : static_cast<long>(gen.rand64());
  }
  FLARE_INLINE_FUNCTION
  static long draw(Generator& gen, const long& range) {
    // FIXME (mfh 26 Oct 2014) It would be better to select the
    // return value at compile time, using something like enable_if.
    return sizeof(long) == 4
               ? static_cast<long>(gen.rand(static_cast<int>(range)))
               : static_cast<long>(gen.rand64(range));
  }
  FLARE_INLINE_FUNCTION
  static long draw(Generator& gen, const long& start, const long& end) {
    // FIXME (mfh 26 Oct 2014) It would be better to select the
    // return value at compile time, using something like enable_if.
    return sizeof(long) == 4
               ? static_cast<long>(
                     gen.rand(static_cast<int>(start), static_cast<int>(end)))
               : static_cast<long>(gen.rand64(start, end));
  }
};

template <class Generator>
struct rand<Generator, unsigned long> {
  FLARE_INLINE_FUNCTION
  static unsigned long max() {
    // FIXME (mfh 26 Oct 2014) It would be better to select the
    // return value at compile time, using something like enable_if.
    return sizeof(unsigned long) == 4
               ? static_cast<unsigned long>(Generator::MAX_URAND)
               : static_cast<unsigned long>(Generator::MAX_URAND64);
  }
  FLARE_INLINE_FUNCTION
  static unsigned long draw(Generator& gen) {
    // FIXME (mfh 26 Oct 2014) It would be better to select the
    // return value at compile time, using something like enable_if.
    return sizeof(unsigned long) == 4
               ? static_cast<unsigned long>(gen.urand())
               : static_cast<unsigned long>(gen.urand64());
  }
  FLARE_INLINE_FUNCTION
  static unsigned long draw(Generator& gen, const unsigned long& range) {
    // FIXME (mfh 26 Oct 2014) It would be better to select the
    // return value at compile time, using something like enable_if.
    return sizeof(unsigned long) == 4
               ? static_cast<unsigned long>(
                     gen.urand(static_cast<unsigned int>(range)))
               : static_cast<unsigned long>(gen.urand64(range));
  }
  FLARE_INLINE_FUNCTION
  static unsigned long draw(Generator& gen, const unsigned long& start,
                            const unsigned long& end) {
    // FIXME (mfh 26 Oct 2014) It would be better to select the
    // return value at compile time, using something like enable_if.
    return sizeof(unsigned long) == 4
               ? static_cast<unsigned long>(
                     gen.urand(static_cast<unsigned int>(start),
                               static_cast<unsigned int>(end)))
               : static_cast<unsigned long>(gen.urand64(start, end));
  }
};

// NOTE (mfh 26 oct 2014) This is a partial specialization for long
// long, a C99 / C++11 signed type which is guaranteed to be at
// least 64 bits.  Do NOT write a partial specialization for
// int64_t!!!  This is just an alias!  It could be either long or
// long long.  We don't know which a priori, and I've seen both.
// The types long and long long are guaranteed to differ, so it's
// always safe to specialize for both.
template <class Generator>
struct rand<Generator, long long> {
  FLARE_INLINE_FUNCTION
  static long long max() {
    // FIXME (mfh 26 Oct 2014) It's legal for long long to be > 64 bits.
    return Generator::MAX_RAND64;
  }
  FLARE_INLINE_FUNCTION
  static long long draw(Generator& gen) {
    // FIXME (mfh 26 Oct 2014) It's legal for long long to be > 64 bits.
    return gen.rand64();
  }
  FLARE_INLINE_FUNCTION
  static long long draw(Generator& gen, const long long& range) {
    // FIXME (mfh 26 Oct 2014) It's legal for long long to be > 64 bits.
    return gen.rand64(range);
  }
  FLARE_INLINE_FUNCTION
  static long long draw(Generator& gen, const long long& start,
                        const long long& end) {
    // FIXME (mfh 26 Oct 2014) It's legal for long long to be > 64 bits.
    return gen.rand64(start, end);
  }
};

// NOTE (mfh 26 oct 2014) This is a partial specialization for
// unsigned long long, a C99 / C++11 unsigned type which is
// guaranteed to be at least 64 bits.  Do NOT write a partial
// specialization for uint64_t!!!  This is just an alias!  It could
// be either unsigned long or unsigned long long.  We don't know
// which a priori, and I've seen both.  The types unsigned long and
// unsigned long long are guaranteed to differ, so it's always safe
// to specialize for both.
template <class Generator>
struct rand<Generator, unsigned long long> {
  FLARE_INLINE_FUNCTION
  static unsigned long long max() {
    // FIXME (mfh 26 Oct 2014) It's legal for unsigned long long to be > 64
    // bits.
    return Generator::MAX_URAND64;
  }
  FLARE_INLINE_FUNCTION
  static unsigned long long draw(Generator& gen) {
    // FIXME (mfh 26 Oct 2014) It's legal for unsigned long long to be > 64
    // bits.
    return gen.urand64();
  }
  FLARE_INLINE_FUNCTION
  static unsigned long long draw(Generator& gen,
                                 const unsigned long long& range) {
    // FIXME (mfh 26 Oct 2014) It's legal for long long to be > 64 bits.
    return gen.urand64(range);
  }
  FLARE_INLINE_FUNCTION
  static unsigned long long draw(Generator& gen,
                                 const unsigned long long& start,
                                 const unsigned long long& end) {
    // FIXME (mfh 26 Oct 2014) It's legal for long long to be > 64 bits.
    return gen.urand64(start, end);
  }
};

#if defined(FLARE_HALF_T_IS_FLOAT) && !FLARE_HALF_T_IS_FLOAT
template <class Generator>
struct rand<Generator, flare::experimental::half_t> {
  using half = flare::experimental::half_t;
  FLARE_INLINE_FUNCTION
  static half max() { return half(1.0); }
  FLARE_INLINE_FUNCTION
  static half draw(Generator& gen) { return half(gen.frand()); }
  FLARE_INLINE_FUNCTION
  static half draw(Generator& gen, const half& range) {
    return half(gen.frand(float(range)));
  }
  FLARE_INLINE_FUNCTION
  static half draw(Generator& gen, const half& start, const half& end) {
    return half(gen.frand(float(start), float(end)));
  }
};
#endif  // defined(FLARE_HALF_T_IS_FLOAT) && !FLARE_HALF_T_IS_FLOAT

#if defined(FLARE_BHALF_T_IS_FLOAT) && !FLARE_BHALF_T_IS_FLOAT
template <class Generator>
struct rand<Generator, flare::experimental::bhalf_t> {
  using bhalf = flare::experimental::bhalf_t;
  FLARE_INLINE_FUNCTION
  static bhalf max() { return bhalf(1.0); }
  FLARE_INLINE_FUNCTION
  static bhalf draw(Generator& gen) { return bhalf(gen.frand()); }
  FLARE_INLINE_FUNCTION
  static bhalf draw(Generator& gen, const bhalf& range) {
    return bhalf(gen.frand(float(range)));
  }
  FLARE_INLINE_FUNCTION
  static bhalf draw(Generator& gen, const bhalf& start, const bhalf& end) {
    return bhalf(gen.frand(float(start), float(end)));
  }
};
#endif  // defined(FLARE_BHALF_T_IS_FLOAT) && !FLARE_BHALF_T_IS_FLOAT

template <class Generator>
struct rand<Generator, float> {
  FLARE_INLINE_FUNCTION
  static float max() { return 1.0f; }
  FLARE_INLINE_FUNCTION
  static float draw(Generator& gen) { return gen.frand(); }
  FLARE_INLINE_FUNCTION
  static float draw(Generator& gen, const float& range) {
    return gen.frand(range);
  }
  FLARE_INLINE_FUNCTION
  static float draw(Generator& gen, const float& start, const float& end) {
    return gen.frand(start, end);
  }
};

template <class Generator>
struct rand<Generator, double> {
  FLARE_INLINE_FUNCTION
  static double max() { return 1.0; }
  FLARE_INLINE_FUNCTION
  static double draw(Generator& gen) { return gen.drand(); }
  FLARE_INLINE_FUNCTION
  static double draw(Generator& gen, const double& range) {
    return gen.drand(range);
  }
  FLARE_INLINE_FUNCTION
  static double draw(Generator& gen, const double& start, const double& end) {
    return gen.drand(start, end);
  }
};

template <class Generator>
struct rand<Generator, flare::complex<float>> {
  FLARE_INLINE_FUNCTION
  static flare::complex<float> max() {
    return flare::complex<float>(1.0, 1.0);
  }
  FLARE_INLINE_FUNCTION
  static flare::complex<float> draw(Generator& gen) {
    const float re = gen.frand();
    const float im = gen.frand();
    return flare::complex<float>(re, im);
  }
  FLARE_INLINE_FUNCTION
  static flare::complex<float> draw(Generator& gen,
                                     const flare::complex<float>& range) {
    const float re = gen.frand(real(range));
    const float im = gen.frand(imag(range));
    return flare::complex<float>(re, im);
  }
  FLARE_INLINE_FUNCTION
  static flare::complex<float> draw(Generator& gen,
                                     const flare::complex<float>& start,
                                     const flare::complex<float>& end) {
    const float re = gen.frand(real(start), real(end));
    const float im = gen.frand(imag(start), imag(end));
    return flare::complex<float>(re, im);
  }
};

template <class Generator>
struct rand<Generator, flare::complex<double>> {
  FLARE_INLINE_FUNCTION
  static flare::complex<double> max() {
    return flare::complex<double>(1.0, 1.0);
  }
  FLARE_INLINE_FUNCTION
  static flare::complex<double> draw(Generator& gen) {
    const double re = gen.drand();
    const double im = gen.drand();
    return flare::complex<double>(re, im);
  }
  FLARE_INLINE_FUNCTION
  static flare::complex<double> draw(Generator& gen,
                                      const flare::complex<double>& range) {
    const double re = gen.drand(real(range));
    const double im = gen.drand(imag(range));
    return flare::complex<double>(re, im);
  }
  FLARE_INLINE_FUNCTION
  static flare::complex<double> draw(Generator& gen,
                                      const flare::complex<double>& start,
                                      const flare::complex<double>& end) {
    const double re = gen.drand(real(start), real(end));
    const double im = gen.drand(imag(start), imag(end));
    return flare::complex<double>(re, im);
  }
};

template <class DeviceType>
class Random_XorShift1024_Pool;

namespace detail {

template <bool UseCArrayState>
struct Random_XorShift1024_State {
  uint64_t state_[16];
  FLARE_DEFAULTED_FUNCTION
  Random_XorShift1024_State() = default;

  template <class StateTensorType>
  FLARE_FUNCTION Random_XorShift1024_State(const StateTensorType& v,
                                            int state_idx) {
    for (int i = 0; i < 16; i++) state_[i] = v(state_idx, i);
  }

  FLARE_FUNCTION
  uint64_t operator[](const int i) const { return state_[i]; }

  FLARE_FUNCTION
  uint64_t& operator[](const int i) { return state_[i]; }
};

template <>
struct Random_XorShift1024_State<false> {
  uint64_t* state_;
  const int stride_;
  FLARE_FUNCTION
  Random_XorShift1024_State() : state_(nullptr), stride_(1){};

  template <class StateTensorType>
  FLARE_FUNCTION Random_XorShift1024_State(const StateTensorType& v,
                                            int state_idx)
      : state_(&v(state_idx, 0)), stride_(v.stride_1()) {}

  FLARE_FUNCTION
  uint64_t operator[](const int i) const { return state_[i * stride_]; }

  FLARE_FUNCTION
  uint64_t& operator[](const int i) { return state_[i * stride_]; }
};

template <class ExecutionSpace>
struct Random_XorShift1024_UseCArrayState : std::true_type {};

#ifdef FLARE_ON_CUDA_DEVICE
template <>
struct Random_XorShift1024_UseCArrayState<flare::Cuda> : std::false_type {};
#endif

template <class DeviceType>
struct Random_UniqueIndex {
  using locks_tensor_type = Tensor<int**, DeviceType>;
  FLARE_FUNCTION
  static int get_state_idx(const locks_tensor_type) {
    FLARE_IF_ON_HOST(
        (return DeviceType::execution_space::impl_hardware_thread_id();))

    FLARE_IF_ON_DEVICE((return 0;))
  }
};

#if defined(FLARE_ON_CUDA_DEVICE)

#define FLARE_IMPL_EXECUTION_SPACE_CUDA flare::Cuda

template <class MemorySpace>
struct Random_UniqueIndex<
    flare::Device<FLARE_IMPL_EXECUTION_SPACE_CUDA, MemorySpace>> {
  using locks_tensor_type =
      Tensor<int**, flare::Device<FLARE_IMPL_EXECUTION_SPACE_CUDA,
                                 MemorySpace>>;
  FLARE_FUNCTION
  static int get_state_idx(const locks_tensor_type& locks_) {
    FLARE_IF_ON_DEVICE((
        const int i_offset =
            (threadIdx.x * blockDim.y + threadIdx.y) * blockDim.z + threadIdx.z;
        int i =
            (((blockIdx.x * gridDim.y + blockIdx.y) * gridDim.z + blockIdx.z) *
                 blockDim.x * blockDim.y * blockDim.z +
             i_offset) %
            locks_.extent(0);
        while (flare::atomic_compare_exchange(&locks_(i, 0), 0, 1)) {
          i += blockDim.x * blockDim.y * blockDim.z;
          if (i >= static_cast<int>(locks_.extent(0))) {
            i = i_offset;
          }
        }

        return i;))
    FLARE_IF_ON_HOST(((void)locks_; return 0;))
  }
};

#undef FLARE_IMPL_EXECUTION_SPACE_CUDA

#endif

}  // namespace detail

template <class DeviceType>
class Random_XorShift64_Pool;

template <class DeviceType>
class Random_XorShift64 {
 private:
  uint64_t state_;
  const int state_idx_;
  friend class Random_XorShift64_Pool<DeviceType>;

 public:
  using device_type = DeviceType;

  constexpr static uint32_t MAX_URAND   = std::numeric_limits<uint32_t>::max();
  constexpr static uint64_t MAX_URAND64 = std::numeric_limits<uint64_t>::max();
  constexpr static int32_t MAX_RAND     = std::numeric_limits<int32_t>::max();
  constexpr static int64_t MAX_RAND64   = std::numeric_limits<int64_t>::max();

  FLARE_INLINE_FUNCTION
  Random_XorShift64(uint64_t state, int state_idx = 0)
      : state_(state == 0 ? uint64_t(1318319) : state), state_idx_(state_idx) {}

  FLARE_INLINE_FUNCTION
  uint32_t urand() {
    state_ ^= state_ >> 12;
    state_ ^= state_ << 25;
    state_ ^= state_ >> 27;

    uint64_t tmp = state_ * 2685821657736338717ULL;
    tmp          = tmp >> 16;
    return static_cast<uint32_t>(tmp & MAX_URAND);
  }

  FLARE_INLINE_FUNCTION
  uint64_t urand64() {
    state_ ^= state_ >> 12;
    state_ ^= state_ << 25;
    state_ ^= state_ >> 27;
    return (state_ * 2685821657736338717ULL) - 1;
  }

  FLARE_INLINE_FUNCTION
  uint32_t urand(const uint32_t& range) {
    const uint32_t max_val = (MAX_URAND / range) * range;
    uint32_t tmp           = urand();
    while (tmp >= max_val) tmp = urand();
    return tmp % range;
  }

  FLARE_INLINE_FUNCTION
  uint32_t urand(const uint32_t& start, const uint32_t& end) {
    return urand(end - start) + start;
  }

  FLARE_INLINE_FUNCTION
  uint64_t urand64(const uint64_t& range) {
    const uint64_t max_val = (MAX_URAND64 / range) * range;
    uint64_t tmp           = urand64();
    while (tmp >= max_val) tmp = urand64();
    return tmp % range;
  }

  FLARE_INLINE_FUNCTION
  uint64_t urand64(const uint64_t& start, const uint64_t& end) {
    return urand64(end - start) + start;
  }

  FLARE_INLINE_FUNCTION
  int rand() { return static_cast<int>(urand() / 2); }

  FLARE_INLINE_FUNCTION
  int rand(const int& range) {
    const int max_val = (MAX_RAND / range) * range;
    int tmp           = rand();
    while (tmp >= max_val) tmp = rand();
    return tmp % range;
  }

  FLARE_INLINE_FUNCTION
  int rand(const int& start, const int& end) {
    return rand(end - start) + start;
  }

  FLARE_INLINE_FUNCTION
  int64_t rand64() { return static_cast<int64_t>(urand64() / 2); }

  FLARE_INLINE_FUNCTION
  int64_t rand64(const int64_t& range) {
    const int64_t max_val = (MAX_RAND64 / range) * range;
    int64_t tmp           = rand64();
    while (tmp >= max_val) tmp = rand64();
    return tmp % range;
  }

  FLARE_INLINE_FUNCTION
  int64_t rand64(const int64_t& start, const int64_t& end) {
    return rand64(end - start) + start;
  }

  FLARE_INLINE_FUNCTION
  float frand() { return urand64() / static_cast<float>(MAX_URAND64); }

  FLARE_INLINE_FUNCTION
  float frand(const float& range) {
    return range * urand64() / static_cast<float>(MAX_URAND64);
  }

  FLARE_INLINE_FUNCTION
  float frand(const float& start, const float& end) {
    return frand(end - start) + start;
  }

  FLARE_INLINE_FUNCTION
  double drand() { return urand64() / static_cast<double>(MAX_URAND64); }

  FLARE_INLINE_FUNCTION
  double drand(const double& range) {
    return range * urand64() / static_cast<double>(MAX_URAND64);
  }

  FLARE_INLINE_FUNCTION
  double drand(const double& start, const double& end) {
    return drand(end - start) + start;
  }

  // Marsaglia polar method for drawing a standard normal distributed random
  // number
  FLARE_INLINE_FUNCTION
  double normal() {
    double S = 2.0;
    double U;
    while (S >= 1.0) {
      U              = 2.0 * drand() - 1.0;
      const double V = 2.0 * drand() - 1.0;
      S              = U * U + V * V;
    }
    return U * std::sqrt(-2.0 * std::log(S) / S);
  }

  FLARE_INLINE_FUNCTION
  double normal(const double& mean, const double& std_dev = 1.0) {
    return mean + normal() * std_dev;
  }
};

template <class DeviceType = flare::DefaultExecutionSpace>
class Random_XorShift64_Pool {
 public:
  using device_type = typename DeviceType::device_type;

 private:
  using execution_space = typename device_type::execution_space;
  using locks_type      = Tensor<int**, device_type>;
  using state_data_type = Tensor<uint64_t**, device_type>;

  locks_type locks_      = {};
  state_data_type state_ = {};
  int num_states_        = {};
  int padding_           = {};

 public:
  using generator_type = Random_XorShift64<DeviceType>;

  Random_XorShift64_Pool()   = default;
  Random_XorShift64_Pool(uint64_t seed) {
    num_states_ = 0;

    init(seed, execution_space().concurrency());
  }

  void init(uint64_t seed, int num_states) {
    if (seed == 0) seed = uint64_t(1318319);
    // I only want to pad on CPU like archs (less than 1000 threads). 64 is a
    // magic number, or random number I just wanted something not too large and
    // not too small. 64 sounded fine.
    padding_    = num_states < 1000 ? 64 : 1;
    num_states_ = num_states;

    locks_ =
        locks_type("flare::Random_XorShift64::locks", num_states, padding_);
    state_ = state_data_type("flare::Random_XorShift64::state", num_states_,
                             padding_);

    typename state_data_type::HostMirror h_state =
        flare::create_mirror_tensor(flare::WithoutInitializing, state_);
    typename locks_type::HostMirror h_lock =
        flare::create_mirror_tensor(flare::WithoutInitializing, locks_);

    // Execute on the HostMirror's default execution space.
    Random_XorShift64<typename state_data_type::HostMirror::execution_space>
        gen(seed, 0);
    for (int i = 0; i < 17; i++) gen.rand();
    for (int i = 0; i < num_states_; i++) {
      int n1        = gen.rand();
      int n2        = gen.rand();
      int n3        = gen.rand();
      int n4        = gen.rand();
      h_state(i, 0) = (((static_cast<uint64_t>(n1)) & 0xffff) << 00) |
                      (((static_cast<uint64_t>(n2)) & 0xffff) << 16) |
                      (((static_cast<uint64_t>(n3)) & 0xffff) << 32) |
                      (((static_cast<uint64_t>(n4)) & 0xffff) << 48);
      h_lock(i, 0) = 0;
    }
    deep_copy(state_, h_state);
    deep_copy(locks_, h_lock);
  }

  FLARE_INLINE_FUNCTION Random_XorShift64<DeviceType> get_state() const {
    FLARE_EXPECTS(num_states_ > 0);
    const int i = detail::Random_UniqueIndex<device_type>::get_state_idx(locks_);
    return Random_XorShift64<DeviceType>(state_(i, 0), i);
  }

  // NOTE: state_idx MUST be unique and less than num_states
  FLARE_INLINE_FUNCTION
  Random_XorShift64<DeviceType> get_state(const int state_idx) const {
    return Random_XorShift64<DeviceType>(state_(state_idx, 0), state_idx);
  }

  FLARE_INLINE_FUNCTION
  void free_state(const Random_XorShift64<DeviceType>& state) const {
    state_(state.state_idx_, 0) = state.state_;
    // Release the lock only after the state has been updated in memory
    flare::memory_fence();
    locks_(state.state_idx_, 0) = 0;
  }
};

template <class DeviceType>
class Random_XorShift1024 {
  using execution_space = typename DeviceType::execution_space;

 private:
  int p_;
  const int state_idx_;
  detail::Random_XorShift1024_State<
      detail::Random_XorShift1024_UseCArrayState<execution_space>::value>
      state_;
  friend class Random_XorShift1024_Pool<DeviceType>;

 public:
  using pool_type   = Random_XorShift1024_Pool<DeviceType>;
  using device_type = DeviceType;

  constexpr static uint32_t MAX_URAND   = std::numeric_limits<uint32_t>::max();
  constexpr static uint64_t MAX_URAND64 = std::numeric_limits<uint64_t>::max();
  constexpr static int32_t MAX_RAND     = std::numeric_limits<int32_t>::max();
  constexpr static int64_t MAX_RAND64   = std::numeric_limits<int64_t>::max();

  FLARE_INLINE_FUNCTION
  Random_XorShift1024(const typename pool_type::state_data_type& state, int p,
                      int state_idx = 0)
      : p_(p), state_idx_(state_idx), state_(state, state_idx) {}

  FLARE_INLINE_FUNCTION
  uint32_t urand() {
    uint64_t state_0 = state_[p_];
    uint64_t state_1 = state_[p_ = (p_ + 1) & 15];
    state_1 ^= state_1 << 31;
    state_1 ^= state_1 >> 11;
    state_0 ^= state_0 >> 30;
    uint64_t tmp = (state_[p_] = state_0 ^ state_1) * 1181783497276652981ULL;
    tmp          = tmp >> 16;
    return static_cast<uint32_t>(tmp & MAX_URAND);
  }

  FLARE_INLINE_FUNCTION
  uint64_t urand64() {
    uint64_t state_0 = state_[p_];
    uint64_t state_1 = state_[p_ = (p_ + 1) & 15];
    state_1 ^= state_1 << 31;
    state_1 ^= state_1 >> 11;
    state_0 ^= state_0 >> 30;
    return ((state_[p_] = state_0 ^ state_1) * 1181783497276652981LL) - 1;
  }

  FLARE_INLINE_FUNCTION
  uint32_t urand(const uint32_t& range) {
    const uint32_t max_val = (MAX_URAND / range) * range;
    uint32_t tmp           = urand();
    while (tmp >= max_val) tmp = urand();
    return tmp % range;
  }

  FLARE_INLINE_FUNCTION
  uint32_t urand(const uint32_t& start, const uint32_t& end) {
    return urand(end - start) + start;
  }

  FLARE_INLINE_FUNCTION
  uint64_t urand64(const uint64_t& range) {
    const uint64_t max_val = (MAX_URAND64 / range) * range;
    uint64_t tmp           = urand64();
    while (tmp >= max_val) tmp = urand64();
    return tmp % range;
  }

  FLARE_INLINE_FUNCTION
  uint64_t urand64(const uint64_t& start, const uint64_t& end) {
    return urand64(end - start) + start;
  }

  FLARE_INLINE_FUNCTION
  int rand() { return static_cast<int>(urand() / 2); }

  FLARE_INLINE_FUNCTION
  int rand(const int& range) {
    const int max_val = (MAX_RAND / range) * range;
    int tmp           = rand();
    while (tmp >= max_val) tmp = rand();
    return tmp % range;
  }

  FLARE_INLINE_FUNCTION
  int rand(const int& start, const int& end) {
    return rand(end - start) + start;
  }

  FLARE_INLINE_FUNCTION
  int64_t rand64() { return static_cast<int64_t>(urand64() / 2); }

  FLARE_INLINE_FUNCTION
  int64_t rand64(const int64_t& range) {
    const int64_t max_val = (MAX_RAND64 / range) * range;
    int64_t tmp           = rand64();
    while (tmp >= max_val) tmp = rand64();
    return tmp % range;
  }

  FLARE_INLINE_FUNCTION
  int64_t rand64(const int64_t& start, const int64_t& end) {
    return rand64(end - start) + start;
  }

  FLARE_INLINE_FUNCTION
  float frand() { return urand64() / static_cast<float>(MAX_URAND64); }

  FLARE_INLINE_FUNCTION
  float frand(const float& range) {
    return range * urand64() / static_cast<float>(MAX_URAND64);
  }

  FLARE_INLINE_FUNCTION
  float frand(const float& start, const float& end) {
    return frand(end - start) + start;
  }

  FLARE_INLINE_FUNCTION
  double drand() { return urand64() / static_cast<double>(MAX_URAND64); }

  FLARE_INLINE_FUNCTION
  double drand(const double& range) {
    return range * urand64() / static_cast<double>(MAX_URAND64);
  }

  FLARE_INLINE_FUNCTION
  double drand(const double& start, const double& end) {
    return drand(end - start) + start;
  }

  // Marsaglia polar method for drawing a standard normal distributed random
  // number
  FLARE_INLINE_FUNCTION
  double normal() {
    double S = 2.0;
    double U;
    while (S >= 1.0) {
      U              = 2.0 * drand() - 1.0;
      const double V = 2.0 * drand() - 1.0;
      S              = U * U + V * V;
    }
    return U * std::sqrt(-2.0 * std::log(S) / S);
  }

  FLARE_INLINE_FUNCTION
  double normal(const double& mean, const double& std_dev = 1.0) {
    return mean + normal() * std_dev;
  }
};

template <class DeviceType = flare::DefaultExecutionSpace>
class Random_XorShift1024_Pool {
 public:
  using device_type = typename DeviceType::device_type;

 private:
  using execution_space = typename device_type::execution_space;
  using locks_type      = Tensor<int**, device_type>;
  using int_tensor_type   = Tensor<int**, device_type>;
  using state_data_type = Tensor<uint64_t * [16], device_type>;

  locks_type locks_      = {};
  state_data_type state_ = {};
  int_tensor_type p_       = {};
  int num_states_        = {};
  int padding_           = {};
  friend class Random_XorShift1024<DeviceType>;

 public:
  using generator_type = Random_XorShift1024<DeviceType>;

  Random_XorShift1024_Pool() = default;

  Random_XorShift1024_Pool(uint64_t seed) {
    num_states_ = 0;

    init(seed, execution_space().concurrency());
  }

  void init(uint64_t seed, int num_states) {
    if (seed == 0) seed = uint64_t(1318319);
    // I only want to pad on CPU like archs (less than 1000 threads). 64 is a
    // magic number, or random number I just wanted something not too large and
    // not too small. 64 sounded fine.
    padding_    = num_states < 1000 ? 64 : 1;
    num_states_ = num_states;
    locks_ =
        locks_type("flare::Random_XorShift1024::locks", num_states_, padding_);
    state_ = state_data_type("flare::Random_XorShift1024::state", num_states_);
    p_ = int_tensor_type("flare::Random_XorShift1024::p", num_states_, padding_);

    typename state_data_type::HostMirror h_state =
        flare::create_mirror_tensor(flare::WithoutInitializing, state_);
    typename locks_type::HostMirror h_lock =
        flare::create_mirror_tensor(flare::WithoutInitializing, locks_);
    typename int_tensor_type::HostMirror h_p =
        flare::create_mirror_tensor(flare::WithoutInitializing, p_);

    // Execute on the HostMirror's default execution space.
    Random_XorShift64<typename state_data_type::HostMirror::execution_space>
        gen(seed, 0);
    for (int i = 0; i < 17; i++) gen.rand();
    for (int i = 0; i < num_states_; i++) {
      for (int j = 0; j < 16; j++) {
        int n1        = gen.rand();
        int n2        = gen.rand();
        int n3        = gen.rand();
        int n4        = gen.rand();
        h_state(i, j) = (((static_cast<uint64_t>(n1)) & 0xffff) << 00) |
                        (((static_cast<uint64_t>(n2)) & 0xffff) << 16) |
                        (((static_cast<uint64_t>(n3)) & 0xffff) << 32) |
                        (((static_cast<uint64_t>(n4)) & 0xffff) << 48);
      }
      h_p(i, 0)    = 0;
      h_lock(i, 0) = 0;
    }
    deep_copy(state_, h_state);
    deep_copy(locks_, h_lock);
  }

  FLARE_INLINE_FUNCTION
  Random_XorShift1024<DeviceType> get_state() const {
    FLARE_EXPECTS(num_states_ > 0);
    const int i = detail::Random_UniqueIndex<device_type>::get_state_idx(locks_);
    return Random_XorShift1024<DeviceType>(state_, p_(i, 0), i);
  };

  // NOTE: state_idx MUST be unique and less than num_states
  FLARE_INLINE_FUNCTION
  Random_XorShift1024<DeviceType> get_state(const int state_idx) const {
    return Random_XorShift1024<DeviceType>(state_, p_(state_idx, 0), state_idx);
  }

  FLARE_INLINE_FUNCTION
  void free_state(const Random_XorShift1024<DeviceType>& state) const {
    for (int i = 0; i < 16; i++) state_(state.state_idx_, i) = state.state_[i];
    p_(state.state_idx_, 0) = state.p_;
    // Release the lock only after the state has been updated in memory
    flare::memory_fence();
    locks_(state.state_idx_, 0) = 0;
  }
};

namespace detail {

template <class TensorType, class RandomPool, int loops, int rank,
          class IndexType>
struct fill_random_functor_begin_end;

template <class TensorType, class RandomPool, int loops, class IndexType>
struct fill_random_functor_begin_end<TensorType, RandomPool, loops, 0,
                                     IndexType> {
  TensorType a;
  RandomPool rand_pool;
  typename TensorType::const_value_type begin, end;

  using Rand = rand<typename RandomPool::generator_type,
                    typename TensorType::non_const_value_type>;

  fill_random_functor_begin_end(TensorType a_, RandomPool rand_pool_,
                                typename TensorType::const_value_type begin_,
                                typename TensorType::const_value_type end_)
      : a(a_), rand_pool(rand_pool_), begin(begin_), end(end_) {}

  FLARE_INLINE_FUNCTION
  void operator()(IndexType) const {
    typename RandomPool::generator_type gen = rand_pool.get_state();
    a()                                     = Rand::draw(gen, begin, end);
    rand_pool.free_state(gen);
  }
};

template <class TensorType, class RandomPool, int loops, class IndexType>
struct fill_random_functor_begin_end<TensorType, RandomPool, loops, 1,
                                     IndexType> {
  TensorType a;
  RandomPool rand_pool;
  typename TensorType::const_value_type begin, end;

  using Rand = rand<typename RandomPool::generator_type,
                    typename TensorType::non_const_value_type>;

  fill_random_functor_begin_end(TensorType a_, RandomPool rand_pool_,
                                typename TensorType::const_value_type begin_,
                                typename TensorType::const_value_type end_)
      : a(a_), rand_pool(rand_pool_), begin(begin_), end(end_) {}

  FLARE_INLINE_FUNCTION
  void operator()(IndexType i) const {
    typename RandomPool::generator_type gen = rand_pool.get_state();
    for (IndexType j = 0; j < loops; j++) {
      const IndexType idx = i * loops + j;
      if (idx < static_cast<IndexType>(a.extent(0)))
        a(idx) = Rand::draw(gen, begin, end);
    }
    rand_pool.free_state(gen);
  }
};

template <class TensorType, class RandomPool, int loops, class IndexType>
struct fill_random_functor_begin_end<TensorType, RandomPool, loops, 2,
                                     IndexType> {
  TensorType a;
  RandomPool rand_pool;
  typename TensorType::const_value_type begin, end;

  using Rand = rand<typename RandomPool::generator_type,
                    typename TensorType::non_const_value_type>;

  fill_random_functor_begin_end(TensorType a_, RandomPool rand_pool_,
                                typename TensorType::const_value_type begin_,
                                typename TensorType::const_value_type end_)
      : a(a_), rand_pool(rand_pool_), begin(begin_), end(end_) {}

  FLARE_INLINE_FUNCTION
  void operator()(IndexType i) const {
    typename RandomPool::generator_type gen = rand_pool.get_state();
    for (IndexType j = 0; j < loops; j++) {
      const IndexType idx = i * loops + j;
      if (idx < static_cast<IndexType>(a.extent(0))) {
        for (IndexType k = 0; k < static_cast<IndexType>(a.extent(1)); k++)
          a(idx, k) = Rand::draw(gen, begin, end);
      }
    }
    rand_pool.free_state(gen);
  }
};

template <class TensorType, class RandomPool, int loops, class IndexType>
struct fill_random_functor_begin_end<TensorType, RandomPool, loops, 3,
                                     IndexType> {
  TensorType a;
  RandomPool rand_pool;
  typename TensorType::const_value_type begin, end;

  using Rand = rand<typename RandomPool::generator_type,
                    typename TensorType::non_const_value_type>;

  fill_random_functor_begin_end(TensorType a_, RandomPool rand_pool_,
                                typename TensorType::const_value_type begin_,
                                typename TensorType::const_value_type end_)
      : a(a_), rand_pool(rand_pool_), begin(begin_), end(end_) {}

  FLARE_INLINE_FUNCTION
  void operator()(IndexType i) const {
    typename RandomPool::generator_type gen = rand_pool.get_state();
    for (IndexType j = 0; j < loops; j++) {
      const IndexType idx = i * loops + j;
      if (idx < static_cast<IndexType>(a.extent(0))) {
        for (IndexType k = 0; k < static_cast<IndexType>(a.extent(1)); k++)
          for (IndexType l = 0; l < static_cast<IndexType>(a.extent(2)); l++)
            a(idx, k, l) = Rand::draw(gen, begin, end);
      }
    }
    rand_pool.free_state(gen);
  }
};

template <class TensorType, class RandomPool, int loops, class IndexType>
struct fill_random_functor_begin_end<TensorType, RandomPool, loops, 4,
                                     IndexType> {
  TensorType a;
  RandomPool rand_pool;
  typename TensorType::const_value_type begin, end;

  using Rand = rand<typename RandomPool::generator_type,
                    typename TensorType::non_const_value_type>;

  fill_random_functor_begin_end(TensorType a_, RandomPool rand_pool_,
                                typename TensorType::const_value_type begin_,
                                typename TensorType::const_value_type end_)
      : a(a_), rand_pool(rand_pool_), begin(begin_), end(end_) {}

  FLARE_INLINE_FUNCTION
  void operator()(IndexType i) const {
    typename RandomPool::generator_type gen = rand_pool.get_state();
    for (IndexType j = 0; j < loops; j++) {
      const IndexType idx = i * loops + j;
      if (idx < static_cast<IndexType>(a.extent(0))) {
        for (IndexType k = 0; k < static_cast<IndexType>(a.extent(1)); k++)
          for (IndexType l = 0; l < static_cast<IndexType>(a.extent(2)); l++)
            for (IndexType m = 0; m < static_cast<IndexType>(a.extent(3)); m++)
              a(idx, k, l, m) = Rand::draw(gen, begin, end);
      }
    }
    rand_pool.free_state(gen);
  }
};

template <class TensorType, class RandomPool, int loops, class IndexType>
struct fill_random_functor_begin_end<TensorType, RandomPool, loops, 5,
                                     IndexType> {
  TensorType a;
  RandomPool rand_pool;
  typename TensorType::const_value_type begin, end;

  using Rand = rand<typename RandomPool::generator_type,
                    typename TensorType::non_const_value_type>;

  fill_random_functor_begin_end(TensorType a_, RandomPool rand_pool_,
                                typename TensorType::const_value_type begin_,
                                typename TensorType::const_value_type end_)
      : a(a_), rand_pool(rand_pool_), begin(begin_), end(end_) {}

  FLARE_INLINE_FUNCTION
  void operator()(IndexType i) const {
    typename RandomPool::generator_type gen = rand_pool.get_state();
    for (IndexType j = 0; j < loops; j++) {
      const IndexType idx = i * loops + j;
      if (idx < static_cast<IndexType>(a.extent(0))) {
        for (IndexType l = 0; l < static_cast<IndexType>(a.extent(1)); l++)
          for (IndexType m = 0; m < static_cast<IndexType>(a.extent(2)); m++)
            for (IndexType n = 0; n < static_cast<IndexType>(a.extent(3)); n++)
              for (IndexType o = 0; o < static_cast<IndexType>(a.extent(4));
                   o++)
                a(idx, l, m, n, o) = Rand::draw(gen, begin, end);
      }
    }
    rand_pool.free_state(gen);
  }
};

template <class TensorType, class RandomPool, int loops, class IndexType>
struct fill_random_functor_begin_end<TensorType, RandomPool, loops, 6,
                                     IndexType> {
  TensorType a;
  RandomPool rand_pool;
  typename TensorType::const_value_type begin, end;

  using Rand = rand<typename RandomPool::generator_type,
                    typename TensorType::non_const_value_type>;

  fill_random_functor_begin_end(TensorType a_, RandomPool rand_pool_,
                                typename TensorType::const_value_type begin_,
                                typename TensorType::const_value_type end_)
      : a(a_), rand_pool(rand_pool_), begin(begin_), end(end_) {}

  FLARE_INLINE_FUNCTION
  void operator()(IndexType i) const {
    typename RandomPool::generator_type gen = rand_pool.get_state();
    for (IndexType j = 0; j < loops; j++) {
      const IndexType idx = i * loops + j;
      if (idx < static_cast<IndexType>(a.extent(0))) {
        for (IndexType k = 0; k < static_cast<IndexType>(a.extent(1)); k++)
          for (IndexType l = 0; l < static_cast<IndexType>(a.extent(2)); l++)
            for (IndexType m = 0; m < static_cast<IndexType>(a.extent(3)); m++)
              for (IndexType n = 0; n < static_cast<IndexType>(a.extent(4));
                   n++)
                for (IndexType o = 0; o < static_cast<IndexType>(a.extent(5));
                     o++)
                  a(idx, k, l, m, n, o) = Rand::draw(gen, begin, end);
      }
    }
    rand_pool.free_state(gen);
  }
};

template <class TensorType, class RandomPool, int loops, class IndexType>
struct fill_random_functor_begin_end<TensorType, RandomPool, loops, 7,
                                     IndexType> {
  TensorType a;
  RandomPool rand_pool;
  typename TensorType::const_value_type begin, end;

  using Rand = rand<typename RandomPool::generator_type,
                    typename TensorType::non_const_value_type>;

  fill_random_functor_begin_end(TensorType a_, RandomPool rand_pool_,
                                typename TensorType::const_value_type begin_,
                                typename TensorType::const_value_type end_)
      : a(a_), rand_pool(rand_pool_), begin(begin_), end(end_) {}

  FLARE_INLINE_FUNCTION
  void operator()(IndexType i) const {
    typename RandomPool::generator_type gen = rand_pool.get_state();
    for (IndexType j = 0; j < loops; j++) {
      const IndexType idx = i * loops + j;
      if (idx < static_cast<IndexType>(a.extent(0))) {
        for (IndexType k = 0; k < static_cast<IndexType>(a.extent(1)); k++)
          for (IndexType l = 0; l < static_cast<IndexType>(a.extent(2)); l++)
            for (IndexType m = 0; m < static_cast<IndexType>(a.extent(3)); m++)
              for (IndexType n = 0; n < static_cast<IndexType>(a.extent(4));
                   n++)
                for (IndexType o = 0; o < static_cast<IndexType>(a.extent(5));
                     o++)
                  for (IndexType p = 0; p < static_cast<IndexType>(a.extent(6));
                       p++)
                    a(idx, k, l, m, n, o, p) = Rand::draw(gen, begin, end);
      }
    }
    rand_pool.free_state(gen);
  }
};

template <class TensorType, class RandomPool, int loops, class IndexType>
struct fill_random_functor_begin_end<TensorType, RandomPool, loops, 8,
                                     IndexType> {
  TensorType a;
  RandomPool rand_pool;
  typename TensorType::const_value_type begin, end;

  using Rand = rand<typename RandomPool::generator_type,
                    typename TensorType::non_const_value_type>;

  fill_random_functor_begin_end(TensorType a_, RandomPool rand_pool_,
                                typename TensorType::const_value_type begin_,
                                typename TensorType::const_value_type end_)
      : a(a_), rand_pool(rand_pool_), begin(begin_), end(end_) {}

  FLARE_INLINE_FUNCTION
  void operator()(IndexType i) const {
    typename RandomPool::generator_type gen = rand_pool.get_state();
    for (IndexType j = 0; j < loops; j++) {
      const IndexType idx = i * loops + j;
      if (idx < static_cast<IndexType>(a.extent(0))) {
        for (IndexType k = 0; k < static_cast<IndexType>(a.extent(1)); k++)
          for (IndexType l = 0; l < static_cast<IndexType>(a.extent(2)); l++)
            for (IndexType m = 0; m < static_cast<IndexType>(a.extent(3)); m++)
              for (IndexType n = 0; n < static_cast<IndexType>(a.extent(4));
                   n++)
                for (IndexType o = 0; o < static_cast<IndexType>(a.extent(5));
                     o++)
                  for (IndexType p = 0; p < static_cast<IndexType>(a.extent(6));
                       p++)
                    for (IndexType q = 0;
                         q < static_cast<IndexType>(a.extent(7)); q++)
                      a(idx, k, l, m, n, o, p, q) = Rand::draw(gen, begin, end);
      }
    }
    rand_pool.free_state(gen);
  }
};

template <class ExecutionSpace, class TensorType, class RandomPool,
          class IndexType = int64_t>
void fill_random(const ExecutionSpace& exec, TensorType a, RandomPool g,
                 typename TensorType::const_value_type begin,
                 typename TensorType::const_value_type end) {
  int64_t LDA = a.extent(0);
  if (LDA > 0)
    parallel_for(
        "flare::fill_random",
        flare::RangePolicy<ExecutionSpace>(exec, 0, (LDA + 127) / 128),
        detail::fill_random_functor_begin_end<TensorType, RandomPool, 128,
                                            TensorType::rank, IndexType>(
            a, g, begin, end));
}

}  // namespace detail

template <class ExecutionSpace, class TensorType, class RandomPool,
          class IndexType = int64_t>
void fill_random(const ExecutionSpace& exec, TensorType a, RandomPool g,
                 typename TensorType::const_value_type begin,
                 typename TensorType::const_value_type end) {
  detail::apply_to_tensor_of_static_rank(
      [&](auto dst) { flare::detail::fill_random(exec, dst, g, begin, end); },
      a);
}

template <class ExecutionSpace, class TensorType, class RandomPool,
          class IndexType = int64_t>
void fill_random(const ExecutionSpace& exec, TensorType a, RandomPool g,
                 typename TensorType::const_value_type range) {
  fill_random(exec, a, g, 0, range);
}

template <class TensorType, class RandomPool, class IndexType = int64_t>
void fill_random(TensorType a, RandomPool g,
                 typename TensorType::const_value_type begin,
                 typename TensorType::const_value_type end) {
  fill_random(typename TensorType::execution_space{}, a, g, begin, end);
}

template <class TensorType, class RandomPool, class IndexType = int64_t>
void fill_random(TensorType a, RandomPool g,
                 typename TensorType::const_value_type range) {
  fill_random(typename TensorType::execution_space{}, a, g, 0, range);
}

}  // namespace flare

#endif  // FLARE_RANDOM_H_
