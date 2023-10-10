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


#ifndef FLARE_CORE_BIT_UTILS_H_
#define FLARE_CORE_BIT_UTILS_H_

#include <flare/core/defines.h>

namespace flare::detail {

// POP COUNT function returns the number of set bits
#if defined(FLARE_ON_CUDA_DEVICE)
    FLARE_FORCEINLINE_FUNCTION int pop_count(unsigned i) { return __popc(i); }

    FLARE_FORCEINLINE_FUNCTION
    int pop_count(unsigned long i) { return __popcll(i); }

    FLARE_FORCEINLINE_FUNCTION
    int pop_count(unsigned long long i) { return __popcll(i); }

    FLARE_FORCEINLINE_FUNCTION
    int pop_count(int i) { return __popc(i); }

    FLARE_FORCEINLINE_FUNCTION
    int pop_count(long i) { return __popcll(i); }

    FLARE_FORCEINLINE_FUNCTION
    int pop_count(long long i) { return __popcll(i); }

#elif defined(__INTEL_COMPILER)
    FLARE_FORCEINLINE_FUNCTION
    int pop_count(unsigned i) { return _popcnt32(i); }
    FLARE_FORCEINLINE_FUNCTION
    int pop_count(unsigned long i) { return _popcnt64(i); }

    FLARE_FORCEINLINE_FUNCTION
    int pop_count(unsigned long long i) { return _popcnt64(i); }

    FLARE_FORCEINLINE_FUNCTION
    int pop_count(int i) { return _popcnt32(i); }

    FLARE_FORCEINLINE_FUNCTION
    int pop_count(long i) { return _popcnt64(i); }

    FLARE_FORCEINLINE_FUNCTION
    int pop_count(long long i) { return _popcnt64(i); }

#elif defined(__GNUC__) || defined(__GNUG__)

    FLARE_FORCEINLINE_FUNCTION
    int pop_count(unsigned i) { return __builtin_popcount(i); }

    FLARE_FORCEINLINE_FUNCTION
    int pop_count(unsigned long i) { return __builtin_popcountl(i); }

    FLARE_FORCEINLINE_FUNCTION
    int pop_count(unsigned long long i) { return __builtin_popcountll(i); }

    FLARE_FORCEINLINE_FUNCTION
    int pop_count(int i) { return __builtin_popcount(i); }

    FLARE_FORCEINLINE_FUNCTION
    int pop_count(long i) { return __builtin_popcountl(i); }

    FLARE_FORCEINLINE_FUNCTION
    int pop_count(long long i) { return __builtin_popcountll(i); }

#elif defined(__ibmxl_vrm__)
    // See
    // https://www.ibm.com/support/knowledgecenter/SSGH3R_16.1.0/com.ibm.xlcpp161.aix.doc/compiler_ref/compiler_builtins.html
    // link gives info about builtin names for xlclang++
    FLARE_FORCEINLINE_FUNCTION
    int pop_count(unsigned i) { return __builtin_popcnt4(i); }
    FLARE_FORCEINLINE_FUNCTION
    int pop_count(unsigned long i) { return __builtin_popcnt8(i); }

    FLARE_FORCEINLINE_FUNCTION
    int pop_count(unsigned long long i) { return __builtin_popcnt8(i); }

    FLARE_FORCEINLINE_FUNCTION
    int pop_count(int i) { return __builtin_popcnt4(i); }

    FLARE_FORCEINLINE_FUNCTION
    int pop_count(long i) { return __builtin_popcnt8(i); }

    FLARE_FORCEINLINE_FUNCTION
    int pop_count(long long i) { return __builtin_popcnt8(i); }

    #elif defined(__IBMCPP__) || defined(__IBMC__)
    FLARE_FORCEINLINE_FUNCTION
    int pop_count(unsigned i) { return __popcnt4(i); }
    FLARE_FORCEINLINE_FUNCTION
    int pop_count(unsigned long i) { return __popcnt8(i); }

    FLARE_FORCEINLINE_FUNCTION
    int pop_count(unsigned long long i) { return __popcnt8(i); }

    FLARE_FORCEINLINE_FUNCTION
    int pop_count(int i) { return __popcnt4(i); }

    FLARE_FORCEINLINE_FUNCTION
    int pop_count(long i) { return __popcnt8(i); }

    FLARE_FORCEINLINE_FUNCTION
    int pop_count(long long i) { return __popcnt8(i); }

#elif defined(FLARE_COMPILER_MSVC)
    FLARE_FORCEINLINE_FUNCTION
    int pop_count(unsigned i) { return __popcnt(i); }
    FLARE_FORCEINLINE_FUNCTION
    int pop_count(unsigned long i) { return __popcnt(i); }

    FLARE_FORCEINLINE_FUNCTION
    int pop_count(unsigned long long i) { return __popcnt64(i); }

    FLARE_FORCEINLINE_FUNCTION
    int pop_count(int i) { return __popcnt(i); }

    FLARE_FORCEINLINE_FUNCTION
    int pop_count(long i) { return __popcnt(i); }

    FLARE_FORCEINLINE_FUNCTION
    int pop_count(long long i) { return __popcnt64(i); }

#else
#error "Popcount function is not defined for this compiler. Please report this with the compiler you are using to flare."
#endif

// least_set_bit function returns the position of right most set bit

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    FLARE_FORCEINLINE_FUNCTION
int least_set_bit(unsigned i) { return __ffs(i); }

FLARE_FORCEINLINE_FUNCTION
int least_set_bit(unsigned long i) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __ffsll(static_cast<unsigned long long>(i));
#else
  return __ffsll(i);
#endif
}

FLARE_FORCEINLINE_FUNCTION
int least_set_bit(unsigned long long i) { return __ffsll(i); }

FLARE_FORCEINLINE_FUNCTION
int least_set_bit(int i) { return __ffs(i); }

FLARE_FORCEINLINE_FUNCTION
int least_set_bit(long i) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __ffsll(static_cast<long long>(i));
#else
  return __ffsll(i);
#endif
}

FLARE_FORCEINLINE_FUNCTION
int least_set_bit(long long i) { return __ffsll(i); }

/*
#elif defined ( __INTEL_COMPILER )
FLARE_FORCEINLINE_FUNCTION
int least_set_bit( unsigned i ){
  return _bit_scan_forward(i) + 1;
}


FLARE_FORCEINLINE_FUNCTION
int least_set_bit( unsigned long long i ){
  const int llsize = sizeof(unsigned long long) * 8;
  const int intsize = sizeof(int) * 8;
  const int iteration = llsize / intsize;
  unsigned long long tmp = i;

  for (int j = 0; j < iteration; ++j){
    unsigned castint = (tmp >> (intsize * j));
    if (castint) return least_set_bit(castint) + intsize * j;
  }
  return -1;
}

FLARE_FORCEINLINE_FUNCTION
int least_set_bit( unsigned long i ){
  return least_set_bit( unsigned long long(i) );
}



FLARE_FORCEINLINE_FUNCTION
int least_set_bit(int i ){
  return _bit_scan_forward(i) + 1;
}

FLARE_FORCEINLINE_FUNCTION
int least_set_bit( long i ){
  return least_set_bit( unsigned long long(i) );
}


FLARE_FORCEINLINE_FUNCTION
int least_set_bit( long long i ){
  return least_set_bit( unsigned long long(i) );
}
*/

#elif defined(__INTEL_COMPILER) || defined(FLARE_COMPILER_IBM) || \
    defined(__GNUC__) || defined(__GNUG__)

    FLARE_FORCEINLINE_FUNCTION
    int least_set_bit(unsigned i) { return __builtin_ffs(i); }

    FLARE_FORCEINLINE_FUNCTION
    int least_set_bit(unsigned long i) { return __builtin_ffsl(i); }

    FLARE_FORCEINLINE_FUNCTION
    int least_set_bit(unsigned long long i) { return __builtin_ffsll(i); }

    FLARE_FORCEINLINE_FUNCTION
    int least_set_bit(int i) { return __builtin_ffs(i); }

    FLARE_FORCEINLINE_FUNCTION
    int least_set_bit(long i) { return __builtin_ffsl(i); }

    FLARE_FORCEINLINE_FUNCTION
    int least_set_bit(long long i) { return __builtin_ffsll(i); }

#elif defined(FLARE_COMPILER_MSVC)
    FLARE_FORCEINLINE_FUNCTION
    int least_set_bit(unsigned i) { return __lzcnt(i); }
    FLARE_FORCEINLINE_FUNCTION
    int least_set_bit(unsigned long i) { return __lzcnt(i); }

    FLARE_FORCEINLINE_FUNCTION
    int least_set_bit(unsigned long long i) { return __lzcnt64(i); }

    FLARE_FORCEINLINE_FUNCTION
    int least_set_bit(int i) { return __lzcnt(i); }
    FLARE_FORCEINLINE_FUNCTION
    int least_set_bit(long i) { return __lzcnt(i); }

    FLARE_FORCEINLINE_FUNCTION
    int least_set_bit(long long i) { return __lzcnt64(i); }

#else
#error "least_set_bit function is not defined for this compiler. Please report this with the compiler you are using to flare."
#endif

}  // namespace flare::detail

#endif  // FLARE_CORE_BIT_UTILS_H_
