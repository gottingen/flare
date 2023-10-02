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


#ifndef FLARE_CORE_ATOMIC_FETCH_OP_CUDA_H_
#define FLARE_CORE_ATOMIC_FETCH_OP_CUDA_H_

#ifndef FLARE_CUDA_ARCH_IS_PRE_VOLTA

#define FLARE_HAVE_CUDA_ATOMICS_ASM

#include <flare/core/atomic/cuda/cuda_asm.h>

#else

namespace flare {
namespace detail {

// clang-format off
inline __device__                int device_atomic_fetch_add(               int* ptr,                int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAdd(ptr,  val); }
inline __device__       unsigned int device_atomic_fetch_add(      unsigned int* ptr,       unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAdd(ptr,  val); }
inline __device__ unsigned long long device_atomic_fetch_add(unsigned long long* ptr, unsigned long long val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAdd(ptr,  val); }
inline __device__              float device_atomic_fetch_add(             float* ptr,              float val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAdd(ptr,  val); }
#ifndef FLARE_CUDA_ARCH_IS_PRE_PASCAL
inline __device__             double device_atomic_fetch_add(            double* ptr,             double val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAdd(ptr,  val); }
#endif

inline __device__                int device_atomic_fetch_sub(               int* ptr,                int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicSub(ptr,  val); }
inline __device__       unsigned int device_atomic_fetch_sub(      unsigned int* ptr,       unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicSub(ptr,  val); }
inline __device__ unsigned long long device_atomic_fetch_sub(unsigned long long* ptr, unsigned long long val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAdd(ptr, -val); }
inline __device__              float device_atomic_fetch_sub(             float* ptr,              float val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAdd(ptr, -val); }
#ifndef FLARE_CUDA_ARCH_IS_PRE_PASCAL
inline __device__             double device_atomic_fetch_sub(            double* ptr,             double val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAdd(ptr, -val); }
#endif

inline __device__                int device_atomic_fetch_min(               int* ptr,                int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicMin(ptr,  val); }
inline __device__       unsigned int device_atomic_fetch_min(      unsigned int* ptr,       unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicMin(ptr,  val); }
inline __device__ unsigned long long device_atomic_fetch_min(unsigned long long* ptr, unsigned long long val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicMin(ptr,  val); }

inline __device__                int device_atomic_fetch_max(               int* ptr,                int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicMax(ptr,  val); }
inline __device__       unsigned int device_atomic_fetch_max(      unsigned int* ptr,       unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicMax(ptr,  val); }
inline __device__ unsigned long long device_atomic_fetch_max(unsigned long long* ptr, unsigned long long val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicMax(ptr,  val); }

inline __device__                int device_atomic_fetch_and(               int* ptr,                int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAnd(ptr,  val); }
inline __device__       unsigned int device_atomic_fetch_and(      unsigned int* ptr,       unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAnd(ptr,  val); }
inline __device__ unsigned long long device_atomic_fetch_and(unsigned long long* ptr, unsigned long long val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAnd(ptr,  val); }

inline __device__                int device_atomic_fetch_or (               int* ptr,                int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicOr (ptr,  val); }
inline __device__       unsigned int device_atomic_fetch_or (      unsigned int* ptr,       unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicOr (ptr,  val); }
inline __device__ unsigned long long device_atomic_fetch_or (unsigned long long* ptr, unsigned long long val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicOr (ptr,  val); }

inline __device__                int device_atomic_fetch_xor(               int* ptr,                int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicXor(ptr,  val); }
inline __device__       unsigned int device_atomic_fetch_xor(      unsigned int* ptr,       unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicXor(ptr,  val); }
inline __device__ unsigned long long device_atomic_fetch_xor(unsigned long long* ptr, unsigned long long val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicXor(ptr,  val); }

inline __device__                int device_atomic_fetch_inc(               int* ptr,                         MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAdd(ptr, 1   ); }
inline __device__       unsigned int device_atomic_fetch_inc(      unsigned int* ptr,                         MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAdd(ptr, 1u  ); }
inline __device__ unsigned long long device_atomic_fetch_inc(unsigned long long* ptr,                         MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAdd(ptr, 1ull); }

inline __device__                int device_atomic_fetch_dec(               int* ptr,                         MemoryOrderRelaxed, MemoryScopeDevice) { return atomicSub(ptr,  1  ); }
inline __device__       unsigned int device_atomic_fetch_dec(      unsigned int* ptr,                         MemoryOrderRelaxed, MemoryScopeDevice) { return atomicSub(ptr,  1u ); }
inline __device__ unsigned long long device_atomic_fetch_dec(unsigned long long* ptr,                         MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAdd(ptr, -1ull);}

inline __device__       unsigned int device_atomic_fetch_inc_mod(  unsigned int* ptr,       unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicInc(ptr,  val); }
inline __device__       unsigned int device_atomic_fetch_dec_mod(  unsigned int* ptr,       unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicDec(ptr,  val); }
// clang-format on

#define FLARE_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP(OP, TYPE)                               \
  template <class MemoryOrder>                                                         \
  __device__ TYPE device_atomic_fetch_##OP(                                            \
      TYPE* ptr, TYPE val, MemoryOrder, MemoryScopeDevice) {                           \
    __threadfence();                                                                   \
    TYPE return_val =                                                                  \
        device_atomic_fetch_##OP(ptr, val, MemoryOrderRelaxed(), MemoryScopeDevice()); \
    __threadfence();                                                                   \
    return return_val;                                                                 \
  }                                                                                    \
  template <class MemoryOrder>                                                         \
  __device__ TYPE device_atomic_fetch_##OP(                                            \
      TYPE* ptr, TYPE val, MemoryOrder, MemoryScopeCore) {                             \
    return device_atomic_fetch_##OP(ptr, val, MemoryOrder(), MemoryScopeDevice());     \
  }

#define FLARE_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP_INTEGRAL(OP) \
  FLARE_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP(OP, int)           \
  FLARE_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP(OP, unsigned int)  \
  FLARE_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP(OP, unsigned long long)

#ifdef FLARE_CUDA_ARCH_IS_PRE_PASCAL

#define FLARE_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP_FLOATING_POINT(OP) \
  FLARE_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP(OP, float)

#else

#define FLARE_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP_FLOATING_POINT(OP) \
  FLARE_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP(OP, float)               \
  FLARE_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP(OP, double)

#endif

FLARE_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP_INTEGRAL(min)
FLARE_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP_INTEGRAL(max)
FLARE_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP_INTEGRAL(and)
FLARE_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP_INTEGRAL(or)
FLARE_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP_INTEGRAL(xor)

FLARE_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP_FLOATING_POINT(add)
FLARE_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP_INTEGRAL(add)
FLARE_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP_FLOATING_POINT(sub)
FLARE_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP_INTEGRAL(sub)

FLARE_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP_INTEGRAL(inc)
FLARE_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP_INTEGRAL(dec)

FLARE_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP(inc_mod, unsigned int)
FLARE_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP(dec_mod, unsigned int)

#undef FLARE_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP_FLOATING_POINT
#undef FLARE_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP_INTEGRAL
#undef FLARE_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP

}  // namespace detail
}  // namespace flare

#endif

#endif  // FLARE_CORE_ATOMIC_FETCH_OP_CUDA_H_
