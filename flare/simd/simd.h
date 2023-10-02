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


#ifndef FLARE_SIMD_SIMD_H_
#define FLARE_SIMD_SIMD_H_

#include <flare/simd/common.h>

// suppress NVCC warnings with the [[nodiscard]] attribute on overloaded
// operators implemented as hidden friends
#if defined(FLARE_COMPILER_NVCC) && FLARE_COMPILER_NVCC < 1130
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
#endif

#include <flare/simd/scalar.h>

#include <flare/core/defines.h>

#if defined(FLARE_ARCH_AVX) && !defined(__AVX__)
#error "__AVX__ must be defined for FLARE_ARCH_AVX"
#endif

#if defined(FLARE_ARCH_AVX2)
#if !defined(__AVX2__)
#error "__AVX2__ must be defined for FLARE_ARCH_AVX2"
#endif
#include <flare/simd/avx2.h>
#endif

#if defined(FLARE_ARCH_AVX512XEON)
#if !defined(__AVX512F__)
#error "__AVX512F__ must be defined for FLARE_ARCH_AVX512XEON"
#endif
#include <flare/simd/avx512.h>
#endif

#ifdef __ARM_NEON
#include <flare/simd/neon.h>
#endif

#if defined(FLARE_COMPILER_NVCC) && FLARE_COMPILER_NVCC < 1130
#pragma GCC diagnostic pop
#endif

namespace flare {
namespace experimental {

namespace simd_abi {

namespace detail {

#if defined(FLARE_ARCH_AVX512XEON)
using host_native = avx512_fixed_size<8>;
#elif defined(FLARE_ARCH_AVX2)
using host_native  = avx2_fixed_size<4>;
#elif defined(__ARM_NEON)
using host_native  = neon_fixed_size<2>;
#else
using host_native   = scalar;
#endif

template <class T>
struct ForSpace;

#ifdef FLARE_ENABLE_SERIAL
template <>
struct ForSpace<flare::Serial> {
  using type = host_native;
};
#endif

#ifdef FLARE_ON_CUDA_DEVICE
template <>
struct ForSpace<flare::Cuda> {
  using type = scalar;
};
#endif

#ifdef FLARE_ENABLE_THREADS
template <>
struct ForSpace<flare::Threads> {
  using type = host_native;
};
#endif

#ifdef FLARE_ENABLE_OPENMP
template <>
struct ForSpace<flare::OpenMP> {
  using type = host_native;
};
#endif


}  // namespace detail

template <class Space>
using ForSpace = typename detail::ForSpace<typename Space::execution_space>::type;

template <class T>
using native = ForSpace<flare::DefaultExecutionSpace>;

}  // namespace simd_abi

template <class T>
using native_simd = simd<T, simd_abi::native<T>>;
template <class T>
using native_simd_mask = simd_mask<T, simd_abi::native<T>>;

namespace detail {

template <class... Abis>
class abi_set {};

template <typename... Ts>
class data_types {};

#if defined(FLARE_ARCH_AVX512XEON)
using host_abi_set  = abi_set<simd_abi::scalar, simd_abi::avx512_fixed_size<8>>;
using data_type_set = data_types<std::int32_t, std::uint32_t, std::int64_t,
                                 std::uint64_t, double, float>;
#elif defined(FLARE_ARCH_AVX2)
using host_abi_set = abi_set<simd_abi::scalar, simd_abi::avx2_fixed_size<4>>;
using data_type_set =
    data_types<std::int32_t, std::int64_t, std::uint64_t, double, float>;
#elif defined(__ARM_NEON)
using host_abi_set = abi_set<simd_abi::scalar, simd_abi::neon_fixed_size<2>>;
using data_type_set =
    data_types<std::int32_t, std::int64_t, std::uint64_t, double, float>;
#else
using host_abi_set  = abi_set<simd_abi::scalar>;
using data_type_set = data_types<std::int32_t, std::uint32_t, std::int64_t,
                                 std::uint64_t, double, float>;
#endif

using device_abi_set = abi_set<simd_abi::scalar>;

}  // namespace detail

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_SIMD_SIMD_H_
