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

#ifndef FLARE_REDUCTION_IDENTITY_H_
#define FLARE_REDUCTION_IDENTITY_H_
#include <flare/core/defines.h>
#include <cfloat>
#include <climits>

namespace flare {

template <class T>
struct reduction_identity; /*{
  FLARE_FORCEINLINE_FUNCTION constexpr static T sum() { return T(); }  // 0
  FLARE_FORCEINLINE_FUNCTION constexpr static T prod()  // 1
    { static_assert( false, "Missing specialization of
flare::reduction_identity for custom prod reduction type"); return T(); }
  FLARE_FORCEINLINE_FUNCTION constexpr static T max()   // minimum value
    { static_assert( false, "Missing specialization of
flare::reduction_identity for custom max reduction type"); return T(); }
  FLARE_FORCEINLINE_FUNCTION constexpr static T min()   // maximum value
    { static_assert( false, "Missing specialization of
flare::reduction_identity for custom min reduction type"); return T(); }
  FLARE_FORCEINLINE_FUNCTION constexpr static T bor()   // 0, only for integer
type { static_assert( false, "Missing specialization of
flare::reduction_identity for custom bor reduction type"); return T(); }
  FLARE_FORCEINLINE_FUNCTION constexpr static T band()  // !0, only for integer
type { static_assert( false, "Missing specialization of
flare::reduction_identity for custom band reduction type"); return T(); }
  FLARE_FORCEINLINE_FUNCTION constexpr static T lor()   // 0, only for integer
type { static_assert( false, "Missing specialization of
flare::reduction_identity for custom lor reduction type"); return T(); }
  FLARE_FORCEINLINE_FUNCTION constexpr static T land()  // !0, only for integer
type { static_assert( false, "Missing specialization of
flare::reduction_identity for custom land reduction type"); return T(); }
};*/

template <>
struct reduction_identity<char> {
  FLARE_FORCEINLINE_FUNCTION constexpr static char sum() {
    return static_cast<char>(0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static char prod() {
    return static_cast<char>(1);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static char max() { return CHAR_MIN; }
  FLARE_FORCEINLINE_FUNCTION constexpr static char min() { return CHAR_MAX; }
  FLARE_FORCEINLINE_FUNCTION constexpr static char bor() {
    return static_cast<char>(0x0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static char band() {
    return ~static_cast<char>(0x0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static char lor() {
    return static_cast<char>(0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static char land() {
    return static_cast<char>(1);
  }
};

template <>
struct reduction_identity<signed char> {
  FLARE_FORCEINLINE_FUNCTION constexpr static signed char sum() {
    return static_cast<signed char>(0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static signed char prod() {
    return static_cast<signed char>(1);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static signed char max() {
    return SCHAR_MIN;
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static signed char min() {
    return SCHAR_MAX;
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static signed char bor() {
    return static_cast<signed char>(0x0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static signed char band() {
    return ~static_cast<signed char>(0x0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static signed char lor() {
    return static_cast<signed char>(0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static signed char land() {
    return static_cast<signed char>(1);
  }
};

template <>
struct reduction_identity<bool> {
  FLARE_FORCEINLINE_FUNCTION constexpr static bool lor() {
    return static_cast<bool>(false);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static bool land() {
    return static_cast<bool>(true);
  }
};

template <>
struct reduction_identity<short> {
  FLARE_FORCEINLINE_FUNCTION constexpr static short sum() {
    return static_cast<short>(0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static short prod() {
    return static_cast<short>(1);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static short max() { return SHRT_MIN; }
  FLARE_FORCEINLINE_FUNCTION constexpr static short min() { return SHRT_MAX; }
  FLARE_FORCEINLINE_FUNCTION constexpr static short bor() {
    return static_cast<short>(0x0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static short band() {
    return ~static_cast<short>(0x0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static short lor() {
    return static_cast<short>(0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static short land() {
    return static_cast<short>(1);
  }
};

template <>
struct reduction_identity<int> {
  FLARE_FORCEINLINE_FUNCTION constexpr static int sum() {
    return static_cast<int>(0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static int prod() {
    return static_cast<int>(1);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static int max() { return INT_MIN; }
  FLARE_FORCEINLINE_FUNCTION constexpr static int min() { return INT_MAX; }
  FLARE_FORCEINLINE_FUNCTION constexpr static int bor() {
    return static_cast<int>(0x0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static int band() {
    return ~static_cast<int>(0x0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static int lor() {
    return static_cast<int>(0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static int land() {
    return static_cast<int>(1);
  }
};

template <>
struct reduction_identity<long> {
  FLARE_FORCEINLINE_FUNCTION constexpr static long sum() {
    return static_cast<long>(0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static long prod() {
    return static_cast<long>(1);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static long max() { return LONG_MIN; }
  FLARE_FORCEINLINE_FUNCTION constexpr static long min() { return LONG_MAX; }
  FLARE_FORCEINLINE_FUNCTION constexpr static long bor() {
    return static_cast<long>(0x0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static long band() {
    return ~static_cast<long>(0x0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static long lor() {
    return static_cast<long>(0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static long land() {
    return static_cast<long>(1);
  }
};

template <>
struct reduction_identity<long long> {
  FLARE_FORCEINLINE_FUNCTION constexpr static long long sum() {
    return static_cast<long long>(0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static long long prod() {
    return static_cast<long long>(1);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static long long max() {
    return LLONG_MIN;
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static long long min() {
    return LLONG_MAX;
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static long long bor() {
    return static_cast<long long>(0x0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static long long band() {
    return ~static_cast<long long>(0x0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static long long lor() {
    return static_cast<long long>(0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static long long land() {
    return static_cast<long long>(1);
  }
};

template <>
struct reduction_identity<unsigned char> {
  FLARE_FORCEINLINE_FUNCTION constexpr static unsigned char sum() {
    return static_cast<unsigned char>(0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static unsigned char prod() {
    return static_cast<unsigned char>(1);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static unsigned char max() {
    return static_cast<unsigned char>(0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static unsigned char min() {
    return UCHAR_MAX;
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static unsigned char bor() {
    return static_cast<unsigned char>(0x0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static unsigned char band() {
    return ~static_cast<unsigned char>(0x0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static unsigned char lor() {
    return static_cast<unsigned char>(0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static unsigned char land() {
    return static_cast<unsigned char>(1);
  }
};

template <>
struct reduction_identity<unsigned short> {
  FLARE_FORCEINLINE_FUNCTION constexpr static unsigned short sum() {
    return static_cast<unsigned short>(0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static unsigned short prod() {
    return static_cast<unsigned short>(1);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static unsigned short max() {
    return static_cast<unsigned short>(0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static unsigned short min() {
    return USHRT_MAX;
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static unsigned short bor() {
    return static_cast<unsigned short>(0x0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static unsigned short band() {
    return ~static_cast<unsigned short>(0x0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static unsigned short lor() {
    return static_cast<unsigned short>(0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static unsigned short land() {
    return static_cast<unsigned short>(1);
  }
};

template <>
struct reduction_identity<unsigned int> {
  FLARE_FORCEINLINE_FUNCTION constexpr static unsigned int sum() {
    return static_cast<unsigned int>(0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static unsigned int prod() {
    return static_cast<unsigned int>(1);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static unsigned int max() {
    return static_cast<unsigned int>(0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static unsigned int min() {
    return UINT_MAX;
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static unsigned int bor() {
    return static_cast<unsigned int>(0x0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static unsigned int band() {
    return ~static_cast<unsigned int>(0x0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static unsigned int lor() {
    return static_cast<unsigned int>(0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static unsigned int land() {
    return static_cast<unsigned int>(1);
  }
};

template <>
struct reduction_identity<unsigned long> {
  FLARE_FORCEINLINE_FUNCTION constexpr static unsigned long sum() {
    return static_cast<unsigned long>(0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static unsigned long prod() {
    return static_cast<unsigned long>(1);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static unsigned long max() {
    return static_cast<unsigned long>(0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static unsigned long min() {
    return ULONG_MAX;
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static unsigned long bor() {
    return static_cast<unsigned long>(0x0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static unsigned long band() {
    return ~static_cast<unsigned long>(0x0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static unsigned long lor() {
    return static_cast<unsigned long>(0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static unsigned long land() {
    return static_cast<unsigned long>(1);
  }
};

template <>
struct reduction_identity<unsigned long long> {
  FLARE_FORCEINLINE_FUNCTION constexpr static unsigned long long sum() {
    return static_cast<unsigned long long>(0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static unsigned long long prod() {
    return static_cast<unsigned long long>(1);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static unsigned long long max() {
    return static_cast<unsigned long long>(0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static unsigned long long min() {
    return ULLONG_MAX;
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static unsigned long long bor() {
    return static_cast<unsigned long long>(0x0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static unsigned long long band() {
    return ~static_cast<unsigned long long>(0x0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static unsigned long long lor() {
    return static_cast<unsigned long long>(0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static unsigned long long land() {
    return static_cast<unsigned long long>(1);
  }
};

template <>
struct reduction_identity<float> {
  FLARE_FORCEINLINE_FUNCTION constexpr static float sum() {
    return static_cast<float>(0.0f);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static float prod() {
    return static_cast<float>(1.0f);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static float max() { return -FLT_MAX; }
  FLARE_FORCEINLINE_FUNCTION constexpr static float min() { return FLT_MAX; }
};

template <>
struct reduction_identity<double> {
  FLARE_FORCEINLINE_FUNCTION constexpr static double sum() {
    return static_cast<double>(0.0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static double prod() {
    return static_cast<double>(1.0);
  }
  FLARE_FORCEINLINE_FUNCTION constexpr static double max() { return -DBL_MAX; }
  FLARE_FORCEINLINE_FUNCTION constexpr static double min() { return DBL_MAX; }
};

// No __host__ __device__ annotation because long double treated as double in
// device code.  May be revisited later if that is not true any more.
template <>
struct reduction_identity<long double> {
  constexpr static long double sum() { return static_cast<long double>(0.0); }
  constexpr static long double prod() { return static_cast<long double>(1.0); }
  constexpr static long double max() { return -LDBL_MAX; }
  constexpr static long double min() { return LDBL_MAX; }
};

}  // namespace flare

#endif  // FLARE_REDUCTION_IDENTITY_H_
