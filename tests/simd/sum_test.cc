#include "sum_test.h"
#if FLARE_SIMD_ENABLE_AVX
template float sum::operator()(flare::simd::avx, float const*, unsigned);
#endif
