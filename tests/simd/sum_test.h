#ifndef SIMD_TEST_SUM_H_
#define SIMD_TEST_SUM_H_
#include <flare/simd/simd.h>
#ifndef FLARE_SIMD_NO_SUPPORTED_ARCHITECTURE

struct sum
{
    // NOTE: no inline definition here otherwise extern template instantiation
    // doesn't prevent implicit instantiation.
    template <class Arch, class T>
    T operator()(Arch, T const* data, unsigned size);
};

template <class Arch, class T>
T sum::operator()(Arch, T const* data, unsigned size)
{
    using batch = flare::simd::batch<T, Arch>;
    batch acc(static_cast<T>(0));
    const unsigned n = size / batch::size * batch::size;
    for (unsigned i = 0; i != n; i += batch::size)
        acc += batch::load_unaligned(data + i);
    T star_acc = flare::simd::reduce_add(acc);
    for (unsigned i = n; i < size; ++i)
        star_acc += data[i];
    return star_acc;
}

#if FLARE_SIMD_ENABLE_AVX
extern template float sum::operator()(flare::simd::avx, float const*, unsigned);
#endif

#endif

#endif  // SIMD_TEST_SUM_H_
