// Copyright 2023 The EA Authors.
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

#include <flare.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <fly/dim4.hpp>
#include <fly/traits.hpp>
#include <stdexcept>
#include <string>
#include <vector>

using fly::array;
using fly::cdouble;
using fly::cfloat;
using fly::dim4;
using fly::dtype;
using fly::dtype_traits;
using fly::fft;
using fly::fft2Norm;
using fly::fft3Norm;
using fly::fftC2R;
using fly::fftNorm;
using fly::fftR2C;
using fly::randu;
using std::abs;
using std::string;
using std::vector;

template<typename T>
class FFT_REAL : public ::testing::Test {};

typedef ::testing::Types<cfloat, cdouble> TestTypes;
TYPED_TEST_SUITE(FFT_REAL, TestTypes);

template<int rank>
array fft(const array &in, double norm) {
    switch (rank) {
        case 1: return fftNorm(in, norm);
        case 2: return fft2Norm(in, norm);
        case 3: return fft3Norm(in, norm);
        default: return in;
    }
}

#define MY_ASSERT_NEAR(aa, bb, cc) ASSERT_NEAR(abs(aa), abs(bb), (cc))

template<typename Tc, int rank>
void fft_real(dim4 dims) {
    typedef typename dtype_traits<Tc>::base_type Tr;
    SUPPORTED_TYPE_CHECK(Tr);

    dtype ty = (dtype)dtype_traits<Tr>::fly_type;
    array a  = randu(dims, ty);

    bool is_odd = dims[0] & 1;

    int dim0 = dims[0] / 2 + 1;

    double norm = 1;
    for (int i = 0; i < rank; i++) norm *= dims[i];
    norm = 1 / norm;

    array as = fftR2C<rank>(a, norm);
    array fly = fft<rank>(a, norm);

    vector<Tc> has(as.elements());
    vector<Tc> haf(fly.elements());

    as.host(&has[0]);
    fly.host(&haf[0]);

    for (int j = 0; j < a.elements() / dims[0]; j++) {
        for (int i = 0; i < dim0; i++) {
            MY_ASSERT_NEAR(haf[j * dims[0] + i], has[j * dim0 + i], 1E-2)
                << "at " << j * dims[0] + i;
        }
    }

    array b = fftC2R<rank>(as, is_odd, 1);

    vector<Tr> ha(a.elements());
    vector<Tr> hb(a.elements());

    a.host(&ha[0]);
    b.host(&hb[0]);

    for (int j = 0; j < a.elements(); j++) { ASSERT_NEAR(ha[j], hb[j], 1E-2); }
}

TYPED_TEST(FFT_REAL, Even1D) { fft_real<TypeParam, 1>(dim4(1024, 256)); }

TYPED_TEST(FFT_REAL, Odd1D) { fft_real<TypeParam, 1>(dim4(625, 256)); }

TYPED_TEST(FFT_REAL, Even2D) { fft_real<TypeParam, 2>(dim4(1024, 256)); }

TYPED_TEST(FFT_REAL, Odd2D) { fft_real<TypeParam, 2>(dim4(625, 256)); }

TYPED_TEST(FFT_REAL, Even3D) { fft_real<TypeParam, 3>(dim4(32, 32, 32)); }

TYPED_TEST(FFT_REAL, Odd3D) { fft_real<TypeParam, 3>(dim4(25, 32, 32)); }
