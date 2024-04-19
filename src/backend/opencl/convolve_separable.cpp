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

#include <convolve.hpp>

#include <Array.hpp>
#include <err_opencl.hpp>
#include <kernel/convolve_separable.hpp>
#include <fly/dim4.hpp>

using fly::dim4;

namespace flare {
namespace opencl {

template<typename T, typename accT>
Array<T> convolve2(Array<T> const& signal, Array<accT> const& c_filter,
                   Array<accT> const& r_filter, const bool expand) {
    const auto cflen = c_filter.elements();
    const auto rflen = r_filter.elements();

    if ((cflen > kernel::MAX_SCONV_FILTER_LEN) ||
        (rflen > kernel::MAX_SCONV_FILTER_LEN)) {
        // TODO call upon fft
        char errMessage[256];
        snprintf(errMessage, sizeof(errMessage),
                 "\nOpenCL Separable convolution doesn't support %llu(coloumn) "
                 "%llu(row) filters\n",
                 cflen, rflen);
        OPENCL_NOT_SUPPORTED(errMessage);
    }

    const dim4& sDims = signal.dims();
    dim4 tDims        = sDims;
    dim4 oDims        = sDims;

    if (expand) {
        tDims[0] += cflen - 1;
        oDims[0] += cflen - 1;
        oDims[1] += rflen - 1;
    }

    Array<T> temp = createEmptyArray<T>(tDims);
    Array<T> out  = createEmptyArray<T>(oDims);

    kernel::convSep<T, accT>(temp, signal, c_filter, 0, expand);
    kernel::convSep<T, accT>(out, temp, r_filter, 1, expand);

    return out;
}

#define INSTANTIATE(T, accT)                                                  \
    template Array<T> convolve2<T, accT>(Array<T> const&, Array<accT> const&, \
                                         Array<accT> const&, const bool);

INSTANTIATE(cdouble, cdouble)
INSTANTIATE(cfloat, cfloat)
INSTANTIATE(double, double)
INSTANTIATE(float, float)
INSTANTIATE(uint, float)
INSTANTIATE(int, float)
INSTANTIATE(uchar, float)
INSTANTIATE(char, float)
INSTANTIATE(short, float)
INSTANTIATE(ushort, float)
INSTANTIATE(intl, float)
INSTANTIATE(uintl, float)

}  // namespace opencl
}  // namespace flare
