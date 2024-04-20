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

#include <copy.hpp>

#include <Array.hpp>
#include <common/half.hpp>
#include <err_cuda.hpp>
#include <kernel/pad_array_borders.hpp>

namespace flare {
namespace cuda {
template<typename T>
Array<T> padArrayBorders(Array<T> const& in, dim4 const& lowerBoundPadding,
                         dim4 const& upperBoundPadding,
                         const fly::borderType btype) {
    const dim4& iDims = in.dims();

    dim4 oDims(lowerBoundPadding[0] + iDims[0] + upperBoundPadding[0],
               lowerBoundPadding[1] + iDims[1] + upperBoundPadding[1],
               lowerBoundPadding[2] + iDims[2] + upperBoundPadding[2],
               lowerBoundPadding[3] + iDims[3] + upperBoundPadding[3]);

    if (oDims == iDims) { return in; }

    auto ret = createEmptyArray<T>(oDims);

    kernel::padBorders<T>(ret, in, lowerBoundPadding, btype);

    return ret;
}

#define INSTANTIATE_PAD_ARRAY_BORDERS(T)                               \
    template Array<T> padArrayBorders<T>(Array<T> const&, dim4 const&, \
                                         dim4 const&, const fly::borderType);

INSTANTIATE_PAD_ARRAY_BORDERS(cfloat)
INSTANTIATE_PAD_ARRAY_BORDERS(cdouble)
INSTANTIATE_PAD_ARRAY_BORDERS(float)
INSTANTIATE_PAD_ARRAY_BORDERS(double)
INSTANTIATE_PAD_ARRAY_BORDERS(int)
INSTANTIATE_PAD_ARRAY_BORDERS(uint)
INSTANTIATE_PAD_ARRAY_BORDERS(intl)
INSTANTIATE_PAD_ARRAY_BORDERS(uintl)
INSTANTIATE_PAD_ARRAY_BORDERS(uchar)
INSTANTIATE_PAD_ARRAY_BORDERS(char)
INSTANTIATE_PAD_ARRAY_BORDERS(ushort)
INSTANTIATE_PAD_ARRAY_BORDERS(short)
INSTANTIATE_PAD_ARRAY_BORDERS(common::half)
}  // namespace cuda
}  // namespace flare
