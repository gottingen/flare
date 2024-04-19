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

#include <fly/array.h>
#include <fly/signal.h>
#include <algorithm>
#include "error.hpp"

namespace fly {

array fftConvolve(const array& signal, const array& filter,
                  const convMode mode) {
    unsigned sN = signal.numdims();
    unsigned fN = filter.numdims();

    switch (std::min(sN, fN)) {
        case 1: return fftConvolve1(signal, filter, mode);
        case 2: return fftConvolve2(signal, filter, mode);
        default:
        case 3: return fftConvolve3(signal, filter, mode);
    }
}

array fftConvolve1(const array& signal, const array& filter,
                   const convMode mode) {
    fly_array out = 0;
    FLY_THROW(fly_fft_convolve1(&out, signal.get(), filter.get(), mode));
    return array(out);
}

array fftConvolve2(const array& signal, const array& filter,
                   const convMode mode) {
    fly_array out = 0;
    FLY_THROW(fly_fft_convolve2(&out, signal.get(), filter.get(), mode));
    return array(out);
}

array fftConvolve3(const array& signal, const array& filter,
                   const convMode mode) {
    fly_array out = 0;
    FLY_THROW(fly_fft_convolve3(&out, signal.get(), filter.get(), mode));
    return array(out);
}

}  // namespace fly
