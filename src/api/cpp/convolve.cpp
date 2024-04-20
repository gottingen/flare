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
#include <fly/compatible.h>
#include <fly/dim4.hpp>
#include <fly/ml.h>
#include <fly/signal.h>
#include <algorithm>
#include "error.hpp"

namespace fly {

array convolve(const array &signal, const array &filter, const convMode mode,
               convDomain domain) {
    unsigned sN = signal.numdims();
    unsigned fN = filter.numdims();

    switch (std::min(sN, fN)) {
        case 1: return convolve1(signal, filter, mode, domain);
        case 2: return convolve2(signal, filter, mode, domain);
        default:
        case 3: return convolve3(signal, filter, mode, domain);
    }
}

array convolve(const array &col_filter, const array &row_filter,
               const array &signal, const convMode mode) {
    fly_array out = 0;
    FLY_THROW(fly_convolve2_sep(&out, col_filter.get(), row_filter.get(),
                              signal.get(), mode));
    return array(out);
}

array convolve1(const array &signal, const array &filter, const convMode mode,
                convDomain domain) {
    fly_array out = 0;
    FLY_THROW(fly_convolve1(&out, signal.get(), filter.get(), mode, domain));
    return array(out);
}

array convolve2(const array &signal, const array &filter, const convMode mode,
                convDomain domain) {
    fly_array out = 0;
    FLY_THROW(fly_convolve2(&out, signal.get(), filter.get(), mode, domain));
    return array(out);
}

array convolve2NN(
    const array &signal, const array &filter,
    const dim4 stride,      // NOLINT(performance-unnecessary-value-param)
    const dim4 padding,     // NOLINT(performance-unnecessary-value-param)
    const dim4 dilation) {  // NOLINT(performance-unnecessary-value-param)
    fly_array out = 0;
    FLY_THROW(fly_convolve2_nn(&out, signal.get(), filter.get(), 2, stride.get(),
                             2, padding.get(), 2, dilation.get()));
    return array(out);
}

array convolve2GradientNN(
    const array &incoming_gradient, const array &original_signal,
    const array &original_filter, const array &convolved_output,
    const dim4 stride,    // NOLINT(performance-unnecessary-value-param)
    const dim4 padding,   // NOLINT(performance-unnecessary-value-param)
    const dim4 dilation,  // NOLINT(performance-unnecessary-value-param)
    fly_conv_gradient_type gradType) {
    fly_array out = 0;
    FLY_THROW(fly_convolve2_gradient_nn(
        &out, incoming_gradient.get(), original_signal.get(),
        original_filter.get(), convolved_output.get(), 2, stride.get(), 2,
        padding.get(), 2, dilation.get(), gradType));
    return array(out);
}

array convolve3(const array &signal, const array &filter, const convMode mode,
                convDomain domain) {
    fly_array out = 0;
    FLY_THROW(fly_convolve3(&out, signal.get(), filter.get(), mode, domain));
    return array(out);
}

array filter(const array &image, const array &kernel) {
    return convolve(image, kernel, FLY_CONV_DEFAULT, FLY_CONV_AUTO);
}

}  // namespace fly
