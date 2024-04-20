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

#include <fly/algorithm.h>
#include <fly/array.h>
#include <fly/compatible.h>
#include <fly/image.h>
#include "error.hpp"

namespace fly {

array histogram(const array& in, const unsigned nbins, const double minval,
                const double maxval) {
    fly_array out = 0;
    FLY_THROW(fly_histogram(&out, in.get(), nbins, minval, maxval));
    return array(out);
}

array histogram(const array& in, const unsigned nbins) {
    fly_array out = 0;
    if (in.numdims() == 0) { return in; }
    FLY_THROW(
        fly_histogram(&out, in.get(), nbins, min<double>(in), max<double>(in)));
    return array(out);
}

array histequal(const array& in, const array& hist) {
    return histEqual(in, hist);
}
array histEqual(const array& in, const array& hist) {
    fly_array temp = 0;
    FLY_THROW(fly_hist_equal(&temp, in.get(), hist.get()));
    return array(temp);
}

}  // namespace fly
