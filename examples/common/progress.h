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

#ifndef __PROGRESS_H
#define __PROGRESS_H

#include <algorithm>
#include <cmath>

static bool progress(unsigned iter_curr, fly::timer t, double time_total) {
    static unsigned iter_prev = 0;
    static double time_prev   = 0;
    static double max_rate    = 0;

    fly::sync();
    double time_curr = fly::timer::stop(t);

    if ((time_curr - time_prev) < 1) return true;

    double rate = (iter_curr - iter_prev) / (time_curr - time_prev);
    printf("  iterations per second: %.0f   (progress %.0f%%)\n", rate,
           100.0f * time_curr / time_total);

    max_rate = std::max(max_rate, rate);

    iter_prev = iter_curr;
    time_prev = time_curr;

    if (time_curr < time_total) return true;

    printf(" ### %f iterations per second (max)\n", max_rate);
    return false;
}

#endif
