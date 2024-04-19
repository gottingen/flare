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

// Investigate rainfall measurements across sites and days
// demonstrating various simple tasks

// Compute various values:
// - total rainfall at each site
// - rain between days 1 and 5
// - number of days with rain
// - total rainfall on each day
// - number of days with over five inches
// - total rainfall at each site

// note: example adapted from
//  "Rapid Problem Solving Using Thrust", Nathan Bell, NVIDIA

#include <flare.h>
#include <stdio.h>
#include <fly/util.h>
#include <cstdlib>
using namespace fly;

int main(int argc, char **argv) {
    try {
        int device = argc > 1 ? atoi(argv[1]) : 0;
        fly::setDevice(device);
        fly::info();

        int days = 9, sites = 4;
        int n                = 10;                              // measurements
        float day_[]         = {0, 0, 1, 2, 5, 5, 6, 6, 7, 8};  // ascending
        float site_[]        = {2, 3, 0, 1, 1, 2, 0, 1, 2, 1};
        float measurement_[] = {9, 5, 6, 3, 3, 8, 2, 6, 5, 10};  // inches
        array day(n, day_);
        array site(n, site_);
        array measurement(n, measurement_);

        array rainfall = constant(0, sites);
        gfor(seq s, sites) { rainfall(s) = sum(measurement * (site == s)); }

        printf("total rainfall at each site:\n");
        fly_print(rainfall);

        array is_between   = 1 <= day && day <= 5;  // days 1 and 5
        float rain_between = sum<float>(measurement * is_between);
        printf("rain between days: %g\n", rain_between);

        printf("number of days with rain: %g\n",
               sum<float>(diff1(day) > 0) + 1);

        array per_day                = constant(0, days);
        gfor(seq d, days) per_day(d) = sum(measurement * (day == d));

        printf("total rainfall each day:\n");
        fly_print(per_day);

        printf("number of days over five: %g\n", sum<float>(per_day > 5));
    } catch (fly::exception &e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    return 0;
}
