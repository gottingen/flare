#!/usr/bin/env python
##########################################################################
# Copyright 2023 The EA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################

import flare as fly

def main():
    try:
        fly.info()

        n = 5
        t = fly.randu(n, n)
        arr_in = fly.matmulNT(t, t) + fly.identity(n, n) * n

        print("Running Cholesky InPlace\n")
        cin_upper = arr_in.copy()
        cin_lower = arr_in.copy()

        fly.cholesky_inplace(cin_upper, True)
        fly.cholesky_inplace(cin_lower, False)

        print(cin_upper)
        print(cin_lower)

        print("\nRunning Cholesky Out of place\n")

        out_upper, upper_success = fly.cholesky(arr_in, True)
        out_lower, lower_success = fly.cholesky(arr_in, False)

        if upper_success == 0:
            print(out_upper)
        if lower_success == 0:
            print(out_lower)

    except Exception as e:
        print('Error: ', str(e))

if __name__ == '__main__':
    main()
