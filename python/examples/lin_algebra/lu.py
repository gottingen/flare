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

        in_array = fly.randu(5,8)

        print("Running LU InPlace\n")
        pivot = fly.lu_inplace(in_array)
        print(in_array)
        print(pivot)

        print("Running LU with Upper Lower Factorization\n")
        lower, upper, pivot = fly.lu(in_array)
        print(lower)
        print(upper)
        print(pivot)

    except Exception as e:
        print('Error: ', str(e))

if __name__ == '__main__':
    main()

