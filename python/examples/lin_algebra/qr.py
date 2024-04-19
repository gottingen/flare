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

        print("Running QR InPlace\n")
        q_in = in_array.copy()
        print(q_in)

        tau = fly.qr_inplace(q_in)

        print(q_in)
        print(tau)

        print("Running QR with Q and R factorization\n")
        q, r, tau = fly.qr(in_array)

        print(q)
        print(r)
        print(tau)

    except Exception as e:
        print("Error: ", str(e))

if __name__ == '__main__':
    main()
