#!/usr/bin/python

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
import sys
from array import array

if __name__ == "__main__":
    if (len(sys.argv) > 1):
        fly.set_device(int(sys.argv[1]))
    fly.info()

    print("\n---- Intro to flare using signed(s32) arrays ----\n")

    h_A = array('i', ( 1,  2,  4, -1,  2,  0,  4,  2,  3))
    h_B = array('i', ( 2,  3,  5,  6,  0, 10,-12,  0,  1))

    A = fly.Array(h_A, (3,3))
    B = fly.Array(h_B, (3,3))

    print("\n---- Sub referencing and sub assignment\n")
    fly.display(A)
    fly.display(A[0,:])
    fly.display(A[:,0])
    A[0,0] = 11
    A[1] = 100
    fly.display(A)
    fly.display(B)
    A[1,:] = B[2,:]
    fly.display(A)

    print("\n---- Bitwise operations\n")
    fly.display(A & B)
    fly.display(A | B)
    fly.display(A ^ B)

    print("\n---- Transpose\n")
    fly.display(A)
    fly.display(fly.transpose(A))

    print("\n---- Flip Vertically / Horizontally\n")
    fly.display(A)
    fly.display(fly.flip(A, 0))
    fly.display(fly.flip(A, 1))

    print("\n---- Sum, Min, Max along row / columns\n")
    fly.display(A)
    fly.display(fly.sum(A, 0))
    fly.display(fly.min(A, 0))
    fly.display(fly.max(A, 0))
    fly.display(fly.sum(A, 1))
    fly.display(fly.min(A, 1))
    fly.display(fly.max(A, 1))

    print("\n---- Get minimum with index\n")
    (min_val, min_idx) = fly.imin(A, 0)
    fly.display(min_val)
    fly.display(min_idx)
