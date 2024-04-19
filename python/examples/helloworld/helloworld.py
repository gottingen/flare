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

try:
    # Display backend information
    fly.info()

    print("Create a 5-by-3 matrix of random floats on the GPU\n")
    A = fly.randu(5, 3, 1, 1, fly.Dtype.f32)
    fly.display(A)

    print("Element-wise arithmetic\n")
    B = fly.sin(A) + 1.5
    fly.display(B)

    print("Negate the first three elements of second column\n")
    B[0:3, 1] = B[0:3, 1] * -1
    fly.display(B)

    print("Fourier transform the result\n");
    C = fly.fft(B);
    fly.display(C);

    print("Grab last row\n");
    c = C[-1,:];
    fly.display(c);

    print("Scan Test\n");
    r = fly.constant(2, 16, 4, 1, 1);
    fly.display(r);

    print("Scan\n");
    S = fly.scan(r, 0, fly.BINARYOP.MUL);
    fly.display(S);

    print("Create 2-by-3 matrix from host data\n");
    d = [ 1, 2, 3, 4, 5, 6 ]
    D = fly.Array(d, (2, 3))
    fly.display(D)

    print("Copy last column onto first\n");
    D[:,0] = D[:, -1]
    fly.display(D);

    print("Sort A and print sorted array and corresponding indices\n");
    [sorted_vals, sorted_idxs] = fly.sort_index(A);
    fly.display(A)
    fly.display(sorted_vals)
    fly.display(sorted_idxs)
except Exception as e:
    print("Error: " + str(e))
