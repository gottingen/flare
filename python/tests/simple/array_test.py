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

import array as host

import flare as fly

from . import _util


def simple_array(verbose=False):
    display_func = _util.display_func(verbose)
    print_func = _util.print_func(verbose)

    a = fly.Array([1, 2, 3])
    display_func(a)
    display_func(a.T)
    display_func(a.H)
    print_func(a.shape)

    b = a.as_type(fly.Dtype.s32)
    display_func(b)

    print_func(a.elements(), a.type(), a.dims(), a.numdims())
    print_func(a.is_empty(), a.is_scalar(), a.is_column(), a.is_row())
    print_func(a.is_complex(), a.is_real(), a.is_double(), a.is_single())
    print_func(a.is_real_floating(), a.is_floating(), a.is_integer(), a.is_bool())

    a = fly.Array(host.array("i", [4, 5, 6]))
    display_func(a)
    print_func(a.elements(), a.type(), a.dims(), a.numdims())
    print_func(a.is_empty(), a.is_scalar(), a.is_column(), a.is_row())
    print_func(a.is_complex(), a.is_real(), a.is_double(), a.is_single())
    print_func(a.is_real_floating(), a.is_floating(), a.is_integer(), a.is_bool())

    a = fly.Array(host.array("I", [7, 8, 9] * 3), (3, 3))
    display_func(a)
    print_func(a.elements(), a.type(), a.dims(), a.numdims())
    print_func(a.is_empty(), a.is_scalar(), a.is_column(), a.is_row())
    print_func(a.is_complex(), a.is_real(), a.is_double(), a.is_single())
    print_func(a.is_real_floating(), a.is_floating(), a.is_integer(), a.is_bool())

    c = a.to_ctype()
    for n in range(a.elements()):
        print_func(c[n])

    c, s = a.to_ctype(True, True)
    for n in range(a.elements()):
        print_func(c[n])
    print_func(s)

    arr = a.to_array()
    lst = a.to_list(True)

    print_func(arr)
    print_func(lst)

    print_func(a.is_sparse())


_util.tests["array"] = simple_array
