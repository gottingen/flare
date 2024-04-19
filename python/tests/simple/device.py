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

from . import _util


def simple_device(verbose=False):
    display_func = _util.display_func(verbose)
    print_func = _util.print_func(verbose)
    print_func(fly.device_info())
    print_func(fly.get_device_count())
    print_func(fly.is_dbl_supported())
    fly.sync()

    curr_dev = fly.get_device()
    print_func(curr_dev)
    for k in range(fly.get_device_count()):
        fly.set_device(k)
        dev = fly.get_device()
        assert(k == dev)

        print_func(fly.is_dbl_supported(k))

        fly.device_gc()

        mem_info_old = fly.device_mem_info()

        a = fly.randu(100, 100)
        fly.sync(dev)
        mem_info = fly.device_mem_info()
        assert(mem_info["alloc"]["buffers"] == 1 + mem_info_old["alloc"]["buffers"])
        assert(mem_info["lock"]["buffers"] == 1 + mem_info_old["lock"]["buffers"])

    fly.set_device(curr_dev)

    a = fly.randu(10, 10)
    display_func(a)
    dev_ptr = fly.get_device_ptr(a)
    print_func(dev_ptr)
    b = fly.Array(src=dev_ptr, dims=a.dims(), dtype=a.dtype(), is_device=True)
    display_func(b)

    c = fly.randu(10, 10)
    fly.lock_array(c)
    fly.unlock_array(c)

    a = fly.constant(1, 3, 3)
    b = fly.constant(2, 3, 3)
    fly.eval(a)
    fly.eval(b)
    print_func(a)
    print_func(b)
    c = a + b
    d = a - b
    fly.eval(c, d)
    print_func(c)
    print_func(d)

    print_func(fly.set_manual_eval_flag(True))
    assert(fly.get_manual_eval_flag())
    print_func(fly.set_manual_eval_flag(False))
    assert(not fly.get_manual_eval_flag())

    display_func(fly.is_locked_array(a))


_util.tests["device"] = simple_device
