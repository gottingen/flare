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


def simple_random(verbose=False):
    display_func = _util.display_func(verbose)

    display_func(fly.randu(3, 3, 1, 2))
    display_func(fly.randu(3, 3, 1, 2, fly.Dtype.b8))
    display_func(fly.randu(3, 3, dtype=fly.Dtype.c32))

    display_func(fly.randn(3, 3, 1, 2))
    display_func(fly.randn(3, 3, dtype=fly.Dtype.c32))

    fly.set_seed(1024)
    assert(fly.get_seed() == 1024)

    engine = fly.Random_Engine(fly.RANDOM_ENGINE.MERSENNE_GP11213, 100)

    display_func(fly.randu(3, 3, 1, 2, engine=engine))
    display_func(fly.randu(3, 3, 1, 2, fly.Dtype.s32, engine=engine))
    display_func(fly.randu(3, 3, dtype=fly.Dtype.c32, engine=engine))

    display_func(fly.randn(3, 3, engine=engine))
    engine.set_seed(100)
    assert(engine.get_seed() == 100)


_util.tests["random"] = simple_random
