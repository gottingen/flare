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

from ._util import tests
from .algorithm import simple_algorithm
from .arith import simple_arith
from .array_test import simple_array
from .blas import simple_blas
from .data import simple_data
from .device import simple_device
from .image import simple_image
from .index import simple_index
from .interop import simple_interop
from .lapack import simple_lapack
from .random import simple_random
from .signal import simple_signal
from .sparse import simple_sparse
from .statistics import simple_statistics

__all__ = [
    "tests",
    "simple_algorithm",
    "simple_arith",
    "simple_array",
    "simple_blas",
    "simple_data",
    "simple_device",
    "simple_image",
    "simple_index",
    "simple_interop",
    "simple_lapack",
    "simple_random",
    "simple_signal",
    "simple_sparse",
    "simple_statistics"
]
