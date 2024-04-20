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


def simple_sparse(verbose=False):
    display_func = _util.display_func(verbose)
    print_func = _util.print_func(verbose)

    dd = fly.randu(5, 5)
    ds = dd * (dd > 0.5)
    sp = fly.create_sparse_from_dense(ds)
    display_func(fly.sparse_get_info(sp))
    display_func(fly.sparse_get_values(sp))
    display_func(fly.sparse_get_row_idx(sp))
    display_func(fly.sparse_get_col_idx(sp))
    print_func(fly.sparse_get_nnz(sp))
    print_func(fly.sparse_get_storage(sp))


_util.tests["sparse"] = simple_sparse
