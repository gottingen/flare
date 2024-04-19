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


def simple_interop(verbose=False):
    if fly.AF_NUMPY_FOUND:
        import numpy as np
        n = np.random.random((5,))
        a = fly.to_array(n)
        n2 = np.array(a)
        assert((n == n2).all())
        n2[:] = 0
        a.to_ndarray(n2)
        assert((n == n2).all())

        n = np.random.random((5, 3))
        a = fly.to_array(n)
        n2 = np.array(a)
        assert((n == n2).all())
        n2[:] = 0
        a.to_ndarray(n2)
        assert((n == n2).all())

        n = np.random.random((5, 3, 2))
        a = fly.to_array(n)
        n2 = np.array(a)
        assert((n == n2).all())
        n2[:] = 0
        a.to_ndarray(n2)
        assert((n == n2).all())

        n = np.random.random((5, 3, 2, 2))
        a = fly.to_array(n)
        n2 = np.array(a)
        assert((n == n2).all())
        n2[:] = 0
        a.to_ndarray(n2)
        assert((n == n2).all())

    if fly.AF_PYCUDA_FOUND and fly.get_active_backend() == "cuda":
        import pycuda.gpuarray as cudaArray
        n = np.random.random((5,))
        c = cudaArray.to_gpu(n)
        a = fly.to_array(c)
        n2 = np.array(a)
        assert((n == n2).all())

        n = np.random.random((5, 3))
        c = cudaArray.to_gpu(n)
        a = fly.to_array(c)
        n2 = np.array(a)
        assert((n == n2).all())

        n = np.random.random((5, 3, 2))
        c = cudaArray.to_gpu(n)
        a = fly.to_array(c)
        n2 = np.array(a)
        assert((n == n2).all())

        n = np.random.random((5, 3, 2, 2))
        c = cudaArray.to_gpu(n)
        a = fly.to_array(c)
        n2 = np.array(a)
        assert((n == n2).all())

    if fly.AF_PYOPENCL_FOUND and fly.backend.name() == "opencl":
        # TODO: This needs fixing upstream
        # import pyopencl as cl
        # import pyopencl.array as clArray
        # ctx = cl.create_some_context()
        # queue = cl.CommandQueue(ctx)

        # n = np.random.random((5,))
        # c = cl.array.to_device(queue, n)
        # a = fly.to_array(c)
        # n2 = np.array(a)
        # assert((n==n2).all())

        # n = np.random.random((5,3))
        # c = cl.array.to_device(queue, n)
        # a = fly.to_array(c)
        # n2 = np.array(a)
        # assert((n==n2).all())

        # n = np.random.random((5,3,2))
        # c = cl.array.to_device(queue, n)
        # a = fly.to_array(c)
        # n2 = np.array(a)
        # assert((n==n2).all())

        # n = np.random.random((5,3,2,2))
        # c = cl.array.to_device(queue, n)
        # a = fly.to_array(c)
        # n2 = np.array(a)
        # assert((n==n2).all())
        pass

    if fly.AF_NUMBA_FOUND and fly.get_active_backend() == "cuda":
        from numba import cuda

        n = np.random.random((5,))
        c = cuda.to_device(n)
        a = fly.to_array(c)
        n2 = np.array(a)
        assert((n == n2).all())

        n = np.random.random((5, 3))
        c = cuda.to_device(n)
        a = fly.to_array(c)
        n2 = np.array(a)
        assert((n == n2).all())

        n = np.random.random((5, 3, 2))
        c = cuda.to_device(n)
        a = fly.to_array(c)
        n2 = np.array(a)
        assert((n == n2).all())

        n = np.random.random((5, 3, 2, 2))
        c = cuda.to_device(n)
        a = fly.to_array(c)
        n2 = np.array(a)
        assert((n == n2).all())


_util.tests["interop"] = simple_interop
