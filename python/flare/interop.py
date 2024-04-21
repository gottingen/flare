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

"""
Interop with other python packages.

This module provides helper functions to copy data to flare from the following modules:

     1. numpy - numpy.ndarray
     2. pycuda - pycuda.gpuarray
     4. numba - numba.cuda.cudadrv.devicearray.DeviceNDArray

"""

from .array import *
from .device import *


def _fc_to_fly_array(in_ptr, in_shape, in_dtype, is_device=False, copy = True):
    """
    Fortran Contiguous to af array
    """
    res = Array(in_ptr, in_shape, in_dtype, is_device=is_device)

    if not is_device:
        return res

    lock_array(res)
    return res.copy() if copy else res

def _cc_to_fly_array(in_ptr, ndim, in_shape, in_dtype, is_device=False, copy = True):
    """
    C Contiguous to af array
    """
    if ndim == 1:
        return _fc_to_fly_array(in_ptr, in_shape, in_dtype, is_device, copy)
    else:
        shape = tuple(reversed(in_shape))
        res = Array(in_ptr, shape, in_dtype, is_device=is_device)
        if is_device: lock_array(res)
        return res._reorder()

_nptype_to_aftype = {'b1' : Dtype.b8,
		     'u1' : Dtype.u8,
		     'u2' : Dtype.u16,
		     'i2' : Dtype.s16,
		     's4' : Dtype.u32,
		     'i4' : Dtype.s32,
		     'f4' : Dtype.f32,
		     'c8' : Dtype.c32,
		     's8' : Dtype.u64,
		     'i8' : Dtype.s64,
                     'f8' : Dtype.f64,
                     'c16' : Dtype.c64}

try:
    import numpy as np
except ImportError:
    FLY_NUMPY_FOUND=False
else:
    from numpy import ndarray as NumpyArray
    from .data import reorder

    FLY_NUMPY_FOUND=True

    def np_to_fly_array(np_arr, copy=True):
        """
        Convert numpy.ndarray to flare.Array.

        Parameters
        ----------
        np_arr  : numpy.ndarray()

        copy : Bool specifying if array is to be copied.
               Default is true.
               Can only be False if array is fortran contiguous.

        Returns
        ---------
        fly_arr  : flare.Array()
        """

        in_shape = np_arr.shape
        in_ptr = np_arr.ctypes.data_as(c_void_ptr_t)
        in_dtype = _nptype_to_aftype[np_arr.dtype.str[1:]]

        if not copy:
            raise RuntimeError("Copy can not be False for numpy arrays")

        if (np_arr.flags['F_CONTIGUOUS']):
            return _fc_to_fly_array(in_ptr, in_shape, in_dtype)
        elif (np_arr.flags['C_CONTIGUOUS']):
            return _cc_to_fly_array(in_ptr, np_arr.ndim, in_shape, in_dtype)
        else:
            return np_to_fly_array(np_arr.copy())

    from_ndarray = np_to_fly_array

try:
    import pycuda.gpuarray
except ImportError:
    FLY_PYCUDA_FOUND=False
else:
    from pycuda.gpuarray import GPUArray as CudaArray
    FLY_PYCUDA_FOUND=True

    def pycuda_to_fly_array(pycu_arr, copy=True):
        """
        Convert pycuda.gpuarray to flare.Array

        Parameters
        -----------
        pycu_arr  : pycuda.GPUArray()

        copy : Bool specifying if array is to be copied.
               Default is true.
               Can only be False if array is fortran contiguous.

        Returns
        ----------
        fly_arr    : flare.Array()

        Note
        ----------
        The input array is copied to fly.Array
        """

        in_ptr = pycu_arr.ptr
        in_shape = pycu_arr.shape
        in_dtype = pycu_arr.dtype.char

        if not copy and not pycu_arr.flags.f_contiguous:
            raise RuntimeError("Copy can only be False when arr.flags.f_contiguous is True")

        if (pycu_arr.flags.f_contiguous):
            return _fc_to_fly_array(in_ptr, in_shape, in_dtype, True, copy)
        elif (pycu_arr.flags.c_contiguous):
            return _cc_to_fly_array(in_ptr, pycu_arr.ndim, in_shape, in_dtype, True, copy)
        else:
            return pycuda_to_fly_array(pycu_arr.copy())

try:
    import numba
except ImportError:
    FLY_NUMBA_FOUND=False
else:
    from numba import cuda
    NumbaCudaArray = cuda.cudadrv.devicearray.DeviceNDArray
    FLY_NUMBA_FOUND=True

    def numba_to_fly_array(nb_arr, copy=True):
        """
        Convert numba.gpuarray to flare.Array

        Parameters
        -----------
        nb_arr  : numba.cuda.cudadrv.devicearray.DeviceNDArray()

        copy : Bool specifying if array is to be copied.
               Default is true.
               Can only be False if array is fortran contiguous.

        Returns
        ----------
        fly_arr    : flare.Array()

        Note
        ----------
        The input array is copied to fly.Array
        """

        in_ptr = nb_arr.device_ctypes_pointer.value
        in_shape = nb_arr.shape
        in_dtype = _nptype_to_aftype[nb_arr.dtype.str[1:]]

        if not copy and not nb_arr.flags.f_contiguous:
            raise RuntimeError("Copy can only be False when arr.flags.f_contiguous is True")

        if (nb_arr.is_f_contiguous()):
            return _fc_to_fly_array(in_ptr, in_shape, in_dtype, True, copy)
        elif (nb_arr.is_c_contiguous()):
            return _cc_to_fly_array(in_ptr, nb_arr.ndim, in_shape, in_dtype, True, copy)
        else:
            return numba_to_fly_array(nb_arr.copy())

def to_array(in_array, copy = True):
    """
    Helper function to convert input from a different module to fly.Array

    Parameters
    -------------

    in_array : array like object
             Can be one of the following:
             - numpy.ndarray
             - pycuda.GPUArray
             - numba.cuda.cudadrv.devicearray.DeviceNDArray
             - array.array
             - list
    copy : Bool specifying if array is to be copied.
          Default is true.
          Can only be False if array is fortran contiguous.

    Returns
    --------------
    fly.Array of same dimensions as input after copying the data from the input

    """
    if FLY_NUMPY_FOUND and isinstance(in_array, NumpyArray):
        return np_to_fly_array(in_array, copy)
    if FLY_PYCUDA_FOUND and isinstance(in_array, CudaArray):
        return pycuda_to_fly_array(in_array, copy)
    if FLY_NUMBA_FOUND and isinstance(in_array, NumbaCudaArray):
        return numba_to_fly_array(in_array, copy)
    return Array(src=in_array)
