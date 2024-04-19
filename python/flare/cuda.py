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
Functions specific to CUDA backend.

This module provides interoperability with other CUDA libraries.
"""

def get_stream(idx):
    """
    Get the CUDA stream used for the device `idx` by flare.

    Parameters
    ----------

    idx : int.
        Specifies the index of the device.

    Returns
    -----------
    stream : integer denoting the stream id.
    """

    import ctypes as ct
    from .util import safe_call as safe_call
    from .library import backend as backend

    if (backend.name() != "cuda"):
        raise RuntimeError("Invalid backend loaded")

    stream = c_void_ptr_t(0)
    safe_call(backend.get().afcu_get_stream(c_pointer(stream), idx))
    return stream.value

def get_native_id(idx):
    """
    Get native (unsorted) CUDA device ID

    Parameters
    ----------

    idx : int.
        Specifies the (sorted) index of the device.

    Returns
    -----------
    native_idx : integer denoting the native cuda id.
    """

    import ctypes as ct
    from .util import safe_call as safe_call
    from .library import backend as backend

    if (backend.name() != "cuda"):
        raise RuntimeError("Invalid backend loaded")

    native = c_int_t(0)
    safe_call(backend.get().afcu_get_native_id(c_pointer(native), idx))
    return native.value

def set_native_id(idx):
    """
    Set native (unsorted) CUDA device ID

    Parameters
    ----------

    idx : int.
        Specifies the (unsorted) native index of the device.
    """

    import ctypes as ct
    from .util import safe_call as safe_call
    from .library import backend as backend

    if (backend.name() != "cuda"):
        raise RuntimeError("Invalid backend loaded")

    safe_call(backend.get().afcu_set_native_id(idx))
    return

def set_cublas_mode(mode=CUBLAS_MATH_MODE.DEFAULT):
    """
    Set's cuBLAS math mode for CUDA backend. In other backends, this has no effect.
    """
    safe_call(backend().get().afcu_cublasSetMathMode(mode.value))
    return
