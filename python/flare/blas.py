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
BLAS functions (matmul, dot, etc)
"""

from .library import *
from .array import *

def matmul(lhs, rhs, lhs_opts=MATPROP.NONE, rhs_opts=MATPROP.NONE):
    """
    Generalized matrix multiplication for two matrices.

    Parameters
    ----------

    lhs : fly.Array
          A 2 dimensional, real or complex flare array.

    rhs : fly.Array
          A 2 dimensional, real or complex flare array.

    lhs_opts: optional: fly.MATPROP. default: fly.MATPROP.NONE.
              Can be one of
               - fly.MATPROP.NONE   - If no op should be done on `lhs`.
               - fly.MATPROP.TRANS  - If `lhs` has to be transposed before multiplying.
               - fly.MATPROP.CTRANS - If `lhs` has to be hermitian transposed before multiplying.

    rhs_opts: optional: fly.MATPROP. default: fly.MATPROP.NONE.
              Can be one of
               - fly.MATPROP.NONE   - If no op should be done on `rhs`.
               - fly.MATPROP.TRANS  - If `rhs` has to be transposed before multiplying.
               - fly.MATPROP.CTRANS - If `rhs` has to be hermitian transposed before multiplying.

    Returns
    -------

    out : fly.Array
          Output of the matrix multiplication on `lhs` and `rhs`.

    Note
    -----

    - The data types of `lhs` and `rhs` should be the same.
    - Batches are not supported.

    """
    out = Array()
    safe_call(backend.get().fly_matmul(c_pointer(out.arr), lhs.arr, rhs.arr,
                                      lhs_opts.value, rhs_opts.value))
    return out

def matmulTN(lhs, rhs):
    """
    Matrix multiplication after transposing the first matrix.

    Parameters
    ----------

    lhs : fly.Array
          A 2 dimensional, real or complex flare array.

    rhs : fly.Array
          A 2 dimensional, real or complex flare array.

    Returns
    -------

    out : fly.Array
          Output of the matrix multiplication on `transpose(lhs)` and `rhs`.

    Note
    -----

    - The data types of `lhs` and `rhs` should be the same.
    - Batches are not supported.

    """
    out = Array()
    safe_call(backend.get().fly_matmul(c_pointer(out.arr), lhs.arr, rhs.arr,
                                      MATPROP.TRANS.value, MATPROP.NONE.value))
    return out

def matmulNT(lhs, rhs):
    """
    Matrix multiplication after transposing the second matrix.

    Parameters
    ----------

    lhs : fly.Array
          A 2 dimensional, real or complex flare array.

    rhs : fly.Array
          A 2 dimensional, real or complex flare array.

    Returns
    -------

    out : fly.Array
          Output of the matrix multiplication on `lhs` and `transpose(rhs)`.

    Note
    -----

    - The data types of `lhs` and `rhs` should be the same.
    - Batches are not supported.

    """
    out = Array()
    safe_call(backend.get().fly_matmul(c_pointer(out.arr), lhs.arr, rhs.arr,
                                      MATPROP.NONE.value, MATPROP.TRANS.value))
    return out

def matmulTT(lhs, rhs):
    """
    Matrix multiplication after transposing both inputs.

    Parameters
    ----------

    lhs : fly.Array
          A 2 dimensional, real or complex flare array.

    rhs : fly.Array
          A 2 dimensional, real or complex flare array.

    Returns
    -------

    out : fly.Array
          Output of the matrix multiplication on `transpose(lhs)` and `transpose(rhs)`.

    Note
    -----

    - The data types of `lhs` and `rhs` should be the same.
    - Batches are not supported.

    """
    out = Array()
    safe_call(backend.get().fly_matmul(c_pointer(out.arr), lhs.arr, rhs.arr,
                                      MATPROP.TRANS.value, MATPROP.TRANS.value))
    return out

def dot(lhs, rhs, lhs_opts=MATPROP.NONE, rhs_opts=MATPROP.NONE, return_scalar = False):
    """
    Dot product of two input vectors.

    Parameters
    ----------

    lhs : fly.Array
          A 1 dimensional, real or complex flare array.

    rhs : fly.Array
          A 1 dimensional, real or complex flare array.

    lhs_opts: optional: fly.MATPROP. default: fly.MATPROP.NONE.
              Can be one of
               - fly.MATPROP.NONE   - If no op should be done on `lhs`.
               - No other options are currently supported.

    rhs_opts: optional: fly.MATPROP. default: fly.MATPROP.NONE.
              Can be one of
               - fly.MATPROP.NONE   - If no op should be done on `rhs`.
               - No other options are currently supported.

    return_scalar: optional: bool. default: False.
               - When set to true, the input arrays are flattened and the output is a scalar

    Returns
    -------

    out : fly.Array or scalar
          Output of dot product of `lhs` and `rhs`.

    Note
    -----

    - The data types of `lhs` and `rhs` should be the same.
    - Batches are not supported.

    """
    if return_scalar:
        real = c_double_t(0)
        imag = c_double_t(0)
        safe_call(backend.get().fly_dot_all(c_pointer(real), c_pointer(imag),
                                           lhs.arr, rhs.arr, lhs_opts.value, rhs_opts.value))
        real = real.value
        imag = imag.value
        return real if imag == 0 else real + imag * 1j
    else:
        out = Array()
        safe_call(backend.get().fly_dot(c_pointer(out.arr), lhs.arr, rhs.arr,
                                       lhs_opts.value, rhs_opts.value))
        return out

def gemm(lhs, rhs, alpha=1.0, beta=0.0, lhs_opts=MATPROP.NONE, rhs_opts=MATPROP.NONE, C=None):
    """
    BLAS general matrix multiply (GEMM) of two fly_array objects.

    This provides a general interface to the BLAS level 3 general matrix multiply (GEMM), which is generally defined as:

    C = alpha * opA(A) opB(B) + beta * C

    where alpha and beta are both scalars; A and B are the matrix multiply operands;
    and opA and opB are noop (if AF_MAT_NONE) or transpose (if AF_MAT_TRANS) operations
    on A or B before the actual GEMM operation.
    Batched GEMM is supported if at least either A or B have more than two dimensions
    (see af::matmul for more details on broadcasting).
    However, only one alpha and one beta can be used for all of the batched matrix operands.

    Parameters
    ----------

    lhs : fly.Array
          A 2 dimensional, real or complex flare array.

    rhs : fly.Array
          A 2 dimensional, real or complex flare array.

    alpha : scalar

    beta : scalar

    lhs_opts: optional: fly.MATPROP. default: fly.MATPROP.NONE.
              Can be one of
               - fly.MATPROP.NONE   - If no op should be done on `lhs`.
               - fly.MATPROP.TRANS  - If `lhs` has to be transposed before multiplying.
               - fly.MATPROP.CTRANS - If `lhs` has to be hermitian transposed before multiplying.

    rhs_opts: optional: fly.MATPROP. default: fly.MATPROP.NONE.
              Can be one of
               - fly.MATPROP.NONE   - If no op should be done on `rhs`.
               - fly.MATPROP.TRANS  - If `rhs` has to be transposed before multiplying.
               - fly.MATPROP.CTRANS - If `rhs` has to be hermitian transposed before multiplying.

    Returns
    -------

    out : fly.Array
          Output of the matrix multiplication on `lhs` and `rhs`.

    Note
    -----

    - The data types of `lhs` and `rhs` should be the same.
    - Batches are not supported.

    """
    if C is None:
        out = Array()
    else:
        out = C

    ltype = lhs.dtype()

    if ltype == Dtype.f32:
        aptr = c_cast(c_pointer(c_float_t(alpha)),c_void_ptr_t)
        bptr = c_cast(c_pointer(c_float_t(beta)), c_void_ptr_t)
    elif ltype == Dtype.c32:
        if isinstance(alpha, fly_cfloat_t):
            aptr = c_cast(c_pointer(alpha), c_void_ptr_t)
        elif isinstance(alpha, tuple):
            aptr = c_cast(c_pointer(fly_cfloat_t(alpha[0], alpha[1])), c_void_ptr_t)
        else:
            aptr = c_cast(c_pointer(fly_cfloat_t(alpha)), c_void_ptr_t)

        if isinstance(beta, fly_cfloat_t):
            bptr = c_cast(c_pointer(beta), c_void_ptr_t)
        elif isinstance(beta, tuple):
            bptr = c_cast(c_pointer(fly_cfloat_t(beta[0], beta[1])), c_void_ptr_t)
        else:
            bptr = c_cast(c_pointer(fly_cfloat_t(beta)), c_void_ptr_t)

    elif ltype == Dtype.f64:
        aptr = c_cast(c_pointer(c_double_t(alpha)),c_void_ptr_t)
        bptr = c_cast(c_pointer(c_double_t(beta)), c_void_ptr_t)
    elif ltype == Dtype.c64:
        if isinstance(alpha, fly_cdouble_t):
            aptr = c_cast(c_pointer(alpha), c_void_ptr_t)
        elif isinstance(alpha, tuple):
            aptr = c_cast(c_pointer(fly_cdouble_t(alpha[0], alpha[1])), c_void_ptr_t)
        else:
            aptr = c_cast(c_pointer(fly_cdouble_t(alpha)), c_void_ptr_t)

        if isinstance(beta, fly_cdouble_t):
            bptr = c_cast(c_pointer(beta), c_void_ptr_t)
        elif isinstance(beta, tuple):
            bptr = c_cast(c_pointer(fly_cdouble_t(beta[0], beta[1])), c_void_ptr_t)
        else:
            bptr = c_cast(c_pointer(fly_cdouble_t(beta)), c_void_ptr_t)
    elif ltype == Dtype.f16:
        raise TypeError("fp16 currently unsupported gemm() input type")
    else:
        raise TypeError("unsupported input type")


    safe_call(backend.get().fly_gemm(c_pointer(out.arr),
                                    lhs_opts.value, rhs_opts.value,
                                    aptr, lhs.arr, rhs.arr, bptr))
    return out
