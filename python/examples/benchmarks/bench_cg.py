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


import sys
from time import time
import flare as fly

try:
    import numpy as np
except ImportError:
    np = None

try:
    from scipy import sparse as sp
    from scipy.sparse import linalg
except ImportError:
    sp = None


def to_numpy(A):
    return np.asarray(A.to_list(), dtype=np.float32)


def to_sparse(A):
    return fly.sparse.create_sparse_from_dense(A)


def to_scipy_sparse(spA, fmt='csr'):
    vals = np.asarray(fly.sparse.sparse_get_values(spA).to_list(),
                      dtype = np.float32)
    rows = np.asarray(fly.sparse.sparse_get_row_idx(spA).to_list(),
                      dtype = np.int)
    cols = np.asarray(fly.sparse.sparse_get_col_idx(spA).to_list(),
                      dtype = np.int)
    return sp.csr_matrix((vals, cols, rows), dtype=np.float32)


def setup_input(n, sparsity=7):
    T = fly.randu(n, n, dtype=fly.Dtype.f32)
    A = fly.floor(T*1000)
    A = A * ((A % sparsity) == 0) / 1000
    A = A.T + A + n*fly.identity(n, n, dtype=fly.Dtype.f32)
    x0 = fly.randu(n, dtype=fly.Dtype.f32)
    b = fly.matmul(A, x0)
    # printing
    # nnz = fly.sum((A != 0))
    # print "Sparsity of A: %2.2f %%" %(100*nnz/n**2,)
    return A, b, x0


def input_info(A, Asp):
    m, n = A.dims()
    nnz = fly.sum((A != 0))
    print("    matrix size:                %i x %i" %(m, n))
    print("    matrix sparsity:            %2.2f %%" %(100*nnz/n**2,))
    print("    dense matrix memory usage:  ")
    print("    sparse matrix memory usage: ")


def calc_flare(A, b, x0, maxiter=10):
    x = fly.constant(0, b.dims()[0], dtype=fly.Dtype.f32)
    r = b - fly.matmul(A, x)
    p = r
    for i in range(maxiter):
        Ap = fly.matmul(A, p)
        alpha_num = fly.dot(r, r)
        alpha_den = fly.dot(p, Ap)
        alpha = alpha_num/alpha_den
        r -= fly.tile(alpha, Ap.dims()[0]) * Ap
        x += fly.tile(alpha, Ap.dims()[0]) * p
        beta_num = fly.dot(r, r)
        beta = beta_num/alpha_num
        p = r + fly.tile(beta, p.dims()[0]) * p
    fly.eval(x)
    res = x0 - x
    return x, fly.dot(res, res)


def calc_numpy(A, b, x0, maxiter=10):
    x = np.zeros(len(b), dtype=np.float32)
    r = b - np.dot(A, x)
    p = r.copy()
    for i in range(maxiter):
        Ap = np.dot(A, p)
        alpha_num = np.dot(r, r)
        alpha_den = np.dot(p, Ap)
        alpha = alpha_num/alpha_den
        r -= alpha * Ap
        x += alpha * p
        beta_num = np.dot(r, r)
        beta = beta_num/alpha_num
        p = r + beta * p
    res = x0 - x
    return x, np.dot(res, res)


def calc_scipy_sparse(A, b, x0, maxiter=10):
    x = np.zeros(len(b), dtype=np.float32)
    r = b - A*x
    p = r.copy()
    for i in range(maxiter):
        Ap = A*p
        alpha_num = np.dot(r, r)
        alpha_den = np.dot(p, Ap)
        alpha = alpha_num/alpha_den
        r -= alpha * Ap
        x += alpha * p
        beta_num = np.dot(r, r)
        beta = beta_num/alpha_num
        p = r + beta * p
    res = x0 - x
    return x, np.dot(res, res)


def calc_scipy_sparse_linalg_cg(A, b, x0, maxiter=10):
    x = np.zeros(len(b), dtype=np.float32)
    x, _ = linalg.cg(A, b, x, tol=0., maxiter=maxiter)
    res = x0 - x
    return x, np.dot(res, res)


def timeit(calc, iters, args):
    t0 = time()
    for i in range(iters):
        calc(*args)
    dt = time() - t0
    return 1000*dt/iters  # ms


def test():
    print("\nTesting benchmark functions...")
    A, b, x0 = setup_input(n=50, sparsity=7)  # dense A
    Asp = to_sparse(A)
    x1, _ = calc_flare(A, b, x0)
    x2, _ = calc_flare(Asp, b, x0)
    if fly.sum(fly.abs(x1 - x2)/x2 > 1e-5):
        raise ValueError("flare test failed")
    if np:
        An = to_numpy(A)
        bn = to_numpy(b)
        x0n = to_numpy(x0)
        x3, _ = calc_numpy(An, bn, x0n)
        if not np.allclose(x3, x1.to_list()):
            raise ValueError("numpy test failed")
    if sp:
        Asc = to_scipy_sparse(Asp)
        x4, _ = calc_scipy_sparse(Asc, bn, x0n)
        if not np.allclose(x4, x1.to_list()):
            raise ValueError("scipy.sparse test failed")
        x5, _ = calc_scipy_sparse_linalg_cg(Asc, bn, x0n)
        if not np.allclose(x5, x1.to_list()):
            raise ValueError("scipy.sparse.linalg.cg test failed")
    print("    all tests passed...")


def bench(n=4*1024, sparsity=7, maxiter=10, iters=10):

    # generate data
    print("\nGenerating benchmark data for n = %i ..." %n)
    A, b, x0 = setup_input(n, sparsity)  # dense A
    Asp = to_sparse(A)  # sparse A
    input_info(A, Asp)

    # make benchmarks
    print("Benchmarking CG solver for n = %i ..." %n)
    t1 = timeit(calc_flare, iters, args=(A, b, x0, maxiter))
    print("    flare - dense:            %f ms" %t1)
    t2 = timeit(calc_flare, iters, args=(Asp, b, x0, maxiter))
    print("    flare - sparse:           %f ms" %t2)
    if np:
        An = to_numpy(A)
        bn = to_numpy(b)
        x0n = to_numpy(x0)
        t3 = timeit(calc_numpy, iters, args=(An, bn, x0n, maxiter))
        print("    numpy     - dense:            %f ms" %t3)
    if sp:
        Asc = to_scipy_sparse(Asp)
        t4 = timeit(calc_scipy_sparse, iters, args=(Asc, bn, x0n, maxiter))
        print("    scipy     - sparse:           %f ms" %t4)
        t5 = timeit(calc_scipy_sparse_linalg_cg, iters, args=(Asc, bn, x0n, maxiter))
        print("    scipy     - sparse.linalg.cg: %f ms" %t5)

if __name__ == "__main__":
    #fly.set_backend('cpu', unsafe=True)

    if (len(sys.argv) > 1):
        fly.set_device(int(sys.argv[1]))

    fly.info()
    test()

    for n in (128, 256, 512, 1024, 2048, 4096):
        bench(n)
