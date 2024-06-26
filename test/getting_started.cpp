// Copyright 2023 The EA Authors.
// part of Elastic AI Search
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include <flare.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <complex>
#include <vector>

using namespace fly;
using std::abs;
using std::vector;

TEST(GettingStarted, SNIPPET_getting_started_gen) {
    //! [ex_getting_started_constructors]
    // Arrays may be created using the array constructor and dimensioned
    // as 1D, 2D, 3D; however, the values in these arrays will be undefined
    array undefined_1D(100);         // 1D array with 100 elements
    array undefined_2D(10, 100);     // 2D array of size 10 x 100
    array undefined_3D(10, 10, 10);  // 3D array of size 10 x 10 x 10
    //! [ex_getting_started_constructors]

    //! [ex_getting_started_gen]
    // Generate an array of size three filled with zeros.
    // If no data type is specified, Flare defaults to f32.
    // The constant function generates the data on the device.
    array zeros = constant(0, 3);

    // Generate a 1x4 array of uniformly distributed [0,1] random numbers
    // The randu function generates the data on the device.
    array rand1 = randu(1, 4);

    // Generate a 2x2 array (or matrix, if you prefer) of random numbers
    // sampled from a normal distribution.
    // The randn function generates data on the device.
    array rand2 = randn(2, 2);

    // Generate a 3x3 identity matrix. The data is generated on the device.
    array iden = identity(3, 3);

    // Lastly, create a 2x1 array (column vector) of uniformly distributed
    // 32-bit complex numbers (c32 data type):
    array randcplx = randu(2, 1, c32);
    //! [ex_getting_started_gen]

    {
        vector<float> output;
        output.resize(zeros.elements());
        zeros.host(&output.front());
        ASSERT_EQ(f32, zeros.type());
        for (dim_t i = 0; i < zeros.elements(); i++)
            ASSERT_FLOAT_EQ(0, output[i]);
    }

    if (!noDoubleTests(f64)) {
        array ones = constant(1, 3, 2, f64);
        vector<double> output(ones.elements());
        ones.host(&output.front());
        ASSERT_EQ(f64, ones.type());
        for (dim_t i = 0; i < ones.elements(); i++)
            ASSERT_FLOAT_EQ(1, output[i]);
    }

    {
        vector<float> output;
        output.resize(iden.elements());
        iden.host(&output.front());
        for (dim_t i = 0; i < iden.dims(0); i++)
            for (dim_t j = 0; j < iden.dims(1); j++)
                if (i == j)
                    ASSERT_FLOAT_EQ(1, output[i * iden.dims(0) + j]);
                else
                    ASSERT_FLOAT_EQ(0, output[i * iden.dims(0) + j]);
    }
}

TEST(GettingStarted, SNIPPET_getting_started_init) {
    //! [ex_getting_started_init]
    // Create a six-element array on the host
    float hA[] = {0, 1, 2, 3, 4, 5};

    // Which can be copied into an Flare Array using the pointer copy
    // constructor. Here we copy the data into a 2x3 matrix:
    array A(2, 3, hA);

    // Flare provides a convenince function for printing array
    // objects in case you wish to see how the data is stored:
    fly_print(A);

    // This technique can also be used to populate an array with complex
    // data (stored in {{real, imaginary}, {real, imaginary},  ... } format
    // as found in C's complex.h and C++'s <complex>.
    // Below we create a 3x1 column vector of complex data values:
    array dB(3, 1, (cfloat *)hA);  // 3x1 column vector of complex numbers
    fly_print(dB);

    //! [ex_getting_started_init]

    vector<float> out(A.elements());
    A.host(&out.front());
    for (unsigned int i = 0; i < out.size(); i++)
        ASSERT_FLOAT_EQ(hA[i], out[i]);
}

TEST(GettingStarted, SNIPPET_getting_started_print) {
    //! [ex_getting_started_print]
    // Generate two arrays
    array a = randu(2, 2);
    array b = constant(1, 2, 1);

    // Print them to the console using fly_print
    fly_print(a);
    fly_print(b);

    // Print the results of an expression involving arrays:
    fly_print(a.col(0) + b + .4);
    //! [ex_getting_started_print]
    array result = a.col(0) + b + 0.4;
    vector<float> outa(a.elements());
    vector<float> outb(b.elements());
    vector<float> out(result.elements());

    a.host(&outa.front());
    b.host(&outb.front());
    result.host(&out.front());

    for (unsigned i = 0; i < outb.size(); i++)
        ASSERT_FLOAT_EQ(outa[i] + outb[i] + 0.4, out[i]);
}

TEST(GettingStarted, SNIPPET_getting_started_dims) {
    //! [ex_getting_started_dims]
    // Create a 4x5x2 array of uniformly distributed random numbers
    array a = randu(4, 5, 2);
    // Determine the number of dimensions using the numdims() function:
    printf("numdims(a)  %d\n", a.numdims());  // 3

    // We can also find the size of the individual dimentions using either
    // the `dims` function:
    printf("dims = [%lld %lld]\n", a.dims(0), a.dims(1));  // 4,5

    // Or the elements of a dim4 object:
    dim4 dims = a.dims();
    printf("dims = [%lld %lld]\n", dims[0], dims[1]);  // 4,5
    //! [ex_getting_started_dims]

    //! [ex_getting_started_prop]
    // Get the type stored in the array. This will be one of the many
    // `fly_dtype`s presented above:
    printf("underlying type: %d\n", a.type());

    // Arrays also have several conveience functions to determine if
    // an Array contains complex or real values:
    printf("is complex? %d    is real? %d\n", a.iscomplex(), a.isreal());

    // if it is a column or row vector
    printf("is vector? %d  column? %d  row? %d\n", a.isvector(), a.iscolumn(),
           a.isrow());

    // and whether or not the array is empty and how much memory it takes on
    // the device:
    printf("empty? %d  total elements: %lld  bytes: %zu\n", a.isempty(),
           a.elements(), a.bytes());
    //! [ex_getting_started_prop]

    ASSERT_EQ(f32, a.type());
    ASSERT_TRUE(a.isreal());
    ASSERT_FALSE(a.iscomplex());
    ASSERT_FALSE(a.isvector());
    ASSERT_FALSE(a.iscolumn());
    ASSERT_FALSE(a.isrow());
    ASSERT_FALSE(a.isempty());
    ASSERT_EQ(40, a.elements());

    ASSERT_EQ(f32, a.type());
    ASSERT_EQ(f32, a.type());

    ASSERT_EQ(4, dims[0]);
    ASSERT_EQ(4, a.dims(0));
    ASSERT_EQ(5, dims[1]);
    ASSERT_EQ(5, a.dims(1));
}

TEST(GettingStarted, SNIPPET_getting_started_arith) {
    //! [ex_getting_started_arith]
    array R = randu(3, 3);
    fly_print(constant(1, 3, 3) + complex(sin(R)));  // will be c32

    // rescale complex values to unit circle
    array a = randn(5, c32);
    fly_print(a / abs(a));

    // calculate L2 norm of vectors
    array X = randn(3, 4);
    fly_print(sqrt(sum(pow(X, 2))));     // norm of every column vector
    fly_print(sqrt(sum(pow(X, 2), 0)));  // same as above
    fly_print(sqrt(sum(pow(X, 2), 1)));  // norm of every row vector
    //! [ex_getting_started_arith]
}

TEST(GettingStarted, SNIPPET_getting_started_dev_ptr) {
#ifdef __CUDACC__
    //! [ex_getting_started_dev_ptr]
    // Create an array on the host, copy it into an Flare 2x3 Flare
    // array
    float host_ptr[] = {0, 1, 2, 3, 4, 5};
    array a(2, 3, host_ptr);

    // Create a CUDA device pointer, populate it with data from the host
    float *device_ptr;
    cudaMalloc((void **)&device_ptr, 6 * sizeof(float));
    cudaMemcpy(device_ptr, host_ptr, 6 * sizeof(float), cudaMemcpyHostToDevice);

    // Convert the CUDA-allocated device memory into an Flare array:
    array b(2, 3, device_ptr, flyDevice);  // Note: flyDevice (default: flyHost)
    // Note that Flare takes ownership over `device_ptr`, so memory will
    // be freed when `b` id destructed. Do not call cudaFree(device_ptr)!

    //! [ex_getting_started_dev_ptr]
#endif  //__CUDACC__
}

TEST(GettingStarted, SNIPPET_getting_started_ptr) {
#ifdef __CUDACC__
    //! [ex_getting_started_ptr]
    // Create an array consisting of 3 random numbers
    array a = randu(3, f32);

    // Copy an array on the device to the host:
    float *host_a = a.host<float>();
    // access the host data as a normal array
    printf("host_a[2] = %g\n", host_a[2]);  // last element
    // and free memory using freeHost:
    freeHost(host_a);

    // Get access to the device memory for a CUDA kernel
    float *d_cuda = a.device<float>();  // no need to free this
    float value;
    cudaMemcpy(&value, d_cuda + 2, sizeof(float), cudaMemcpyDeviceToHost);
    printf("d_cuda[2] = %g\n", value);
    a.unlock();  // unlock to allow garbage collection if necessary

    //! [ex_getting_started_ptr]
#endif  //__CUDACC__
}

TEST(GettingStarted, SNIPPET_getting_started_scalar) {
    //! [ex_getting_started_scalar]
    array a   = randu(3);
    float val = a.scalar<float>();
    printf("scalar value: %g\n", val);
    //! [ex_getting_started_scalar]
}

TEST(GettingStarted, SNIPPET_getting_started_bit) {
    //! [ex_getting_started_bit]
    int h_A[] = {1, 1, 0, 0, 4, 0, 0, 2, 0};
    int h_B[] = {1, 0, 1, 0, 1, 0, 1, 1, 1};
    array A = array(3, 3, h_A), B = array(3, 3, h_B);
    fly_print(A);
    fly_print(B);

    array A_and_B = A & B;
    fly_print(A_and_B);
    array A_or_B = A | B;
    fly_print(A_or_B);
    array A_xor_B = A ^ B;
    fly_print(A_xor_B);
    //! [ex_getting_started_bit]

    vector<int> Andout(A_and_B.elements());
    vector<int> Orout(A_or_B.elements());
    vector<int> Xorout(A_xor_B.elements());
    A_and_B.host(&Andout.front());
    A_or_B.host(&Orout.front());
    A_xor_B.host(&Xorout.front());

    for (unsigned int i = 0; i < Andout.size(); i++)
        ASSERT_FLOAT_EQ(h_A[i] & h_B[i], Andout[i]);
    for (unsigned int i = 0; i < Orout.size(); i++)
        ASSERT_FLOAT_EQ(h_A[i] | h_B[i], Orout[i]);
    for (unsigned int i = 0; i < Xorout.size(); i++)
        ASSERT_FLOAT_EQ(h_A[i] ^ h_B[i], Xorout[i]);
}

TEST(GettingStarted, SNIPPET_getting_started_constants) {
    //! [ex_getting_started_constants]
    array A          = randu(5, 5);
    A(where(A > .5)) = NaN;

    array x = randu(10e6), y = randu(10e6);
    double pi_est = 4 * sum<float>(hypot(x, y) < 1) / 10e6;
    printf("estimation error: %g\n", fabs(Pi - pi_est));
    //! [ex_getting_started_constants]

    ASSERT_LE(fabs(Pi - pi_est), 0.005);
}

TEST(GettingStarted, SNIPPET_JohnTest) {
    array a = iota(dim4(2, 3));
    array b = sum(a);     // sum across the first axis, same as sum(a, 0)
    array c = sum(a, 1);  // sum across the second axis
    array d = sum(a, 2);  // sum across the third axis
    array e = sum(a, 3);  // sum acorss the fourth axis
    // array f = sum(a, 4); fails due to stepping out of bounds
    fly_print(a);
    fly_print(b);
    fly_print(c);
    fly_print(d);
    fly_print(e);
}