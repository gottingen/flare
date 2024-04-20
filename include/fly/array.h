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

#pragma once
#include <fly/compilers.h>
#include <fly/defines.h>
#include <fly/device.h>
#include <fly/dim4.hpp>
#include <fly/exception.h>
#include <fly/index.h>
#include <fly/seq.h>
#include <fly/util.h>

#ifdef __cplusplus
#include <fly/traits.hpp>

#if FLY_COMPILER_CXX_GENERALIZED_INITIALIZERS
#include <initializer_list>
#endif

namespace fly
{

    class dim4;

    ///
    /// \brief A multi dimensional data container
    /// \ingroup flare_class
    class FLY_API array {
        fly_array   arr;


    public:
        ///
        /// \brief Updates the internal \ref fly_array object
        ///
        /// \note This function will reduce the reference of the previous
        ///       \ref fly_array object
        ///
        void set(fly_array tmp);

        ///
        /// \brief Intermediate data class. Used for assignment and indexing.
        ///
        /// \note This class is for internal book keeping while indexing. This
        ///       class is not intended for use in user code.
        ///
        class FLY_API array_proxy
        {
            struct array_proxy_impl;    //forward declaration
            array_proxy_impl *impl;     // implementation

        public:
            array_proxy(array& par, fly_index_t *ssss, bool linear = false);
            array_proxy(const array_proxy &other);
#if FLY_COMPILER_CXX_RVALUE_REFERENCES
            array_proxy(array_proxy &&other);
            array_proxy & operator=(array_proxy &&other);
#endif
            ~array_proxy();

            // Implicit conversion operators
            operator array() const;
            operator array();

#define ASSIGN(OP)                                                  \
            array_proxy& operator OP(const array_proxy &a);         \
            array_proxy& operator OP(const array &a);               \
            array_proxy& operator OP(const double &a);              \
            array_proxy& operator OP(const cdouble &a);             \
            array_proxy& operator OP(const cfloat &a);              \
            array_proxy& operator OP(const float &a);               \
            array_proxy& operator OP(const int &a);                 \
            array_proxy& operator OP(const unsigned &a);            \
            array_proxy& operator OP(const bool &a);                \
            array_proxy& operator OP(const char &a);                \
            array_proxy& operator OP(const unsigned char &a);       \
            array_proxy& operator OP(const long  &a);               \
            array_proxy& operator OP(const unsigned long &a);       \
            array_proxy& operator OP(const long long  &a);          \
            array_proxy& operator OP(const unsigned long long &a);

            ASSIGN(=)
            ASSIGN(+=)
            ASSIGN(-=)
            ASSIGN(*=)
            ASSIGN(/=)
#undef ASSIGN

#define ASSIGN(OP)                                                  \
            array_proxy& operator OP(const short &a);               \
            array_proxy& operator OP(const unsigned short &a);

            ASSIGN(=)
            ASSIGN(+=)
            ASSIGN(-=)
            ASSIGN(*=)
            ASSIGN(/=)
#undef ASSIGN

            // fly::array member functions. same behavior as those below
            fly_array get();
            fly_array get() const;
            dim_t elements() const;
            template<typename T> T* host() const;
            void host(void *ptr) const;
            dtype type() const;
            dim4 dims() const;
            dim_t dims(unsigned dim) const;
            unsigned numdims() const;
            size_t bytes() const;
            size_t allocated() const;
            array copy() const;
            bool isempty() const;
            bool isscalar() const;
            bool isvector() const;
            bool isrow() const;
            bool iscolumn() const;
            bool iscomplex() const;
            inline bool isreal() const { return !iscomplex(); }
            bool isdouble() const;
            bool issingle() const;
            bool ishalf() const;
            bool isrealfloating() const;
            bool isfloating() const;
            bool isinteger() const;
            bool isbool() const;
            bool issparse() const;
            void eval() const;
            array as(dtype type) const;
            array T() const;
            array H() const;
            template<typename T> T scalar() const;
            template<typename T> T* device() const;
            void unlock() const;
            void lock() const;

            bool isLocked() const;

                  array::array_proxy row(int index);
            const array::array_proxy row(int index) const;

                  array::array_proxy rows(int first, int last);
            const array::array_proxy rows(int first, int last) const;

                  array::array_proxy col(int index);
            const array::array_proxy col(int index) const;
                  array::array_proxy cols(int first, int last);
            const array::array_proxy cols(int first, int last) const;

                  array::array_proxy slice(int index);
            const array::array_proxy slice(int index) const;

                  array::array_proxy slices(int first, int last);
            const array::array_proxy slices(int first, int last) const;
        };

        /**
            Create an uninitialized array (no data, undefined size)

            \code
            array A, B, C;   // creates three arrays called A, B and C
            \endcode
        */
        array();

#if FLY_COMPILER_CXX_RVALUE_REFERENCES
        /**
            Move constructor

            Moves the \p other fly::array into the current fly::array. After this
            operation, the \p other array will not be left uninitialized.

            \param[in] other The array to be moved
        */
        array(array &&other) FLY_NOEXCEPT;

        /**
            Move assignment operator

            Moves the array into the current array. After this operation the
            \p other array is left uninitialized. The previously referenced
            fly_array of the current object is released.

            \param[in] other The array to be moved
            \returns the reference to the current array
        */
        array &operator=(array &&other) FLY_NOEXCEPT;
#endif
        /**
            Creates an array from an \ref fly_array handle. Does not increment
            a reference counter: the array assumes ownership of the handle. To
            share the array between multiple objects, use this in conjunction
            with \ref fly_retain_array.
            \param handle the fly_array object.
         */
        explicit
        array(const fly_array handle);

        /**
            Creates a copy to the \p in array.

            \param in The input \ref array
         */
        array(const array& in);

        /**
            Allocate a one-dimensional array of a specified size with undefined
            contents

            Declare a two-dimensional array by passing the
            number of rows and the number of columns as the first two parameters.
            The (optional) second parameter is the type of the array. The default
            type is f32 or 4-byte single-precision floating-point numbers.

            \code
            // allocate space for an array with 10 rows
            array A(10);          // type is the default f32

            // allocate space for a column vector with 100 rows
            array A(100, f64);    // f64 = double precision
            \endcode

            \param[in] dim0 number of columns in the array
            \param[in] ty   optional label describing the data type
                       (default is f32)

        */
        explicit
        array(dim_t dim0, dtype ty = f32);

        /**
            Allocate a two-dimensional array of a specified size with undefined
            contents

            Declare a two-dimensional array by passing the
            number of rows and the number of columns as the first two parameters.
            The (optional) third parameter is the type of the array. The default
            type is f32 or 4-byte single-precision floating-point numbers.

            \code
            // allocate space for an array with 10 rows and 8 columns
            array A(10, 8);          // type is the default f32

            // allocate space for a column vector with 100 rows (and 1 column)
            array A(100, 1, f64);    // f64 = double precision
            \endcode

            \param[in] dim0 number of columns in the array
            \param[in] dim1 number of rows in the array
            \param[in] ty optional label describing the data type
                       (default is f32)

        */
        explicit
        array(dim_t dim0, dim_t dim1, dtype ty = f32);

        /**
            Allocate a three-dimensional (3D) array of a specified size with
            undefined contents

            This is useful to quickly declare a three-dimensional array by
            passing the size as the first three parameters. The (optional)
            fourth parameter is the type of the array. The default type is f32
            or 4-byte single-precision floating point numbers.

            \code
            // allocate space for a 10 x 10 x 10 array
            array A(10, 10, 10);          // type is the default f32

            // allocate space for a 3D, double precision array
            array A(10, 10, 10, f64);     // f64 = double precision
            \endcode

            \param[in] dim0 first dimension of the array
            \param[in] dim1 second dimension of the array
            \param[in] dim2 third dimension of the array
            \param[in] ty optional label describing the data type
                       (default is f32)

        */
        explicit
        array(dim_t dim0, dim_t dim1, dim_t dim2, dtype ty = f32);

        /**
            Allocate a four-dimensional (4D) array of a specified size with
            undefined contents

            This is useful to quickly declare a four-dimensional array by
            passing the size as the first four parameters. The (optional) fifth
            parameter is the type of the array. The default type is f32 or
            4-byte floating point numbers.

            \code
            // allocate space for a 10 x 10 x 10 x 20 array
            array A(10, 10, 10, 20);          // type is the default f32

            // allocate space for a 4D, double precision array
            array A(10, 10, 10, 20, f64);     // f64 = double precision
            \endcode

            \param[in] dim0 first dimension of the array
            \param[in] dim1 second dimension of the array
            \param[in] dim2 third dimension of the array
            \param[in] dim3 fourth dimension of the array
            \param[in] ty optional label describing the data type
                       (default is f32)

        */
        explicit
        array(dim_t dim0, dim_t dim1, dim_t dim2, dim_t dim3, dtype ty = f32);

        /**
            Allocate an array of a specified size with undefined contents

            This can be useful when the dimensions of the array are calculated
            somewhere else within the code. The first parameter specifies the
            size of the array via dim4(). The second parameter is the type of
            the array. The default type is f32 or 4-byte
            single-precision floating point numbers.

            \code

            // create a two-dimensional 10 x 10 array
            dim4 dims(10, 10);       // converted to (10, 10, 1, 1)
            array a1(dims);          // create the array (type is f32, the default)

            // create a three-dimensional 10 x 10 x 20 array
            dim4 dims(10, 10, 20);   // converted to (10, 10, 20, 1)
            array a2(dims,f64);      // f64 = double precision

            \endcode

            \param[in] dims size of the array
            \param[in] ty optional label describing the data type
                       (default is f32)
        */
        explicit
        array(const dim4& dims, dtype ty = f32);

        /**
            Create a column vector on the device using a host/device pointer

            \param[in] dim0     number of elements in the column vector
            \param[in] pointer  pointer (points to a buffer on the host/device)
            \param[in] src      source of the data (default is flyHost, can also
                                be flyDevice)

            \code
            // allocate data on the host
            int h_buffer[] = {23, 34, 18, 99, 34};

            array A(4, h_buffer);   // copy host data to device
                                    //
                                    // A = [23]
                                    //     [34]
                                    //     [18]
                                    //     [99]

            \endcode

            \note If \p src is \ref flyHost, the first \p dim0 elements are
                  copied. If \p src is \ref flyDevice, no copy is done; the
                  array object wraps the device pointer AND takes ownership
                  of the underlying memory.

        */
        template<typename T>
        explicit
        array(dim_t dim0,
              const T *pointer, fly::source src=flyHost);


        /**
            Create a 2D array on the device using a host/device pointer

            \param[in] dim0     number of rows
            \param[in] dim1     number of columns
            \param[in] pointer  pointer (points to a buffer on the host/device)
            \param[in] src      source of the data (default is flyHost, can also
                                be \ref flyDevice)

            \code
            int h_buffer[] = {0, 1, 2, 3, 4, 5};  // host array
            array A(2, 3, h_buffer);              // copy host data to device
            \endcode

            \image html 2dArray.png

            \note If \p src is \ref flyHost, the first \p dim0 * \p dim1 elements
                  are copied. If \p src is \ref flyDevice, no copy is done; the
                  array object wraps the device pointer AND takes ownership of
                  the underlying memory. The data is treated as column major
                  format when performing linear algebra operations.
        */
        template<typename T>
        explicit
        array(dim_t dim0, dim_t dim1,
              const T *pointer, fly::source src=flyHost);


        /**
            Create a 3D array on the device using a host/device pointer

            \param[in] dim0     first dimension
            \param[in] dim1     second dimension
            \param[in] dim2     third dimension
            \param[in] pointer  pointer (points to a buffer on the host/device)
            \param[in] src      source of the data (default is \ref flyHost, can
                                also be \ref flyDevice)

            \code
            int h_buffer[] = {0, 1, 2, 3, 4, 5, 6, 7, 8
                              9, 0, 1, 2, 3, 4, 5, 6, 7};   // host array

            array A(3, 3, 2,  h_buffer);   // copy host data to 3D device array
            \endcode

            \note If \p src is \ref flyHost, the first \p dim0 * \p dim1 *
                  \p dim2 elements are copied. If \p src is \ref flyDevice, no
                  copy is done; the array object just wraps the device pointer
                  and does not take ownership of the underlying memory. The data
                  is treated as column major format when performing linear
                  algebra operations.

            \image html 3dArray.png
        */
        template<typename T>
        explicit
        array(dim_t dim0, dim_t dim1, dim_t dim2,
              const T *pointer, fly::source src=flyHost);


        /**
            Create a 4D array on the device using a host/device pointer

            \param[in] dim0     first dimension
            \param[in] dim1     second dimension
            \param[in] dim2     third dimension
            \param[in] dim3     fourth dimension
            \param[in] pointer  pointer (points to a buffer on the host/device)
            \param[in] src      source of the data (default is flyHost, can also
                                be \ref flyDevice)

            \code
            int h_buffer[] = {0, 1, 2, 3,
                              4, 5, 6, 7,
                              8, 9, 0, 1,
                              2, 3, 4, 5};   // host array with 16 elements

            array A(2, 2, 2, 2, h_buffer);   // copy host data to 4D device array
            \endcode

            \note If \p src is \ref flyHost, the first \p dim0 * \p dim1 *
                  \p dim2 * \p dim3 elements are copied. If \p src is
                  \ref flyDevice, no copy is done; the array object just wraps
                  the device pointer and does not take ownership of the
                  underlying memory. The data is treated as column major format
                  when performing linear algebra operations.

        */
        template<typename T>
        explicit
        array(dim_t dim0, dim_t dim1, dim_t dim2, dim_t dim3,
              const T *pointer, fly::source src=flyHost);

        /**
            Create an array of specified size on the device using a host/device
            pointer

            This function copies data from the location specified by the
            pointer to a 1D/2D/3D/4D array on the device. The data is arranged
            in "column-major" format (similar to that used by FORTRAN).

            \param[in] dims    vector data type containing the dimension of the
                               \ref array
            \param[in] pointer pointer (points to a buffer on the host/device)
            \param[in] src     source of the data (default is flyHost, can also
                                be \ref flyDevice)

            \code
            int h_buffer[] = {0, 1, 2, 3,    // host array with 16 elements
                              4, 5, 6, 7,    // written in "row-major" format
                              8, 9, 0, 1,
                              2, 3, 4, 5};

            dim4 dims(4, 4);

            array A(dims, h_buffer);         // A  =  0  4  8  2
                                             //       1  5  9  3
                                             //       2  6  0  4
                                             //       3  7  1  5

                                             // Note the "column-major" ordering
                                             // used in Flare
            \endcode

            \note If \p src is \ref flyHost, the first dims.elements() elements
                  are copied. If \p src is \ref flyDevice, no copy is done; the
                  array object just wraps the device pointer and does not take
                  ownership of the underlying memory. The data is treated as
                  column major format when performing linear algebra operations.

        */
        template<typename T>
        explicit
        array(const dim4& dims,
              const T *pointer, fly::source src=flyHost);

#if FLY_COMPILER_CXX_GENERALIZED_INITIALIZERS
        /// \brief Initializer list constructor
        template <typename T, typename = typename std::enable_if<
                                  std::is_fundamental<T>::value, void>::type>
        array(std::initializer_list<T> list)
        : arr(nullptr) {
          dim_t size = list.size();
          if (fly_err __aferr = fly_create_array(&arr, list.begin(), 1, &size,
                              static_cast<fly_dtype>(fly::dtype_traits<T>::fly_type))) {
            char *msg = NULL;
            fly_get_last_error(&msg, NULL);
            fly::exception ex(msg, __PRETTY_FUNCTION__, "include/fly/array.h",
                             __LINE__, __aferr);
            fly_free_host(msg);
            throw std::move(ex);
          }
        }

        /// \brief Initializer list constructor
        template <typename T, typename = typename std::enable_if<
                                  std::is_fundamental<T>::value, void>::type>
        array(const fly::dim4 &dims, std::initializer_list<T> list)
            : arr(nullptr) {
          const dim_t *size = dims.get();
          if (fly_err __aferr = fly_create_array(
              &arr, list.begin(), FLY_MAX_DIMS, size,
              static_cast<fly_dtype>(fly::dtype_traits<T>::fly_type))) {
            char *msg = NULL;
            fly_get_last_error(&msg, NULL);
            fly::exception ex(msg, __PRETTY_FUNCTION__, "include/fly/array.h",
                             __LINE__, __aferr);
            fly_free_host(msg);
            throw std::move(ex);
          }
        }
#endif

        /**
           Adjust the dimensions of an N-D array (fast).

           This operation simply rearranges the description of the array.
           No memory transfers or transformations are  performed. The total
           number of elements must not change.

           \code
           float f[] = {1,2,3,4};
           array a(2,2,f);
           //a=[1 3]
           //  [2 4]

           array b = array(a, dim4(4));
           //b=[1]
           //  [2]
           //  [3]
           //  [4]

           array c = array(a, b.T().dims() );
           //c=[1 2 3 4]
           \endcode

           \param[in] input
           \param[in] dims total number of elements must not change.
        */
        array(const array& input, const dim4& dims);

        /**
           Adjust the dimensions of an N-D array (fast).

           This operation simply rearranges the description of the array.
           No memory transfers or transformations are  performed. The total
           number of elements must not change.

           \code

           float f[] = {1,2,3,4};
           array a(2,2,f);
           //a=[1 3]
           //  [2 4]

           array b = array(a, 4);
           //b=[1]
           //  [2]
           //  [3]
           //  [4]

           array c = array(a, 1, 4);
           //c=[1 2 3 4]
           \endcode

           \param[in] input
           \param[in] dim0 first dimension
           \param[in] dim1 second dimension
           \param[in] dim2 third dimension
           \param[in] dim3 fourth dimension
        */
        array(  const array& input,
                const dim_t dim0, const dim_t dim1 = 1,
                const dim_t dim2 = 1, const dim_t dim3 = 1);

        /**
           get the \ref fly_array handle
        */
        fly_array get();

        /**
           get the \ref fly_array handle
        */
        fly_array get() const;

        /**
           Get the total number of elements across all dimensions of the array
        */
        dim_t elements() const;

        /**
           Copy array data to host and return host pointer
        */
        template<typename T> T* host() const;

        /**
           Copy array data to existing host pointer
        */
        void host(void *ptr) const;

        /**
           Perform deep copy from host/device pointer to an existing array
        */
        template<typename T> void write(const T *ptr, const size_t bytes, fly::source src = flyHost);

        /**
           Get array data type
        */
        dtype type() const;

        /**
           Get dimensions of the array
        */
        dim4 dims() const;

        /**
           Get dimensions of the array
        */
        dim_t dims(unsigned dim) const;

        /**
           Get the number of dimensions of the array
        */
        unsigned numdims() const;

        /**
           Get the size of the array in bytes
        */
        size_t bytes() const;

        /**
           Get the size of the array in memory. This will return the parent's
           bytes() if the array is indexed.
        */
        size_t allocated() const;

        /**
           Perform deep copy of the array
        */
        array copy() const;

        /**
           \brief Returns true of the array is empty
         */
        bool isempty() const;

        /**
           \brief Returns true of the array contains only one value
         */
        bool isscalar() const;

        /**
           \brief Returns true if only one of the array dimensions has more
                  than one element
        */
        bool isvector() const;

        /**
           \brief Returns true if only the second dimension has more than one
                  element
        */
        bool isrow() const;

        /**
           \brief Returns true if only the first dimension has more than one
                  element
        */
        bool iscolumn() const;

        /**
           \brief Returns true if the array type is \ref c32 or \ref c64
        */
        bool iscomplex() const;

        /**
           \brief Returns true if the array type is neither \ref c32 nor \ref c64
        */
        inline bool isreal() const { return !iscomplex(); }

        /**
           \brief Returns true if the array type is \ref f64 or \ref c64
        */
        bool isdouble() const;

        /**
           \brief Returns true if the array type is either \ref f32 nor \ref c32
        */
        bool issingle() const;

        /**
           \brief Returns true if the array type is \ref f16
        */
        bool ishalf() const;

        /**
           \brief Returns true if the array type is \ref f16 \ref f32 or \ref f64
        */
        bool isrealfloating() const;

        /**
           \brief Returns true if the array type is \ref f16 \ref f32, \ref f64,
                  \ref c32 or \ref c64
        */
        bool isfloating() const;

        /**
           \brief Returns true if the array type is \ref u8, \ref b8, \ref s32
                  \ref u32, \ref s64, \ref u64, \ref s16, \ref u16
        */
        bool isinteger() const;

        /**
           \brief Returns true if the array type is \ref b8
        */
        bool isbool() const;

        /**
           \brief Returns true if the array is a sparse array
        */
        bool issparse() const;

        /**
           \brief Evaluate any JIT expressions to generate data for the array
        */
        void eval() const;

        /**
           \brief Get the first element of the array as a scalar

           \note The scalar function is recommended for use while debugging.
                 Calling this method often will affect performance.
        */
        template<typename T> T scalar() const;

        /**
           \brief Get the device pointer from the array and lock the buffer in memory manager.

           The device memory returned by this function is not freed until unlock() is called.
        */
        template<typename T> T* device() const;

        // INDEXING
        // Single arguments

        /**
            \brief This operator returns a reference of the original array at a given coordinate.

            You can pass \ref fly::seq, \ref fly::array, or an int as its parameters.
            These references can be used for assignment or returning references
            to \ref fly::array objects.

            If the \ref fly::array is a multi-dimensional array then this coordinate
            will treated as the data as a linear array.

            \param[in] s0   is sequence of linear indices

            \returns A reference to the array at the given index

            \ingroup array_mem_operator_paren

        */
        array::array_proxy operator()(const index &s0);

        /**
            \copydoc operator()(const index &)

            \ingroup array_mem_operator_paren
        */
        const array::array_proxy operator()(const index &s0) const;


        /**
            \brief This operator returns a reference of the original array at a
            given coordinate.

            You can pass \ref fly::seq, \ref fly::array, or an int as it's parameters.
            These references can be used for assignment or returning references
            to \ref fly::array objects.

            \param[in] s0   is sequence of indices along the first dimension
            \param[in] s1   is sequence of indices along the second dimension
            \param[in] s2   is sequence of indices along the third dimension
            \param[in] s3   is sequence of indices along the fourth dimension

            \returns A reference to the array at the given index

            \ingroup array_mem_operator_paren
        */
        array::array_proxy operator()(const index &s0,
                                      const index &s1,
                                      const index &s2 = span,
                                      const index &s3 = span);

        /**
            \copydoc operator()(const index &, const index &, const index &, const index &)

            \ingroup array_mem_operator_paren
        */
        const array::array_proxy operator()(const index &s0,
                                            const index &s1,
                                            const index &s2 = span,
                                            const index &s3 = span) const;


        /// \ingroup array_mem_row
        /// @{
        ///
        /// \brief Returns a reference to a row
        ///
        /// \copydetails array_mem_row
        ///
        /// \param[in]  index is the index of the row to be returned
        ///
        /// \returns a reference to a row defined by \p index
        ///
              array::array_proxy row(int index);
        const array::array_proxy row(int index) const; ///< \copydoc row

        ///
        /// \brief Returns a reference to sequence of rows
        ///
        /// \copydetails array_mem_row
        ///
        /// \param[in]  first is the index of the row to be returned
        /// \param[in]  last is the index of the row to be returned
        ///
        /// \returns a reference to a set of rows
              array::array_proxy rows(int first, int last);
        const array::array_proxy rows(int first, int last) const; ///< \copydoc rows
        /// @}

        /// \ingroup array_mem_col
        /// @{
        ///
        /// \brief Returns a reference to a col
        ///
        /// \copydetails array_mem_col
        ///
        /// \param[in]  index is the index of the col to be returned
        ///
        /// \returns a reference to a col defined by \p index
        ///
              array::array_proxy col(int index);
        const array::array_proxy col(int index) const; ///< \copydoc col

        ///
        /// \brief Returns a reference to sequence of columns
        ///
        /// \copydetails array_mem_col
        ///
        /// \param[in]  first is the index of the columns to be returned
        /// \param[in]  last is the index of the columns to be returned
        ///
        /// \returns a reference to a set of columns
              array::array_proxy cols(int first, int last);
        const array::array_proxy cols(int first, int last) const; ///< \copydoc cols
        /// @}

        /// \ingroup array_mem_slice
        /// @{
        ///
        /// \brief Returns a reference to a matrix in a volume
        ///
        /// \copydetails array_mem_slice
        ///
        /// \param[in]  index is the index of the slice to be returned
        ///
        /// \returns a reference to a col
        ///
              array::array_proxy slice(int index);
        const array::array_proxy slice(int index) const; ///< \copydoc slice

        /// \brief Returns a reference to a matrix in a volume
        ///
        /// \copydetails array_mem_slice
        ///
        /// \param[in]  first is the index of the slices to be returned
        /// \param[in]  last is the index of the slices to be returned
        ///
        /// \returns a reference to a set of slice
              array::array_proxy slices(int first, int last);
        const array::array_proxy slices(int first, int last) const; ///< \copydoc slices
        /// @}

        /// \brief Casts the array into another data type
        ///
        /// \note Consecitive casting operations may be may be optimized out if
        /// the original type of the fly::array is the same as the final type.
        /// For example if the original type is f64 which is then cast to f32
        /// and then back to f64, then the cast to f32 will be skipped and that
        /// operation will *NOT* be performed by Flare. The following table
        /// shows which casts will be optimized out. outer -> inner -> outer
        /// | inner-> | f32 | f64 | c32 | c64 | s32 | u32 | u8 | b8 | s64 | u64 | s16 | u16 | f16 |
        /// |---------|-----|-----|-----|-----|-----|-----|----|----|-----|-----|-----|-----|-----|
        /// | f32     | x   | x   | x   | x   |     |     |    |    |     |     |     |     | x   |
        /// | f64     | x   | x   | x   | x   |     |     |    |    |     |     |     |     | x   |
        /// | c32     | x   | x   | x   | x   |     |     |    |    |     |     |     |     | x   |
        /// | c64     | x   | x   | x   | x   |     |     |    |    |     |     |     |     | x   |
        /// | s32     | x   | x   | x   | x   | x   | x   |    |    | x   | x   |     |     | x   |
        /// | u32     | x   | x   | x   | x   | x   | x   |    |    | x   | x   |     |     | x   |
        /// | u8      | x   | x   | x   | x   | x   | x   | x  | x  | x   | x   | x   | x   | x   |
        /// | b8      | x   | x   | x   | x   | x   | x   | x  | x  | x   | x   | x   | x   | x   |
        /// | s64     | x   | x   | x   | x   |     |     |    |    | x   | x   |     |     | x   |
        /// | u64     | x   | x   | x   | x   |     |     |    |    | x   | x   |     |     | x   |
        /// | s16     | x   | x   | x   | x   | x   | x   |    |    | x   | x   | x   | x   | x   |
        /// | u16     | x   | x   | x   | x   | x   | x   |    |    | x   | x   | x   | x   | x   |
        /// | f16     | x   | x   | x   | x   |     |     |    |    |     |     |     |     | x   |
        /// If you want to avoid this behavior use fly_eval after the first cast
        /// operation. This will ensure that the cast operation is performed on
        /// the fly::array
        ///
        /// \param[in] type is the desired type(f32, s64, etc.)
        /// \returns an array with the type specified by \p type
        const array as(dtype type) const;


        ~array();

        /// \brief Get the transposed the array
        ///
        /// \returns Transposed matrix
        array T() const;
        /// \brief Get the conjugate-transpose of the current array
        ///
        /// \returns conjugate-transpose matrix
        array H() const;

#define ASSIGN_(OP2)                                                                      \
        array& OP2(const array &val);                                                     \
        array& OP2(const double &val);              /**< \copydoc OP2##(const array &) */ \
        array& OP2(const cdouble &val);             /**< \copydoc OP2##(const array &) */ \
        array& OP2(const cfloat &val);              /**< \copydoc OP2##(const array &) */ \
        array& OP2(const float &val);               /**< \copydoc OP2##(const array &) */ \
        array& OP2(const int &val);                 /**< \copydoc OP2##(const array &) */ \
        array& OP2(const unsigned &val);            /**< \copydoc OP2##(const array &) */ \
        array& OP2(const bool &val);                /**< \copydoc OP2##(const array &) */ \
        array& OP2(const char &val);                /**< \copydoc OP2##(const array &) */ \
        array& OP2(const unsigned char &val);       /**< \copydoc OP2##(const array &) */ \
        array& OP2(const long  &val);               /**< \copydoc OP2##(const array &) */ \
        array& OP2(const unsigned long &val);       /**< \copydoc OP2##(const array &) */ \
        array& OP2(const long long  &val);          /**< \copydoc OP2##(const array &) */ \
        array& OP2(const unsigned long long &val);


#define ASSIGN(OP)                                                                        \
        ASSIGN_(OP)                                                                       \
          array& OP(const short  &val);              /**< \copydoc OP##(const array &) */ \
          array& OP(const unsigned short &val);



        /// \ingroup array_mem_operator_eq
        /// @{
        /// \brief Assignes the value(s) of val to the elements of the array.
        ///
        /// \param[in] val  is the value to be assigned to the /ref fly::array
        /// \returns the reference to this
        ///
        /// \note   This is a copy on write operation. The copy only occurs when the
        ///          operator() is used on the left hand side.
        ASSIGN(operator=)
        /// @}

        /// \ingroup array_mem_operator_plus_eq
        /// @{
        /// \brief Adds the value(s) of val to the elements of the array.
        ///
        /// \param[in] val  is the value to be assigned to the /ref fly::array
        /// \returns the reference to this
        ///
        /// \note   This is a copy on write operation. The copy only occurs when the
        ///          operator() is used on the left hand side.
        ASSIGN(operator+=)
        /// @}

        /// \ingroup array_mem_operator_minus_eq
        /// @{
        /// \brief Subtracts the value(s) of val to the elements of the array.
        ///
        /// \param[in] val  is the value to be assigned to the /ref fly::array
        /// \returns the reference to this
        ///
        /// \note   This is a copy on write operation. The copy only occurs when the
        ///          operator() is used on the left hand side.
        ASSIGN(operator-=)
        /// @}

        /// \ingroup array_mem_operator_multiply_eq
        /// @{
        /// \brief Multiplies the value(s) of val to the elements of the array.
        ///
        /// \param[in] val  is the value to be assigned to the /ref fly::array
        /// \returns the reference to this
        ///
        /// \note   This is a copy on write operation. The copy only occurs when the
        ///          operator() is used on the left hand side.
        ASSIGN(operator*=)
        /// @}

        /// \ingroup array_mem_operator_divide_eq
        /// @{
        /// \brief Divides the value(s) of val to the elements of the array.
        ///
        /// \param[in] val  is the value to be assigned to the /ref fly::array
        /// \returns the reference to this
        ///
        /// \note   This is a copy on write operation. The copy only occurs when the
        ///          operator() is used on the left hand side.
        /// \ingroup array_mem_operator_divide_eq
        ASSIGN(operator/=)
        /// @}


#undef ASSIGN
#undef ASSIGN_

        ///
        /// \brief Negates the values of the array
        /// \ingroup arith_func_neg
        ///
        /// \returns an \ref array with negated values
        array operator -() const;

        ///
        /// \brief Performs a not operation on the values of the array
        /// \ingroup arith_func_not
        ///
        /// \returns an \ref array with negated values
        array operator !() const;

        ///
        /// \brief Performs a bitwise not operation on the values of the array
        /// \ingroup arith_func_bitnot
        ///
        /// \returns an \ref array with inverted values
        array operator ~() const;

        ///
        /// \brief Get the count of non-zero elements in the array
        ///
        /// For dense matrix, this is the same as count<int>(arr);
        int nonzeros() const;


        ///
        /// \brief Locks the device buffer in the memory manager.
        ///
        /// This method can be called to take control of the device pointer from the memory manager.
        /// While a buffer is locked, the memory manager doesn't free the memory until unlock() is invoked.
        void lock() const;


        ///
        /// \brief Query if the array has been locked by the user.
        ///
        /// An array can be locked by the user by calling `arry.lock` or `arr.device`
        /// or `getRawPtr` function.
        bool isLocked() const;


        ///
        /// \brief Unlocks the device buffer in the memory manager.
        ///
        /// This method can be called after called after calling \ref array::lock()
        /// Calling this method gives back the control of the device pointer to the memory manager.
        void unlock() const;
    };
    // end of class array

#define BIN_OP_(OP)                                                                                                      \
    FLY_API array OP (const array& lhs, const array& rhs);                                                                 \
    FLY_API array OP (const bool& lhs, const array& rhs);               /**< \copydoc OP##(const array&, const array&) */ \
    FLY_API array OP (const int& lhs, const array& rhs);                /**< \copydoc OP##(const array&, const array&) */ \
    FLY_API array OP (const unsigned& lhs, const array& rhs);           /**< \copydoc OP##(const array&, const array&) */ \
    FLY_API array OP (const char& lhs, const array& rhs);               /**< \copydoc OP##(const array&, const array&) */ \
    FLY_API array OP (const unsigned char& lhs, const array& rhs);      /**< \copydoc OP##(const array&, const array&) */ \
    FLY_API array OP (const long& lhs, const array& rhs);               /**< \copydoc OP##(const array&, const array&) */ \
    FLY_API array OP (const unsigned long& lhs, const array& rhs);      /**< \copydoc OP##(const array&, const array&) */ \
    FLY_API array OP (const long long& lhs, const array& rhs);          /**< \copydoc OP##(const array&, const array&) */ \
    FLY_API array OP (const unsigned long long& lhs, const array& rhs); /**< \copydoc OP##(const array&, const array&) */ \
    FLY_API array OP (const double& lhs, const array& rhs);             /**< \copydoc OP##(const array&, const array&) */ \
    FLY_API array OP (const float& lhs, const array& rhs);              /**< \copydoc OP##(const array&, const array&) */ \
    FLY_API array OP (const cfloat& lhs, const array& rhs);             /**< \copydoc OP##(const array&, const array&) */ \
    FLY_API array OP (const cdouble& lhs, const array& rhs);            /**< \copydoc OP##(const array&, const array&) */ \
    FLY_API array OP (const array& lhs, const bool& rhs);               /**< \copydoc OP##(const array&, const array&) */ \
    FLY_API array OP (const array& lhs, const int& rhs);                /**< \copydoc OP##(const array&, const array&) */ \
    FLY_API array OP (const array& lhs, const unsigned& rhs);           /**< \copydoc OP##(const array&, const array&) */ \
    FLY_API array OP (const array& lhs, const char& rhs);               /**< \copydoc OP##(const array&, const array&) */ \
    FLY_API array OP (const array& lhs, const unsigned char& rhs);      /**< \copydoc OP##(const array&, const array&) */ \
    FLY_API array OP (const array& lhs, const long& rhs);               /**< \copydoc OP##(const array&, const array&) */ \
    FLY_API array OP (const array& lhs, const unsigned long& rhs);      /**< \copydoc OP##(const array&, const array&) */ \
    FLY_API array OP (const array& lhs, const long long& rhs);          /**< \copydoc OP##(const array&, const array&) */ \
    FLY_API array OP (const array& lhs, const unsigned long long& rhs); /**< \copydoc OP##(const array&, const array&) */ \
    FLY_API array OP (const array& lhs, const double& rhs);             /**< \copydoc OP##(const array&, const array&) */ \
    FLY_API array OP (const array& lhs, const float& rhs);              /**< \copydoc OP##(const array&, const array&) */ \
    FLY_API array OP (const array& lhs, const cfloat& rhs);             /**< \copydoc OP##(const array&, const array&) */ \
    FLY_API array OP (const array& lhs, const cdouble& rhs);

#define BIN_OP(OP)                                                                                                        \
        BIN_OP_(OP)                                                                                                       \
        FLY_API array OP (const short& lhs, const array& rhs);           /**< \copydoc OP##(const array&, const array&) */ \
        FLY_API array OP (const unsigned short& lhs, const array& rhs);  /**< \copydoc OP##(const array&, const array&) */ \
        FLY_API array OP (const array& lhs, const short& rhs);           /**< \copydoc OP##(const array&, const array&) */ \
        FLY_API array OP (const array& lhs, const unsigned short& rhs);


    /// \ingroup arith_func_add
    /// @{
    /// \brief Adds two arrays or an array and a value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns an array which is the sum of the \p lhs and \p rhs
    BIN_OP(operator+ )
    /// @}

    /// \ingroup arith_func_sub
    /// @{
    /// \brief Subtracts two arrays or an array and a value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns an array which is the subtraction of the \p lhs and \p rhs
    BIN_OP(operator- )
    /// @}

    /// \ingroup arith_func_mul
    /// @{
    /// \brief Multiplies two arrays or an array and a value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns an array which is the product of the \p lhs and \p rhs
    BIN_OP(operator* )
    /// @}

    /// \ingroup arith_func_div
    /// @{
    /// \brief Divides two arrays or an array and a value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns an array which is the quotient of the \p lhs and \p rhs
    BIN_OP(operator/ )
    /// @}

    /// \ingroup arith_func_eq
    /// @{
    /// \brief Performs an equality operation on two arrays or an array and a value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns an array of type b8 with the equality operation performed on each element
    BIN_OP(operator==)
    /// @}

    /// \ingroup arith_func_neq
    /// @{
    /// \brief Performs an inequality operation on two arrays or an array and a value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns    an array of type b8 with the != operation performed on each element
    ///             of \p lhs and \p rhs
    BIN_OP(operator!=)
    /// @}

    /// \ingroup arith_func_lt
    /// @{
    /// \brief Performs a lower than operation on two arrays or an array and a value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns    an array of type b8 with the < operation performed on each element
    ///             of \p lhs and \p rhs
    BIN_OP(operator< )
    /// @}

    /// \ingroup arith_func_le
    /// @{
    /// \brief Performs an lower or equal operation on two arrays or an array and a value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns    an array of type b8 with the <= operation performed on each element
    BIN_OP(operator<=)
    /// @}

    /// \ingroup arith_func_gt
    /// @{
    /// \brief Performs an greater than operation on two arrays or an array and a value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns    an array of type b8 with the > operation performed on each element
    ///             of \p lhs and \p rhs
    BIN_OP(operator> )
    /// @}

    /// \ingroup arith_func_ge
    /// @{
    /// \brief Performs an greater or equal operation on two arrays or an array and a value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns    an array of type b8 with the >= operation performed on each element
    ///             of \p lhs and \p rhs
    BIN_OP(operator>=)
    /// @}

    /// \ingroup arith_func_or
    /// @{
    /// \brief  Performs an logical OR operation on two arrays or an array and a
    ///         value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns    an array of type b8 with a logical OR operation performed on each
    ///             element of \p lhs and \p rhs
    BIN_OP(operator||)
    /// @}

    /// \ingroup arith_func_mod
    /// @{
    /// \brief Performs an modulo operation on two arrays or an array and a value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns    an array with a modulo operation performed on each
    ///             element of \p lhs and \p rhs
    BIN_OP(operator% )
    /// @}

    /// \ingroup arith_func_bitor
    /// @{
    /// \brief  Performs an bitwise OR operation on two arrays or an array and
    ///         a value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns    an array with a bitwise OR operation performed on each
    ///             element of \p lhs and \p rhs
    BIN_OP(operator| )
    /// @}

    /// \ingroup arith_func_bitxor
    /// @{
    /// \brief  Performs an bitwise XOR operation on two arrays or an array and
    ///         a value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns    an array with a bitwise OR operation performed on each
    ///             element of \p lhs and \p rhs
    BIN_OP(operator^ )
    /// @}

    /// \ingroup arith_func_shiftl
    /// @{
    /// \brief  Performs an left shift operation on two arrays or an array and a
    ///          value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns    an array with a left shift operation performed on each
    ///             element of \p lhs and \p rhs
    BIN_OP(operator<<)
    /// @}

    /// \ingroup arith_func_shiftr
    /// @{
    /// \brief  Performs an right shift operation on two arrays or an array and a
    ///          value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns    an array with a right shift operation performed on each
    ///             element of \p lhs and \p rhs
    BIN_OP(operator>>)
    /// @}

#undef BIN_OP
#undef BIN_OP_

    /// \ingroup arith_func_bitand
    /// @{
    /// \brief  Performs an bitwise AND operation on two arrays or an array and
    ///         a value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns    an array with a bitwise AND operation performed on each
    ///             element of \p lhs and \p rhs
    FLY_API array operator&(const array& lhs, const array& rhs);
    FLY_API array operator&(const array& lhs, const bool& rhs);
    FLY_API array operator&(const array& lhs, const cdouble& rhs);
    FLY_API array operator&(const array& lhs, const cfloat& rhs);
    FLY_API array operator&(const array& lhs, const char& rhs);
    FLY_API array operator&(const array& lhs, const double& rhs);
    FLY_API array operator&(const array& lhs, const float& rhs);
    FLY_API array operator&(const array& lhs, const int& rhs);
    FLY_API array operator&(const array& lhs, const long long& rhs);
    FLY_API array operator&(const array& lhs, const long& rhs);
    FLY_API array operator&(const array& lhs, const short& rhs);
    FLY_API array operator&(const array& lhs, const unsigned char& rhs);
    FLY_API array operator&(const array& lhs, const unsigned long long& rhs);
    FLY_API array operator&(const array& lhs, const unsigned long& rhs);
    FLY_API array operator&(const array& lhs, const unsigned short& rhs);
    FLY_API array operator&(const array& lhs, const unsigned& rhs);
    FLY_API array operator&(const bool& lhs, const array& rhs);
    FLY_API array operator&(const cdouble& lhs, const array& rhs);
    FLY_API array operator&(const cfloat& lhs, const array& rhs);
    FLY_API array operator&(const char& lhs, const array& rhs);
    FLY_API array operator&(const double& lhs, const array& rhs);
    FLY_API array operator&(const float& lhs, const array& rhs);
    FLY_API array operator&(const int& lhs, const array& rhs);
    FLY_API array operator&(const long long& lhs, const array& rhs);
    FLY_API array operator&(const long& lhs, const array& rhs);
    FLY_API array operator&(const short& lhs, const array& rhs);
    FLY_API array operator&(const unsigned char& lhs, const array& rhs);
    FLY_API array operator&(const unsigned long long& lhs, const array& rhs);
    FLY_API array operator&(const unsigned long& lhs, const array& rhs);
    FLY_API array operator&(const unsigned short& lhs, const array& rhs);
    FLY_API array operator&(const unsigned& lhs, const array& rhs);
    /// @}

    /// \ingroup arith_func_and
    /// @{
    /// \brief  Performs a logical AND operation on two arrays or an array and a
    ///         value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns    an array of type b8 with a logical AND operation performed on each
    ///             element of \p lhs and \p rhs
    FLY_API array operator&&(const array& lhs, const array& rhs);
    FLY_API array operator&&(const array& lhs, const bool& rhs);
    FLY_API array operator&&(const array& lhs, const cdouble& rhs);
    FLY_API array operator&&(const array& lhs, const cfloat& rhs);
    FLY_API array operator&&(const array& lhs, const char& rhs);
    FLY_API array operator&&(const array& lhs, const double& rhs);
    FLY_API array operator&&(const array& lhs, const float& rhs);
    FLY_API array operator&&(const array& lhs, const int& rhs);
    FLY_API array operator&&(const array& lhs, const long long& rhs);
    FLY_API array operator&&(const array& lhs, const long& rhs);
    FLY_API array operator&&(const array& lhs, const short& rhs);
    FLY_API array operator&&(const array& lhs, const unsigned char& rhs);
    FLY_API array operator&&(const array& lhs, const unsigned long long& rhs);
    FLY_API array operator&&(const array& lhs, const unsigned long& rhs);
    FLY_API array operator&&(const array& lhs, const unsigned short& rhs);
    FLY_API array operator&&(const array& lhs, const unsigned& rhs);
    FLY_API array operator&&(const bool& lhs, const array& rhs);
    FLY_API array operator&&(const cdouble& lhs, const array& rhs);
    FLY_API array operator&&(const cfloat& lhs, const array& rhs);
    FLY_API array operator&&(const char& lhs, const array& rhs);
    FLY_API array operator&&(const double& lhs, const array& rhs);
    FLY_API array operator&&(const float& lhs, const array& rhs);
    FLY_API array operator&&(const int& lhs, const array& rhs);
    FLY_API array operator&&(const long long& lhs, const array& rhs);
    FLY_API array operator&&(const long& lhs, const array& rhs);
    FLY_API array operator&&(const short& lhs, const array& rhs);
    FLY_API array operator&&(const unsigned char& lhs, const array& rhs);
    FLY_API array operator&&(const unsigned long long& lhs, const array& rhs);
    FLY_API array operator&&(const unsigned long& lhs, const array& rhs);
    FLY_API array operator&&(const unsigned short& lhs, const array& rhs);
    FLY_API array operator&&(const unsigned& lhs, const array& rhs);
    /// @}


    /// Evaluate an expression (nonblocking).
    /**
       \ingroup data_mat
       @{
    */
    inline array &eval(array &a) { a.eval(); return a; }

    ///
    /// Evaluate multiple arrays simultaneously
    ///
    FLY_API void eval(int num, array **arrays);

    inline void eval(array &a, array &b)
    {
        array *arrays[] = {&a, &b};
        return eval(2, arrays);
    }

    inline void eval(array &a, array &b, array &c)
    {
        array *arrays[] = {&a, &b, &c};
        return eval(3, arrays);
    }

    inline void eval(array &a, array &b, array &c, array &d)
    {
        array *arrays[] = {&a, &b, &c, &d};
        return eval(4, arrays);
    }

    inline void eval(array &a, array &b, array &c, array &d, array &e)
    {
        array *arrays[] = {&a, &b, &c, &d, &e};
        return eval(5, arrays);
    }

    inline void eval(array &a, array &b, array &c, array &d, array &e, array &f)
    {
        array *arrays[] = {&a, &b, &c, &d, &e, &f};
        return eval(6, arrays);
    }


    /// Evaluate an expression (nonblocking).
    inline const array &eval(const array &a) { a.eval(); return a; }

#if FLY_COMPILER_CXX_VARIADIC_TEMPLATES
    template <typename... ARRAYS>
    inline void eval(ARRAYS... in) {
        array *arrays[] = {const_cast<array *>(&in)...};
        eval((int)sizeof...(in), arrays);
    }

#else

    inline void eval(const array &a, const array &b)
    {
        const array *arrays[] = {&a, &b};
        return eval(2, const_cast<array **>(arrays));
    }

    inline void eval(const array &a, const array &b, const array &c)
    {
        const array *arrays[] = {&a, &b, &c};
        return eval(3, const_cast<array **>(arrays));
    }

    inline void eval(const array &a, const array &b, const array &c,
                     const array &d)
    {
        const array *arrays[] = {&a, &b, &c, &d};
        return eval(4, const_cast<array **>(arrays));
    }

    inline void eval(const array &a, const array &b, const array &c,
                     const array &d, const array &e)
    {
        const array *arrays[] = {&a, &b, &c, &d, &e};
        return eval(5, const_cast<array **>(arrays));
    }

    inline void eval(const array &a, const array &b, const array &c,
                     const array &d, const array &e, const array &f)
    {
        const array *arrays[] = {&a, &b, &c, &d, &e, &f};
        return eval(6, const_cast<array **>(arrays));
    }
#endif // FLY_COMPILER_CXX_VARIADIC_TEMPLATES

    ///
    /// Turn the manual eval flag on or off
    ///
    FLY_API void setManualEvalFlag(bool flag);

    /// Get the manual eval flag
    FLY_API bool getManualEvalFlag();

    /**
       @}
    */

}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    /**
       \ingroup c_api_mat
       @{
    */

    /**
       Create an \ref fly_array handle initialized with user defined data

       This function will create an \ref fly_array handle from the memory provided in \p data.

       \param[out]  arr The pointer to the returned object.
       \param[in]   data The data which will be loaded into the array
       \param[in]   ndims The number of dimensions read from the \p dims parameter
       \param[in]   dims A C pointer with \p ndims elements. Each value represents the size of that dimension
       \param[in]   type The type of the \ref fly_array object

       \returns \ref FLY_SUCCESS if the operation was a success
    */
    FLY_API fly_err fly_create_array(fly_array *arr, const void * const data, const unsigned ndims, const dim_t * const dims, const fly_dtype type);

    /**
       Create fly_array handle

       To release the memory allocated by this call you would have to
       call \ref fly_release_array once your use of this \ref fly_array is complete.

       \param[out]  arr The pointer to the retured object.
       \param[in]   ndims The number of dimensions read from the \p dims parameter
       \param[in]   dims A C pointer with \p ndims elements. Each value represents the size of that dimension
       \param[in]   type The type of the \ref fly_array object

       \returns \ref FLY_SUCCESS if the operation was a success
    */
    FLY_API fly_err fly_create_handle(fly_array *arr, const unsigned ndims, const dim_t * const dims, const fly_dtype type);

    /**
       Deep copy an array to another
    */
    FLY_API fly_err fly_copy_array(fly_array *arr, const fly_array in);

    /**
       Copy data from a C pointer (host/device) to an existing array.
    */
    FLY_API fly_err fly_write_array(fly_array arr, const void *data, const size_t bytes, fly_source src);

    /**
       Copy data from an fly_array to a C pointer.

       Needs to used in conjunction with the two functions above
    */
    FLY_API fly_err fly_get_data_ptr(void *data, const fly_array arr);

    /**
       \brief Reduce the reference count of the \ref fly_array

       \note Zero initialized fly_arrays can be accepted after version 3.7
    */
    FLY_API fly_err fly_release_array(fly_array arr);

    /**
       Increments an \ref fly_array reference count.
    */
    FLY_API fly_err fly_retain_array(fly_array *out, const fly_array in);

    /**
       Get the reference count of \ref fly_array
    */
    FLY_API fly_err fly_get_data_ref_count(int *use_count, const fly_array in);

    /**
       Evaluate any expressions in the Array
    */
    FLY_API fly_err fly_eval(fly_array in);

    /**
       Evaluate multiple arrays together
    */
    FLY_API fly_err fly_eval_multiple(const int num, fly_array *arrays);

    /**
       Turn the manual eval flag on or off
    */
    FLY_API fly_err fly_set_manual_eval_flag(bool flag);

    /**
       Get the manual eval flag
    */
    FLY_API fly_err fly_get_manual_eval_flag(bool *flag);

    /**
        \brief Get the total number of elements across all dimensions of the array

        \param[out] elems is the output that contains number of elements of \p arr
        \param[in] arr is the input array

        \returns error codes
    */
    FLY_API fly_err fly_get_elements(dim_t *elems, const fly_array arr);

    /**
        \brief Gets the type of an array.

        \param[out] type is the output that contains the type of \p arr
        \param[in] arr is the input array

        \returns error codes
    */
    FLY_API fly_err fly_get_type(fly_dtype *type, const fly_array arr);

    /**
        \brief Gets the dimensions of an array.

        \param[out] d0 is the output that contains the size of first dimension of \p arr
        \param[out] d1 is the output that contains the size of second dimension of \p arr
        \param[out] d2 is the output that contains the size of third dimension of \p arr
        \param[out] d3 is the output that contains the size of fourth dimension of \p arr
        \param[in] arr is the input array

        \returns error codes
    */
    FLY_API fly_err fly_get_dims(dim_t *d0, dim_t *d1, dim_t *d2, dim_t *d3,
                             const fly_array arr);

    /**
        \brief Gets the number of dimensions of an array.

        \param[out] result is the output that contains the number of dims of \p arr
        \param[in] arr is the input array

        \returns error codes
    */
    FLY_API fly_err fly_get_numdims(unsigned *result, const fly_array arr);

    /**
        \brief Check if an array is empty.

        \param[out] result is true if elements of arr is 0, otherwise false
        \param[in] arr is the input array

        \returns error codes
    */
    FLY_API fly_err fly_is_empty        (bool *result, const fly_array arr);

    /**
        \brief Check if an array is scalar, ie. single element.

        \param[out] result is true if elements of arr is 1, otherwise false
        \param[in] arr is the input array

        \returns error codes
    */
    FLY_API fly_err fly_is_scalar       (bool *result, const fly_array arr);

    /**
        \brief Check if an array is row vector.

        \param[out] result is true if arr has dims [1 x 1 1], false otherwise
        \param[in] arr is the input array

        \returns error codes
    */
    FLY_API fly_err fly_is_row          (bool *result, const fly_array arr);

    /**
        \brief Check if an array is a column vector

        \param[out] result is true if arr has dims [x 1 1 1], false otherwise
        \param[in] arr is the input array

        \returns error codes
    */
    FLY_API fly_err fly_is_column       (bool *result, const fly_array arr);

    /**
        \brief Check if an array is a vector

        A vector is any array that has exactly 1 dimension not equal to 1.

        \param[out] result is true if arr is a vector, false otherwise
        \param[in] arr is the input array

        \returns error codes
    */
    FLY_API fly_err fly_is_vector       (bool *result, const fly_array arr);

    /**
        \brief Check if an array is complex type

        \param[out] result is true if arr is of type \ref c32 or \ref c64, otherwise false
        \param[in] arr is the input array

        \returns error codes
    */
    FLY_API fly_err fly_is_complex      (bool *result, const fly_array arr);

    /**
        \brief Check if an array is real type

        This is mutually exclusive to \ref fly_is_complex

        \param[out] result is true if arr is NOT \ref c32 or \ref c64, otherwise false
        \param[in] arr is the input array

        \returns error codes
    */
    FLY_API fly_err fly_is_real         (bool *result, const fly_array arr);

    /**
        \brief Check if an array is double precision type

        \param[out] result is true if arr is of type \ref f64 or \ref c64, otherwise false
        \param[in] arr is the input array

        \returns error codes
    */
    FLY_API fly_err fly_is_double       (bool *result, const fly_array arr);

    /**
        \brief Check if an array is single precision type

        \param[out] result is true if arr is of type \ref f32 or \ref c32, otherwise false
        \param[in] arr is the input array

        \returns error codes
    */
    FLY_API fly_err fly_is_single       (bool *result, const fly_array arr);

    /**
        \brief Check if an array is 16 bit floating point type

        \param[out] result is true if arr is of type \ref f16 otherwise false
        \param[in] arr     is the input array

        \returns error codes
    */
    FLY_API fly_err fly_is_half(bool *result, const fly_array arr);

    /**
        \brief Check if an array is real floating point type

        \param[out] result is true if arr is of type \ref f32 or \ref f64, otherwise false
        \param[in] arr is the input array

        \returns error codes
    */
    FLY_API fly_err fly_is_realfloating (bool *result, const fly_array arr);

    /**
        \brief Check if an array is floating precision type

        This is a combination of \ref fly_is_realfloating and \ref fly_is_complex

        \param[out] result is true if arr is of type \ref f16 \ref f32, \ref
                           f64, \ref c32 or \ref c64, otherwise false
        \param[in] arr is the input array

        \returns error codes
    */
    FLY_API fly_err fly_is_floating     (bool *result, const fly_array arr);

    /**
        \brief Check if an array is integer type

        \param[out] result is true if arr is of integer types, otherwise false
        \param[in] arr is the input array

        \returns error codes
    */
    FLY_API fly_err fly_is_integer      (bool *result, const fly_array arr);

    /**
        \brief Check if an array is bool type

        \param[out] result is true if arr is of \ref b8 type, otherwise false
        \param[in] arr is the input array

        \returns error codes
    */
    FLY_API fly_err fly_is_bool         (bool *result, const fly_array arr);

    /**
        \brief Check if an array is sparse

        \param[out] result is true if arr is sparse, otherwise false
        \param[in] arr is the input array

        \returns error codes
    */
    FLY_API fly_err fly_is_sparse       (bool *result, const fly_array arr);

    /**
        \brief Get first element from an array

        \param[out] output_value is the element requested
        \param[in] arr is the input array
        \return \ref FLY_SUCCESS if the execution completes properly
    */
    FLY_API fly_err fly_get_scalar(void* output_value, const fly_array arr);

    /**
        @}
    */

#ifdef __cplusplus
}
#endif
