/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <fly/defines.h>

#ifdef __cplusplus
namespace fly
{
    class array;

    /**
        \param[in] exp is an expression, generally the name of the array
        \param[in] arr is the input array

        \ingroup print_func_print
    */
    FLY_API void print(const char *exp, const array &arr);

#if FLY_API_VERSION >= 31
    /**
        \param[in] exp is an expression, generally the name of the array
        \param[in] arr is the input array
        \param[in] precision is the precision length for display

        \ingroup print_func_print
    */
    FLY_API void print(const char *exp, const array &arr, const int precision);
#endif

#if FLY_API_VERSION >= 31
    /**
        \param[in] key is an expression used as tag/key for the array during \ref readArray
        \param[in] arr is the array to be written
        \param[in] filename is the path to the location on disk
        \param[in] append is used to append to an existing file when true and create or
        overwrite an existing file when false

        \returns index of the saved array in the file

        \ingroup stream_func_save
    */
    FLY_API int saveArray(const char *key, const array &arr, const char *filename, const bool append = false);
#endif

#if FLY_API_VERSION >= 31
    /**
        \param[in] filename is the path to the location on disk
        \param[in] index is the 0-based sequential location of the array to be read

        \returns array read from the index location

        \note This function will throw an exception if the index is out of bounds

        \ingroup stream_func_read
    */
    FLY_API array readArray(const char *filename, const unsigned index);
#endif

#if FLY_API_VERSION >= 31
    /**
        \param[in] filename is the path to the location on disk
        \param[in] key is the tag/name of the array to be read. The key needs to have an exact match.

        \returns array read by key

        \note This function will throw an exception if the key is not found.

        \ingroup stream_func_read
    */
    FLY_API array readArray(const char *filename, const char *key);
#endif

#if FLY_API_VERSION >= 31
    /**
        When reading by key, it may be a good idea to run this function first to check for the key
        and then call the readArray using the index. This will avoid exceptions in case of key not found.

        \param[in] filename is the path to the location on disk
        \param[in] key is the tag/name of the array to be read. The key needs to have an exact match.

        \returns index of the array in the file if the key is found. -1 if key is not found.

        \ingroup stream_func_read
    */
    FLY_API int readArrayCheck(const char *filename, const char *key);
#endif

#if FLY_API_VERSION >= 31
    /**
        \param[out] output is the pointer to the c-string that will hold the data. The memory for
        output is allocated by the function. The user is responsible for deleting the memory using
        fly::freeHost() or fly_free_host().
        \param[in] exp is an expression, generally the name of the array
        \param[in] arr is the input array
        \param[in] precision is the precision length for display
        \param[in] transpose determines whether or not to transpose the array before storing it in
        the string

        \ingroup print_func_tostring
    */
    FLY_API void toString(char **output, const char *exp, const array &arr,
                        const int precision = 4, const bool transpose = true);
#endif

#if FLY_API_VERSION >= 33
    /**
        \param[in] exp is an expression, generally the name of the array
        \param[in] arr is the input array
        \param[in] precision is the precision length for display
        \param[in] transpose determines whether or not to transpose the array before storing it in
        the string

        \return output is the pointer to the c-string that will hold the data. The memory for
        output is allocated by the function. The user is responsible for deleting the memory using
        fly::freeHost() or fly_free_host().

        \ingroup print_func_tostring
    */
    FLY_API const char* toString(const char *exp, const array &arr,
                               const int precision = 4, const bool transpose = true);
#endif

    // Purpose of Addition: "How to add Function" documentation
    FLY_API array exampleFunction(const array& in, const fly_someenum_t param);

#if FLY_API_VERSION >= 34
    ///
    /// Get the size of the type represented by an fly_dtype enum
    ///
    FLY_API size_t getSizeOf(fly::dtype type);
#endif
}

#if FLY_API_VERSION >= 31

#define FLY_PRINT1(exp)            fly::print(#exp, exp);
#define FLY_PRINT2(exp, precision) fly::print(#exp, exp, precision);

#define GET_PRINT_MACRO(_1, _2, NAME, ...) NAME

#define fly_print(...) GET_PRINT_MACRO(__VA_ARGS__, FLY_PRINT2, FLY_PRINT1)(__VA_ARGS__)

#else // FLY_API_VERSION

#define fly_print(exp) fly::print(#exp, exp);

#endif // FLY_API_VERSION

#endif //__cplusplus

#ifdef __cplusplus
extern "C" {
#endif

    /**
        \param[in] arr is the input array

        \returns error codes

        \ingroup print_func_print
    */
    FLY_API fly_err fly_print_array(fly_array arr);

#if FLY_API_VERSION >= 31
    /**
        \param[in] exp is the expression or name of the array
        \param[in] arr is the input array
        \param[in] precision precision for the display

        \returns error codes

        \ingroup print_func_print
    */
    FLY_API fly_err fly_print_array_gen(const char *exp, const fly_array arr, const int precision);
#endif

#if FLY_API_VERSION >= 31
    /**
        \param[out] index is the index location of the array in the file
        \param[in] key is an expression used as tag/key for the array during \ref fly::readArray()
        \param[in] arr is the array to be written
        \param[in] filename is the path to the location on disk
        \param[in] append is used to append to an existing file when true and create or
        overwrite an existing file when false

        \ingroup stream_func_save
    */
    FLY_API fly_err fly_save_array(int *index, const char* key, const fly_array arr, const char *filename, const bool append);
#endif

#if FLY_API_VERSION >= 31
    /**
        \param[out] out is the array read from index
        \param[in] filename is the path to the location on disk
        \param[in] index is the 0-based sequential location of the array to be read

        \note This function will throw an exception if the key is not found.

        \ingroup stream_func_read
    */
    FLY_API fly_err fly_read_array_index(fly_array *out, const char *filename, const unsigned index);
#endif

#if FLY_API_VERSION >= 31
    /**
        \param[out] out is the array read from key
        \param[in] filename is the path to the location on disk
        \param[in] key is the tag/name of the array to be read. The key needs to have an exact match.

        \note This function will throw an exception if the key is not found.

        \ingroup stream_func_read
    */
    FLY_API fly_err fly_read_array_key(fly_array *out, const char *filename, const char* key);
#endif

#if FLY_API_VERSION >= 31
    /**
        When reading by key, it may be a good idea to run this function first to check for the key
        and then call the readArray using the index. This will avoid exceptions in case of key not found.

        \param[out] index of the array in the file if the key is found. -1 if key is not found.
        \param[in] filename is the path to the location on disk
        \param[in] key is the tag/name of the array to be read. The key needs to have an exact match.

        \ingroup stream_func_read
    */
    FLY_API fly_err fly_read_array_key_check(int *index, const char *filename, const char* key);
#endif

#if FLY_API_VERSION >= 31
    /**
        \param[out] output is the pointer to the c-string that will hold the data. The memory for
        output is allocated by the function. The user is responsible for deleting the memory.
        \param[in] exp is an expression, generally the name of the array
        \param[in] arr is the input array
        \param[in] precision is the precision length for display
        \param[in] transpose determines whether or not to transpose the array before storing it in
        the string

        \ingroup print_func_tostring
    */
    FLY_API fly_err fly_array_to_string(char **output, const char *exp, const fly_array arr,
                                    const int precision, const bool transpose);
#endif

    // Purpose of Addition: "How to add Function" documentation
    FLY_API fly_err fly_example_function(fly_array* out,
                                     const fly_array in,
                                     const fly_someenum_t param);

    ///
    /// Get the version information of the library
    ///
    FLY_API fly_err fly_get_version(int *major, int *minor, int *patch);


#if FLY_API_VERSION >= 33
    ///
    /// Get the revision (commit) information of the library.
    /// This returns a constant string from compile time and should not be
    /// freed by the user.
    ///
    FLY_API const char *fly_get_revision();
#endif

#if FLY_API_VERSION >= 34
    ///
    /// Get the size of the type represented by an fly_dtype enum
    ///
    FLY_API fly_err fly_get_size_of(size_t *size, fly_dtype type);
#endif

#if FLY_API_VERSION >= 37
    /// Enable(default) or disable error messages that display the stacktrace.
    ///
    /// \param[in] is_enabled If zero stacktraces are not shown with the error
    ///                       messages
    /// \returns Always returns FLY_SUCCESS
    FLY_API fly_err fly_set_enable_stacktrace(int is_enabled);
#endif

#ifdef __cplusplus
}
#endif
