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
#include <fly/defines.h>
#include <fly/seq.h>

///
/// \brief Struct used to index an fly_array
///
/// This struct represents objects which can be used to index into an fly_array
/// Object. It contains a union object which can be an \ref fly_seq or an
/// \ref fly_array. Indexing with an int can be represented using a \ref fly_seq
/// object with the same \ref fly_seq::begin and \ref fly_seq::end with an
/// fly_seq::step of 1
typedef struct fly_index_t {
    union {
        fly_array arr;   ///< The fly_array used for indexing
        fly_seq   seq;   ///< The fly_seq used for indexing
    } idx;

    bool     isSeq;     ///< If true the idx value represents a seq
    bool     isBatch;   ///< If true the seq object is a batch parameter
} fly_index_t;


#if __cplusplus
namespace fly
{

class dim4;
class array;
class seq;

///
/// \brief Wrapper for fly_index.
///
/// This class is a wrapper for the fly_index struct in the C interface. It
/// allows implicit type conversion from valid indexing types like int,
/// \ref fly::seq, \ref fly_seq, and \ref fly::array.
///
/// \note This is a helper class and does not necessarily need to be created
/// explicitly. It is used in the operator() overloads to simplify the API.
///
/// \ingroup flare_class
class FLY_API index {

    fly_index_t impl;
    public:
    ///
    /// \brief Default constructor. Equivalent to \ref fly::span
    ///
    index();
    ~index();

    ///
    /// \brief Implicit int converter
    ///
    /// Indexes the fly::array at index \p idx
    ///
    /// \param[in] idx is the id of the index
    ///
    /// \sa indexing
    ///
    index(const int idx);

    ///
    /// \brief Implicit seq converter
    ///
    /// Indexes the fly::array using an \ref fly::seq object
    ///
    /// \param[in] s0 is the set of indices to parse
    ///
    /// \sa indexing
    ///
    index(const fly::seq& s0);

    ///
    /// \brief Implicit seq converter
    ///
    /// Indexes the fly::array using an \ref fly_seq object
    ///
    /// \param[in] s0 is the set of indices to parse
    ///
    /// \sa indexing
    ///
    index(const fly_seq& s0);

    ///
    /// \brief Implicit int converter
    ///
    /// Indexes the fly::array using an \ref fly::array object
    ///
    /// \param[in] idx0 is the set of indices to parse
    ///
    /// \sa indexing
    ///
    index(const fly::array& idx0);

    ///
    /// \brief Copy constructor
    ///
    /// \param[in] idx0 is index to copy.
    ///
    /// \sa indexing
    ///
    index(const index& idx0);

    ///
    /// \brief Returns true if the \ref fly::index represents a fly::span object
    ///
    /// \returns true if the fly::index is an fly::span
    ///
    bool isspan() const;

    ///
    /// \brief Gets the underlying fly_index_t object
    ///
    /// \returns the fly_index_t represented by this object
    ///
    const fly_index_t& get() const;

    ///
    /// \brief Assigns idx0 to this index
    ///
    /// \param[in] idx0 is the index to be assigned to the /ref fly::index
    /// \returns the reference to this
    ///
    ///
    index & operator=(const index& idx0);

#if FLY_COMPILER_CXX_RVALUE_REFERENCES
    ///
    /// \brief Move constructor
    ///
    /// \param[in] idx0 is index to copy.
    ///
    index(index &&idx0);
    ///
    /// \brief Move assignment operator
    ///
    /// \param[in] idx0 is the index to be assigned to the /ref fly::index
    /// \returns a reference to this
    ///
    index& operator=(index &&idx0);
#endif
};

///
/// Lookup the values of an input array by indexing with another array
///
/// \param[in] in is the input array that will be queried
/// \param[in] idx are the lookup indices
/// \param[in] dim specifies the dimension for indexing
/// \returns an array containing values of \p in at locations specified by \p index
///
/// \ingroup index_func_lookup
///
FLY_API array lookup(const array &in, const array &idx, const int dim = -1);

///
/// Copy the values of an input array based on index
///
/// \param[out] dst The destination array
/// \param[in] src The source array
/// \param[in] idx0 The first index
/// \param[in] idx1 The second index (defaults to \ref fly::span)
/// \param[in] idx2 The third index (defaults to \ref fly::span)
/// \param[in] idx3 The fourth index (defaults to \ref fly::span)
/// \ingroup index_func_index
///
FLY_API void copy(array &dst, const array &src,
                const index &idx0,
                const index &idx1 = span,
                const index &idx2 = span,
                const index &idx3 = span);

}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    ///
    /// Lookup the values of input array based on sequences
    ///
    /// \param[out] out  output array containing values indexed by the
    ///                  sequences
    /// \param[in] in    is the input array
    /// \param[in] ndims is the number of sequences provided
    /// \param[in] index is an array of sequences
    ///
    /// \ingroup index_func_index
    FLY_API fly_err fly_index(  fly_array *out,
                            const fly_array in,
                            const unsigned ndims, const fly_seq* const index);


    ///
    /// Lookup the values of an input array by indexing with another array
    ///
    /// \param[out] out      output array containing values of \p in at locations
    ///                      specified by \p indices
    /// \param[in] in        is the input array that will be queried
    /// \param[in] indices   are the lookup indices
    /// \param[in] dim       specifies the dimension for indexing
    ///
    /// \ingroup index_func_lookup
    ///
    FLY_API fly_err fly_lookup( fly_array *out,
                            const fly_array in, const fly_array indices,
                            const unsigned dim);

    ///
    /// Copy and write values in the locations specified by the sequences
    ///
    /// \param[out] out     output array with values of \p rhs copied to
    ///                     locations specified by \p index and values from
    ///                     \p lhs in all other locations.
    /// \param[in] lhs      is array whose values are used for indices NOT
    ///                     specified by \p index
    /// \param[in] ndims    is the number of sequences provided
    /// \param[in] indices  is an array of sequences
    /// \param[in] rhs      is the array whose values are used for indices
    ///                     specified by \p index
    ///
    /// \ingroup index_func_assign
    ///
    FLY_API fly_err fly_assign_seq( fly_array *out,
                                const fly_array lhs,
                                const unsigned ndims, const fly_seq* const indices,
                                const fly_array rhs);

    ///
    /// \brief Indexing an array using \ref fly_seq, or \ref fly_array
    ///
    /// Generalized indexing function that accepts either fly_array or fly_seq
    /// along a dimension to index the input array and create the corresponding
    /// output array
    ///
    /// \param[out] out     output array containing values at indexed by
    ///                     the sequences
    /// \param[in] in       is the input array
    /// \param[in] ndims    is the number of \ref fly_index_t provided
    /// \param[in] indices  is an array of \ref fly_index_t objects
    ///
    /// \ingroup index_func_index
    ///
    FLY_API fly_err fly_index_gen(  fly_array *out,
                                const fly_array in,
                                const dim_t ndims, const fly_index_t* indices);

    ///
    /// \brief Assignment of an array using \ref fly_seq, or \ref fly_array
    ///
    /// Generalized assignment function that accepts either fly_array or fly_seq
    /// along a dimension to assign elements form an input array to an output
    /// array
    ///
    /// \param[out] out     output array containing values at indexed by
    ///                     the sequences
    /// \param[in] lhs      is the input array
    /// \param[in] ndims    is the number of \ref fly_index_t provided
    /// \param[in] indices  is an fly_array of \ref fly_index_t objects
    /// \param[in] rhs      is the array whose values will be assigned to \p lhs
    ///
    /// \ingroup index_func_assign
    ///
    FLY_API fly_err fly_assign_gen( fly_array *out,
                                const fly_array lhs,
                                const dim_t ndims, const fly_index_t* indices,
                                const fly_array rhs);

    ///
    /// \brief Create an quadruple of fly_index_t array
    ///
    /// \snippet test/index.cpp ex_index_util_0
    ///
    /// \param[out] indexers pointer to location where quadruple fly_index_t array is created
    /// \returns \ref fly_err error code
    ///
    /// \ingroup index_func_index
    ///
    FLY_API fly_err fly_create_indexers(fly_index_t** indexers);

    ///
    /// \brief set \p dim to given indexer fly_array \p idx
    ///
    /// \snippet test/index.cpp ex_index_util_0
    ///
    /// \param[in] indexer pointer to location where quadruple fly_index_t array was created
    /// \param[in] idx is the fly_array indexer for given dimension \p dim
    /// \param[in] dim is the dimension to be indexed
    /// \returns \ref fly_err error code
    ///
    /// \ingroup index_func_index
    ///
    FLY_API fly_err fly_set_array_indexer(fly_index_t* indexer, const fly_array idx, const dim_t dim);

    ///
    /// \brief set \p dim to given indexer fly_array \p idx
    ///
    /// This function is similar to \ref fly_set_array_indexer in terms of functionality except
    /// that this version accepts object of type \ref fly_seq instead of \ref fly_array.
    ///
    /// \snippet test/index.cpp ex_index_util_0
    ///
    /// \param[in] indexer pointer to location where quadruple fly_index_t array was created
    /// \param[in] idx is the fly_seq indexer for given dimension \p dim
    /// \param[in] dim is the dimension to be indexed
    /// \param[in] is_batch indicates if the sequence based indexing is inside a batch operation
    ///
    /// \ingroup index_func_index
    ///
    FLY_API fly_err fly_set_seq_indexer(fly_index_t* indexer, const fly_seq* idx,
                                  const dim_t dim, const bool is_batch);

    ///
    /// \brief set \p dim to given indexer fly_array \p idx
    ///
    ///  This function is alternative to \ref fly_set_seq_indexer where instead of passing
    ///  in an already prepared \ref fly_seq object, you pass the arguments necessary for
    ///  creating an fly_seq directly.
    ///
    /// \param[in] indexer pointer to location where quadruple fly_index_t array was created
    /// \param[in] begin is the beginning index of along dimension \p dim
    /// \param[in] end is the beginning index of along dimension \p dim
    /// \param[in] step size along dimension \p dim
    /// \param[in] dim is the dimension to be indexed
    /// \param[in] is_batch indicates if the sequence based indexing is inside a batch operation
    /// \returns \ref fly_err error code
    ///
    /// \ingroup index_func_index
    ///
    FLY_API fly_err fly_set_seq_param_indexer(fly_index_t* indexer,
                                        const double begin, const double end, const double step,
                                        const dim_t dim, const bool is_batch);

    ///
    /// \brief Release's the memory resource used by the quadruple fly_index_t array
    ///
    /// \snippet test/index.cpp ex_index_util_0
    ///
    /// \param[in] indexers is pointer to location where quadruple fly_index_t array is created
    //  \returns \ref fly_err error code
    ///
    /// \ingroup index_func_index
    ///
    FLY_API fly_err fly_release_indexers(fly_index_t* indexers);

#ifdef __cplusplus
}
#endif
