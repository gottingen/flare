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

///
/// \brief Handle for a random engine object.
///
/// This handle is used to reference the internal random engine object.
///
/// \ingroup random_mat
typedef void * fly_random_engine;

#ifdef __cplusplus
namespace fly
{
    class array;
    class dim4;
    /// C++ Interface - Random Number Generation Engine Class
    ///
    /// The \ref fly::randomEngine class is used to set the type and seed of
    /// random number generation engine based on \ref fly::randomEngineType.
    ///
    /// \ingroup flare_class
    /// \ingroup random_mat
    class FLY_API randomEngine {
    private:
      ///
      /// \brief Handle to the interal random engine object
      fly_random_engine engine;

    public:
      /**
          C++ Interface to create a \ref fly::randomEngine object with a \ref
          fly::randomEngineType and a seed.

          \code
            // create a random engine of default type with seed = 1
            randomEngine r(FLY_RANDOM_ENGINE_DEFAULT, 1);
          \endcode
      */
      explicit randomEngine(randomEngineType typeIn = FLY_RANDOM_ENGINE_DEFAULT,
                            unsigned long long seedIn = 0);

      /**
          C++ Interface copy constructor for a \ref fly::randomEngine.

          \param[in] other input random engine object
      */
      randomEngine(const randomEngine &other);

      /**
          C++ Interface to create a copy of the random engine object from a
          \ref fly_random_engine handle.

          \param[in] engine The input random engine object
      */
      randomEngine(fly_random_engine engine);

      /**
          C++ Interface destructor for a \ref fly::randomEngine.
      */
      ~randomEngine();

      /**
          C++ Interface to assign the internal state of randome engine.

          \param[in] other object to be assigned to the random engine

          \return the reference to this
      */
      randomEngine &operator=(const randomEngine &other);

      /**
          C++ Interface to set the random type of the random engine.

          \param[in] type type of the random number generator
      */
      void setType(const randomEngineType type);

      /**
          C++ Interface to get the random type of the random engine.

          \return \ref fly::randomEngineType associated with random engine
      */
      randomEngineType getType(void);

      /**
          C++ Interface to set the seed of the random engine.

          \param[in] seed initializing seed of the random number generator
      */
      void setSeed(const unsigned long long seed);

      /**
          C++ Interface to return the seed of the random engine.

          \return seed associated with random engine
      */
      unsigned long long getSeed(void) const;

      /**
          C++ Interface to return the fly_random_engine handle of this object.

          \return handle to the fly_random_engine associated with this random
                  engine
      */
      fly_random_engine get(void) const;
    };

    /**
        C++ Interface to create an array of random numbers uniformly
        distributed.

        \param[in] dims dimensions of the array to be generated
        \param[in] ty   type of the array
        \param[in] r    random engine object
        \return    random number array of size `dims`

        \ingroup random_func_randu
    */
    FLY_API array randu(const dim4 &dims, const dtype ty, randomEngine &r);

    /**
        C++ Interface to create an array of random numbers normally
        distributed.

        \param[in] dims dimensions of the array to be generated
        \param[in] ty   type of the array
        \param[in] r    random engine object
        \return    random number array of size `dims`

        \ingroup random_func_randn
    */
    FLY_API array randn(const dim4 &dims, const dtype ty, randomEngine &r);

    /**
        C++ Interface to create an array of random numbers uniformly
        distributed.

        \param[in] dims dimensions of the array to be generated
        \param[in] ty   type of the array

        \ingroup random_func_randu
    */
    FLY_API array randu(const dim4 &dims, const dtype ty=f32);

    /**
        C++ Interface to create an array of random numbers uniformly
        distributed.

        \param[in] d0 size of the first dimension
        \param[in] ty type of the array
        \return    random number array of size `d0`

        \ingroup random_func_randu
    */
    FLY_API array randu(const dim_t d0, const dtype ty=f32);

    /**
        C++ Interface to create an array of random numbers uniformly
        distributed.

        \param[in] d0 size of the first dimension
        \param[in] d1 size of the second dimension
        \param[in] ty type of the array
        \return    random number array of size `d0` x `d1`

        \ingroup random_func_randu
    */
    FLY_API array randu(const dim_t d0,
                      const dim_t d1, const dtype ty=f32);

    /**
        C++ Interface to create an array of random numbers uniformly
        distributed.

        \param[in] d0 size of the first dimension
        \param[in] d1 size of the second dimension
        \param[in] d2 size of the third dimension
        \param[in] ty type of the array
        \return    random number array of size `d0` x `d1` x `d2`

        \ingroup random_func_randu
    */
    FLY_API array randu(const dim_t d0,
                      const dim_t d1, const dim_t d2, const dtype ty=f32);

    /**
        C++ Interface to create an array of random numbers uniformly
        distributed.

        \param[in] d0 size of the first dimension
        \param[in] d1 size of the second dimension
        \param[in] d2 size of the third dimension
        \param[in] d3 size of the fourth dimension
        \param[in] ty type of the array
        \return    random number array of size `d0` x `d1` x `d2` x `d3`

        \ingroup random_func_randu
    */
    FLY_API array randu(const dim_t d0,
                      const dim_t d1, const dim_t d2,
                      const dim_t d3, const dtype ty=f32);

    /**
        C++ Interface to create an array of random numbers normally
        distributed.

        \param[in] dims dimensions of the array to be generated
        \param[in] ty   type of the array
        \return    random number array of size `dims`

        \ingroup random_func_randn
    */
    FLY_API array randn(const dim4 &dims, const dtype ty=f32);

    /**
        C++ Interface to create an array of random numbers normally
        distributed.

        \param[in] d0 size of the first dimension
        \param[in] ty type of the array
        \return    random number array of size `d0`

        \ingroup random_func_randn
    */
    FLY_API array randn(const dim_t d0, const dtype ty=f32);
    /**
        C++ Interface to create an array of random numbers normally
        distributed.

        \param[in] d0 size of the first dimension
        \param[in] d1 size of the second dimension
        \param[in] ty type of the array
        \return    random number array of size `d0` x `d1`

        \ingroup random_func_randn
    */
    FLY_API array randn(const dim_t d0,
                      const dim_t d1, const dtype ty=f32);
    /**
        C++ Interface to create an array of random numbers normally
        distributed.

        \param[in] d0 size of the first dimension
        \param[in] d1 size of the second dimension
        \param[in] d2 size of the third dimension
        \param[in] ty type of the array
        \return    random number array of size `d0` x `d1` x `d2`

        \ingroup random_func_randn
    */
    FLY_API array randn(const dim_t d0,
                      const dim_t d1, const dim_t d2, const dtype ty=f32);

    /**
        C++ Interface to create an array of random numbers normally
        distributed.

        \param[in] d0 size of the first dimension
        \param[in] d1 size of the second dimension
        \param[in] d2 size of the third dimension
        \param[in] d3 size of the fourth dimension
        \param[in] ty type of the array
        \return    random number array of size `d0` x `d1` x `d2` x `d3`

        \ingroup random_func_randn
    */
    FLY_API array randn(const dim_t d0,
                      const dim_t d1, const dim_t d2,
                      const dim_t d3, const dtype ty=f32);

    /**
        C++ Interface to set the default random engine type.

        \param[in] rtype type of the random number generator

        \ingroup random_func_set_default_engine
    */
    FLY_API void setDefaultRandomEngineType(randomEngineType rtype);

    /**
        C++ Interface to get the default random engine type.

        \return \ref fly::randomEngine object for the default random engine

        \ingroup random_func_get_default_engine
    */
    FLY_API randomEngine getDefaultRandomEngine(void);

    /**
        C++ Interface to set the seed of the default random number generator.

        \param[in] seed 64-bit unsigned integer

        \ingroup random_func_set_seed
    */
    FLY_API void setSeed(const unsigned long long seed);

    /**
        C++ Interface to get the seed of the default random number generator.

        \return seed 64-bit unsigned integer

        \ingroup random_func_get_seed
    */
    FLY_API unsigned long long getSeed();

}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    /**
       C Interface to create a random engine.

       \param[out] engine pointer to the returned random engine object
       \param[in]  rtype  type of the random number generator
       \param[in]  seed   initializing seed of the random number generator
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup random_func_random_engine
    */
    FLY_API fly_err fly_create_random_engine(fly_random_engine *engine,
                                         fly_random_engine_type rtype,
                                         unsigned long long seed);

    /**
       C Interface to retain a random engine.

       \param[out] out    pointer to the returned random engine object
       \param[in]  engine random engine object
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup random_func_random_engine
    */
    FLY_API fly_err fly_retain_random_engine(fly_random_engine *out,
                                         const fly_random_engine engine);

    /**
       C Interface to change random engine type.

       \param[in]  engine random engine object
       \param[in]  rtype  type of the random number generator
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup random_func_random_engine
    */
    FLY_API fly_err fly_random_engine_set_type(fly_random_engine *engine,
                                           const fly_random_engine_type rtype);

    /**
       C Interface to get random engine type.

       \param[out] rtype  type of the random number generator
       \param[in]  engine random engine object
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup random_func_random_engine
    */
    FLY_API fly_err fly_random_engine_get_type(fly_random_engine_type *rtype,
                                           const fly_random_engine engine);

    /**
       C Interface to create an array of uniform numbers using a random engine.

       \param[out] out    pointer to the returned object
       \param[in]  ndims  number of dimensions
       \param[in]  dims   C pointer with `ndims` elements; each value
                          represents the size of that dimension
       \param[in]  type   type of the \ref fly_array object
       \param[in]  engine random engine object
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup random_func_randu
    */
    FLY_API fly_err fly_random_uniform(fly_array *out, const unsigned ndims,
                                   const dim_t * const dims, const fly_dtype type,
                                   fly_random_engine engine);

    /**
       C Interface to create an array of normal numbers using a random engine.

       \param[out] out    pointer to the returned object
       \param[in]  ndims  number of dimensions
       \param[in]  dims   C pointer with `ndims` elements; each value
                          represents the size of that dimension
       \param[in]  type   type of the \ref fly_array object
       \param[in]  engine random engine object
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup random_func_randn
    */
    FLY_API fly_err fly_random_normal(fly_array *out, const unsigned ndims,
                                  const dim_t * const dims, const fly_dtype type,
                                  fly_random_engine engine);

    /**
       C Interface to set the seed of a random engine.

       \param[out] engine pointer to the returned random engine object
       \param[in]  seed   initializing seed of the random number generator
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup random_func_random_engine
    */
    FLY_API fly_err fly_random_engine_set_seed(fly_random_engine *engine,
                                           const unsigned long long seed);

    /**
       C Interface to get the default random engine.

       \param[out] engine pointer to the returned default random engine object
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup random_func_get_default_engine
    */
    FLY_API fly_err fly_get_default_random_engine(fly_random_engine *engine);

    /**
       C Interface to set the type of the default random engine.

       \param[in]  rtype type of the random number generator
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup random_func_set_default_engine
    */
    FLY_API fly_err fly_set_default_random_engine_type(const fly_random_engine_type rtype);

    /**
       C Interface to get the seed of a random engine.

       \param[out] seed   pointer to the returned seed
       \param[in]  engine random engine object
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup random_func_random_engine
    */
    FLY_API fly_err fly_random_engine_get_seed(unsigned long long * const seed,
                                           fly_random_engine engine);

    /**
       C Interface to release a random engine.

       \param[in] engine random engine object
       \return    \ref FLY_SUCCESS, if function returns successfully, else
                  an \ref fly_err code is given

       \ingroup random_func_random_engine
    */
    FLY_API fly_err fly_release_random_engine(fly_random_engine engine);

    /**
       \param[out] out   generated array
       \param[in]  ndims number of dimensions
       \param[in]  dims  array containing sizes of the dimension
       \param[in]  type  type of array to generate
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup random_func_randu
    */
    FLY_API fly_err fly_randu(fly_array *out, const unsigned ndims,
                          const dim_t * const dims, const fly_dtype type);

    /**
       \param[out] out   generated array
       \param[in]  ndims number of dimensions
       \param[in]  dims  array containing sizes of the dimension
       \param[in]  type  type of array to generate
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

       \ingroup random_func_randn
    */
    FLY_API fly_err fly_randn(fly_array *out, const unsigned ndims,
                          const dim_t * const dims, const fly_dtype type);

    /**
       \param[in] seed a 64-bit unsigned integer
       \return    \ref FLY_SUCCESS, if function returns successfully, else
                  an \ref fly_err code is given

        \ingroup random_func_set_seed
    */
    FLY_API fly_err fly_set_seed(const unsigned long long seed);

    /**
       \param[out] seed a 64-bit unsigned integer
       \return     \ref FLY_SUCCESS, if function returns successfully, else
                   an \ref fly_err code is given

        \ingroup random_func_get_seed
    */
    FLY_API fly_err fly_get_seed(unsigned long long *seed);

#ifdef __cplusplus
}
#endif
