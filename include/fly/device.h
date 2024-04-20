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

#ifdef __cplusplus
namespace fly
{
    /**
       \defgroup device_func_info info

       Display Flare and device info

       @{

       \ingroup flare_func
       \ingroup device_mat
    */
    FLY_API void info();
    /**
       @}
    */

    /**
       \defgroup device_func_info_string infoString

       Get fly::info() as a string

       @{

       \brief Returns the output of fly::info() as a string

       \param[in] verbose flag to return verbose info

       \returns string containing output of fly::info()

       \ingroup flare_func
       \ingroup device_mat
    */
    FLY_API const char* infoString(const bool verbose = false);
    /**
       @}
    */

    /**
        \copydoc device_func_prop

        \ingroup device_func_prop
    */
    FLY_API void deviceInfo(char* d_name, char* d_platform, char *d_toolkit, char* d_compute);

    /// \brief Gets the number of devices
    ///
    /// \copydoc device_func_count
    /// \returns the number of devices on the system
    /// \ingroup device_func_count
    FLY_API int getDeviceCount();

    /// \brief Gets the current device ID
    ///
    /// \copydoc device_func_get
    /// \returns the device ID of the current device
    /// \ingroup device_func_get
    FLY_API int getDevice();

    /// \brief Queries the current device for double precision floating point
    ///        support
    ///
    /// \param[in] device the ID of the device to query
    ///
    /// \returns true if the \p device supports double precision operations.
    ///          false otherwise
    /// \ingroup device_func_dbl
    FLY_API bool isDoubleAvailable(const int device);

    /// \brief Queries the current device for half precision floating point
    ///        support
    ///
    /// \param[in] device the ID of the device to query
    ///
    /// \returns true if the \p device supports half precision operations.
    ///          false otherwise
    /// \ingroup device_func_half
    FLY_API bool isHalfAvailable(const int device);

    /// \brief Sets the current device
    ///
    /// \param[in] device The ID of the target device
    /// \ingroup device_func_set
    FLY_API void setDevice(const int device);

    /// \brief Blocks until the \p device is finished processing
    ///
    /// \param[in] device is the target device
    /// \ingroup device_func_sync
    FLY_API void sync(const int device = -1);

    /// \ingroup device_func_alloc
    /// @{
    /// \brief Allocates memory using Flare's memory manager
    ///
    /// \param[in] elements the number of elements to allocate
    /// \param[in] type is the type of the elements to allocate
    /// \returns Pointer to the device memory on the current device. This is a
    ///          CUDA device pointer for the CUDA backend. and a C pointer
    ///          for the CPU backend
    ///
    /// \note The device memory returned by this function is only freed if
    ///       fly::free() is called explicitly
    FLY_DEPRECATED("Use fly::fly_alloc instead")
    FLY_API void *alloc(const size_t elements, const dtype type);

    /// \brief Allocates memory using Flare's memory manager
    ///
    /// \param[in] bytes the number of bytes to allocate
    /// \returns Pointer to the device memory on the current device. This is a
    ///          CUDA device pointer for the CUDA backend.and a C pointer for the CPU backend
    ///
    /// \note The device memory returned by this function is only freed if
    ///       fly::fly_free() is called explicitly
    FLY_API void *fly_alloc(const size_t bytes);

    /// \brief Allocates memory using Flare's memory manager
    //
    /// \param[in] elements the number of elements to allocate
    /// \returns Pointer to the device memory on the current device. This is a
    ///          CUDA device pointer for the CUDA backend. and a C pointer
    ///          for the CPU backend
    ///
    /// \note the size of the memory allocated is the number of \p elements *
    ///       sizeof(type)
    /// \note The device memory returned by this function is only freed if
    ///       fly::free() is called explicitly
    template <typename T>
    FLY_DEPRECATED("Use fly::fly_alloc instead")
    T *alloc(const size_t elements);
    /// @}

    /// \ingroup device_func_free
    ///
    /// \copydoc device_func_free
    /// \param[in] ptr the memory allocated by the fly::alloc function that
    ///                will be freed
    ///
    /// \note This function will free a device pointer even if it has been
    ///       previously locked.
    FLY_DEPRECATED("Use fly::fly_free instead")
    FLY_API void free(const void *ptr);

    /// \ingroup device_func_free
    /// \copydoc device_func_free
    /// \param[in] ptr The pointer returned by fly::fly_alloc
    ///
    /// This function will free a device pointer even if it has been previously
    /// locked.
    FLY_API void fly_free(const void *ptr);

    /// \ingroup device_func_pinned
    /// @{
    /// \copydoc device_func_pinned
    ///
    /// \param[in] elements the number of elements to allocate
    /// \param[in] type is the type of the elements to allocate
    /// \returns the pointer to the memory
    FLY_API void *pinned(const size_t elements, const dtype type);

    /// \copydoc device_func_pinned
    ///
    /// \param[in] elements the number of elements to allocate
    /// \returns the pointer to the memory
    template<typename T>
    T* pinned(const size_t elements);
    /// @}

    /// \ingroup device_func_free_pinned
    ///
    /// \copydoc device_func_free_pinned
    /// \param[in] ptr the memory to free
    FLY_API void freePinned(const void *ptr);

    /// \brief Allocate memory on host
    ///
    /// \copydoc device_func_alloc_host
    ///
    /// \param[in] elements the number of elements to allocate
    /// \param[in] type is the type of the elements to allocate
    /// \returns the pointer to the memory
    ///
    /// \ingroup device_func_alloc_host
    FLY_API void *allocHost(const size_t elements, const dtype type);

    /// \brief Allocate memory on host
    ///
    /// \copydoc device_func_alloc_host
    ///
    /// \param[in] elements the number of elements to allocate
    /// \returns the pointer to the memory
    ///
    /// \note the size of the memory allocated is the number of \p elements *
    ///         sizeof(type)
    ///
    /// \ingroup device_func_alloc_host
    template<typename T>
    FLY_API T* allocHost(const size_t elements);

    /// \brief Free memory allocated internally by Flare
    //
    /// \copydoc device_func_free_host
    ///
    /// \param[in] ptr the memory to free
    ///
    /// \ingroup device_func_free_host
    FLY_API void freeHost(const void *ptr);

    /// \ingroup device_func_mem
    /// @{
    /// \brief Gets information about the memory manager
    ///
    /// \param[out] alloc_bytes the number of bytes allocated by the memory
    //                          manager
    /// \param[out] alloc_buffers   the number of buffers created by the memory
    //                              manager
    /// \param[out] lock_bytes The number of bytes in use
    /// \param[out] lock_buffers The number of buffers in use
    ///
    /// \note This function performs a synchronization operation
    FLY_API void deviceMemInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                             size_t *lock_bytes, size_t *lock_buffers);

    ///
    /// Prints buffer details from the Flare Device Manager
    //
    /// \param [in] msg A message to print before the table
    /// \param [in] device_id print the memory info of the specified device.
    ///  -1 signifies active device.
    //
    /// \ingroup device_func_mem
    ///
    /// \note This function performs a synchronization operation
    FLY_API void printMemInfo(const char *msg = NULL, const int device_id = -1);

    /// \brief Call the garbage collection function in the memory manager
    ///
    /// \ingroup device_func_mem
    FLY_API void deviceGC();
    /// @}

    /// \brief Set the resolution of memory chunks. Works only with the default
    /// memory manager - throws if a custom memory manager is set.
    ///
    /// \ingroup device_func_mem
    FLY_API void setMemStepSize(const size_t size);

    /// \brief Get the resolution of memory chunks. Works only with the default
    /// memory manager - throws if a custom memory manager is set.
    ///
    /// \ingroup device_func_mem
    FLY_API size_t getMemStepSize();
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    /**
       \ingroup device_func_info
    */
    FLY_API fly_err fly_info();

    /**
       \ingroup device_func_info
    */
    FLY_API fly_err fly_init();

    /**
       \brief Gets the output of fly_info() as a string

       \param[out] str contains the string
       \param[in] verbose flag to return verbose info

       \ingroup device_func_info_string
    */
    FLY_API fly_err fly_info_string(char** str, const bool verbose);

    /**
        \copydoc device_func_prop

        \ingroup device_func_prop
    */
    FLY_API fly_err fly_device_info(char* d_name, char* d_platform, char *d_toolkit, char* d_compute);

    /**
       \ingroup device_func_count
    */
    FLY_API fly_err fly_get_device_count(int *num_of_devices);

    /**
       \ingroup device_func_dbl
    */
    FLY_API fly_err fly_get_dbl_support(bool* available, const int device);

    /**
       \ingroup device_func_half
    */
    FLY_API fly_err fly_get_half_support(bool *available, const int device);

    /**
       \ingroup device_func_set
    */
    FLY_API fly_err fly_set_device(const int device);

    /**
       \ingroup device_func_set
    */
    FLY_API fly_err fly_get_device(int *device);

    /**
       \ingroup device_func_sync
    */
    FLY_API fly_err fly_sync(const int device);

    /**
       \brief Allocates memory using Flare's memory manager
       \ingroup device_func_alloc

       This device memory returned by this function can only be freed using
       fly_free_device

       \param [out] ptr Pointer to the device memory on the current device. This
                        is a CUDA device pointer for the CUDA backend. and a C pointer
                        for the CPU backend
       \param [in] bytes The number of bites to allocate on the device

       \returns FLY_SUCCESS if a pointer could be allocated. FLY_ERR_NO_MEM if
                there is no memory
    */
    FLY_API fly_err fly_alloc_device(void **ptr, const dim_t bytes);

    /**
       \brief Returns memory to Flare's memory manager.

       This function will free a device pointer even if it has been previously
       locked.

       \param[in] ptr The pointer allocated by fly_alloc_device to be freed

       \ingroup device_func_free
    */
    FLY_API fly_err fly_free_device(void *ptr);

    /**
       \ingroup device_func_pinned
    */
    FLY_API fly_err fly_alloc_pinned(void **ptr, const dim_t bytes);

    /**
       \ingroup device_func_free_pinned
    */
    FLY_API fly_err fly_free_pinned(void *ptr);

    /**
       \ingroup device_func_alloc_host
    */
    FLY_API fly_err fly_alloc_host(void **ptr, const dim_t bytes);

    /**
       \ingroup device_func_free_host
    */
    FLY_API fly_err fly_free_host(void *ptr);

    /**
       Create array from device memory
       \ingroup c_api_mat
    */
    FLY_API fly_err fly_device_array(fly_array *arr, void *data, const unsigned ndims, const dim_t * const dims, const fly_dtype type);

    /**
       Get memory information from the memory manager
       \ingroup device_func_mem
    */
    FLY_API fly_err fly_device_mem_info(size_t *alloc_bytes, size_t *alloc_buffers,
                                    size_t *lock_bytes, size_t *lock_buffers);

    /**
       Prints buffer details from the Flare Device Manager.

       The result is a table with several columns:

        * POINTER:   The hex address of the array's device or pinned-memory
                     pointer
        * SIZE:      Human-readable size of the array
        * FLY LOCK:   Indicates whether Flare is using this chunk of memory.
                     If not, the chunk is ready for reuse.
        * USER LOCK: If set, Flare is prevented from freeing this memory.
                     The chunk is not ready for re-use even if all Flare's
                     references to it go out of scope.

       \param [in] msg A message to print before the table
       \param [in] device_id print the memory info of the specified device.
       -1 signifies active device.

       \returns FLY_SUCCESS if successful

       \ingroup device_func_mem
    */
    FLY_API fly_err fly_print_mem_info(const char *msg, const int device_id);

    /**
       Call the garbage collection routine
       \ingroup device_func_mem
    */
    FLY_API fly_err fly_device_gc();

    /**
       Set the minimum memory chunk size. Works only with the default
       memory manager - returns an error if a custom memory manager is set.

       \ingroup device_func_mem
    */
    FLY_API fly_err fly_set_mem_step_size(const size_t step_bytes);

    /**
       Get the minimum memory chunk size. Works only with the default
       memory manager - returns an error if a custom memory manager is set.

       \ingroup device_func_mem
    */
    FLY_API fly_err fly_get_mem_step_size(size_t *step_bytes);

    /**
       Lock the device buffer in the memory manager.

       Locked buffers are not freed by memory manager until \ref fly_unlock_array is called.
       \ingroup device_func_mem
    */
    FLY_DEPRECATED("Use fly_lock_array instead")
    FLY_API fly_err fly_lock_device_ptr(const fly_array arr);

    /**
       Unlock device buffer in the memory manager.

       This function will give back the control over the device pointer to the memory manager.
       \ingroup device_func_mem
    */
    FLY_DEPRECATED("Use fly_unlock_array instead")
    FLY_API fly_err fly_unlock_device_ptr(const fly_array arr);

    /**
       Lock the device buffer in the memory manager.

       Locked buffers are not freed by memory manager until \ref fly_unlock_array is called.
       \ingroup device_func_mem
    */
    FLY_API fly_err fly_lock_array(const fly_array arr);

    /**
       Unlock device buffer in the memory manager.

       This function will give back the control over the device pointer to the memory manager.
       \ingroup device_func_mem
    */
    FLY_API fly_err fly_unlock_array(const fly_array arr);

    /**
       Query if the array has been locked by the user.

       An array can be locked by the user by calling `fly_lock_array`
       or `fly_get_device_ptr` or `fly_get_raw_ptr` function.

       \ingroup device_func_mem
    */
    FLY_API fly_err fly_is_locked_array(bool *res, const fly_array arr);

    /**
       Get the device pointer and lock the buffer in memory manager.

       The device pointer \p ptr is notfreed by memory manager until \ref fly_unlock_device_ptr is called.
       \ingroup device_func_mem
    */
    FLY_API fly_err fly_get_device_ptr(void **ptr, const fly_array arr);

    /**
       Sets the path where the kernels generated at runtime will be cached

       Sets the path where the kernels generated at runtime will be stored to
       cache for later use. The files in this directory can be safely deleted.
       The default location for these kernels is in $HOME/.flare on Unix
       systems and in the Flare temp directory on Windows.

       \param[in] path The location where the kernels will be stored
       \param[in] override_env if true this path will take precedence over the
                               FLY_JIT_KERNEL_CACHE_DIRECTORY environment variable.
                               If false, the environment variable takes precedence
                               over this path.

       \returns FLY_SUCCESS if the variable is set. FLY_ERR_ARG if path is NULL.
       \ingroup device_func_mem
    */
    FLY_API fly_err fly_set_kernel_cache_directory(const char* path,
                                               int override_env);

    /**
       Gets the path where the kernels generated at runtime will be cached

       Gets the path where the kernels generated at runtime will be stored to
       cache for later use. The files in this directory can be safely deleted.
       The default location for these kernels is in $HOME/.flare on Unix
       systems and in the Flare temp directory on Windows.

       \param[out] length The length of the path array. If \p path is NULL, the
                          length of the current path is assigned to this pointer
       \param[out] path The path of the runtime generated kernel cache
                         variable. If NULL, the current path length is assigned
                         to \p length
       \returns FLY_SUCCESS if the variable is set.
                FLY_ERR_ARG if path and length are null at the same time.
                FLY_ERR_SIZE if \p length not sufficient enought to store the
                            path
       \ingroup device_func_mem
    */
    FLY_API fly_err fly_get_kernel_cache_directory(size_t *length, char *path);


#ifdef __cplusplus
}
#endif
