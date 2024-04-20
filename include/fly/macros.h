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
#include <stdio.h>

///
/// Print a line on screen using printf syntax.
/// Usage: Uses same syntax and semantics as printf.
/// Output: \<filename\>:\<line number\>: \<message\>
///
#ifndef FLY_MSG
#define FLY_MSG(fmt,...) do {            \
        printf("%s:%d: " fmt "\n",      \
                 __FILE__, __LINE__, ##__VA_ARGS__);      \
        } while (0);
#endif

/**
 * FLY_MEM_INFO macro can be used to print the current stats of Flare's memory
 * manager.
 *
 * FLY_MEM_INFO print 4 values:
 *
 * ---------------------------------------------------
 *  Name                    | Description
 * -------------------------|-------------------------
 *  Allocated Bytes         | Total number of bytes allocated by the memory manager
 *  Allocated Buffers       | Total number of buffers allocated
 *  Locked (In Use) Bytes   | Number of bytes that are in use by active arrays
 *  Locked (In Use) Buffers | Number of buffers that are in use by active arrays
 * ---------------------------------------------------
 *
 *  The `Allocated Bytes` is always a multiple of the memory step size. The
 *  default step size is 1024 bytes. This means when a buffer is to be
 *  allocated, the size is always rounded up to a multiple of the step size.
 *  You can use fly::getMemStepSize() to check the current step size and
 *  fly::setMemStepSize() to set a custom resolution size.
 *
 *  The `Allocated Buffers` is the number of buffers that use up the allocated
 *  bytes. This includes buffers currently in scope, as well as buffers marked
 *  as free, ie, from arrays gone out of scope. The free buffers are available
 *  for use by new arrays that might be created.
 *
 *  The `Locked Bytes` is the number of bytes in use that cannot be
 *  reallocated at the moment. The difference of Allocated Bytes and Locked
 *  Bytes is the total bytes available for reallocation.
 *
 *  The `Locked Buffers` is the number of buffer in use that cannot be
 *  reallocated at the moment. The difference of Allocated Buffers and Locked
 *  Buffers is the number of buffers available for reallocation.
 *
 * The FLY_MEM_INFO macro can accept a string an argument that is printed to screen
 *
 * \param[in] msg (Optional) A message that is printed to screen
 *
 * \code
 *     FLY_MEM_INFO("At start");
 * \endcode
 *
 * Output:
 *
 *     FLY Memory at /workspace/myfile.cpp:41: At Start
 *     Allocated [ Bytes | Buffers ] = [ 4096 | 4 ]
 *     In Use    [ Bytes | Buffers ] = [ 2048 | 2 ]
 */
#define FLY_MEM_INFO(msg) do {                                                           \
    size_t abytes = 0, abuffs = 0, lbytes = 0, lbuffs = 0;                              \
    fly_err err = fly_device_mem_info(&abytes, &abuffs, &lbytes, &lbuffs);                \
    if(err == FLY_SUCCESS) {                                                             \
        printf("FLY Memory at %s:%d: " msg "\n", __FILE__, __LINE__);                    \
        printf("Allocated [ Bytes | Buffers ] = [ %ld | %ld ]\n", abytes, abuffs);      \
        printf("In Use    [ Bytes | Buffers ] = [ %ld | %ld ]\n", lbytes, lbuffs);      \
    } else {                                                                            \
        fprintf(stderr, "FLY Memory at %s:%d: " msg "\nFLY Error %d\n",                   \
                __FILE__, __LINE__, err);                                               \
    }                                                                                   \
} while(0)
