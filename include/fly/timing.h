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

#if defined(_WIN32) || defined(_MSC_VER)
  #include <windows.h>
#elif defined(__APPLE__) && defined(__MACH__)
  // http://developer.apple.com/qa/qa2004/qa1398.html
  #include <mach/mach_time.h>
#else // Linux
  #ifndef FLY_DOC
    #include <sys/time.h>
  #endif
#endif

namespace fly {

/// Internal timer object
    typedef struct timer {
    #if defined(_WIN32) || defined(_MSC_VER)
      LARGE_INTEGER val;
    #elif defined(__APPLE__) && defined(__MACH__)
      uint64_t val;
    #else // Linux
      struct timeval val;
    #endif

    FLY_API static timer start();

    FLY_API static double stop();
    FLY_API static double stop(timer start);

} timer;

FLY_API double timeit(void(*fn)());
}

#endif
