// Copyright 2023 The Elastic-AI Authors.
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

#include <flare/core.h>
#include <flare/core/memory/host_space_deepcopy.h>

namespace flare {

namespace detail {

void hostspace_fence(const DefaultHostExecutionSpace& exec) {
  exec.fence("HostSpace fence");
}

void hostspace_parallel_deepcopy(void* dst, const void* src, ptrdiff_t n) {
  flare::DefaultHostExecutionSpace exec;
  hostspace_parallel_deepcopy_async(exec, dst, src, n);
}

// DeepCopy called with an execution space that can't access HostSpace
void hostspace_parallel_deepcopy_async(void* dst, const void* src,
                                       ptrdiff_t n) {
  flare::DefaultHostExecutionSpace exec;
  hostspace_parallel_deepcopy_async(exec, dst, src, n);
  exec.fence(
      "flare::detail::hostspace_parallel_deepcopy_async: fence after copy");
}

void hostspace_parallel_deepcopy_async(const DefaultHostExecutionSpace& exec,
                                       void* dst, const void* src,
                                       ptrdiff_t n) {
  using policy_t = flare::RangePolicy<flare::DefaultHostExecutionSpace>;

  constexpr int host_deep_copy_serial_limit = 10 * 8192;
  if ((n < host_deep_copy_serial_limit) ||
      (DefaultHostExecutionSpace().concurrency() == 1)) {
    if (0 < n) std::memcpy(dst, src, n);
    return;
  }

  // Both src and dst are aligned the same way with respect to 8 byte words
  if (reinterpret_cast<ptrdiff_t>(src) % 8 ==
      reinterpret_cast<ptrdiff_t>(dst) % 8) {
    char* dst_c       = reinterpret_cast<char*>(dst);
    const char* src_c = reinterpret_cast<const char*>(src);
    int count         = 0;
    // get initial bytes copied
    while (reinterpret_cast<ptrdiff_t>(dst_c) % 8 != 0) {
      *dst_c = *src_c;
      dst_c++;
      src_c++;
      count++;
    }

    // copy the bulk of the data
    double* dst_p       = reinterpret_cast<double*>(dst_c);
    const double* src_p = reinterpret_cast<const double*>(src_c);
    flare::parallel_for("flare::detail::host_space_deepcopy_double",
                         policy_t(exec, 0, (n - count) / 8),
                         [=](const ptrdiff_t i) { dst_p[i] = src_p[i]; });

    // get final data copied
    dst_c += ((n - count) / 8) * 8;
    src_c += ((n - count) / 8) * 8;
    char* dst_end = reinterpret_cast<char*>(dst) + n;
    while (dst_c != dst_end) {
      *dst_c = *src_c;
      dst_c++;
      src_c++;
    }
    return;
  }

  // Both src and dst are aligned the same way with respect to 4 byte words
  if (reinterpret_cast<ptrdiff_t>(src) % 4 ==
      reinterpret_cast<ptrdiff_t>(dst) % 4) {
    char* dst_c       = reinterpret_cast<char*>(dst);
    const char* src_c = reinterpret_cast<const char*>(src);
    int count         = 0;
    // get initial bytes copied
    while (reinterpret_cast<ptrdiff_t>(dst_c) % 4 != 0) {
      *dst_c = *src_c;
      dst_c++;
      src_c++;
      count++;
    }

    // copy the bulk of the data
    int32_t* dst_p       = reinterpret_cast<int32_t*>(dst_c);
    const int32_t* src_p = reinterpret_cast<const int32_t*>(src_c);
    flare::parallel_for("flare::detail::host_space_deepcopy_int",
                         policy_t(exec, 0, (n - count) / 4),
                         [=](const ptrdiff_t i) { dst_p[i] = src_p[i]; });

    // get final data copied
    dst_c += ((n - count) / 4) * 4;
    src_c += ((n - count) / 4) * 4;
    char* dst_end = reinterpret_cast<char*>(dst) + n;
    while (dst_c != dst_end) {
      *dst_c = *src_c;
      dst_c++;
      src_c++;
    }
    return;
  }

  // Src and dst are not aligned the same way, we can only to byte wise copy.
  {
    char* dst_p       = reinterpret_cast<char*>(dst);
    const char* src_p = reinterpret_cast<const char*>(src);
    flare::parallel_for("flare::detail::host_space_deepcopy_char",
                         policy_t(exec, 0, n),
                         [=](const ptrdiff_t i) { dst_p[i] = src_p[i]; });
  }
}

}  // namespace detail

}  // namespace flare
