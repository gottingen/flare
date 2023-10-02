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

#include <cstring>
#include <cstdlib>

#include <iostream>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <flare/core/common/error.h>
#include <flare/core/common/stacktrace.h>
#include <flare/backend/cuda/cuda_error.h>


namespace flare {

#ifdef FLARE_ON_CUDA_DEVICE
    namespace experimental {

void CudaRawMemoryAllocationFailure::append_additional_error_information(
    std::ostream &o) const {
  if (m_error_code != cudaSuccess) {
    o << "  The Cuda allocation returned the error code \""
      << cudaGetErrorName(m_error_code) << "\".";
  }
}

}  // end namespace experimental
#endif

}  // namespace flare