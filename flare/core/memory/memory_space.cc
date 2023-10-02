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

/** @file memory_space.cc
 *
 *  Operations common to memory space instances, or at least default
 *  implementations thereof.
 */

#include <flare/core/memory/memory_space.h>

#include <iostream>
#include <string>
#include <sstream>

namespace flare {
namespace detail {

void safe_throw_allocation_with_header_failure(
    std::string const& space_name, std::string const& label,
    flare::experimental::RawMemoryAllocationFailure const& failure) {
  auto generate_failure_message = [&](std::ostream& o) {
    o << "flare failed to allocate memory for label \"" << label
      << "\".  Allocation using MemorySpace named \"" << space_name
      << "\" failed with the following error:  ";
    failure.print_error_message(o);
    if (failure.failure_mode() ==
        flare::experimental::RawMemoryAllocationFailure::FailureMode::
            AllocationNotAligned) {
      // TODO: delete the misaligned memory?
      o << "Warning: Allocation failed due to misalignment; memory may "
           "be leaked.\n";
    }
    o.flush();
  };
  try {
    std::ostringstream sstr;
    generate_failure_message(sstr);
    flare::detail::throw_runtime_exception(sstr.str());
  } catch (std::bad_alloc const&) {
    // Probably failed to allocate the string because we're so close to out
    // of memory. Try printing to std::cerr instead
    try {
      generate_failure_message(std::cerr);
    } catch (std::bad_alloc const&) {
      // oh well, we tried...
    }
    flare::detail::throw_runtime_exception(
        "flare encountered an allocation failure, then another allocation "
        "failure while trying to create the error message.");
  }
}

}  // end namespace detail
}  // end namespace flare
