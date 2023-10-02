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

#ifndef FLARE_CORE_COMMON_STACKTRACE_H_
#define FLARE_CORE_COMMON_STACKTRACE_H_

#include <functional>
#include <ostream>
#include <string>

namespace flare {
namespace detail {

/// \brief Return the demangled version of the input symbol, or the
///   original input if demangling is not possible.
std::string demangle(const std::string& name);

/// \brief Save the current stacktrace.
///
/// You may only save one stacktrace at a time.  If you call this
/// twice, the second call will overwrite the result of the first
/// call.
void save_stacktrace();

/// \brief Print the raw form of the currently saved stacktrace, if
///   any, to the given output stream.
void print_saved_stacktrace(std::ostream& out);

/// \brief Print the currently saved, demangled stacktrace, if any, to
///   the given output stream.
///
/// Demangling is best effort only.
void print_demangled_saved_stacktrace(std::ostream& out);

/// \brief Set the std::terminate handler so that it prints the
///   currently saved stack trace, then calls user_post.
///
/// This is useful if you want to call, say, MPI_Abort instead of
/// std::abort.  The MPI Standard frowns upon calling MPI functions
/// without including their header file, and flare does not depend on
/// MPI, so there's no way for flare to depend on MPI_Abort in a
/// portable way.
void set_flare_terminate_handler(std::function<void()> user_post = nullptr);

}  // namespace detail
}  // namespace flare

#endif  // FLARE_CORE_COMMON_STACKTRACE_H_
