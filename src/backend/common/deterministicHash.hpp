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

#include <fly/span.hpp>
#include <string>

#include <common/Source.hpp>

/// Return the FNV-1a hash of the provided bata.
///
/// \param[in] data Binary data to hash
/// \param[in] byteSize Size of the data in bytes
/// \param[in] optional prevHash Hash of previous parts when string is split
///
/// \returns An unsigned integer representing the hash of the data
constexpr std::size_t FNV1A_BASE_OFFSET = 0x811C9DC5;
constexpr std::size_t FNV1A_PRIME       = 0x01000193;
std::size_t deterministicHash(const void* data, std::size_t byteSize,
                              const std::size_t prevHash = FNV1A_BASE_OFFSET);

// This is just a wrapper around the above function.
std::size_t deterministicHash(const std::string& data,
                              const std::size_t prevHash = FNV1A_BASE_OFFSET);

// This concatenates strings in the vector and computes hash
std::size_t deterministicHash(nonstd::span<const std::string> list,
                              const std::size_t prevHash = FNV1A_BASE_OFFSET);

// This concatenates hashes of multiple sources
std::size_t deterministicHash(
    nonstd::span<const flare::common::Source> list);
