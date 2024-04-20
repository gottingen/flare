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

#include <common/deterministicHash.hpp>

#include <fly/span.hpp>
#include <numeric>
#include <string>

using nonstd::span;
using std::accumulate;
using std::string;

size_t deterministicHash(const void* data, size_t byteSize, size_t prevHash) {
    // Fowler-Noll-Vo "1a" 32 bit hash
    // https://en.wikipedia.org/wiki/Fowler-Noll-Vo_hash_function
    const auto* byteData = static_cast<const uint8_t*>(data);
    return accumulate(
        byteData, byteData + byteSize, prevHash,
        [&](size_t hash, uint8_t data) { return (hash ^ data) * FNV1A_PRIME; });
}

size_t deterministicHash(const string& data, const size_t prevHash) {
    return deterministicHash(data.data(), data.size(), prevHash);
}

size_t deterministicHash(span<const string> list, const size_t prevHash) {
    size_t hash = prevHash;
    for (auto s : list) { hash = deterministicHash(s.data(), s.size(), hash); }
    return hash;
}

size_t deterministicHash(span<const flare::common::Source> list) {
    // Combine the different source codes, via their hashes
    size_t hash = FNV1A_BASE_OFFSET;
    for (auto s : list) {
        size_t h = s.hash ? s.hash : deterministicHash(s.ptr, s.length);
        hash     = deterministicHash(&h, sizeof(size_t), hash);
    }
    return hash;
}
