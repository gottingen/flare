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
#include <string>

// Some compilers create these macros in the header. Causes
// some errors in the Version struct constructor
#ifdef major
#undef major
#endif
#ifdef minor
#undef minor
#endif

namespace flare {
namespace common {
class Version {
    int major_ = -1;
    int minor_ = -1;
    int patch_ = -1;

   public:
    /// Checks if the major version is defined before minor and minor is defined
    /// before patch
    constexpr static bool validate(int major_, int minor_,
                                   int patch_) noexcept {
        return !(major_ < 0 && (minor_ >= 0 || patch_ >= 0)) &&
               !(minor_ < 0 && patch_ >= 0);
    }

    constexpr int major() const { return major_; }
    constexpr int minor() const { return minor_; }
    constexpr int patch() const { return patch_; }

    constexpr Version(const int ver_major, const int ver_minor = -1,
                      const int ver_patch = -1) noexcept
        : major_(ver_major), minor_(ver_minor), patch_(ver_patch) {}
};

constexpr bool operator==(const Version& lhs, const Version& rhs) {
    return lhs.major() == rhs.major() && lhs.minor() == rhs.minor() &&
           lhs.patch() == rhs.patch();
}

constexpr bool operator!=(const Version& lhs, const Version& rhs) {
    return !(lhs == rhs);
}

constexpr static Version NullVersion{-1, -1, -1};

constexpr bool operator<(const Version& lhs, const Version& rhs) {
    if (lhs == NullVersion || rhs == NullVersion) return false;
    if (lhs.major() != -1 && rhs.major() != -1 && lhs.major() < rhs.major())
        return true;
    if (lhs.minor() != -1 && rhs.minor() != -1 && lhs.minor() < rhs.minor())
        return true;
    if (lhs.patch() != -1 && rhs.patch() != -1 && lhs.patch() < rhs.patch())
        return true;
    return false;
}

inline Version fromCudaVersion(size_t version_int) {
    return {static_cast<int>(version_int / 1000),
            static_cast<int>(version_int % 1000) / 10,
            static_cast<int>(version_int % 10)};
}

inline std::string int_version_to_string(int version) {
    return std::to_string(version / 1000) + "." +
           std::to_string(static_cast<int>((version % 1000) / 10.));
}

}  // namespace common
}  // namespace flare
