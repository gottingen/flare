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

#ifndef FLARE_CORE_MEMORY_MEMORY_TRAITS_H_
#define FLARE_CORE_MEMORY_MEMORY_TRAITS_H_

#include <flare/core/common/traits.h>

//----------------------------------------------------------------------------

namespace flare {

/** \brief  Memory access traits for views, an extension point.
 *
 *  These traits should be orthogonal.  If there are dependencies then
 *  the MemoryTraits template must detect and enforce dependencies.
 *
 *  A zero value is the default for a View, indicating that none of
 *  these traits are present.
 */
enum MemoryTraitsFlags {
  Unmanaged    = 0x01,
  RandomAccess = 0x02,
  Atomic       = 0x04,
  Restrict     = 0x08,
  Aligned      = 0x10
};

template <unsigned T>
struct MemoryTraits {
  //! Tag this class as a flare memory traits:
  using memory_traits = MemoryTraits<T>;

  static constexpr unsigned impl_value = T;

  static constexpr bool is_unmanaged =
      (unsigned(0) != (T & unsigned(flare::Unmanaged)));
  static constexpr bool is_random_access =
      (unsigned(0) != (T & unsigned(flare::RandomAccess)));
  static constexpr bool is_atomic =
      (unsigned(0) != (T & unsigned(flare::Atomic)));
  static constexpr bool is_restrict =
      (unsigned(0) != (T & unsigned(flare::Restrict)));
  static constexpr bool is_aligned =
      (unsigned(0) != (T & unsigned(flare::Aligned)));
};

}  // namespace flare

//----------------------------------------------------------------------------

namespace flare {

using MemoryManaged   = flare::MemoryTraits<0>;
using MemoryUnmanaged = flare::MemoryTraits<flare::Unmanaged>;
using MemoryRandomAccess =
    flare::MemoryTraits<flare::Unmanaged | flare::RandomAccess>;

}  // namespace flare

//----------------------------------------------------------------------------

namespace flare {
namespace detail {

static_assert((0 < int(FLARE_MEMORY_ALIGNMENT)) &&
                  (0 == (int(FLARE_MEMORY_ALIGNMENT) &
                         (int(FLARE_MEMORY_ALIGNMENT) - 1))),
              "FLARE_MEMORY_ALIGNMENT must be a power of two");

/** \brief Memory alignment settings
 *
 *  Sets global value for memory alignment.  Must be a power of two!
 *  Enable compatibility of views from different devices with static stride.
 *  Use compiler flag to enable overwrites.
 */
enum : unsigned {
  MEMORY_ALIGNMENT           = FLARE_MEMORY_ALIGNMENT,
  MEMORY_ALIGNMENT_THRESHOLD = FLARE_MEMORY_ALIGNMENT_THRESHOLD
};

// ------------------------------------------------------------------ //
//  this identifies the default memory trait
//
template <typename Tp>
struct is_default_memory_trait : std::false_type {};

template <>
struct is_default_memory_trait<flare::MemoryTraits<0>> : std::true_type {};

}  // namespace detail
}  // namespace flare

#endif  // FLARE_CORE_MEMORY_MEMORY_TRAITS_H_
