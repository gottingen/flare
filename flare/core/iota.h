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

#ifndef FLARE_CORE_IOTA_H_
#define FLARE_CORE_IOTA_H_

#include <type_traits>
#include <cstddef>
#include <flare/core/pair.h>

namespace flare::detail {

    /*! \class Iota
      \brief A class that mimics a small subset of flare::Tensor

      \tparam T the type returned by operator()
      \tparam SizeType a custom offset type

      \typedef size_type SizeType
      \typedef value_type T
      \typedef non_const_value_type non-const T
      \typedef device_type void
      \typedef data_type const value_type *
      \enum rank always 1

      Iota::operator() returns offset + i
      Meant to be used in place of a flare::Tensor where entry i holds i + offset.
      Unlike a flare::Tensor, Iota is not materialized in memory.

      Constructing with a size less than 0 yeilds a 0-size Iota
    */
    template<typename T, typename SizeType = size_t>
    class Iota {
    public:
        using size_type = SizeType;
        using value_type = T;
        using non_const_value_type = std::remove_const_t<value_type>;
        using device_type = void;
        using data_type = const value_type *;

        /*! \brief construct an Iota where iota(i) -> offset + i

            \param[in] size the number of entries
            \param[in] offset the offset of the first entry

            Constructing with size < 0 yeilds a 0-size Iota
        */
        FLARE_INLINE_FUNCTION
        constexpr Iota(const size_type &size, const value_type offset)
                : size_(size), offset_(offset) {
            if constexpr (std::is_signed_v<size_type>) {
                if (size_ < size_type(0)) {
                    size_ = 0;
                }
            }
        }

        /*! \brief construct an Iota where iota(i) ->  i

            \param[in] size the number of entries
        */
        FLARE_INLINE_FUNCTION
        explicit constexpr Iota(const size_type &size) : Iota(size, 0) {}

        /*! \brief construct a zero-sized iota
         */
        FLARE_INLINE_FUNCTION
        constexpr Iota() : size_(0), offset_(0) {}

        /*! \brief Construct Iota subtensor

          Like the flare::Tensor 1D subtensor constructor:
          \verbatim
          flare::Tensor a(10); // size = 10
          flare::Tensor b(a, flare::pair{3,7}); // entries 3,4,5,6 of a

          Iota a(10);
          Iota b(a, flare::pair{3,7}); // entries // 3,4,5,6 of a
          \endverbatim

          Creating a subtensor outside of the base Iota yeilds undefined behavior
        */
        template<typename P1, typename P2>
        FLARE_INLINE_FUNCTION constexpr Iota(const Iota &base,
                                             const flare::pair<P1, P2> &range)
                : Iota(range.second - range.first, base.offset_ + range.first) {}

        /*! \brief Construct Iota subtensor

           i >= size() or i < 0 yields undefined behavior.
        */
        FLARE_INLINE_FUNCTION
        constexpr T operator()(size_type i) const noexcept {
            return value_type(i + offset_);
        };

        /// \brief return the size of the iota
        FLARE_INLINE_FUNCTION
        constexpr size_t size() const noexcept { return size_; }

        /// \brief Iotas are always like a rank-1 flare::Tensor
        enum {
            rank = 1
        };

    private:
        size_type size_;
        value_type offset_;
    };

    /// \class is_iota
    /// \brief is_iota<T>::value is true if T is a Iota<...>, false otherwise
    template<typename>
    struct is_iota : public std::false_type {
    };
    template<typename... P>
    struct is_iota<Iota<P...>> : public std::true_type {
    };
    template<typename... P>
    struct is_iota<const Iota<P...>> : public std::true_type {
    };
    template<typename... P>
    inline constexpr bool is_iota_v = is_iota<P...>::value;

}  // namespace flare::detail

#endif  // FLARE_CORE_IOTA_H_
