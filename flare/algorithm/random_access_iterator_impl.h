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

#ifndef FLARE_ALGORITHM_RANDOM_ACCESS_ITERATOR_IMPL_H_
#define FLARE_ALGORITHM_RANDOM_ACCESS_ITERATOR_IMPL_H_

#include <iterator>
#include <flare/core/defines.h>
#include <flare/core/tensor/view.h>
#include <flare/algorithm/constraints_impl.h>

namespace flare {
namespace experimental {
namespace detail {

template <class T>
class RandomAccessIterator;

template <class DataType, class... Args>
class RandomAccessIterator< ::flare::View<DataType, Args...> > {
 public:
  using view_type     = ::flare::View<DataType, Args...>;
  using iterator_type = RandomAccessIterator<view_type>;

  using iterator_category = std::random_access_iterator_tag;
  using value_type        = typename view_type::value_type;
  using difference_type   = ptrdiff_t;
  using pointer           = typename view_type::pointer_type;
  using reference         = typename view_type::reference_type;

  static_assert(view_type::rank == 1 &&
                    (std::is_same<typename view_type::traits::array_layout,
                                  flare::LayoutLeft>::value ||
                     std::is_same<typename view_type::traits::array_layout,
                                  flare::LayoutRight>::value ||
                     std::is_same<typename view_type::traits::array_layout,
                                  flare::LayoutStride>::value),
                "RandomAccessIterator only supports 1D Views with LayoutLeft, "
                "LayoutRight, LayoutStride.");

  FLARE_DEFAULTED_FUNCTION RandomAccessIterator() = default;

  explicit FLARE_FUNCTION RandomAccessIterator(const view_type view)
      : m_view(view) {}
  explicit FLARE_FUNCTION RandomAccessIterator(const view_type view,
                                                ptrdiff_t current_index)
      : m_view(view), m_current_index(current_index) {}

  FLARE_FUNCTION
  iterator_type& operator++() {
    ++m_current_index;
    return *this;
  }

  FLARE_FUNCTION
  iterator_type operator++(int) {
    auto tmp = *this;
    ++*this;
    return tmp;
  }

  FLARE_FUNCTION
  iterator_type& operator--() {
    --m_current_index;
    return *this;
  }

  FLARE_FUNCTION
  iterator_type operator--(int) {
    auto tmp = *this;
    --*this;
    return tmp;
  }

  FLARE_FUNCTION
  reference operator[](difference_type n) const {
    return m_view(m_current_index + n);
  }

  FLARE_FUNCTION
  iterator_type& operator+=(difference_type n) {
    m_current_index += n;
    return *this;
  }

  FLARE_FUNCTION
  iterator_type& operator-=(difference_type n) {
    m_current_index -= n;
    return *this;
  }

  FLARE_FUNCTION
  iterator_type operator+(difference_type n) const {
    return iterator_type(m_view, m_current_index + n);
  }

  FLARE_FUNCTION
  iterator_type operator-(difference_type n) const {
    return iterator_type(m_view, m_current_index - n);
  }

  FLARE_FUNCTION
  difference_type operator-(iterator_type it) const {
    return m_current_index - it.m_current_index;
  }

  FLARE_FUNCTION
  bool operator==(iterator_type other) const {
    return m_current_index == other.m_current_index &&
           m_view.data() == other.m_view.data();
  }

  FLARE_FUNCTION
  bool operator!=(iterator_type other) const {
    return m_current_index != other.m_current_index ||
           m_view.data() != other.m_view.data();
  }

  FLARE_FUNCTION
  bool operator<(iterator_type other) const {
    return m_current_index < other.m_current_index;
  }

  FLARE_FUNCTION
  bool operator<=(iterator_type other) const {
    return m_current_index <= other.m_current_index;
  }

  FLARE_FUNCTION
  bool operator>(iterator_type other) const {
    return m_current_index > other.m_current_index;
  }

  FLARE_FUNCTION
  bool operator>=(iterator_type other) const {
    return m_current_index >= other.m_current_index;
  }

  FLARE_FUNCTION
  reference operator*() const { return m_view(m_current_index); }

 private:
  view_type m_view;
  ptrdiff_t m_current_index = 0;
};

}  // namespace detail
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_RANDOM_ACCESS_ITERATOR_IMPL_H_
