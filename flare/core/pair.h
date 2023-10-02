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

/// \file pair.h
/// \brief Declaration and definition of flare::pair.
///
/// This header file declares and defines flare::pair and its related
/// nonmember functions.

#ifndef FLARE_CORE_PAIR_H_
#define FLARE_CORE_PAIR_H_

#include <flare/core/defines.h>
#include <utility>

namespace flare {
/// \struct pair
/// \brief Replacement for std::pair that works on CUDA devices.
///
/// The instance methods of std::pair, including its constructors, are
/// not marked as <tt>__device__</tt> functions.  Thus, they cannot be
/// called on a CUDA device, such as an NVIDIA GPU.  This struct
/// implements the same interface as std::pair, but can be used on a
/// CUDA device as well as on the host.
template <class T1, class T2>
struct pair {
  //! The first template parameter of this class.
  using first_type = T1;
  //! The second template parameter of this class.
  using second_type = T2;

  //! The first element of the pair.
  first_type first;
  //! The second element of the pair.
  second_type second;

  /// \brief Default constructor.
  ///
  /// This calls the default constructors of T1 and T2.  It won't
  /// compile if those default constructors are not defined and
  /// public.
  FLARE_DEFAULTED_FUNCTION constexpr pair() = default;

  /// \brief Constructor that takes both elements of the pair.
  ///
  /// This calls the copy constructors of T1 and T2.  It won't compile
  /// if those copy constructors are not defined and public.
#if defined(FLARE_COMPILER_NVHPC) && FLARE_COMPILER_NVHPC < 230700
  FLARE_FORCEINLINE_FUNCTION
#else
  FLARE_FORCEINLINE_FUNCTION constexpr
#endif
  pair(first_type const& f, second_type const& s) : first(f), second(s) {}

  /// \brief Copy constructor.
  ///
  /// This calls the copy constructors of T1 and T2.  It won't compile
  /// if those copy constructors are not defined and public.
  template <class U, class V>
#if defined(FLARE_COMPILER_NVHPC) && FLARE_COMPILER_NVHPC < 230700
  FLARE_FORCEINLINE_FUNCTION
#else
  FLARE_FORCEINLINE_FUNCTION constexpr
#endif
  pair(const pair<U, V>& p)
      : first(p.first), second(p.second) {
  }

  /// \brief Assignment operator.
  ///
  /// This calls the assignment operators of T1 and T2.  It won't
  /// compile if the assignment operators are not defined and public.
  template <class U, class V>
  FLARE_FORCEINLINE_FUNCTION pair<T1, T2>& operator=(const pair<U, V>& p) {
    first  = p.first;
    second = p.second;
    return *this;
  }

  // from std::pair<U,V>
  template <class U, class V>
  pair(const std::pair<U, V>& p) : first(p.first), second(p.second) {}

  /// \brief Return the std::pair version of this object.
  ///
  /// This is <i>not</i> a device function; you may not call it on a
  /// CUDA device.  It is meant to be called on the host, if the user
  /// wants an std::pair instead of a flare::pair.
  ///
  /// \note This is not a conversion operator, since defining a
  ///   conversion operator made the relational operators have
  ///   ambiguous definitions.
  std::pair<T1, T2> to_std_pair() const {
    return std::make_pair(first, second);
  }
};

template <class T1, class T2>
struct pair<T1&, T2&> {
  //! The first template parameter of this class.
  using first_type = T1&;
  //! The second template parameter of this class.
  using second_type = T2&;

  //! The first element of the pair.
  first_type first;
  //! The second element of the pair.
  second_type second;

  /// \brief Constructor that takes both elements of the pair.
  ///
  /// This calls the copy constructors of T1 and T2.  It won't compile
  /// if those copy constructors are not defined and public.
  FLARE_FORCEINLINE_FUNCTION constexpr pair(first_type f, second_type s)
      : first(f), second(s) {}

  /// \brief Copy constructor.
  ///
  /// This calls the copy constructors of T1 and T2.  It won't compile
  /// if those copy constructors are not defined and public.
  template <class U, class V>
  FLARE_FORCEINLINE_FUNCTION constexpr pair(const pair<U, V>& p)
      : first(p.first), second(p.second) {}

  // from std::pair<U,V>
  template <class U, class V>
  pair(const std::pair<U, V>& p) : first(p.first), second(p.second) {}

  /// \brief Assignment operator.
  ///
  /// This calls the assignment operators of T1 and T2.  It won't
  /// compile if the assignment operators are not defined and public.
  template <class U, class V>
  FLARE_FORCEINLINE_FUNCTION pair<first_type, second_type>& operator=(
      const pair<U, V>& p) {
    first  = p.first;
    second = p.second;
    return *this;
  }

  /// \brief Return the std::pair version of this object.
  ///
  /// This is <i>not</i> a device function; you may not call it on a
  /// CUDA device.  It is meant to be called on the host, if the user
  /// wants an std::pair instead of a flare::pair.
  ///
  /// \note This is not a conversion operator, since defining a
  ///   conversion operator made the relational operators have
  ///   ambiguous definitions.
  std::pair<T1, T2> to_std_pair() const {
    return std::make_pair(first, second);
  }
};

template <class T1, class T2>
struct pair<T1, T2&> {
  //! The first template parameter of this class.
  using first_type = T1;
  //! The second template parameter of this class.
  using second_type = T2&;

  //! The first element of the pair.
  first_type first;
  //! The second element of the pair.
  second_type second;

  /// \brief Constructor that takes both elements of the pair.
  ///
  /// This calls the copy constructors of T1 and T2.  It won't compile
  /// if those copy constructors are not defined and public.
  FLARE_FORCEINLINE_FUNCTION constexpr pair(first_type const& f, second_type s)
      : first(f), second(s) {}

  /// \brief Copy constructor.
  ///
  /// This calls the copy constructors of T1 and T2.  It won't compile
  /// if those copy constructors are not defined and public.
  template <class U, class V>
  FLARE_FORCEINLINE_FUNCTION constexpr pair(const pair<U, V>& p)
      : first(p.first), second(p.second) {}

  // from std::pair<U,V>
  template <class U, class V>
  pair(const std::pair<U, V>& p) : first(p.first), second(p.second) {}

  /// \brief Assignment operator.
  ///
  /// This calls the assignment operators of T1 and T2.  It won't
  /// compile if the assignment operators are not defined and public.
  template <class U, class V>
  FLARE_FORCEINLINE_FUNCTION pair<first_type, second_type>& operator=(
      const pair<U, V>& p) {
    first  = p.first;
    second = p.second;
    return *this;
  }

  /// \brief Return the std::pair version of this object.
  ///
  /// This is <i>not</i> a device function; you may not call it on a
  /// CUDA device.  It is meant to be called on the host, if the user
  /// wants an std::pair instead of a flare::pair.
  ///
  /// \note This is not a conversion operator, since defining a
  ///   conversion operator made the relational operators have
  ///   ambiguous definitions.
  std::pair<T1, T2> to_std_pair() const {
    return std::make_pair(first, second);
  }
};

template <class T1, class T2>
struct pair<T1&, T2> {
  //! The first template parameter of this class.
  using first_type = T1&;
  //! The second template parameter of this class.
  using second_type = T2;

  //! The first element of the pair.
  first_type first;
  //! The second element of the pair.
  second_type second;

  /// \brief Constructor that takes both elements of the pair.
  ///
  /// This calls the copy constructors of T1 and T2.  It won't compile
  /// if those copy constructors are not defined and public.
  FLARE_FORCEINLINE_FUNCTION constexpr pair(first_type f, second_type const& s)
      : first(f), second(s) {}

  /// \brief Copy constructor.
  ///
  /// This calls the copy constructors of T1 and T2.  It won't compile
  /// if those copy constructors are not defined and public.
  template <class U, class V>
  FLARE_FORCEINLINE_FUNCTION constexpr pair(const pair<U, V>& p)
      : first(p.first), second(p.second) {}

  // from std::pair<U,V>
  template <class U, class V>
  pair(const std::pair<U, V>& p) : first(p.first), second(p.second) {}

  /// \brief Assignment operator.
  ///
  /// This calls the assignment operators of T1 and T2.  It won't
  /// compile if the assignment operators are not defined and public.
  template <class U, class V>
  FLARE_FORCEINLINE_FUNCTION pair<first_type, second_type>& operator=(
      const pair<U, V>& p) {
    first  = p.first;
    second = p.second;
    return *this;
  }

  /// \brief Return the std::pair version of this object.
  ///
  /// This is <i>not</i> a device function; you may not call it on a
  /// CUDA device.  It is meant to be called on the host, if the user
  /// wants an std::pair instead of a flare::pair.
  ///
  /// \note This is not a conversion operator, since defining a
  ///   conversion operator made the relational operators have
  ///   ambiguous definitions.
  std::pair<T1, T2> to_std_pair() const {
    return std::make_pair(first, second);
  }
};

//! Equality operator for flare::pair.
template <class T1, class T2>
FLARE_FORCEINLINE_FUNCTION constexpr bool operator==(const pair<T1, T2>& lhs,
                                                      const pair<T1, T2>& rhs) {
  return lhs.first == rhs.first && lhs.second == rhs.second;
}

//! Inequality operator for flare::pair.
template <class T1, class T2>
FLARE_FORCEINLINE_FUNCTION constexpr bool operator!=(const pair<T1, T2>& lhs,
                                                      const pair<T1, T2>& rhs) {
  return !(lhs == rhs);
}

//! Less-than operator for flare::pair.
template <class T1, class T2>
FLARE_FORCEINLINE_FUNCTION constexpr bool operator<(const pair<T1, T2>& lhs,
                                                     const pair<T1, T2>& rhs) {
  return lhs.first < rhs.first ||
         (!(rhs.first < lhs.first) && lhs.second < rhs.second);
}

//! Less-than-or-equal-to operator for flare::pair.
template <class T1, class T2>
FLARE_FORCEINLINE_FUNCTION constexpr bool operator<=(const pair<T1, T2>& lhs,
                                                      const pair<T1, T2>& rhs) {
  return !(rhs < lhs);
}

//! Greater-than operator for flare::pair.
template <class T1, class T2>
FLARE_FORCEINLINE_FUNCTION constexpr bool operator>(const pair<T1, T2>& lhs,
                                                     const pair<T1, T2>& rhs) {
  return rhs < lhs;
}

//! Greater-than-or-equal-to operator for flare::pair.
template <class T1, class T2>
FLARE_FORCEINLINE_FUNCTION constexpr bool operator>=(const pair<T1, T2>& lhs,
                                                      const pair<T1, T2>& rhs) {
  return !(lhs < rhs);
}

/// \brief Return a new pair.
///
/// This is a "nonmember constructor" for flare::pair.  It works just
/// like std::make_pair.
template <class T1, class T2>
FLARE_FORCEINLINE_FUNCTION constexpr pair<T1, T2> make_pair(T1 x, T2 y) {
  return (pair<T1, T2>(x, y));
}

/// \brief Return a pair of references to the input arguments.
///
/// This compares to std::tie (new in C++11).  You can use it to
/// assign to two variables at once, from the result of a function
/// that returns a pair.  For example (<tt>__device__</tt> and
/// <tt>__host__</tt> attributes omitted for brevity):
/// \code
/// // Declaration of the function to call.
/// // First return value: operation count.
/// // Second return value: whether all operations succeeded.
/// flare::pair<int, bool> someFunction ();
///
/// // Code that uses flare::tie.
/// int myFunction () {
///   int count = 0;
///   bool success = false;
///
///   // This assigns to both count and success.
///   flare::tie (count, success) = someFunction ();
///
///   if (! success) {
///     // ... Some operation failed;
///     //     take corrective action ...
///   }
///   return count;
/// }
/// \endcode
///
/// The line that uses tie() could have been written like this:
/// \code
///   flare::pair<int, bool> result = someFunction ();
///   count = result.first;
///   success = result.second;
/// \endcode
///
/// Using tie() saves two lines of code and avoids a copy of each
/// element of the pair.  The latter could be significant if one or
/// both elements of the pair are more substantial objects than \c int
/// or \c bool.
template <class T1, class T2>
FLARE_FORCEINLINE_FUNCTION pair<T1&, T2&> tie(T1& x, T2& y) {
  return (pair<T1&, T2&>(x, y));
}

//
// Specialization of flare::pair for a \c void second argument.  This
// is not actually a "pair"; it only contains one element, the first.
//
template <class T1>
struct pair<T1, void> {
  using first_type  = T1;
  using second_type = void;

  first_type first;
  enum { second = 0 };

  FLARE_DEFAULTED_FUNCTION constexpr pair() = default;

  FLARE_FORCEINLINE_FUNCTION constexpr pair(const first_type& f) : first(f) {}

  FLARE_FORCEINLINE_FUNCTION constexpr pair(const first_type& f, int)
      : first(f) {}

  template <class U>
  FLARE_FORCEINLINE_FUNCTION constexpr pair(const pair<U, void>& p)
      : first(p.first) {}

  template <class U>
  FLARE_FORCEINLINE_FUNCTION pair<T1, void>& operator=(
      const pair<U, void>& p) {
    first = p.first;
    return *this;
  }
};

//
// Specialization of relational operators for flare::pair<T1,void>.
//

template <class T1>
FLARE_FORCEINLINE_FUNCTION constexpr bool operator==(
    const pair<T1, void>& lhs, const pair<T1, void>& rhs) {
  return lhs.first == rhs.first;
}

template <class T1>
FLARE_FORCEINLINE_FUNCTION constexpr bool operator!=(
    const pair<T1, void>& lhs, const pair<T1, void>& rhs) {
  return !(lhs == rhs);
}

template <class T1>
FLARE_FORCEINLINE_FUNCTION constexpr bool operator<(
    const pair<T1, void>& lhs, const pair<T1, void>& rhs) {
  return lhs.first < rhs.first;
}

template <class T1>
FLARE_FORCEINLINE_FUNCTION constexpr bool operator<=(
    const pair<T1, void>& lhs, const pair<T1, void>& rhs) {
  return !(rhs < lhs);
}

template <class T1>
FLARE_FORCEINLINE_FUNCTION constexpr bool operator>(
    const pair<T1, void>& lhs, const pair<T1, void>& rhs) {
  return rhs < lhs;
}

template <class T1>
FLARE_FORCEINLINE_FUNCTION constexpr bool operator>=(
    const pair<T1, void>& lhs, const pair<T1, void>& rhs) {
  return !(lhs < rhs);
}

namespace detail {

template <class T>
struct is_pair_like : std::false_type {};
template <class T, class U>
struct is_pair_like<flare::pair<T, U>> : std::true_type {};
template <class T, class U>
struct is_pair_like<std::pair<T, U>> : std::true_type {};

}  // end namespace detail

}  // namespace flare

#endif  // FLARE_CORE_PAIR_H_
