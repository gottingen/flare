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
#include <fly/defines.h>

/**
    \struct fly_seq

    \brief C-style struct to creating sequences for indexing

    \ingroup index_mat
*/
typedef struct fly_seq {
    /// Start position of the sequence
    double begin;

    /// End position of the sequence (inclusive)
    double end;

    /// Step size between sequence values
    double step;
} fly_seq;

static const fly_seq fly_span = {1, 1, 0};

#ifdef __cplusplus
namespace fly
{
class array;

/**
    \class seq

    \brief seq is used to create sequences for indexing fly::array

    \ingroup flare_class
*/
class FLY_API seq
{
public:
    ///
    /// \brief Get the \ref fly_seq C-style struct
    ///
    fly_seq s;

    ///
    /// \brief Get's the length of the sequence
    ///
    size_t size;

    ///
    /// \brief Flag for gfor
    ///
    bool m_gfor;

    /**
        \brief Creates a sequence of size length as [0, 1, 2..., length - 1]

        The sequence has begin as 0, end as length - 1 and step as 1.

        \note When doing seq(-n), where n is > 0, then the sequence is generated as
        0...-n but step remains +1. This is because when such a sequence is
        used for indexing fly::array, then -n represents n elements from the
        end. That is, seq(-2) will imply indexing an array 0...dimSize - 2.

        \code
                            // [begin, end, step]
        seq a(10);          // [0, 9, 1]    => 0, 1, 2....9
        \endcode

        \param[in] length is the size of the seq to be created.
    */
    seq(double length = 0);

    /**
        \brief Destructor
    */
    ~seq();

    /**
        \brief Creates a sequence starting at begin,
        ending at or before end (inclusive) with increments as step.

        The sequence will be [begin, begin + step, begin + 2 * step...., begin + n * step]
        where the begin + n * step <= end.

        \code
                            // [begin, end, step]
        seq a(10, 20);      // [10, 20, 1]  => 10, 11, 12....20
        seq b(10, 20, 2);   // [10, 20, 2]  => 10, 12, 14....20
        seq c(-5, 5);       // [-5, 5, 1]   => -5, -4, -3....0, 1....5
        seq d(-5, -15, -1); // [-5,-15, -1] => -5, -6, -7....-15
        seq e(-15, -5, 1);  // [-15, -5, 1] => -15, -14, -13....-5
        \endcode

        \param[in] begin is the start of the sequence
        \param[in] end is the maximum value a sequence can take (inclusive)
        \param[in] step is the increment or decrement size (default is 1)
    */
    seq(double begin, double end, double step = 1);

    /**
        \brief Copy constructor

        Creates a copy seq from another sequence.

        \param[in] other seqence to be copies
        \param[in] is_gfor is the gfor flag
    */
    seq(seq other, bool is_gfor);

    /**
        \brief Create a seq object from an \ref fly_seq struct

        \param[in] s_ is the \ref fly_seq struct
    */
    seq(const fly_seq& s_);

    /**
        \brief Assignment operator to create a new sequence from an fly_seq

        This operator creates a new sequence using the begin, end and step
        from the input sequence.

        \param[in] s is the input sequence
    */
    seq& operator=(const fly_seq& s);

    /**
        \brief Negation operator creates a sequence with the signs negated

        begin is changed to -begin
        end is changed to -end
        step is changed to -step

        \code
                        // [begin, end, step]
        seq a(1, 10);   // [ 1, 10, 1] => 1, 2, 3....10
        seq b = -a;     // [-1,-10,-1] => -1, -2, -3...-10
        \endcode
    */
    inline seq operator-()         { return seq(-s.begin, -s.end,  -s.step); }

    /**
        \brief Addition operator offsets the begin and end by x. There is no
        change in step.

        begin is changed to begin + x
        end is changed to end + x

        \code
                            // [begin, end, step]
        seq a(2, 20, 2);    // [2, 20, 2] => 2, 4, 6....20
        seq b = a + 3;      // [5, 23, 2] => 5, 7, 9....23
        \endcode
    */
    inline seq operator+(double x) { return seq(s.begin + x, s.end + x, s.step); }

    /**
        \brief Subtraction operator offsets the begin and end by x. There is no
        change in step.

        begin is changed to begin - x
        end is changed to end - x

        \code
                            // [begin, end, step]
        seq a(10, 20, 2);   // [10, 20, 2] => 10, 12, 14....20
        seq b(2, 10);       // [ 2, 10, 1] => 2, 3, 4....10
        seq c = a - 3;      // [ 7, 17, 2] => 7, 9, 11....17
        seq d = b - 3;      // [-1,  7, 2] => -1, 1, 3....7
        \endcode
    */
    inline seq operator-(double x) { return seq(s.begin - x, s.end - x, s.step); }

    /**
        \brief Multiplication operator spaces the sequence by a factor x.

        begin is changed to begin * x
        end is changed to end * x
        step is changed to step * x

        \code
                            // [begin, end, step]
        seq a(10, 20, 2);   // [10, 20, 2] => 10, 12, 14....20
        seq b(-5, 5);       // [-5, 5, 1] => -5, -4, -3....0, 1....5
        seq c = a * 3;      // [30, 60, 6] => 30, 36, 42....60
        seq d = b * 3;      // [-15, 15, 3] => -15, -12, -9....0, 3....15
        seq e = a * 0.5;    // [5, 10, 1] => 5, 6, 7....10
        \endcode
    */
    inline seq operator*(double x) { return seq(s.begin * x, s.end * x, s.step * x); }

    friend inline seq operator+(double x, seq y) { return  y + x; }

    friend inline seq operator-(double x, seq y) { return -y + x; }

    friend inline seq operator*(double x, seq y) { return  y * x; }

    /**
        \brief Implicit conversion operator from seq to fly::array

        Convertes a seq object into an fly::array object. The contents of the
        fly::array will be the explicit values from the seq.

        \note Do not use this to create arrays of sequences. Use \ref range.

        \code
                            // [begin, end, step]
        seq s(10, 20, 2);   // [10, 20, 2] => 10, 12, 14....20
        array arr = s;
        fly_print(arr);      // 10    12    14    16    18    20
        \endcode
    */
    operator array() const;

    private:
    void init(double begin, double end, double step);
};

/// A special value representing the last value of an axis
extern FLY_API int end;

/// A special value representing the entire axis of an fly::array
extern FLY_API seq span;

}
#endif

#ifdef __cplusplus
extern "C" {
#endif

/// Create a new fly_seq object.
FLY_API fly_seq fly_make_seq(double begin, double end, double step);

#ifdef __cplusplus
}
#endif
