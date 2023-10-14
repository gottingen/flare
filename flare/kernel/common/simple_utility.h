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


#ifndef FLARE_KERNEL_COMMON_SIMPLE_UTILITY_H_
#define FLARE_KERNEL_COMMON_SIMPLE_UTILITY_H_

#include <flare/core.h>
#include <flare/core/numeric_traits.h>
#include <flare/core/arith_traits.h>

#define FLARE_MACRO_MIN(x, y) ((x) < (y) ? (x) : (y))
#define FLARE_MACRO_MAX(x, y) ((x) < (y) ? (y) : (x))
#define FLARE_MACRO_ABS(x) \
  flare::ArithTraits<typename std::decay<decltype(x)>::type>::abs(x)

namespace flare::detail {

    template<class TensorType>
    class SquareRootFunctor {
    public:
        typedef typename TensorType::execution_space execution_space;
        typedef typename TensorType::size_type size_type;

        SquareRootFunctor(const TensorType &theTensor) : theTensor_(theTensor) {}

        FLARE_INLINE_FUNCTION void operator()(const size_type i) const {
            typedef typename TensorType::value_type value_type;
            theTensor_(i) = flare::ArithTraits<value_type>::sqrt(theTensor_(i));
        }

    private:
        TensorType theTensor_;
    };

    template<typename tensor_t>
    struct ExclusiveParallelPrefixSum {
        typedef typename tensor_t::value_type value_type;
        tensor_t array_sum;

        ExclusiveParallelPrefixSum(tensor_t arr_) : array_sum(arr_) {}

        FLARE_INLINE_FUNCTION
        void operator()(const size_t ii, value_type &update, const bool final) const {
            value_type val =
                    (ii == array_sum.extent(0) - 1) ? value_type(0) : array_sum(ii);
            if (final) {
                array_sum(ii) = value_type(update);
            }
            update += val;
        }
    };

    template<typename array_type>
    struct InclusiveParallelPrefixSum {
        typedef typename array_type::value_type idx;
        array_type array_sum;

        InclusiveParallelPrefixSum(array_type arr_) : array_sum(arr_) {}

        FLARE_INLINE_FUNCTION
        void operator()(const size_t ii, size_t &update, const bool final) const {
            update += array_sum(ii);
            if (final) {
                array_sum(ii) = idx(update);
            }
        }
    };

    /***
     * \brief Function performs the exclusive parallel prefix sum. That is each
     * entry holds the sum until itself.
     * \param exec: the execution space instance on which to run
     * \param num_elements: size of the array
     * \param arr: the array for which the prefix sum will be performed.
     */
    template<typename MyExecSpace, typename tensor_t>
    inline void flare_exclusive_parallel_prefix_sum(
            const MyExecSpace &exec, typename tensor_t::value_type num_elements,
            tensor_t arr) {
        typedef flare::RangePolicy<MyExecSpace> my_exec_space;
        flare::parallel_scan("flare::Common::PrefixSum",
                             my_exec_space(exec, 0, num_elements),
                             ExclusiveParallelPrefixSum<tensor_t>(arr));
    }

    /***
     * \brief Function performs the exclusive parallel prefix sum. That is each
     * entry holds the sum until itself.
     * \param num_elements: size of the array
     * \param arr: the array for which the prefix sum will be performed.
     */
    template<typename MyExecSpace, typename tensor_t>
    inline void flare_exclusive_parallel_prefix_sum(
            typename tensor_t::value_type num_elements, tensor_t arr) {
        flare_exclusive_parallel_prefix_sum(MyExecSpace(), num_elements, arr);
    }

    /***
     * \brief Function performs the exclusive parallel prefix sum. That is each
     * entry holds the sum until itself. This version also returns the final sum
     * equivalent to the sum-reduction of arr before doing the scan.
     * \param exec: the execution space instance on which to run
     * \param num_elements: size of the array
     * \param arr: the array for which the prefix sum will be performed.
     * \param finalSum: will be set to arr[num_elements - 1] after computing the
     * prefix sum.
     */
    template<typename MyExecSpace, typename tensor_t>
    inline void flare_exclusive_parallel_prefix_sum(
            const MyExecSpace &exec, typename tensor_t::value_type num_elements,
            tensor_t arr, typename tensor_t::non_const_value_type &finalSum) {
        typedef flare::RangePolicy<MyExecSpace> my_exec_space;
        flare::parallel_scan("flare::Common::PrefixSum",
                             my_exec_space(exec, 0, num_elements),
                             ExclusiveParallelPrefixSum<tensor_t>(arr), finalSum);
    }

    /***
     * \brief Function performs the exclusive parallel prefix sum. That is each
     * entry holds the sum until itself. This version also returns the final sum
     * equivalent to the sum-reduction of arr before doing the scan.
     * \param num_elements: size of the array
     * \param arr: the array for which the prefix sum will be performed.
     * \param finalSum: will be set to arr[num_elements - 1] after computing the
     * prefix sum.
     */
    template<typename MyExecSpace, typename tensor_t>
    inline void flare_exclusive_parallel_prefix_sum(
            typename tensor_t::value_type num_elements, tensor_t arr,
            typename tensor_t::non_const_value_type &finalSum) {
        flare_exclusive_parallel_prefix_sum(MyExecSpace(), num_elements, arr, finalSum);
    }

    ///
    /// \brief Function performs the inclusive parallel prefix sum. That is each
    ///        entry holds the sum until itself including itself.
    /// \param my_exec_space: The execution space instance
    /// \param num_elements: size of the array
    /// \param arr: the array for which the prefix sum will be performed.
    ///
    template<typename MyExecSpace, typename forward_array_type>
    void flare_inclusive_parallel_prefix_sum(
            MyExecSpace my_exec_space,
            typename forward_array_type::value_type num_elements,
            forward_array_type arr) {
        typedef flare::RangePolicy<MyExecSpace> range_policy_t;
        flare::parallel_scan("flare::Common::PrefixSum",
                             range_policy_t(my_exec_space, 0, num_elements),
                             InclusiveParallelPrefixSum<forward_array_type>(arr));
    }

    ///
    /// \brief Function performs the inclusive parallel prefix sum. That is each
    ///        entry holds the sum until itself including itself.
    /// \param num_elements: size of the array
    /// \param arr: the array for which the prefix sum will be performed.
    ///
    template<typename MyExecSpace, typename forward_array_type>
    void flare_inclusive_parallel_prefix_sum(
            typename forward_array_type::value_type num_elements,
            forward_array_type arr) {
        MyExecSpace my_exec_space;
        return flare_inclusive_parallel_prefix_sum(my_exec_space, num_elements, arr);
    }

    template<typename tensor_t>
    struct ReductionFunctor {
        tensor_t array_sum;

        ReductionFunctor(tensor_t arr_) : array_sum(arr_) {}

        FLARE_INLINE_FUNCTION
        void operator()(const size_t ii, typename tensor_t::value_type &update) const {
            update += array_sum(ii);
        }
    };

    template<typename tensor_t>
    struct ReductionFunctor2 {
        tensor_t array_sum;

        ReductionFunctor2(tensor_t arr_) : array_sum(arr_) {}

        FLARE_INLINE_FUNCTION
        void operator()(const size_t ii, size_t &update) const {
            update += array_sum(ii);
        }
    };

    template<typename tensor_t, typename tensor2_t>
    struct DiffReductionFunctor {
        tensor_t array_begins;
        tensor2_t array_ends;

        DiffReductionFunctor(tensor_t begins, tensor2_t ends)
                : array_begins(begins), array_ends(ends) {}

        FLARE_INLINE_FUNCTION
        void operator()(const size_t ii,
                        typename tensor_t::non_const_value_type &update) const {
            update += (array_ends(ii) - array_begins(ii));
        }
    };

    template<typename tensor_t, typename tensor2_t, typename MyExecSpace>
    inline void flare_reduce_diff_tensor(
            size_t num_elements, tensor_t smaller, tensor2_t bigger,
            typename tensor_t::non_const_value_type &reduction) {
        typedef flare::RangePolicy<MyExecSpace> my_exec_space;
        flare::parallel_reduce(
                "flare::Common::ReduceDiffTensor", my_exec_space(0, num_elements),
                DiffReductionFunctor<tensor_t, tensor2_t>(smaller, bigger), reduction);
    }

    template<typename it>
    struct DiffReductionFunctorP {
        const it *array_begins;
        const it *array_ends;

        DiffReductionFunctorP(const it *begins, const it *ends)
                : array_begins(begins), array_ends(ends) {}

        FLARE_INLINE_FUNCTION
        void operator()(const size_t ii, it &update) const {
            update += (array_ends[ii] - array_begins[ii]);
        }
    };

    template<typename it, typename MyExecSpace>
    inline void kkp_reduce_diff_tensor(const size_t num_elements, const it *smaller,
                                     const it *bigger, it &reduction) {
        typedef flare::RangePolicy<MyExecSpace> my_exec_space;
        flare::parallel_reduce(
                "flare::Common::ReduceDiffTensor", my_exec_space(0, num_elements),
                DiffReductionFunctorP<it>(smaller, bigger), reduction);
    }

    /***
     * \brief Function performs the a reduction
     * until itself.
     * \param num_elements: size of the array
     * \param arr: the array for which the prefix sum will be performed.
     */
    template<typename tensor_t, typename MyExecSpace>
    inline void flare_reduce_tensor(size_t num_elements, tensor_t arr,
                               typename tensor_t::value_type &reduction) {
        typedef flare::RangePolicy<MyExecSpace> my_exec_space;
        flare::parallel_reduce("flare::Common::ReduceTensor",
                               my_exec_space(0, num_elements),
                               ReductionFunctor<tensor_t>(arr), reduction);
    }

    template<typename tensor_t, typename MyExecSpace>
    inline void flare_reduce_tensor2(size_t num_elements, tensor_t arr,
                                size_t &reduction) {
        typedef flare::RangePolicy<MyExecSpace> my_exec_space;
        flare::parallel_reduce("flare::Common::ReduceTensor2",
                               my_exec_space(0, num_elements),
                               ReductionFunctor2<tensor_t>(arr), reduction);
    }

    template<typename tensor_type1, typename tensor_type2,
            typename eps_type = typename flare::ArithTraits<
                    typename tensor_type2::non_const_value_type>::mag_type>
    struct IsIdenticalFunctor {
        tensor_type1 tensor1;
        tensor_type2 tensor2;
        eps_type eps;

        IsIdenticalFunctor(tensor_type1 tensor1_, tensor_type2 tensor2_, eps_type eps_)
                : tensor1(tensor1_), tensor2(tensor2_), eps(eps_) {}

        FLARE_INLINE_FUNCTION
        void operator()(const size_t &i, size_t &is_equal) const {
            typedef typename tensor_type2::non_const_value_type val_type;
            typedef flare::ArithTraits<val_type> KAT;
            typedef typename KAT::mag_type mag_type;
            const mag_type val_diff = KAT::abs(tensor1(i) - tensor2(i));

            if (val_diff > eps) {
                is_equal += 1;
            }
        }
    };

    template<typename tensor_type1, typename tensor_type2, typename eps_type,
            typename MyExecSpace>
    bool flare_is_identical_tensor(tensor_type1 tensor1, tensor_type2 tensor2, eps_type eps) {
        if (tensor1.extent(0) != tensor2.extent(0)) {
            return false;
        }

        size_t num_elements = tensor1.extent(0);

        typedef flare::RangePolicy<MyExecSpace> my_exec_space;
        size_t issame = 0;
        flare::parallel_reduce(
                "flare::Common::IsIdenticalTensor", my_exec_space(0, num_elements),
                IsIdenticalFunctor<tensor_type1, tensor_type2, eps_type>(tensor1, tensor2, eps),
                issame);
        MyExecSpace().fence();
        if (issame > 0) {
            return false;
        } else {
            return true;
        }
    }

    template<typename tensor_type1, typename tensor_type2,
            typename eps_type = typename flare::ArithTraits<
                    typename tensor_type2::non_const_value_type>::mag_type>
    struct IsRelativelyIdenticalFunctor {
        tensor_type1 tensor1;
        tensor_type2 tensor2;
        eps_type eps;

        IsRelativelyIdenticalFunctor(tensor_type1 tensor1_, tensor_type2 tensor2_,
                                     eps_type eps_)
                : tensor1(tensor1_), tensor2(tensor2_), eps(eps_) {}

        FLARE_INLINE_FUNCTION
        void operator()(const size_t &i, size_t &num_diffs) const {
            typedef typename tensor_type2::non_const_value_type val_type;
            typedef flare::ArithTraits<val_type> KAT;
            typedef typename KAT::mag_type mag_type;
            typedef flare::ArithTraits<mag_type> KATM;

            mag_type val_diff = KATM::zero();
            if (KAT::abs(tensor1(i)) > mag_type(eps) ||
                KAT::abs(tensor2(i)) > mag_type(eps)) {
                val_diff = KAT::abs(tensor1(i) - tensor2(i)) /
                           (KAT::abs(tensor1(i)) + KAT::abs(tensor2(i)));
            }

            if (val_diff > mag_type(eps)) {
#if defined(FLARE_DISABLE_PRINTF)
                FLARE_IMPL_DO_NOT_USE_PRINTF(
                        "Values at index %d, %.6f + %.6fi and %.6f + %.6fi, differ too much "
                        "(eps = %e)\n",
                        (int) i, KAT::real(tensor1(i)), KAT::imag(tensor1(i)), KAT::real(tensor2(i)),
                        KAT::imag(tensor2(i)), eps);
#else
                flare::printf(
      "Values at index %d, %.6f + %.6fi and %.6f + %.6fi, differ too much "
      "(eps = %e)\n",
      (int)i, KAT::real(tensor1(i)), KAT::imag(tensor1(i)), KAT::real(tensor2(i)),
      KAT::imag(tensor2(i)), eps);
#endif
                num_diffs++;
            }
        }
    };

    template<typename tensor_type1, typename tensor_type2, typename eps_type,
            typename MyExecSpace>
    bool flare_is_relatively_identical_tensor(tensor_type1 tensor1, tensor_type2 tensor2,
                                         eps_type eps) {
        if (tensor1.extent(0) != tensor2.extent(0)) {
            return false;
        }

        size_t num_elements = tensor1.extent(0);

        typedef flare::RangePolicy<MyExecSpace> my_exec_space;
        size_t numDifferences = 0;
        flare::parallel_reduce(
                "flare::Common::IsRelativelyIdenticalTensor",
                my_exec_space(0, num_elements),
                IsRelativelyIdenticalFunctor<tensor_type1, tensor_type2, eps_type>(
                        tensor1, tensor2, eps),
                numDifferences);
        return numDifferences == 0;
    }

    template<typename tensor_type>
    struct ReduceMaxFunctor {
        tensor_type tensor_to_reduce;
        typedef typename tensor_type::non_const_value_type value_type;
        const value_type min_val;

        ReduceMaxFunctor(tensor_type tensor_to_reduce_)
                : tensor_to_reduce(tensor_to_reduce_),
                  min_val((std::numeric_limits<value_type>::lowest())) {}

        FLARE_INLINE_FUNCTION
        void operator()(const size_t &i, value_type &max_reduction) const {
            value_type val = tensor_to_reduce(i);
            if (max_reduction < val) {
                max_reduction = val;
            }
        }

        FLARE_INLINE_FUNCTION
        void join(value_type &dst, const value_type &src) const {
            if (dst < src) {
                dst = src;
            }
        }

        FLARE_INLINE_FUNCTION
        void init(value_type &dst) const {
            // The identity under max is -Inf.
            // flare does not come with a portable way to access
            // floating -point Inf and NaN. flare does , however;
            // see flare :: ArithTraits in the Tpetra package.
            dst = min_val;
        }
    };

    template<typename tensor_type, typename MyExecSpace>
    void flare_tensor_reduce_max(
            size_t num_elements, tensor_type tensor_to_reduce,
            typename tensor_type::non_const_value_type &max_reduction) {
        typedef flare::RangePolicy<MyExecSpace> my_exec_space;
        flare::parallel_reduce(
                "flare::Common::ReduceMax", my_exec_space(0, num_elements),
                ReduceMaxFunctor<tensor_type>(tensor_to_reduce), max_reduction);
    }

    // xorshift hash/pseudorandom function (supported for 32- and 64-bit integer
    // types only)
    template<typename Value>
    FLARE_FORCEINLINE_FUNCTION Value xorshiftHash(Value v) {
        static_assert(std::is_unsigned<Value>::value,
                      "xorshiftHash: value must be an unsigned integer type");
        uint64_t x = v;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        return std::is_same<Value, uint32_t>::value
               ? static_cast<Value>((x * 2685821657736338717ULL - 1) >> 16)
               : static_cast<Value>(x * 2685821657736338717ULL - 1);
    }

    struct TensorHashFunctor {
        TensorHashFunctor(const uint8_t *data_) : data(data_) {}

        FLARE_INLINE_FUNCTION void operator()(size_t i, uint32_t &lhash) const {
            // Compute a hash/digest of both the index i, and data[i]. Then add that to
            // overall hash.
            uint32_t x = uint32_t(i);
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x ^= uint32_t(data[i]);
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            lhash += x;
        }

        const uint8_t *data;
    };

    /// \brief Compute a hash of a tensor.
    /// \param v: the tensor to hash. Must be contiguous, and its element type must
    /// not contain any padding bytes.
    template<typename Tensor>
    uint32_t hashTensor(const Tensor &v) {
        assert(v.span_is_contiguous());
        // Note: This type trait is supposed to be part of C++17,
        // but it's not defined on Intel 19 (with GCC 7.2.0 standard library).
        // So just check if it's available before using.
#ifdef __cpp_lib_has_unique_object_representations
        static_assert(std::has_unique_object_representations<
                              typename Tensor::non_const_value_type>::value,
                      "flare::Impl::hashTensor: the tensor's element type must "
                      "not have any padding bytes.");
#endif
        size_t nbytes = v.span() * sizeof(typename Tensor::value_type);
        uint32_t h;
        flare::parallel_reduce(
                flare::RangePolicy<typename Tensor::execution_space, size_t>(0, nbytes),
                TensorHashFunctor(reinterpret_cast<const uint8_t *>(v.data())), h);
        return h;
    }

    template<typename V>
    struct SequentialFillFunctor {
        using size_type = typename V::size_type;
        using val_type = typename V::non_const_value_type;

        SequentialFillFunctor(const V &v_, val_type start_) : v(v_), start(start_) {}

        FLARE_INLINE_FUNCTION void operator()(size_type i) const {
            v(i) = start + (val_type) i;
        }

        V v;
        val_type start;
    };

    template<typename V>
    void sequential_fill(const V &v, typename V::non_const_value_type start = 0) {
        flare::parallel_for(
                flare::RangePolicy<typename V::execution_space>(0, v.extent(0)),
                SequentialFillFunctor<V>(v, start));
    }

}  // namespace flare::detail


#endif  // FLARE_KERNEL_COMMON_SIMPLE_UTILITY_H_
