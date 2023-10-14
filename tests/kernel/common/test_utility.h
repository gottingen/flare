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


#ifndef FLARE_TEST_UTILITY_H
#define FLARE_TEST_UTILITY_H

#include <flare/core.h>
#include <flare/random.h>
#include <doctest.h>
#include <random>
#include <flare/core/arith_traits.h>
#include <flare/core/exec_space_utils.h>
#include <flare/kernel/common/io_utility.h>
#include <flare/kernel/common/utility.h>

#define FLARE_TEST_ALL_TYPES

#if defined(FLARE_INST_LAYOUTLEFT) || \
    defined(FLARE_TEST_ALL_TYPES)
#define FLARE_TEST_LAYOUTLEFT
#endif
#if defined(FLARE_INST_LAYOUTRIGHT) || \
    defined(FLARE_TEST_ALL_TYPES)
#define FLARE_TEST_LAYOUTRIGHT
#endif
#if defined(FLARE_INST_LAYOUTSTRIDE) || \
    defined(FLARE_TEST_ALL_TYPES)
#define FLARE_TEST_LAYOUTSTRIDE
#endif
#if defined(FLARE_INST_FLOAT) || defined(FLARE_TEST_ALL_TYPES)
#define FLARE_TEST_FLOAT
#endif
#if defined(FLARE_INST_DOUBLE) || defined(FLARE_TEST_ALL_TYPES)
#define FLARE_TEST_DOUBLE
#endif
#if defined(FLARE_INST_INT) || defined(FLARE_TEST_ALL_TYPES)
#define FLARE_TEST_INT
#endif
#if defined(FLARE_INST_COMPLEX_FLOAT) || \
    defined(FLARE_TEST_ALL_TYPES)
#define FLARE_TEST_COMPLEX_FLOAT
#endif
#if defined(FLARE_INST_COMPLEX_DOUBLE) || \
    defined(FLARE_TEST_ALL_TYPES)
#define FLARE_TEST_COMPLEX_DOUBLE
#endif

namespace Test {

    // Utility class for testing kernels with rank-1 and rank-2 tensors that may be
    // LayoutStride. Simplifies making a LayoutStride tensor of a given size that is
    // actually noncontiguous, and host-device transfers for checking results on
    // host.
    //
    // Constructed with label and extent(s), and then provides 5 tensors as members:
    //  - d_tensor, and a const-valued alias d_tensor_const
    //  - h_tensor
    //  - d_base
    //  - h_base
    // d_tensor is of type TensorType, and has the extents passed to the constructor.
    // h_tensor is a mirror of d_tensor.
    // d_base (and its mirror h_base) are contiguous tensors, so they can be
    // deep-copied to each other. d_tensor aliases d_base, and h_tensor aliases h_base.
    // This means that copying between d_base and h_base
    //    also copies between d_tensor and h_tensor.
    //
    // If the Boolean template parameter 'createMirrorTensor' is:
    // - 'true' (default value), then this utility class will use
    //   flare::create_mirror_tensor();
    // - 'false', then this utility class will use flare::create_mirror()
    template <class TensorType, bool createMirrorTensor = true>
    struct tensor_stride_adapter {
        static_assert(flare::is_tensor_v<TensorType>,
                      "tensor_stride_adapter: TensorType must be a flare::Tensor");
        static_assert(TensorType::rank >= 1 && TensorType::rank <= 2,
                      "tensor_stride_adapter: TensorType must be rank 1 or rank 2");

        static constexpr bool strided = std::is_same<typename TensorType::array_layout,
                flare::LayoutStride>::value;
        static constexpr int rank     = TensorType::rank;

        using DTensor = TensorType;
        using HTensor = typename DTensor::HostMirror;
        // If not strided, the base tensor types are the same as DTensor/HTensor.
        // But if strided, the base tensors have one additional dimension, so that
        // d_tensor/h_tensor have stride > 1 between consecutive elements.
        using DTensorBase = std::conditional_t<
        strided,
        flare::Tensor<typename TensorType::data_type*, flare::LayoutRight,
                typename TensorType::device_type>,
        DTensor>;
        using HTensorBase = typename DTensorBase::HostMirror;

        tensor_stride_adapter(const std::string& label, int m, int n = 1) {
            if constexpr (rank == 1) {
                if constexpr (strided) {
                    d_base = DTensorBase(label, m, 2);
                    h_base = createMirrorTensor ? flare::create_mirror_tensor(d_base)
                                              : flare::create_mirror(d_base);
                    d_tensor = flare::subtensor(d_base, flare::ALL(), 0);
                    h_tensor = flare::subtensor(h_base, flare::ALL(), 0);
                } else {
                    d_base = DTensorBase(label, m);
                    h_base = createMirrorTensor ? flare::create_mirror_tensor(d_base)
                                              : flare::create_mirror(d_base);
                    d_tensor = d_base;
                    h_tensor = h_base;
                }
            } else {
                if constexpr (strided) {
                    d_base = DTensorBase(label, m, n, 2);
                    h_base = createMirrorTensor ? flare::create_mirror_tensor(d_base)
                                              : flare::create_mirror(d_base);
                    d_tensor =
                            flare::subtensor(d_base, flare::ALL(), flare::make_pair(0, n), 0);
                    h_tensor =
                            flare::subtensor(h_base, flare::ALL(), flare::make_pair(0, n), 0);
                } else {
                    d_base = DTensorBase(label, m, n);
                    h_base = createMirrorTensor ? flare::create_mirror_tensor(d_base)
                                              : flare::create_mirror(d_base);
                    d_tensor = d_base;
                    h_tensor = h_base;
                }
            }
            d_tensor_const = d_tensor;
        }

        // Have both const and nonconst versions of d_tensor (with same underlying
        // data), since we often test BLAS with both
        DTensor d_tensor;
        typename DTensor::const_type d_tensor_const;
        HTensor h_tensor;
        DTensorBase d_base;
        HTensorBase h_base;
    };

    template <class Scalar1, class Scalar2, class Scalar3>
    void EXPECT_NEAR_KK(Scalar1 val1, Scalar2 val2, Scalar3 tol,
                        std::string msg = "") {
        typedef flare::ArithTraits<Scalar1> AT1;
        typedef flare::ArithTraits<Scalar3> AT3;
        REQUIRE_LE((double)AT1::abs(val1 - val2), (double)AT3::abs(tol));
    }

    template <class Scalar1, class Scalar2, class Scalar3>
    void EXPECT_NEAR_KK_REL(Scalar1 val1, Scalar2 val2, Scalar3 tol,
                            std::string msg = "") {
        typedef typename std::remove_reference<decltype(val1)>::type hv1_type;
        typedef typename std::remove_reference<decltype(val2)>::type hv2_type;
        const auto ahv1 = flare::ArithTraits<hv1_type>::abs(val1);
        const auto ahv2 = flare::ArithTraits<hv2_type>::abs(val2);
        EXPECT_NEAR_KK(val1, val2, tol * flare::max(ahv1, ahv2), msg);
    }

    template <class TensorType1, class TensorType2, class Scalar>
    void EXPECT_NEAR_KK_1DTensor(TensorType1 v1, TensorType2 v2, Scalar tol) {
        size_t v1_size = v1.extent(0);
        size_t v2_size = v2.extent(0);
        REQUIRE_EQ(v1_size, v2_size);

        typename TensorType1::HostMirror h_v1 = flare::create_mirror_tensor(v1);
        typename TensorType2::HostMirror h_v2 = flare::create_mirror_tensor(v2);

        flare::detail::safe_device_to_host_deep_copy(v1.extent(0), v1, h_v1);
        flare::detail::safe_device_to_host_deep_copy(v2.extent(0), v2, h_v2);

        for (size_t i = 0; i < v1_size; ++i) {
            EXPECT_NEAR_KK(h_v1(i), h_v2(i), tol);
        }
    }

    template <class TensorType1, class TensorType2, class Scalar>
    void EXPECT_NEAR_KK_REL_1DTensor(TensorType1 v1, TensorType2 v2, Scalar tol) {
        size_t v1_size = v1.extent(0);
        size_t v2_size = v2.extent(0);
        REQUIRE_EQ(v1_size, v2_size);

        typename TensorType1::HostMirror h_v1 = flare::create_mirror_tensor(v1);
        typename TensorType2::HostMirror h_v2 = flare::create_mirror_tensor(v2);

        flare::detail::safe_device_to_host_deep_copy(v1.extent(0), v1, h_v1);
        flare::detail::safe_device_to_host_deep_copy(v2.extent(0), v2, h_v2);

        for (size_t i = 0; i < v1_size; ++i) {
            EXPECT_NEAR_KK_REL(h_v1(i), h_v2(i), tol);
        }
    }

    /// This function returns a descriptive user defined failure string for
    /// insertion into gtest macros such as FAIL() and REQUIRE_LE(). \param file The
    /// filename where the failure originated \param func The function where the
    /// failure originated \param line The line number where the failure originated
    /// \return a new string containing: "  > from file:func:line\n    > "
    static inline const std::string kk_failure_str(std::string file,
                                                   std::string func,
                                                   const int line) {
        std::string failure_msg = "  > from ";
        failure_msg += (file + ":" + func + ":" + std::to_string(line) + "\n    > ");
        return std::string(failure_msg);
    }

#if defined(FLARE_HALF_T_IS_FLOAT)
    using halfScalarType = flare::experimental::half_t;
#endif  // FLARE_HALF_T_IS_FLOAT

#if defined(FLARE_BHALF_T_IS_FLOAT)
    using bhalfScalarType = flare::experimental::bhalf_t;
#endif  // FLARE_BHALF_T_IS_FLOAT

    template <class TensorTypeA, class TensorTypeB, class TensorTypeC,
            class ExecutionSpace>
    struct SharedVanillaGEMM {
        bool A_t, B_t, A_c, B_c;
        int C_rows, C_cols, A_cols;
        TensorTypeA A;
        TensorTypeB B;
        TensorTypeC C;

        typedef typename TensorTypeA::value_type ScalarA;
        typedef typename TensorTypeB::value_type ScalarB;
        typedef typename TensorTypeC::value_type ScalarC;
        typedef flare::Tensor<ScalarA*, flare::LayoutStride,
                typename TensorTypeA::device_type>
                SubtensorTypeA;
        typedef flare::Tensor<ScalarB*, flare::LayoutStride,
                typename TensorTypeB::device_type>
                SubtensorTypeB;
        typedef flare::ArithTraits<ScalarC> APT;
        typedef typename APT::mag_type mag_type;
        ScalarA alpha;
        ScalarC beta;

        FLARE_INLINE_FUNCTION
        void operator()(
                const typename flare::TeamPolicy<ExecutionSpace>::member_type& team)
        const {
            flare::parallel_for(
                    flare::TeamThreadRange(team, C_rows), [&](const int& i) {
                        // Give each flare thread a vector of A
                        SubtensorTypeA a_vec;
                        if (A_t)
                            a_vec = flare::subtensor(A, flare::ALL(), i);
                        else
                            a_vec = flare::subtensor(A, i, flare::ALL());

                        // Have all vector lanes perform the dot product
                        flare::parallel_for(
                                flare::ThreadVectorRange(team, C_cols), [&](const int& j) {
                                    SubtensorTypeB b_vec;
                                    if (B_t)
                                        b_vec = flare::subtensor(B, j, flare::ALL());
                                    else
                                        b_vec = flare::subtensor(B, flare::ALL(), j);
                                    ScalarC ab = ScalarC(0);
                                    for (int k = 0; k < A_cols; k++) {
                                        auto a = A_c ? APT::conj(a_vec(k)) : a_vec(k);
                                        auto b = B_c ? APT::conj(b_vec(k)) : b_vec(k);
                                        ab += a * b;
                                    }
                                    C(i, j) = beta * C(i, j) + alpha * ab;
                                });
                    });
        }
    };
// C(i,:,:) = alpha * (A(i,:,:) * B(i,:,:)) + beta * C(i,:,:)
    template <class TensorTypeA, class TensorTypeB, class TensorTypeC,
            class ExecutionSpace>
    struct Functor_BatchedVanillaGEMM {
        bool A_t, B_t, A_c, B_c, batch_size_last_dim = false;
        TensorTypeA A;
        TensorTypeB B;
        TensorTypeC C;

        using ScalarA      = typename TensorTypeA::value_type;
        using ScalarB      = typename TensorTypeB::value_type;
        using ScalarC      = typename TensorTypeC::value_type;
        using SubtensorTypeA = typename flare::Tensor<ScalarA**, flare::LayoutStride,
                typename TensorTypeA::device_type>;
        using SubtensorTypeB = typename flare::Tensor<ScalarB**, flare::LayoutStride,
                typename TensorTypeA::device_type>;
        using SubtensorTypeC = typename flare::Tensor<ScalarC**, flare::LayoutStride,
                typename TensorTypeA::device_type>;

        ScalarA alpha;
        ScalarC beta;

        FLARE_INLINE_FUNCTION
        void operator()(
                const typename flare::TeamPolicy<ExecutionSpace>::member_type& team)
        const {
            int i = team.league_rank();
            SubtensorTypeA _A;
            SubtensorTypeB _B;
            SubtensorTypeC _C;

            if (batch_size_last_dim) {
                _A = flare::subtensor(A, flare::ALL(), flare::ALL(), i);
                _B = flare::subtensor(B, flare::ALL(), flare::ALL(), i);
                _C = flare::subtensor(C, flare::ALL(), flare::ALL(), i);
            } else {
                _A = flare::subtensor(A, i, flare::ALL(), flare::ALL());
                _B = flare::subtensor(B, i, flare::ALL(), flare::ALL());
                _C = flare::subtensor(C, i, flare::ALL(), flare::ALL());
            }
            struct SharedVanillaGEMM<SubtensorTypeA, SubtensorTypeB, SubtensorTypeC,
                    ExecutionSpace>
                    vgemm;
            vgemm.A_t    = A_t;
            vgemm.B_t    = B_t;
            vgemm.A_c    = A_c;
            vgemm.B_c    = B_c;
            vgemm.C_rows = batch_size_last_dim ? C.extent(0) : C.extent(1);
            vgemm.C_cols = batch_size_last_dim ? C.extent(1) : C.extent(2);
            vgemm.A_cols = batch_size_last_dim ? (A_t ? A.extent(0) : A.extent(1))
                                               : (A_t ? A.extent(1) : A.extent(2));
            vgemm.A     = _A;
            vgemm.B     = _B;
            vgemm.C     = _C;
            vgemm.alpha = alpha;
            vgemm.beta  = beta;
            vgemm(team);
        }

        inline void run() {
            flare::parallel_for(
                    "Test::VanillaGEMM",
                    flare::TeamPolicy<ExecutionSpace>(
                            batch_size_last_dim ? C.extent(2) : C.extent(0), flare::AUTO,
                            flare::detail::flare_get_max_vector_size<ExecutionSpace>()),
                    *this);
        }
    };

// Compute C := alpha * AB + beta * C
    template <class TensorTypeA, class TensorTypeB, class TensorTypeC>
    void vanillaGEMM(typename TensorTypeC::non_const_value_type alpha,
                     const TensorTypeA& A, const TensorTypeB& B,
                     typename TensorTypeC::non_const_value_type beta,
                     const TensorTypeC& C) {
        using value_type = typename TensorTypeC::non_const_value_type;
        using KAT        = flare::ArithTraits<value_type>;
        int m            = A.extent(0);
        int k            = A.extent(1);
        int n            = B.extent(1);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                value_type sum = KAT::zero();
                for (int ii = 0; ii < k; ii++) {
                    sum += A(i, ii) * B(ii, j);
                }
                C(i, j) = alpha * sum + beta * C(i, j);
            }
        }
    }

    template <class AlphaType, class TensorTypeA, class TensorTypeX, class BetaType,
            class TensorTypeY>
    FLARE_INLINE_FUNCTION void vanillaGEMV(char mode, AlphaType alpha,
                                            const TensorTypeA& A, const TensorTypeX& x,
                                            BetaType beta, const TensorTypeY& y) {
        using ScalarY = typename TensorTypeY::non_const_value_type;
        using KAT_A   = flare::ArithTraits<typename TensorTypeA::non_const_value_type>;
        const bool transposed = mode == 'T' || mode == 'C';
        const bool conjugated = mode == 'C';
        const bool has_beta   = beta != flare::ArithTraits<BetaType>::zero();
        int M                 = A.extent(transposed ? 1 : 0);
        int N                 = A.extent(transposed ? 0 : 1);
        for (int i = 0; i < M; i++) {
            ScalarY y_i{};
            if (has_beta) y_i = beta * y(i);
            for (int j = 0; j < N; j++) {
                const auto a   = transposed ? A(j, i) : A(i, j);
                const auto Aij = conjugated ? KAT_A::conj(a) : a;
                y_i += alpha * Aij * x(j);
            }
            y(i) = y_i;
        }
    }

    template <class T>
    class epsilon {
    public:
        constexpr static double value = std::numeric_limits<T>::epsilon();
    };

// explicit epsilon specializations
#if defined(FLARE_HALF_T_IS_FLOAT) && !FLARE_HALF_T_IS_FLOAT
    template <>
    class epsilon<flare::experimental::half_t> {
        public:
            constexpr static double value = 0.0009765625F;
    };
#endif  // FLARE_HALF_T_IS_FLOAT

// explicit epsilon specializations
#if defined(FLARE_BHALF_T_IS_FLOAT) && !FLARE_BHALF_T_IS_FLOAT
    template <>
    class epsilon<flare::experimental::bhalf_t> {
     public:
        constexpr static double value = 0.0078125F;
    };
#endif  // FLARE_HALF_T_IS_FLOAT

    using flare::detail::getRandomBounds;

    template <typename scalar_t, typename lno_t, typename size_type,
            typename device, typename crsMat_t>
    crsMat_t symmetrize(crsMat_t A) {
        typedef typename crsMat_t::StaticCrsGraphType graph_t;
        typedef typename crsMat_t::values_type::non_const_type scalar_tensor_t;
        typedef typename graph_t::row_map_type::non_const_type lno_tensor_t;
        typedef typename graph_t::entries_type::non_const_type lno_nnz_tensor_t;
        auto host_rowmap =
                flare::create_mirror_tensor_and_copy(flare::HostSpace(), A.graph.row_map);
        auto host_entries =
                flare::create_mirror_tensor_and_copy(flare::HostSpace(), A.graph.entries);
        auto host_values =
                flare::create_mirror_tensor_and_copy(flare::HostSpace(), A.values);
        lno_t numRows = A.numRows();
        // symmetrize as input_mat + input_mat^T, to still have a diagonally dominant
        // matrix
        typedef std::map<lno_t, scalar_t> Row;
        std::vector<Row> symRows(numRows);
        for (lno_t r = 0; r < numRows; r++) {
            auto& row = symRows[r];
            for (size_type i = host_rowmap(r); i < host_rowmap(r + 1); i++) {
                lno_t c   = host_entries(i);
                auto& col = symRows[c];
                auto it   = row.find(c);
                if (it == row.end())
                    row[c] = host_values(i);
                else
                    row[c] += host_values(i);
                it = col.find(r);
                if (it == col.end())
                    col[r] = host_values(i);
                else
                    col[r] += host_values(i);
            }
        }
        // Count entries
        flare::Tensor<size_type*, flare::LayoutLeft, flare::HostSpace>
                                                     new_host_rowmap("Rowmap", numRows + 1);
        size_t accum = 0;
        for (lno_t r = 0; r <= numRows; r++) {
            new_host_rowmap(r) = accum;
            if (r < numRows) accum += symRows[r].size();
        }
        // Allocate new entries/values
        flare::Tensor<lno_t*, flare::LayoutLeft, flare::HostSpace> new_host_entries(
                "Entries", accum);
        flare::Tensor<scalar_t*, flare::LayoutLeft, flare::HostSpace>
                                                    new_host_values("Values", accum);
        for (lno_t r = 0; r < numRows; r++) {
            auto rowIt = symRows[r].begin();
            for (size_type i = new_host_rowmap(r); i < new_host_rowmap(r + 1); i++) {
                new_host_entries(i) = rowIt->first;
                new_host_values(i)  = rowIt->second;
                rowIt++;
            }
        }
        lno_tensor_t new_rowmap("Rowmap", numRows + 1);
        lno_nnz_tensor_t new_entries("Entries", accum);
        scalar_tensor_t new_values("Values", accum);
        flare::deep_copy(new_rowmap, new_host_rowmap);
        flare::deep_copy(new_entries, new_host_entries);
        flare::deep_copy(new_values, new_host_values);
        return crsMat_t("SymA", numRows, numRows, accum, new_values, new_rowmap,
                        new_entries);
    }

// create_random_x_vector and create_random_y_vector can be used together to
// generate a random linear system Ax = y.
    template <typename vec_t>
    vec_t create_random_x_vector(vec_t& kok_x, double max_value = 10.0) {
        typedef typename vec_t::value_type scalar_t;
        auto h_x = flare::create_mirror_tensor(kok_x);
        for (size_t j = 0; j < h_x.extent(1); ++j) {
            for (size_t i = 0; i < h_x.extent(0); ++i) {
                scalar_t r = static_cast<scalar_t>(rand()) /
                             static_cast<scalar_t>(RAND_MAX / max_value);
                h_x.access(i, j) = r;
            }
        }
        flare::deep_copy(kok_x, h_x);
        return kok_x;
    }

/// \brief SharedParamTag class used to specify how to invoke templates within
///                       batched unit tests
/// \var TA Indicates which transpose operation to apply to the A matrix
/// \var TB Indicates which transpose operation to apply to the B matrix
/// \var BL Indicates whether the batch size is in the leftmost or rightmost
///         dimension
    template <typename TA, typename TB, typename BL>
    struct SharedParamTag {
        using transA      = TA;
        using transB      = TB;
        using batchLayout = BL;
    };

/// \brief value_type_name returns a string with the value type name
    template <typename T>
    std::string value_type_name() {
        return "::UnknownValueType";
    }

    template <>
    std::string value_type_name<float>() {
        return "::Float";
    }

    template <>
    std::string value_type_name<double>() {
        return "::Double";
    }

    template <>
    std::string value_type_name<int>() {
        return "::Int";
    }

    template <>
    std::string value_type_name<flare::complex<float>>() {
    return "::ComplexFloat";
}

template <>
std::string value_type_name<flare::complex<double>>() {
return "::ComplexDouble";
}

int string_compare_no_case(const char* str1, const char* str2) {
    std::string str1_s(str1);
    std::string str2_s(str2);
    for (size_t i = 0; i < str1_s.size(); i++)
        str1_s[i] = std::tolower(str1_s[i]);
    for (size_t i = 0; i < str2_s.size(); i++)
        str2_s[i] = std::tolower(str2_s[i]);
    return strcmp(str1_s.c_str(), str2_s.c_str());
}

int string_compare_no_case(const std::string& str1, const std::string& str2) {
    return string_compare_no_case(str1.c_str(), str2.c_str());
}
/// /brief Coo matrix class for testing purposes.
/// \tparam ScalarType
/// \tparam LayoutType
/// \tparam Device
template <class ScalarType, class LayoutType, class Device>
class RandCooMat {
private:
    using ExeSpaceType  = typename Device::execution_space;
    using RowTensorTypeD  = flare::Tensor<int64_t*, LayoutType, Device>;
    using ColTensorTypeD  = flare::Tensor<int64_t*, LayoutType, Device>;
    using DatATensorTypeD = flare::Tensor<ScalarType*, LayoutType, Device>;
    RowTensorTypeD __row_d;
    ColTensorTypeD __col_d;
    DatATensorTypeD __data_d;

    template <class T>
    T __getter_copy_helper(T src) {
        T dst(std::string("RandCooMat.") + typeid(T).name() + " copy",
              src.extent(0));
        flare::deep_copy(dst, src);
        ExeSpaceType().fence();
        return dst;
    }

public:
    std::string info;
    /// Constructs a random coo matrix with negative indices.
    /// \param m The max row id
    /// \param n The max col id
    /// \param n_tuples The number of tuples.
    /// \param min_val The minimum scalar value in the matrix.
    /// \param max_val The maximum scalar value in the matrix.
    RandCooMat(int64_t m, int64_t n, int64_t n_tuples, ScalarType min_val,
               ScalarType max_val) {
        uint64_t ticks =
                std::chrono::high_resolution_clock::now().time_since_epoch().count() %
                UINT32_MAX;

        info = std::string(std::string("RandCooMat<") + typeid(ScalarType).name() +
                           ", " + typeid(LayoutType).name() + ", " +
                           typeid(ExeSpaceType).name() + std::to_string(n) +
                           "...): rand seed: " + std::to_string(ticks) + "\n");
        flare::Random_XorShift64_Pool<ExeSpaceType> random(ticks);

        __row_d = RowTensorTypeD("RandCooMat.RowTensorType", n_tuples);
        flare::fill_random(__row_d, random, -m, m);

        __col_d = ColTensorTypeD("RandCooMat.ColTensorType", n_tuples);
        flare::fill_random(__col_d, random, -n, n);

        __data_d = DatATensorTypeD("RandCooMat.DatATensorType", n_tuples);
        flare::fill_random(__data_d, random, min_val, max_val);

        ExeSpaceType().fence();
    }
    auto get_row() { return __getter_copy_helper(__row_d); }
    auto get_col() { return __getter_copy_helper(__col_d); }
    auto get_data() { return __getter_copy_helper(__data_d); }
};

/// /brief Cs (Compressed Sparse) matrix class for testing purposes.
/// This class is for testing purposes only and will generate a random
/// Crs / Ccs matrix when instantiated. The class is intentionally written
/// without the use of "row" and "column" member names.
/// dim1 refers to either rows for Crs matrix or columns for a Ccs matrix.
/// dim2 refers to either columns for a Crs matrix or rows for a Ccs matrix.
/// \tparam ScalarType
/// \tparam LayoutType
/// \tparam Device
template <class ScalarType, class LayoutType, class Device,
        typename Ordinal = int64_t,
        typename Size    = typename flare::TensorTraits<Ordinal*, Device, void,
                void>::size_type>
class RandCsMatrix {
public:
    using value_type   = ScalarType;
    using array_layout = LayoutType;
    using device_type  = Device;
    using ordinal_type = Ordinal;
    using size_type    = Size;
    using ValTensorTypeD = flare::Tensor<ScalarType*, LayoutType, Device>;
    using IDTensorTypeD  = flare::Tensor<Ordinal*, LayoutType, Device>;
    using MapTensorTypeD = flare::Tensor<Size*, LayoutType, Device>;

private:
    using execution_space = typename Device::execution_space;
    Ordinal __dim2;
    Ordinal __dim1;
    Size __nnz = 0;
    MapTensorTypeD __map_d;
    IDTensorTypeD __ids_d;
    ValTensorTypeD __vals_d;
    using MapTensorTypeH = typename MapTensorTypeD::HostMirror;
    using IDTensorTypeH  = typename IDTensorTypeD::HostMirror;
    using ValTensorTypeH = typename ValTensorTypeD::HostMirror;
    MapTensorTypeH __map;
    IDTensorTypeH __ids;
    ValTensorTypeH __vals;
    bool __fully_sparse;

    /// Generates a random map where (using Ccs terminology):
    ///  1. __map(i) is in [__ids.data(), &row_ids.data()[nnz - 1]
    ///  2. __map(i) > col_map(i - 1) for i > 1
    ///  3. __map(i) == col_map(j) iff __map(i) == col_map(j) == nullptr
    ///  4. __map(i) - col_map(i - 1) is in [0, m]
    void __populate_random_cs_mat(uint64_t ticks) {
        std::srand(ticks);
        for (Ordinal col_idx = 0; col_idx < __dim1; col_idx++) {
            Ordinal r = std::rand() % (__dim2 + 1);
            if (r == 0 || __fully_sparse) {  // 100% sparse vector
                __map(col_idx) = __nnz;
            } else {  // sparse vector with r elements
                // Populate r row ids
                std::vector<Ordinal> v(r);

                for (Ordinal i = 0; i < r; i++) v.at(i) = i;

                std::shuffle(v.begin(), v.end(), std::mt19937(std::random_device()()));

                for (Ordinal i = 0; i < r; i++) __ids(i + __nnz) = v.at(i);

                // Point to new column and accumulate number of non zeros
                __map(col_idx) = __nnz;
                __nnz += r;
            }
        }

        // last entry in map points to end of id list
        __map(__dim1) = __nnz;

        // Copy to device
        flare::deep_copy(__map_d, __map);
        IDTensorTypeD tight_ids(flare::tensor_alloc(flare::WithoutInitializing,
                                                 "RandCsMatrix.IDTensorTypeD"),
                              __nnz);
        flare::deep_copy(
                tight_ids,
                flare::subtensor(__ids, flare::make_pair(0, static_cast<int>(__nnz))));
        __ids_d = tight_ids;
    }

    template <class T>
    T __getter_copy_helper(T src) {
        T dst(std::string("RandCsMatrix.") + typeid(T).name() + " copy",
              src.extent(0));
        flare::deep_copy(dst, src);
        return dst;
    }

public:
    std::string info;
    /// Constructs a random cs matrix.
    /// \param dim1 The first dimension: rows for Crs or columns for Ccs
    /// \param dim2 The second dimension: columns for Crs or rows for Ccs
    /// \param min_val The minimum scalar value in the matrix.
    /// \param max_val The maximum scalar value in the matrix.
    RandCsMatrix(Ordinal dim1, Ordinal dim2, ScalarType min_val,
                 ScalarType max_val, bool fully_sparse = false) {
        __dim1         = dim1;
        __dim2         = dim2;
        __fully_sparse = fully_sparse;
        __map_d        = MapTensorTypeD("RandCsMatrix.ColMapTensorType", __dim1 + 1);
        __map          = flare::create_mirror_tensor(__map_d);
        __ids_d        = IDTensorTypeD("RandCsMatrix.RowIDTensorType",
                                     dim2 * dim1 + 1);  // over-allocated
        __ids          = flare::create_mirror_tensor(__ids_d);

        uint64_t ticks =
                std::chrono::high_resolution_clock::now().time_since_epoch().count() %
                UINT32_MAX;

        info = std::string(
                std::string("RandCsMatrix<") + typeid(ScalarType).name() + ", " +
                typeid(LayoutType).name() + ", " + execution_space().name() + ">(" +
                std::to_string(dim2) + ", " + std::to_string(dim1) +
                "...): rand seed: " + std::to_string(ticks) +
                ", fully sparse: " + (__fully_sparse ? "true" : "false") + "\n");
        flare::Random_XorShift64_Pool<flare::HostSpace> random(ticks);
        __populate_random_cs_mat(ticks);

        __vals_d = ValTensorTypeD("RandCsMatrix.ValTensorType", __nnz + 1);
        __vals   = flare::create_mirror_tensor(__vals_d);
        flare::fill_random(__vals, random, min_val, max_val);  // random scalars
        flare::fence();
        __vals(__nnz) = ScalarType(0);

        // Copy to device
        flare::deep_copy(__vals_d, __vals);
    }

    // O(c), where c is a constant.
    ScalarType operator()(Size idx) { return __vals(idx); }
    size_t get_nnz() { return size_t(__nnz); }
    // dimension2: This is either columns for a Crs matrix or rows for a Ccs
    // matrix.
    Ordinal get_dim2() { return __dim2; }
    // dimension1: This is either rows for Crs matrix or columns for a Ccs matrix.
    Ordinal get_dim1() { return __dim1; }
    ValTensorTypeD get_vals() { return __getter_copy_helper(__vals_d); }
    IDTensorTypeD get_ids() { return __getter_copy_helper(__ids_d); }
    MapTensorTypeD get_map() { return __getter_copy_helper(__map_d); }
};

/// \brief Randomly shuffle the entries in each row (col) of a Crs (Ccs) matrix.
template <typename Rowptrs, typename Entries, typename Values>
void shuffleMatrixEntries(Rowptrs rowptrs, Entries entries, Values values) {
    using size_type    = typename Rowptrs::non_const_value_type;
    using ordinal_type = typename Entries::value_type;
    auto rowptrsHost =
            flare::create_mirror_tensor_and_copy(flare::HostSpace(), rowptrs);
    auto entriesHost =
            flare::create_mirror_tensor_and_copy(flare::HostSpace(), entries);
    auto valuesHost =
            flare::create_mirror_tensor_and_copy(flare::HostSpace(), values);
    ordinal_type numRows =
            rowptrsHost.extent(0) ? (rowptrsHost.extent(0) - 1) : 0;
    for (ordinal_type i = 0; i < numRows; i++) {
        size_type rowBegin = rowptrsHost(i);
        size_type rowEnd   = rowptrsHost(i + 1);
        for (size_type j = rowBegin; j < rowEnd - 1; j++) {
            ordinal_type swapRange = rowEnd - j;
            size_type swapOffset   = j + (rand() % swapRange);
            std::swap(entriesHost(j), entriesHost(swapOffset));
            std::swap(valuesHost(j), valuesHost(swapOffset));
        }
    }
    flare::deep_copy(entries, entriesHost);
    flare::deep_copy(values, valuesHost);
}

}  // namespace Test

#endif //FLARE_TEST_UTILITY_H
