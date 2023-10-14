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

#ifndef FLARE_CORE_TENSOR_COPY_TENSORS_H_
#define FLARE_CORE_TENSOR_COPY_TENSORS_H_

#include <string>
#include <flare/core/parallel/parallel.h>
#include <flare/core/policy/exp_mdrange_policy.h>
#include <flare/core/memory/layout.h>
#include <flare/core/memory/host_space_zero_memset.h>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace flare::detail {

    template<class Layout>
    struct TensorFillLayoutSelector {
    };

    template<>
    struct TensorFillLayoutSelector<flare::LayoutLeft> {
        static const flare::Iterate iterate = flare::Iterate::Left;
    };

    template<>
    struct TensorFillLayoutSelector<flare::LayoutRight> {
        static const flare::Iterate iterate = flare::Iterate::Right;
    };

    template<class TensorType, class Layout, class ExecSpace, typename iType>
    struct TensorFill<TensorType, Layout, ExecSpace, 0, iType> {
        using ST = typename TensorType::non_const_value_type;

        TensorFill(const TensorType &a, const ST &val, const ExecSpace &space) {
            flare::detail::DeepCopy<typename TensorType::memory_space, flare::HostSpace,
                    ExecSpace>(space, a.data(), &val, sizeof(ST));
        }
    };

    template<class TensorType, class Layout, class ExecSpace, typename iType>
    struct TensorFill<TensorType, Layout, ExecSpace, 1, iType> {
        TensorType a;
        typename TensorType::const_value_type val;
        using policy_type = flare::RangePolicy<ExecSpace, flare::IndexType<iType>>;

        TensorFill(const TensorType &a_, typename TensorType::const_value_type &val_,
                 const ExecSpace &space)
                : a(a_), val(val_) {
            flare::parallel_for("flare::TensorFill-1D",
                                policy_type(space, 0, a.extent(0)), *this);
        }

        FLARE_INLINE_FUNCTION
        void operator()(const iType &i) const { a(i) = val; };
    };

    template<class TensorType, class Layout, class ExecSpace, typename iType>
    struct TensorFill<TensorType, Layout, ExecSpace, 2, iType> {
        TensorType a;
        typename TensorType::const_value_type val;

        using iterate_type = flare::Rank<2, TensorFillLayoutSelector<Layout>::iterate,
                TensorFillLayoutSelector<Layout>::iterate>;
        using policy_type =
                flare::MDRangePolicy<ExecSpace, iterate_type, flare::IndexType<iType>>;

        TensorFill(const TensorType &a_, typename TensorType::const_value_type &val_,
                 const ExecSpace &space)
                : a(a_), val(val_) {
            flare::parallel_for("flare::TensorFill-2D",
                                policy_type(space, {0, 0}, {a.extent(0), a.extent(1)}),
                                *this);
        }

        FLARE_INLINE_FUNCTION
        void operator()(const iType &i0, const iType &i1) const { a(i0, i1) = val; };
    };

    template<class TensorType, class Layout, class ExecSpace, typename iType>
    struct TensorFill<TensorType, Layout, ExecSpace, 3, iType> {
        TensorType a;
        typename TensorType::const_value_type val;

        using iterate_type = flare::Rank<3, TensorFillLayoutSelector<Layout>::iterate,
                TensorFillLayoutSelector<Layout>::iterate>;
        using policy_type =
                flare::MDRangePolicy<ExecSpace, iterate_type, flare::IndexType<iType>>;

        TensorFill(const TensorType &a_, typename TensorType::const_value_type &val_,
                 const ExecSpace &space)
                : a(a_), val(val_) {
            flare::parallel_for(
                    "flare::TensorFill-3D",
                    policy_type(space, {0, 0, 0}, {a.extent(0), a.extent(1), a.extent(2)}),
                    *this);
        }

        FLARE_INLINE_FUNCTION
        void operator()(const iType &i0, const iType &i1, const iType &i2) const {
            a(i0, i1, i2) = val;
        };
    };

    template<class TensorType, class Layout, class ExecSpace, typename iType>
    struct TensorFill<TensorType, Layout, ExecSpace, 4, iType> {
        TensorType a;
        typename TensorType::const_value_type val;

        using iterate_type = flare::Rank<4, TensorFillLayoutSelector<Layout>::iterate,
                TensorFillLayoutSelector<Layout>::iterate>;
        using policy_type =
                flare::MDRangePolicy<ExecSpace, iterate_type, flare::IndexType<iType>>;

        TensorFill(const TensorType &a_, typename TensorType::const_value_type &val_,
                 const ExecSpace &space)
                : a(a_), val(val_) {
            flare::parallel_for(
                    "flare::TensorFill-4D",
                    policy_type(space, {0, 0, 0, 0},
                                {a.extent(0), a.extent(1), a.extent(2), a.extent(3)}),
                    *this);
        }

        FLARE_INLINE_FUNCTION
        void operator()(const iType &i0, const iType &i1, const iType &i2,
                        const iType &i3) const {
            a(i0, i1, i2, i3) = val;
        };
    };

    template<class TensorType, class Layout, class ExecSpace, typename iType>
    struct TensorFill<TensorType, Layout, ExecSpace, 5, iType> {
        TensorType a;
        typename TensorType::const_value_type val;

        using iterate_type = flare::Rank<5, TensorFillLayoutSelector<Layout>::iterate,
                TensorFillLayoutSelector<Layout>::iterate>;
        using policy_type =
                flare::MDRangePolicy<ExecSpace, iterate_type, flare::IndexType<iType>>;

        TensorFill(const TensorType &a_, typename TensorType::const_value_type &val_,
                 const ExecSpace &space)
                : a(a_), val(val_) {
            flare::parallel_for("flare::TensorFill-5D",
                                policy_type(space, {0, 0, 0, 0, 0},
                                            {a.extent(0), a.extent(1), a.extent(2),
                                             a.extent(3), a.extent(4)}),
                                *this);
        }

        FLARE_INLINE_FUNCTION
        void operator()(const iType &i0, const iType &i1, const iType &i2,
                        const iType &i3, const iType &i4) const {
            a(i0, i1, i2, i3, i4) = val;
        };
    };

    template<class TensorType, class Layout, class ExecSpace, typename iType>
    struct TensorFill<TensorType, Layout, ExecSpace, 6, iType> {
        TensorType a;
        typename TensorType::const_value_type val;

        using iterate_type = flare::Rank<6, TensorFillLayoutSelector<Layout>::iterate,
                TensorFillLayoutSelector<Layout>::iterate>;
        using policy_type =
                flare::MDRangePolicy<ExecSpace, iterate_type, flare::IndexType<iType>>;

        TensorFill(const TensorType &a_, typename TensorType::const_value_type &val_,
                 const ExecSpace &space)
                : a(a_), val(val_) {
            flare::parallel_for("flare::TensorFill-6D",
                                policy_type(space, {0, 0, 0, 0, 0, 0},
                                            {a.extent(0), a.extent(1), a.extent(2),
                                             a.extent(3), a.extent(4), a.extent(5)}),
                                *this);
        }

        FLARE_INLINE_FUNCTION
        void operator()(const iType &i0, const iType &i1, const iType &i2,
                        const iType &i3, const iType &i4, const iType &i5) const {
            a(i0, i1, i2, i3, i4, i5) = val;
        };
    };

    template<class TensorType, class Layout, class ExecSpace, typename iType>
    struct TensorFill<TensorType, Layout, ExecSpace, 7, iType> {
        TensorType a;
        typename TensorType::const_value_type val;

        using iterate_type = flare::Rank<6, TensorFillLayoutSelector<Layout>::iterate,
                TensorFillLayoutSelector<Layout>::iterate>;
        using policy_type =
                flare::MDRangePolicy<ExecSpace, iterate_type, flare::IndexType<iType>>;

        TensorFill(const TensorType &a_, typename TensorType::const_value_type &val_,
                 const ExecSpace &space)
                : a(a_), val(val_) {
            flare::parallel_for("flare::TensorFill-7D",
                                policy_type(space, {0, 0, 0, 0, 0, 0},
                                            {a.extent(0), a.extent(1), a.extent(2),
                                             a.extent(3), a.extent(5), a.extent(6)}),
                                *this);
        }

        FLARE_INLINE_FUNCTION
        void operator()(const iType &i0, const iType &i1, const iType &i3,
                        const iType &i4, const iType &i5, const iType &i6) const {
            for (iType i2 = 0; i2 < iType(a.extent(2)); i2++)
                a(i0, i1, i2, i3, i4, i5, i6) = val;
        };
    };

    template<class TensorType, class Layout, class ExecSpace, typename iType>
    struct TensorFill<TensorType, Layout, ExecSpace, 8, iType> {
        TensorType a;
        typename TensorType::const_value_type val;

        using iterate_type = flare::Rank<6, TensorFillLayoutSelector<Layout>::iterate,
                TensorFillLayoutSelector<Layout>::iterate>;
        using policy_type =
                flare::MDRangePolicy<ExecSpace, iterate_type, flare::IndexType<iType>>;

        TensorFill(const TensorType &a_, typename TensorType::const_value_type &val_,
                 const ExecSpace &space)
                : a(a_), val(val_) {
            flare::parallel_for("flare::TensorFill-8D",
                                policy_type(space, {0, 0, 0, 0, 0, 0},
                                            {a.extent(0), a.extent(1), a.extent(3),
                                             a.extent(5), a.extent(6), a.extent(7)}),
                                *this);
        }

        FLARE_INLINE_FUNCTION
        void operator()(const iType &i0, const iType &i1, const iType &i3,
                        const iType &i5, const iType &i6, const iType &i7) const {
            for (iType i2 = 0; i2 < iType(a.extent(2)); i2++)
                for (iType i4 = 0; i4 < iType(a.extent(4)); i4++)
                    a(i0, i1, i2, i3, i4, i5, i6, i7) = val;
        };
    };

    template<class TensorTypeA, class TensorTypeB, class Layout, class ExecSpace,
            typename iType>
    struct TensorCopy<TensorTypeA, TensorTypeB, Layout, ExecSpace, 1, iType> {
        TensorTypeA a;
        TensorTypeB b;

        using policy_type = flare::RangePolicy<ExecSpace, flare::IndexType<iType>>;
        using value_type = typename TensorTypeA::value_type;

        TensorCopy(const TensorTypeA &a_, const TensorTypeB &b_,
                 const ExecSpace space = ExecSpace())
                : a(a_), b(b_) {
            flare::parallel_for("flare::TensorCopy-1D",
                                policy_type(space, 0, a.extent(0)), *this);
        }

        FLARE_INLINE_FUNCTION
        void operator()(const iType &i0) const {
            a(i0) = static_cast<value_type>(b(i0));
        };
    };

    template<class TensorTypeA, class TensorTypeB, class Layout, class ExecSpace,
            typename iType>
    struct TensorCopy<TensorTypeA, TensorTypeB, Layout, ExecSpace, 2, iType> {
        TensorTypeA a;
        TensorTypeB b;
        static const flare::Iterate outer_iteration_pattern =
                flare::layout_iterate_type_selector<Layout>::outer_iteration_pattern;
        static const flare::Iterate inner_iteration_pattern =
                flare::layout_iterate_type_selector<Layout>::inner_iteration_pattern;
        using iterate_type =
                flare::Rank<2, outer_iteration_pattern, inner_iteration_pattern>;
        using policy_type =
                flare::MDRangePolicy<ExecSpace, iterate_type, flare::IndexType<iType>>;
        using value_type = typename TensorTypeA::value_type;

        TensorCopy(const TensorTypeA &a_, const TensorTypeB &b_,
                 const ExecSpace space = ExecSpace())
                : a(a_), b(b_) {
            flare::parallel_for("flare::TensorCopy-2D",
                                policy_type(space, {0, 0}, {a.extent(0), a.extent(1)}),
                                *this);
        }

        FLARE_INLINE_FUNCTION
        void operator()(const iType &i0, const iType &i1) const {
            a(i0, i1) = static_cast<value_type>(b(i0, i1));
        };
    };

    template<class TensorTypeA, class TensorTypeB, class Layout, class ExecSpace,
            typename iType>
    struct TensorCopy<TensorTypeA, TensorTypeB, Layout, ExecSpace, 3, iType> {
        TensorTypeA a;
        TensorTypeB b;

        static const flare::Iterate outer_iteration_pattern =
                flare::layout_iterate_type_selector<Layout>::outer_iteration_pattern;
        static const flare::Iterate inner_iteration_pattern =
                flare::layout_iterate_type_selector<Layout>::inner_iteration_pattern;
        using iterate_type =
                flare::Rank<3, outer_iteration_pattern, inner_iteration_pattern>;
        using policy_type =
                flare::MDRangePolicy<ExecSpace, iterate_type, flare::IndexType<iType>>;
        using value_type = typename TensorTypeA::value_type;

        TensorCopy(const TensorTypeA &a_, const TensorTypeB &b_,
                 const ExecSpace space = ExecSpace())
                : a(a_), b(b_) {
            flare::parallel_for(
                    "flare::TensorCopy-3D",
                    policy_type(space, {0, 0, 0}, {a.extent(0), a.extent(1), a.extent(2)}),
                    *this);
        }

        FLARE_INLINE_FUNCTION
        void operator()(const iType &i0, const iType &i1, const iType &i2) const {
            a(i0, i1, i2) = static_cast<value_type>(b(i0, i1, i2));
        };
    };

    template<class TensorTypeA, class TensorTypeB, class Layout, class ExecSpace,
            typename iType>
    struct TensorCopy<TensorTypeA, TensorTypeB, Layout, ExecSpace, 4, iType> {
        TensorTypeA a;
        TensorTypeB b;

        static const flare::Iterate outer_iteration_pattern =
                flare::layout_iterate_type_selector<Layout>::outer_iteration_pattern;
        static const flare::Iterate inner_iteration_pattern =
                flare::layout_iterate_type_selector<Layout>::inner_iteration_pattern;
        using iterate_type =
                flare::Rank<4, outer_iteration_pattern, inner_iteration_pattern>;
        using policy_type =
                flare::MDRangePolicy<ExecSpace, iterate_type, flare::IndexType<iType>>;

        TensorCopy(const TensorTypeA &a_, const TensorTypeB &b_,
                 const ExecSpace space = ExecSpace())
                : a(a_), b(b_) {
            flare::parallel_for(
                    "flare::TensorCopy-4D",
                    policy_type(space, {0, 0, 0, 0},
                                {a.extent(0), a.extent(1), a.extent(2), a.extent(3)}),
                    *this);
        }

        FLARE_INLINE_FUNCTION
        void operator()(const iType &i0, const iType &i1, const iType &i2,
                        const iType &i3) const {
            a(i0, i1, i2, i3) = b(i0, i1, i2, i3);
        };
    };

    template<class TensorTypeA, class TensorTypeB, class Layout, class ExecSpace,
            typename iType>
    struct TensorCopy<TensorTypeA, TensorTypeB, Layout, ExecSpace, 5, iType> {
        TensorTypeA a;
        TensorTypeB b;

        static const flare::Iterate outer_iteration_pattern =
                flare::layout_iterate_type_selector<Layout>::outer_iteration_pattern;
        static const flare::Iterate inner_iteration_pattern =
                flare::layout_iterate_type_selector<Layout>::inner_iteration_pattern;
        using iterate_type =
                flare::Rank<5, outer_iteration_pattern, inner_iteration_pattern>;
        using policy_type =
                flare::MDRangePolicy<ExecSpace, iterate_type, flare::IndexType<iType>>;

        TensorCopy(const TensorTypeA &a_, const TensorTypeB &b_,
                 const ExecSpace space = ExecSpace())
                : a(a_), b(b_) {
            flare::parallel_for("flare::TensorCopy-5D",
                                policy_type(space, {0, 0, 0, 0, 0},
                                            {a.extent(0), a.extent(1), a.extent(2),
                                             a.extent(3), a.extent(4)}),
                                *this);
        }

        FLARE_INLINE_FUNCTION
        void operator()(const iType &i0, const iType &i1, const iType &i2,
                        const iType &i3, const iType &i4) const {
            a(i0, i1, i2, i3, i4) = b(i0, i1, i2, i3, i4);
        };
    };

    template<class TensorTypeA, class TensorTypeB, class Layout, class ExecSpace,
            typename iType>
    struct TensorCopy<TensorTypeA, TensorTypeB, Layout, ExecSpace, 6, iType> {
        TensorTypeA a;
        TensorTypeB b;

        static const flare::Iterate outer_iteration_pattern =
                flare::layout_iterate_type_selector<Layout>::outer_iteration_pattern;
        static const flare::Iterate inner_iteration_pattern =
                flare::layout_iterate_type_selector<Layout>::inner_iteration_pattern;
        using iterate_type =
                flare::Rank<6, outer_iteration_pattern, inner_iteration_pattern>;
        using policy_type =
                flare::MDRangePolicy<ExecSpace, iterate_type, flare::IndexType<iType>>;

        TensorCopy(const TensorTypeA &a_, const TensorTypeB &b_,
                 const ExecSpace space = ExecSpace())
                : a(a_), b(b_) {
            flare::parallel_for("flare::TensorCopy-6D",
                                policy_type(space, {0, 0, 0, 0, 0, 0},
                                            {a.extent(0), a.extent(1), a.extent(2),
                                             a.extent(3), a.extent(4), a.extent(5)}),
                                *this);
        }

        FLARE_INLINE_FUNCTION
        void operator()(const iType &i0, const iType &i1, const iType &i2,
                        const iType &i3, const iType &i4, const iType &i5) const {
            a(i0, i1, i2, i3, i4, i5) = b(i0, i1, i2, i3, i4, i5);
        };
    };

    template<class TensorTypeA, class TensorTypeB, class Layout, class ExecSpace,
            typename iType>
    struct TensorCopy<TensorTypeA, TensorTypeB, Layout, ExecSpace, 7, iType> {
        TensorTypeA a;
        TensorTypeB b;

        static const flare::Iterate outer_iteration_pattern =
                flare::layout_iterate_type_selector<Layout>::outer_iteration_pattern;
        static const flare::Iterate inner_iteration_pattern =
                flare::layout_iterate_type_selector<Layout>::inner_iteration_pattern;
        using iterate_type =
                flare::Rank<6, outer_iteration_pattern, inner_iteration_pattern>;
        using policy_type =
                flare::MDRangePolicy<ExecSpace, iterate_type, flare::IndexType<iType>>;

        TensorCopy(const TensorTypeA &a_, const TensorTypeB &b_,
                 const ExecSpace space = ExecSpace())
                : a(a_), b(b_) {
            flare::parallel_for("flare::TensorCopy-7D",
                                policy_type(space, {0, 0, 0, 0, 0, 0},
                                            {a.extent(0), a.extent(1), a.extent(3),
                                             a.extent(4), a.extent(5), a.extent(6)}),
                                *this);
        }

        FLARE_INLINE_FUNCTION
        void operator()(const iType &i0, const iType &i1, const iType &i3,
                        const iType &i4, const iType &i5, const iType &i6) const {
            for (iType i2 = 0; i2 < iType(a.extent(2)); i2++)
                a(i0, i1, i2, i3, i4, i5, i6) = b(i0, i1, i2, i3, i4, i5, i6);
        };
    };

    template<class TensorTypeA, class TensorTypeB, class Layout, class ExecSpace,
            typename iType>
    struct TensorCopy<TensorTypeA, TensorTypeB, Layout, ExecSpace, 8, iType> {
        TensorTypeA a;
        TensorTypeB b;

        static const flare::Iterate outer_iteration_pattern =
                flare::layout_iterate_type_selector<Layout>::outer_iteration_pattern;
        static const flare::Iterate inner_iteration_pattern =
                flare::layout_iterate_type_selector<Layout>::inner_iteration_pattern;
        using iterate_type =
                flare::Rank<6, outer_iteration_pattern, inner_iteration_pattern>;
        using policy_type =
                flare::MDRangePolicy<ExecSpace, iterate_type, flare::IndexType<iType>>;

        TensorCopy(const TensorTypeA &a_, const TensorTypeB &b_,
                 const ExecSpace space = ExecSpace())
                : a(a_), b(b_) {
            flare::parallel_for("flare::TensorCopy-8D",
                                policy_type(space, {0, 0, 0, 0, 0, 0},
                                            {a.extent(0), a.extent(1), a.extent(3),
                                             a.extent(5), a.extent(6), a.extent(7)}),
                                *this);
        }

        FLARE_INLINE_FUNCTION
        void operator()(const iType &i0, const iType &i1, const iType &i3,
                        const iType &i5, const iType &i6, const iType &i7) const {
            for (iType i2 = 0; i2 < iType(a.extent(2)); i2++)
                for (iType i4 = 0; i4 < iType(a.extent(4)); i4++)
                    a(i0, i1, i2, i3, i4, i5, i6, i7) = b(i0, i1, i2, i3, i4, i5, i6, i7);
        };
    };


    template<class ExecutionSpace, class DstType, class SrcType>
    void tensor_copy(const ExecutionSpace &space, const DstType &dst,
                   const SrcType &src) {
        using dst_memory_space = typename DstType::memory_space;
        using src_memory_space = typename SrcType::memory_space;

        enum {
            ExecCanAccessSrc =
            flare::SpaceAccessibility<ExecutionSpace, src_memory_space>::accessible
        };
        enum {
            ExecCanAccessDst =
            flare::SpaceAccessibility<ExecutionSpace, dst_memory_space>::accessible
        };

        if (!(ExecCanAccessSrc && ExecCanAccessDst)) {
            flare::detail::throw_runtime_exception(
                    "flare::detail::tensor_copy called with invalid execution space");
        } else {
            // Figure out iteration order in case we need it
            int64_t strides[DstType::rank + 1];
            dst.stride(strides);
            flare::Iterate iterate;
            if (flare::is_layouttiled<typename DstType::array_layout>::value) {
                iterate = flare::layout_iterate_type_selector<
                        typename DstType::array_layout>::outer_iteration_pattern;
            } else if (std::is_same<typename DstType::array_layout,
                    flare::LayoutRight>::value) {
                iterate = flare::Iterate::Right;
            } else if (std::is_same<typename DstType::array_layout,
                    flare::LayoutLeft>::value) {
                iterate = flare::Iterate::Left;
            } else if (std::is_same<typename DstType::array_layout,
                    flare::LayoutStride>::value) {
                if (strides[0] > strides[DstType::rank - 1])
                    iterate = flare::Iterate::Right;
                else
                    iterate = flare::Iterate::Left;
            } else {
                if (std::is_same<typename DstType::execution_space::array_layout,
                        flare::LayoutRight>::value)
                    iterate = flare::Iterate::Right;
                else
                    iterate = flare::Iterate::Left;
            }

            if ((dst.span() >= size_t(std::numeric_limits<int>::max())) ||
                (src.span() >= size_t(std::numeric_limits<int>::max()))) {
                if (iterate == flare::Iterate::Right)
                    flare::detail::TensorCopy<
                            typename DstType::uniform_runtime_nomemspace_type,
                            typename SrcType::uniform_runtime_const_nomemspace_type,
                            flare::LayoutRight, ExecutionSpace, DstType::rank, int64_t>(
                            dst, src, space);
                else
                    flare::detail::TensorCopy<
                            typename DstType::uniform_runtime_nomemspace_type,
                            typename SrcType::uniform_runtime_const_nomemspace_type,
                            flare::LayoutLeft, ExecutionSpace, DstType::rank, int64_t>(
                            dst, src, space);
            } else {
                if (iterate == flare::Iterate::Right)
                    flare::detail::TensorCopy<
                            typename DstType::uniform_runtime_nomemspace_type,
                            typename SrcType::uniform_runtime_const_nomemspace_type,
                            flare::LayoutRight, ExecutionSpace, DstType::rank, int>(dst, src,
                                                                                    space);
                else
                    flare::detail::TensorCopy<
                            typename DstType::uniform_runtime_nomemspace_type,
                            typename SrcType::uniform_runtime_const_nomemspace_type,
                            flare::LayoutLeft, ExecutionSpace, DstType::rank, int>(dst, src,
                                                                                   space);
            }
        }
    }

    template<class DstType, class SrcType>
    void tensor_copy(const DstType &dst, const SrcType &src) {
        using dst_execution_space = typename DstType::execution_space;
        using src_execution_space = typename SrcType::execution_space;
        using dst_memory_space = typename DstType::memory_space;
        using src_memory_space = typename SrcType::memory_space;

        enum {
            DstExecCanAccessSrc =
            flare::SpaceAccessibility<dst_execution_space,
                    src_memory_space>::accessible
        };

        enum {
            SrcExecCanAccessDst =
            flare::SpaceAccessibility<src_execution_space,
                    dst_memory_space>::accessible
        };

        if (!DstExecCanAccessSrc && !SrcExecCanAccessDst) {
            std::string message(
                    "Error: flare::deep_copy with no available copy mechanism: ");
            message += src.label();
            message += " to ";
            message += dst.label();
            flare::detail::throw_runtime_exception(message);
        }

        // Figure out iteration order in case we need it
        int64_t strides[DstType::rank + 1];
        dst.stride(strides);
        flare::Iterate iterate;
        if (flare::is_layouttiled<typename DstType::array_layout>::value) {
            iterate = flare::layout_iterate_type_selector<
                    typename DstType::array_layout>::outer_iteration_pattern;
        } else if (std::is_same<typename DstType::array_layout,
                flare::LayoutRight>::value) {
            iterate = flare::Iterate::Right;
        } else if (std::is_same<typename DstType::array_layout,
                flare::LayoutLeft>::value) {
            iterate = flare::Iterate::Left;
        } else if (std::is_same<typename DstType::array_layout,
                flare::LayoutStride>::value) {
            if (strides[0] > strides[DstType::rank - 1])
                iterate = flare::Iterate::Right;
            else
                iterate = flare::Iterate::Left;
        } else {
            if (std::is_same<typename DstType::execution_space::array_layout,
                    flare::LayoutRight>::value)
                iterate = flare::Iterate::Right;
            else
                iterate = flare::Iterate::Left;
        }

        if ((dst.span() >= size_t(std::numeric_limits<int>::max())) ||
            (src.span() >= size_t(std::numeric_limits<int>::max()))) {
            if (DstExecCanAccessSrc) {
                if (iterate == flare::Iterate::Right)
                    flare::detail::TensorCopy<
                            typename DstType::uniform_runtime_nomemspace_type,
                            typename SrcType::uniform_runtime_const_nomemspace_type,
                            flare::LayoutRight, dst_execution_space, DstType::rank, int64_t>(
                            dst, src);
                else
                    flare::detail::TensorCopy<
                            typename DstType::uniform_runtime_nomemspace_type,
                            typename SrcType::uniform_runtime_const_nomemspace_type,
                            flare::LayoutLeft, dst_execution_space, DstType::rank, int64_t>(
                            dst, src);
            } else {
                if (iterate == flare::Iterate::Right)
                    flare::detail::TensorCopy<
                            typename DstType::uniform_runtime_nomemspace_type,
                            typename SrcType::uniform_runtime_const_nomemspace_type,
                            flare::LayoutRight, src_execution_space, DstType::rank, int64_t>(
                            dst, src);
                else
                    flare::detail::TensorCopy<
                            typename DstType::uniform_runtime_nomemspace_type,
                            typename SrcType::uniform_runtime_const_nomemspace_type,
                            flare::LayoutLeft, src_execution_space, DstType::rank, int64_t>(
                            dst, src);
            }
        } else {
            if (DstExecCanAccessSrc) {
                if (iterate == flare::Iterate::Right)
                    flare::detail::TensorCopy<
                            typename DstType::uniform_runtime_nomemspace_type,
                            typename SrcType::uniform_runtime_const_nomemspace_type,
                            flare::LayoutRight, dst_execution_space, DstType::rank, int>(dst,
                                                                                         src);
                else
                    flare::detail::TensorCopy<
                            typename DstType::uniform_runtime_nomemspace_type,
                            typename SrcType::uniform_runtime_const_nomemspace_type,
                            flare::LayoutLeft, dst_execution_space, DstType::rank, int>(dst,
                                                                                        src);
            } else {
                if (iterate == flare::Iterate::Right)
                    flare::detail::TensorCopy<
                            typename DstType::uniform_runtime_nomemspace_type,
                            typename SrcType::uniform_runtime_const_nomemspace_type,
                            flare::LayoutRight, src_execution_space, DstType::rank, int>(dst,
                                                                                         src);
                else
                    flare::detail::TensorCopy<
                            typename DstType::uniform_runtime_nomemspace_type,
                            typename SrcType::uniform_runtime_const_nomemspace_type,
                            flare::LayoutLeft, src_execution_space, DstType::rank, int>(dst,
                                                                                        src);
            }
        }
    }

    template<class DstType, class SrcType, int Rank, class... Args>
    struct CommonSubtensor;

    template<class DstType, class SrcType, class Arg0, class... Args>
    struct CommonSubtensor<DstType, SrcType, 1, Arg0, Args...> {
        using dst_subtensor_type = typename flare::Subtensor<DstType, Arg0>;
        using src_subtensor_type = typename flare::Subtensor<SrcType, Arg0>;
        dst_subtensor_type dst_sub;
        src_subtensor_type src_sub;

        CommonSubtensor(const DstType &dst, const SrcType &src, const Arg0 &arg0,
                      Args...)
                : dst_sub(dst, arg0), src_sub(src, arg0) {}
    };

    template<class DstType, class SrcType, class Arg0, class Arg1, class... Args>
    struct CommonSubtensor<DstType, SrcType, 2, Arg0, Arg1, Args...> {
        using dst_subtensor_type = typename flare::Subtensor<DstType, Arg0, Arg1>;
        using src_subtensor_type = typename flare::Subtensor<SrcType, Arg0, Arg1>;
        dst_subtensor_type dst_sub;
        src_subtensor_type src_sub;

        CommonSubtensor(const DstType &dst, const SrcType &src, const Arg0 &arg0,
                      const Arg1 &arg1, Args...)
                : dst_sub(dst, arg0, arg1), src_sub(src, arg0, arg1) {}
    };

    template<class DstType, class SrcType, class Arg0, class Arg1, class Arg2,
            class... Args>
    struct CommonSubtensor<DstType, SrcType, 3, Arg0, Arg1, Arg2, Args...> {
        using dst_subtensor_type = typename flare::Subtensor<DstType, Arg0, Arg1, Arg2>;
        using src_subtensor_type = typename flare::Subtensor<SrcType, Arg0, Arg1, Arg2>;
        dst_subtensor_type dst_sub;
        src_subtensor_type src_sub;

        CommonSubtensor(const DstType &dst, const SrcType &src, const Arg0 &arg0,
                      const Arg1 &arg1, const Arg2 &arg2, Args...)
                : dst_sub(dst, arg0, arg1, arg2), src_sub(src, arg0, arg1, arg2) {}
    };

    template<class DstType, class SrcType, class Arg0, class Arg1, class Arg2,
            class Arg3, class... Args>
    struct CommonSubtensor<DstType, SrcType, 4, Arg0, Arg1, Arg2, Arg3, Args...> {
        using dst_subtensor_type =
                typename flare::Subtensor<DstType, Arg0, Arg1, Arg2, Arg3>;
        using src_subtensor_type =
                typename flare::Subtensor<SrcType, Arg0, Arg1, Arg2, Arg3>;
        dst_subtensor_type dst_sub;
        src_subtensor_type src_sub;

        CommonSubtensor(const DstType &dst, const SrcType &src, const Arg0 &arg0,
                      const Arg1 &arg1, const Arg2 &arg2, const Arg3 &arg3,
                      const Args...)
                : dst_sub(dst, arg0, arg1, arg2, arg3),
                  src_sub(src, arg0, arg1, arg2, arg3) {}
    };

    template<class DstType, class SrcType, class Arg0, class Arg1, class Arg2,
            class Arg3, class Arg4, class... Args>
    struct CommonSubtensor<DstType, SrcType, 5, Arg0, Arg1, Arg2, Arg3, Arg4,
            Args...> {
        using dst_subtensor_type =
                typename flare::Subtensor<DstType, Arg0, Arg1, Arg2, Arg3, Arg4>;
        using src_subtensor_type =
                typename flare::Subtensor<SrcType, Arg0, Arg1, Arg2, Arg3, Arg4>;
        dst_subtensor_type dst_sub;
        src_subtensor_type src_sub;

        CommonSubtensor(const DstType &dst, const SrcType &src, const Arg0 &arg0,
                      const Arg1 &arg1, const Arg2 &arg2, const Arg3 &arg3,
                      const Arg4 &arg4, const Args...)
                : dst_sub(dst, arg0, arg1, arg2, arg3, arg4),
                  src_sub(src, arg0, arg1, arg2, arg3, arg4) {}
    };

    template<class DstType, class SrcType, class Arg0, class Arg1, class Arg2,
            class Arg3, class Arg4, class Arg5, class... Args>
    struct CommonSubtensor<DstType, SrcType, 6, Arg0, Arg1, Arg2, Arg3, Arg4, Arg5,
            Args...> {
        using dst_subtensor_type =
                typename flare::Subtensor<DstType, Arg0, Arg1, Arg2, Arg3, Arg4, Arg5>;
        using src_subtensor_type =
                typename flare::Subtensor<SrcType, Arg0, Arg1, Arg2, Arg3, Arg4, Arg5>;
        dst_subtensor_type dst_sub;
        src_subtensor_type src_sub;

        CommonSubtensor(const DstType &dst, const SrcType &src, const Arg0 &arg0,
                      const Arg1 &arg1, const Arg2 &arg2, const Arg3 &arg3,
                      const Arg4 &arg4, const Arg5 &arg5, const Args...)
                : dst_sub(dst, arg0, arg1, arg2, arg3, arg4, arg5),
                  src_sub(src, arg0, arg1, arg2, arg3, arg4, arg5) {}
    };

    template<class DstType, class SrcType, class Arg0, class Arg1, class Arg2,
            class Arg3, class Arg4, class Arg5, class Arg6, class... Args>
    struct CommonSubtensor<DstType, SrcType, 7, Arg0, Arg1, Arg2, Arg3, Arg4, Arg5,
            Arg6, Args...> {
        using dst_subtensor_type = typename flare::Subtensor<DstType, Arg0, Arg1, Arg2,
                Arg3, Arg4, Arg5, Arg6>;
        using src_subtensor_type = typename flare::Subtensor<SrcType, Arg0, Arg1, Arg2,
                Arg3, Arg4, Arg5, Arg6>;
        dst_subtensor_type dst_sub;
        src_subtensor_type src_sub;

        CommonSubtensor(const DstType &dst, const SrcType &src, const Arg0 &arg0,
                      const Arg1 &arg1, const Arg2 &arg2, const Arg3 &arg3,
                      const Arg4 &arg4, const Arg5 &arg5, const Arg6 &arg6, Args...)
                : dst_sub(dst, arg0, arg1, arg2, arg3, arg4, arg5, arg6),
                  src_sub(src, arg0, arg1, arg2, arg3, arg4, arg5, arg6) {}
    };

    template<class DstType, class SrcType, class Arg0, class Arg1, class Arg2,
            class Arg3, class Arg4, class Arg5, class Arg6, class Arg7>
    struct CommonSubtensor<DstType, SrcType, 8, Arg0, Arg1, Arg2, Arg3, Arg4, Arg5,
            Arg6, Arg7> {
        using dst_subtensor_type =
                typename flare::Subtensor<DstType, Arg0, Arg1, Arg2, Arg3, Arg4, Arg5,
                        Arg6, Arg7>;
        using src_subtensor_type =
                typename flare::Subtensor<SrcType, Arg0, Arg1, Arg2, Arg3, Arg4, Arg5,
                        Arg6, Arg7>;
        dst_subtensor_type dst_sub;
        src_subtensor_type src_sub;

        CommonSubtensor(const DstType &dst, const SrcType &src, const Arg0 &arg0,
                      const Arg1 &arg1, const Arg2 &arg2, const Arg3 &arg3,
                      const Arg4 &arg4, const Arg5 &arg5, const Arg6 &arg6,
                      const Arg7 &arg7)
                : dst_sub(dst, arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7),
                  src_sub(src, arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7) {}
    };

    template<class DstType, class SrcType,
            class ExecSpace = typename DstType::execution_space,
            int Rank = DstType::rank>
    struct TensorRemap;

    template<class DstType, class SrcType, class ExecSpace>
    struct TensorRemap<DstType, SrcType, ExecSpace, 1> {
        using p_type = flare::pair<int64_t, int64_t>;

        template<typename... OptExecSpace>
        TensorRemap(const DstType &dst, const SrcType &src,
                  const OptExecSpace &... exec_space) {
            static_assert(
                    sizeof...(OptExecSpace) <= 1,
                    "OptExecSpace must be either empty or be an execution space!");

            if (dst.extent(0) == src.extent(0)) {
                tensor_copy(exec_space..., dst, src);
            } else {
                p_type ext0(0, std::min(dst.extent(0), src.extent(0)));
                using sv_adapter_type = CommonSubtensor<DstType, SrcType, 1, p_type>;
                sv_adapter_type common_subtensor(dst, src, ext0);
                tensor_copy(exec_space..., common_subtensor.dst_sub, common_subtensor.src_sub);
            }
        }
    };

    template<class DstType, class SrcType, class ExecSpace>
    struct TensorRemap<DstType, SrcType, ExecSpace, 2> {
        using p_type = flare::pair<int64_t, int64_t>;

        template<typename... OptExecSpace>
        TensorRemap(const DstType &dst, const SrcType &src,
                  const OptExecSpace &... exec_space) {
            static_assert(
                    sizeof...(OptExecSpace) <= 1,
                    "OptExecSpace must be either empty or be an execution space!");

            if (dst.extent(0) == src.extent(0)) {
                if (dst.extent(1) == src.extent(1)) {
                    tensor_copy(exec_space..., dst, src);
                } else {
                    p_type ext1(0, std::min(dst.extent(1), src.extent(1)));
                    using sv_adapter_type =
                            CommonSubtensor<DstType, SrcType, 2, flare::ALL_t, p_type>;
                    sv_adapter_type common_subtensor(dst, src, flare::ALL, ext1);
                    tensor_copy(exec_space..., common_subtensor.dst_sub,
                              common_subtensor.src_sub);
                }
            } else {
                if (dst.extent(1) == src.extent(1)) {
                    p_type ext0(0, std::min(dst.extent(0), src.extent(0)));
                    using sv_adapter_type =
                            CommonSubtensor<DstType, SrcType, 2, p_type, flare::ALL_t>;
                    sv_adapter_type common_subtensor(dst, src, ext0, flare::ALL);
                    tensor_copy(exec_space..., common_subtensor.dst_sub,
                              common_subtensor.src_sub);
                } else {
                    p_type ext0(0, std::min(dst.extent(0), src.extent(0)));
                    p_type ext1(0, std::min(dst.extent(1), src.extent(1)));
                    using sv_adapter_type =
                            CommonSubtensor<DstType, SrcType, 2, p_type, p_type>;
                    sv_adapter_type common_subtensor(dst, src, ext0, ext1);
                    tensor_copy(exec_space..., common_subtensor.dst_sub,
                              common_subtensor.src_sub);
                }
            }
        }
    };

    template<class DstType, class SrcType, class ExecSpace>
    struct TensorRemap<DstType, SrcType, ExecSpace, 3> {
        using p_type = flare::pair<int64_t, int64_t>;

        template<typename... OptExecSpace>
        TensorRemap(const DstType &dst, const SrcType &src,
                  const OptExecSpace &... exec_space) {
            static_assert(
                    sizeof...(OptExecSpace) <= 1,
                    "OptExecSpace must be either empty or be an execution space!");

            if (dst.extent(0) == src.extent(0)) {
                if (dst.extent(2) == src.extent(2)) {
                    p_type ext1(0, std::min(dst.extent(1), src.extent(1)));
                    using sv_adapter_type =
                            CommonSubtensor<DstType, SrcType, 3, flare::ALL_t, p_type,
                                    flare::ALL_t>;
                    sv_adapter_type common_subtensor(dst, src, flare::ALL, ext1,
                                                   flare::ALL);
                    tensor_copy(exec_space..., common_subtensor.dst_sub,
                              common_subtensor.src_sub);
                } else {
                    p_type ext1(0, std::min(dst.extent(1), src.extent(1)));
                    p_type ext2(0, std::min(dst.extent(2), src.extent(2)));
                    using sv_adapter_type =
                            CommonSubtensor<DstType, SrcType, 3, flare::ALL_t, p_type, p_type>;
                    sv_adapter_type common_subtensor(dst, src, flare::ALL, ext1, ext2);
                    tensor_copy(exec_space..., common_subtensor.dst_sub,
                              common_subtensor.src_sub);
                }
            } else {
                if (dst.extent(2) == src.extent(2)) {
                    p_type ext0(0, std::min(dst.extent(0), src.extent(0)));
                    p_type ext1(0, std::min(dst.extent(1), src.extent(1)));
                    using sv_adapter_type =
                            CommonSubtensor<DstType, SrcType, 3, p_type, p_type, flare::ALL_t>;
                    sv_adapter_type common_subtensor(dst, src, ext0, ext1, flare::ALL);
                    tensor_copy(exec_space..., common_subtensor.dst_sub,
                              common_subtensor.src_sub);
                } else {
                    p_type ext0(0, std::min(dst.extent(0), src.extent(0)));
                    p_type ext1(0, std::min(dst.extent(1), src.extent(1)));
                    p_type ext2(0, std::min(dst.extent(2), src.extent(2)));
                    using sv_adapter_type =
                            CommonSubtensor<DstType, SrcType, 3, p_type, p_type, p_type>;
                    sv_adapter_type common_subtensor(dst, src, ext0, ext1, ext2);
                    tensor_copy(exec_space..., common_subtensor.dst_sub,
                              common_subtensor.src_sub);
                }
            }
        }
    };

    template<class DstType, class SrcType, class ExecSpace>
    struct TensorRemap<DstType, SrcType, ExecSpace, 4> {
        using p_type = flare::pair<int64_t, int64_t>;

        template<typename... OptExecSpace>
        TensorRemap(const DstType &dst, const SrcType &src,
                  const OptExecSpace &... exec_space) {
            static_assert(
                    sizeof...(OptExecSpace) <= 1,
                    "OptExecSpace must be either empty or be an execution space!");

            if (dst.extent(0) == src.extent(0)) {
                if (dst.extent(3) == src.extent(3)) {
                    p_type ext1(0, std::min(dst.extent(1), src.extent(1)));
                    p_type ext2(0, std::min(dst.extent(2), src.extent(2)));
                    using sv_adapter_type =
                            CommonSubtensor<DstType, SrcType, 4, flare::ALL_t, p_type, p_type,
                                    flare::ALL_t>;
                    sv_adapter_type common_subtensor(dst, src, flare::ALL, ext1, ext2,
                                                   flare::ALL);
                    tensor_copy(exec_space..., common_subtensor.dst_sub,
                              common_subtensor.src_sub);
                } else {
                    p_type ext1(0, std::min(dst.extent(1), src.extent(1)));
                    p_type ext2(0, std::min(dst.extent(2), src.extent(2)));
                    p_type ext3(0, std::min(dst.extent(3), src.extent(3)));
                    using sv_adapter_type =
                            CommonSubtensor<DstType, SrcType, 4, flare::ALL_t, p_type, p_type,
                                    p_type>;
                    sv_adapter_type common_subtensor(dst, src, flare::ALL, ext1, ext2, ext3);
                    tensor_copy(exec_space..., common_subtensor.dst_sub,
                              common_subtensor.src_sub);
                }
            } else {
                if (dst.extent(7) == src.extent(7)) {
                    p_type ext0(0, std::min(dst.extent(0), src.extent(0)));
                    p_type ext1(0, std::min(dst.extent(1), src.extent(1)));
                    p_type ext2(0, std::min(dst.extent(2), src.extent(2)));
                    using sv_adapter_type = CommonSubtensor<DstType, SrcType, 4, p_type,
                            p_type, p_type, flare::ALL_t>;
                    sv_adapter_type common_subtensor(dst, src, ext0, ext1, ext2, flare::ALL);
                    tensor_copy(exec_space..., common_subtensor.dst_sub,
                              common_subtensor.src_sub);
                } else {
                    p_type ext0(0, std::min(dst.extent(0), src.extent(0)));
                    p_type ext1(0, std::min(dst.extent(1), src.extent(1)));
                    p_type ext2(0, std::min(dst.extent(2), src.extent(2)));
                    p_type ext3(0, std::min(dst.extent(3), src.extent(3)));
                    using sv_adapter_type =
                            CommonSubtensor<DstType, SrcType, 4, p_type, p_type, p_type, p_type>;
                    sv_adapter_type common_subtensor(dst, src, ext0, ext1, ext2, ext3);
                    tensor_copy(exec_space..., common_subtensor.dst_sub,
                              common_subtensor.src_sub);
                }
            }
        }
    };

    template<class DstType, class SrcType, class ExecSpace>
    struct TensorRemap<DstType, SrcType, ExecSpace, 5> {
        using p_type = flare::pair<int64_t, int64_t>;

        template<typename... OptExecSpace>
        TensorRemap(const DstType &dst, const SrcType &src,
                  const OptExecSpace &... exec_space) {
            static_assert(
                    sizeof...(OptExecSpace) <= 1,
                    "OptExecSpace must be either empty or be an execution space!");

            if (dst.extent(0) == src.extent(0)) {
                if (dst.extent(4) == src.extent(4)) {
                    p_type ext1(0, std::min(dst.extent(1), src.extent(1)));
                    p_type ext2(0, std::min(dst.extent(2), src.extent(2)));
                    p_type ext3(0, std::min(dst.extent(3), src.extent(3)));
                    using sv_adapter_type =
                            CommonSubtensor<DstType, SrcType, 5, flare::ALL_t, p_type, p_type,
                                    p_type, flare::ALL_t>;
                    sv_adapter_type common_subtensor(dst, src, flare::ALL, ext1, ext2, ext3,
                                                   flare::ALL);
                    tensor_copy(exec_space..., common_subtensor.dst_sub,
                              common_subtensor.src_sub);
                } else {
                    p_type ext1(0, std::min(dst.extent(1), src.extent(1)));
                    p_type ext2(0, std::min(dst.extent(2), src.extent(2)));
                    p_type ext3(0, std::min(dst.extent(3), src.extent(3)));
                    p_type ext4(0, std::min(dst.extent(4), src.extent(4)));
                    using sv_adapter_type =
                            CommonSubtensor<DstType, SrcType, 5, flare::ALL_t, p_type, p_type,
                                    p_type, p_type>;
                    sv_adapter_type common_subtensor(dst, src, flare::ALL, ext1, ext2, ext3,
                                                   ext4);
                    tensor_copy(exec_space..., common_subtensor.dst_sub,
                              common_subtensor.src_sub);
                }
            } else {
                if (dst.extent(4) == src.extent(4)) {
                    p_type ext0(0, std::min(dst.extent(0), src.extent(0)));
                    p_type ext1(0, std::min(dst.extent(1), src.extent(1)));
                    p_type ext2(0, std::min(dst.extent(2), src.extent(2)));
                    p_type ext3(0, std::min(dst.extent(3), src.extent(3)));
                    using sv_adapter_type =
                            CommonSubtensor<DstType, SrcType, 5, p_type, p_type, p_type, p_type,
                                    flare::ALL_t>;
                    sv_adapter_type common_subtensor(dst, src, ext0, ext1, ext2, ext3,
                                                   flare::ALL);
                    tensor_copy(exec_space..., common_subtensor.dst_sub,
                              common_subtensor.src_sub);
                } else {
                    p_type ext0(0, std::min(dst.extent(0), src.extent(0)));
                    p_type ext1(0, std::min(dst.extent(1), src.extent(1)));
                    p_type ext2(0, std::min(dst.extent(2), src.extent(2)));
                    p_type ext3(0, std::min(dst.extent(3), src.extent(3)));
                    p_type ext4(0, std::min(dst.extent(4), src.extent(4)));
                    using sv_adapter_type = CommonSubtensor<DstType, SrcType, 5, p_type,
                            p_type, p_type, p_type, p_type>;
                    sv_adapter_type common_subtensor(dst, src, ext0, ext1, ext2, ext3, ext4);
                    tensor_copy(exec_space..., common_subtensor.dst_sub,
                              common_subtensor.src_sub);
                }
            }
        }
    };

    template<class DstType, class SrcType, class ExecSpace>
    struct TensorRemap<DstType, SrcType, ExecSpace, 6> {
        using p_type = flare::pair<int64_t, int64_t>;

        template<typename... OptExecSpace>
        TensorRemap(const DstType &dst, const SrcType &src,
                  const OptExecSpace &... exec_space) {
            static_assert(
                    sizeof...(OptExecSpace) <= 1,
                    "OptExecSpace must be either empty or be an execution space!");

            if (dst.extent(0) == src.extent(0)) {
                if (dst.extent(5) == src.extent(5)) {
                    p_type ext1(0, std::min(dst.extent(1), src.extent(1)));
                    p_type ext2(0, std::min(dst.extent(2), src.extent(2)));
                    p_type ext3(0, std::min(dst.extent(3), src.extent(3)));
                    p_type ext4(0, std::min(dst.extent(4), src.extent(4)));
                    using sv_adapter_type =
                            CommonSubtensor<DstType, SrcType, 6, flare::ALL_t, p_type, p_type,
                                    p_type, p_type, flare::ALL_t>;
                    sv_adapter_type common_subtensor(dst, src, flare::ALL, ext1, ext2, ext3,
                                                   ext4, flare::ALL);
                    tensor_copy(exec_space..., common_subtensor.dst_sub,
                              common_subtensor.src_sub);
                } else {
                    p_type ext1(0, std::min(dst.extent(1), src.extent(1)));
                    p_type ext2(0, std::min(dst.extent(2), src.extent(2)));
                    p_type ext3(0, std::min(dst.extent(3), src.extent(3)));
                    p_type ext4(0, std::min(dst.extent(4), src.extent(4)));
                    p_type ext5(0, std::min(dst.extent(5), src.extent(5)));
                    using sv_adapter_type =
                            CommonSubtensor<DstType, SrcType, 6, flare::ALL_t, p_type, p_type,
                                    p_type, p_type, p_type>;
                    sv_adapter_type common_subtensor(dst, src, flare::ALL, ext1, ext2, ext3,
                                                   ext4, ext5);
                    tensor_copy(exec_space..., common_subtensor.dst_sub,
                              common_subtensor.src_sub);
                }
            } else {
                if (dst.extent(5) == src.extent(5)) {
                    p_type ext0(0, std::min(dst.extent(0), src.extent(0)));
                    p_type ext1(0, std::min(dst.extent(1), src.extent(1)));
                    p_type ext2(0, std::min(dst.extent(2), src.extent(2)));
                    p_type ext3(0, std::min(dst.extent(3), src.extent(3)));
                    p_type ext4(0, std::min(dst.extent(4), src.extent(4)));

                    using sv_adapter_type =
                            CommonSubtensor<DstType, SrcType, 6, p_type, p_type, p_type, p_type,
                                    p_type, flare::ALL_t>;
                    sv_adapter_type common_subtensor(dst, src, ext0, ext1, ext2, ext3, ext4,
                                                   flare::ALL);
                    tensor_copy(exec_space..., common_subtensor.dst_sub,
                              common_subtensor.src_sub);
                } else {
                    p_type ext0(0, std::min(dst.extent(0), src.extent(0)));
                    p_type ext1(0, std::min(dst.extent(1), src.extent(1)));
                    p_type ext2(0, std::min(dst.extent(2), src.extent(2)));
                    p_type ext3(0, std::min(dst.extent(3), src.extent(3)));
                    p_type ext4(0, std::min(dst.extent(4), src.extent(4)));
                    p_type ext5(0, std::min(dst.extent(5), src.extent(5)));

                    using sv_adapter_type =
                            CommonSubtensor<DstType, SrcType, 6, p_type, p_type, p_type, p_type,
                                    p_type, p_type>;
                    sv_adapter_type common_subtensor(dst, src, ext0, ext1, ext2, ext3, ext4,
                                                   ext5);
                    tensor_copy(exec_space..., common_subtensor.dst_sub,
                              common_subtensor.src_sub);
                }
            }
        }
    };

    template<class DstType, class SrcType, class ExecSpace>
    struct TensorRemap<DstType, SrcType, ExecSpace, 7> {
        using p_type = flare::pair<int64_t, int64_t>;

        template<typename... OptExecSpace>
        TensorRemap(const DstType &dst, const SrcType &src,
                  const OptExecSpace &... exec_space) {
            static_assert(
                    sizeof...(OptExecSpace) <= 1,
                    "OptExecSpace must be either empty or be an execution space!");

            if (dst.extent(0) == src.extent(0)) {
                if (dst.extent(6) == src.extent(6)) {
                    p_type ext1(0, std::min(dst.extent(1), src.extent(1)));
                    p_type ext2(0, std::min(dst.extent(2), src.extent(2)));
                    p_type ext3(0, std::min(dst.extent(3), src.extent(3)));
                    p_type ext4(0, std::min(dst.extent(4), src.extent(4)));
                    p_type ext5(0, std::min(dst.extent(5), src.extent(5)));
                    using sv_adapter_type =
                            CommonSubtensor<DstType, SrcType, 7, flare::ALL_t, p_type, p_type,
                                    p_type, p_type, p_type, flare::ALL_t>;
                    sv_adapter_type common_subtensor(dst, src, flare::ALL, ext1, ext2, ext3,
                                                   ext4, ext5, flare::ALL);
                    tensor_copy(exec_space..., common_subtensor.dst_sub,
                              common_subtensor.src_sub);
                } else {
                    p_type ext1(0, std::min(dst.extent(1), src.extent(1)));
                    p_type ext2(0, std::min(dst.extent(2), src.extent(2)));
                    p_type ext3(0, std::min(dst.extent(3), src.extent(3)));
                    p_type ext4(0, std::min(dst.extent(4), src.extent(4)));
                    p_type ext5(0, std::min(dst.extent(5), src.extent(5)));
                    p_type ext6(0, std::min(dst.extent(6), src.extent(6)));
                    using sv_adapter_type =
                            CommonSubtensor<DstType, SrcType, 7, flare::ALL_t, p_type, p_type,
                                    p_type, p_type, p_type, p_type>;
                    sv_adapter_type common_subtensor(dst, src, flare::ALL, ext1, ext2, ext3,
                                                   ext4, ext5, ext6);
                    tensor_copy(exec_space..., common_subtensor.dst_sub,
                              common_subtensor.src_sub);
                }
            } else {
                if (dst.extent(6) == src.extent(6)) {
                    p_type ext0(0, std::min(dst.extent(0), src.extent(0)));
                    p_type ext1(0, std::min(dst.extent(1), src.extent(1)));
                    p_type ext2(0, std::min(dst.extent(2), src.extent(2)));
                    p_type ext3(0, std::min(dst.extent(3), src.extent(3)));
                    p_type ext4(0, std::min(dst.extent(4), src.extent(4)));
                    p_type ext5(0, std::min(dst.extent(5), src.extent(5)));
                    using sv_adapter_type =
                            CommonSubtensor<DstType, SrcType, 7, p_type, p_type, p_type, p_type,
                                    p_type, p_type, flare::ALL_t>;
                    sv_adapter_type common_subtensor(dst, src, ext0, ext1, ext2, ext3, ext4,
                                                   ext5, flare::ALL);
                    tensor_copy(exec_space..., common_subtensor.dst_sub,
                              common_subtensor.src_sub);
                } else {
                    p_type ext0(0, std::min(dst.extent(0), src.extent(0)));
                    p_type ext1(0, std::min(dst.extent(1), src.extent(1)));
                    p_type ext2(0, std::min(dst.extent(2), src.extent(2)));
                    p_type ext3(0, std::min(dst.extent(3), src.extent(3)));
                    p_type ext4(0, std::min(dst.extent(4), src.extent(4)));
                    p_type ext5(0, std::min(dst.extent(5), src.extent(5)));
                    p_type ext6(0, std::min(dst.extent(6), src.extent(6)));
                    using sv_adapter_type =
                            CommonSubtensor<DstType, SrcType, 7, p_type, p_type, p_type, p_type,
                                    p_type, p_type, p_type>;
                    sv_adapter_type common_subtensor(dst, src, ext0, ext1, ext2, ext3, ext4,
                                                   ext5, ext6);
                    tensor_copy(exec_space..., common_subtensor.dst_sub,
                              common_subtensor.src_sub);
                }
            }
        }
    };

    template<class DstType, class SrcType, class ExecSpace>
    struct TensorRemap<DstType, SrcType, ExecSpace, 8> {
        using p_type = flare::pair<int64_t, int64_t>;

        template<typename... OptExecSpace>
        TensorRemap(const DstType &dst, const SrcType &src,
                  const OptExecSpace &... exec_space) {
            static_assert(
                    sizeof...(OptExecSpace) <= 1,
                    "OptExecSpace must be either empty or be an execution space!");

            if (dst.extent(0) == src.extent(0)) {
                if (dst.extent(7) == src.extent(7)) {
                    p_type ext1(0, std::min(dst.extent(1), src.extent(1)));
                    p_type ext2(0, std::min(dst.extent(2), src.extent(2)));
                    p_type ext3(0, std::min(dst.extent(3), src.extent(3)));
                    p_type ext4(0, std::min(dst.extent(4), src.extent(4)));
                    p_type ext5(0, std::min(dst.extent(5), src.extent(5)));
                    p_type ext6(0, std::min(dst.extent(6), src.extent(6)));
                    using sv_adapter_type =
                            CommonSubtensor<DstType, SrcType, 8, flare::ALL_t, p_type, p_type,
                                    p_type, p_type, p_type, p_type, flare::ALL_t>;
                    sv_adapter_type common_subtensor(dst, src, flare::ALL, ext1, ext2, ext3,
                                                   ext4, ext5, ext6, flare::ALL);
                    tensor_copy(exec_space..., common_subtensor.dst_sub,
                              common_subtensor.src_sub);
                } else {
                    p_type ext1(0, std::min(dst.extent(1), src.extent(1)));
                    p_type ext2(0, std::min(dst.extent(2), src.extent(2)));
                    p_type ext3(0, std::min(dst.extent(3), src.extent(3)));
                    p_type ext4(0, std::min(dst.extent(4), src.extent(4)));
                    p_type ext5(0, std::min(dst.extent(5), src.extent(5)));
                    p_type ext6(0, std::min(dst.extent(6), src.extent(6)));
                    p_type ext7(0, std::min(dst.extent(7), src.extent(7)));
                    using sv_adapter_type =
                            CommonSubtensor<DstType, SrcType, 8, flare::ALL_t, p_type, p_type,
                                    p_type, p_type, p_type, p_type, p_type>;
                    sv_adapter_type common_subtensor(dst, src, flare::ALL, ext1, ext2, ext3,
                                                   ext4, ext5, ext6, ext7);
                    tensor_copy(exec_space..., common_subtensor.dst_sub,
                              common_subtensor.src_sub);
                }
            } else {
                if (dst.extent(7) == src.extent(7)) {
                    p_type ext0(0, std::min(dst.extent(0), src.extent(0)));
                    p_type ext1(0, std::min(dst.extent(1), src.extent(1)));
                    p_type ext2(0, std::min(dst.extent(2), src.extent(2)));
                    p_type ext3(0, std::min(dst.extent(3), src.extent(3)));
                    p_type ext4(0, std::min(dst.extent(4), src.extent(4)));
                    p_type ext5(0, std::min(dst.extent(5), src.extent(5)));
                    p_type ext6(0, std::min(dst.extent(6), src.extent(6)));
                    using sv_adapter_type =
                            CommonSubtensor<DstType, SrcType, 8, p_type, p_type, p_type, p_type,
                                    p_type, p_type, p_type, flare::ALL_t>;
                    sv_adapter_type common_subtensor(dst, src, ext0, ext1, ext2, ext3, ext4,
                                                   ext5, ext6, flare::ALL);
                    tensor_copy(exec_space..., common_subtensor.dst_sub,
                              common_subtensor.src_sub);
                } else {
                    p_type ext0(0, std::min(dst.extent(0), src.extent(0)));
                    p_type ext1(0, std::min(dst.extent(1), src.extent(1)));
                    p_type ext2(0, std::min(dst.extent(2), src.extent(2)));
                    p_type ext3(0, std::min(dst.extent(3), src.extent(3)));
                    p_type ext4(0, std::min(dst.extent(4), src.extent(4)));
                    p_type ext5(0, std::min(dst.extent(5), src.extent(5)));
                    p_type ext6(0, std::min(dst.extent(6), src.extent(6)));
                    p_type ext7(0, std::min(dst.extent(7), src.extent(7)));
                    using sv_adapter_type =
                            CommonSubtensor<DstType, SrcType, 8, p_type, p_type, p_type, p_type,
                                    p_type, p_type, p_type, p_type>;
                    sv_adapter_type common_subtensor(dst, src, ext0, ext1, ext2, ext3, ext4,
                                                   ext5, ext6, ext7);
                    tensor_copy(exec_space..., common_subtensor.dst_sub,
                              common_subtensor.src_sub);
                }
            }
        }
    };

    template<typename ExecutionSpace, class DT, class... DP>
    inline void contiguous_fill(
            const ExecutionSpace &exec_space, const Tensor<DT, DP...> &dst,
            typename TensorTraits<DT, DP...>::const_value_type &value) {
        using TensorType = Tensor<DT, DP...>;
        using TensorTypeFlat = flare::Tensor<
                typename TensorType::value_type *, flare::LayoutRight,
                flare::Device<typename TensorType::execution_space,
                        std::conditional_t<TensorType::rank == 0,
                                typename TensorType::memory_space,
                                flare::AnonymousSpace>>,
                flare::MemoryTraits<0>>;

        TensorTypeFlat dst_flat(dst.data(), dst.size());
        if (dst.span() < static_cast<size_t>(std::numeric_limits<int>::max())) {
            flare::detail::TensorFill<TensorTypeFlat, flare::LayoutRight, ExecutionSpace,
                    TensorTypeFlat::rank, int>(dst_flat, value,
                                             exec_space);
        } else
            flare::detail::TensorFill<TensorTypeFlat, flare::LayoutRight, ExecutionSpace,
                    TensorTypeFlat::rank, int64_t>(dst_flat, value,
                                                 exec_space);
    }

// Default implementation for execution spaces that don't provide a definition
    template<typename ExecutionSpace, class TensorType>
    struct ZeroMemset {
        ZeroMemset(const ExecutionSpace &exec_space, const TensorType &dst,
                   typename TensorType::const_value_type &value) {
            contiguous_fill(exec_space, dst, value);
        }

        ZeroMemset(const TensorType &dst, typename TensorType::const_value_type &value) {
            contiguous_fill(ExecutionSpace(), dst, value);
        }
    };

    template<typename ExecutionSpace, class DT, class... DP>
    inline std::enable_if_t<
            std::is_trivial<typename TensorTraits<DT, DP...>::value_type>::value &&
            std::is_trivially_copy_assignable<
                    typename TensorTraits<DT, DP...>::value_type>::value>
    contiguous_fill_or_memset(
            const ExecutionSpace &exec_space, const Tensor<DT, DP...> &dst,
            typename TensorTraits<DT, DP...>::const_value_type &value) {
// On A64FX memset seems to do the wrong thing with regards to first touch
// leading to the significant performance issues
#ifndef FLARE_ARCH_A64FX
        if (detail::is_zero_byte(value))
            ZeroMemset<ExecutionSpace, Tensor<DT, DP...>>(exec_space, dst, value);
        else
#endif
            contiguous_fill(exec_space, dst, value);
    }

    template<typename ExecutionSpace, class DT, class... DP>
    inline std::enable_if_t<
            !(std::is_trivial<typename TensorTraits<DT, DP...>::value_type>::value &&
              std::is_trivially_copy_assignable<
                      typename TensorTraits<DT, DP...>::value_type>::value)>
    contiguous_fill_or_memset(
            const ExecutionSpace &exec_space, const Tensor<DT, DP...> &dst,
            typename TensorTraits<DT, DP...>::const_value_type &value) {
        contiguous_fill(exec_space, dst, value);
    }

    template<class DT, class... DP>
    inline std::enable_if_t<
            std::is_trivial<typename TensorTraits<DT, DP...>::value_type>::value &&
            std::is_trivially_copy_assignable<
                    typename TensorTraits<DT, DP...>::value_type>::value>
    contiguous_fill_or_memset(
            const Tensor<DT, DP...> &dst,
            typename TensorTraits<DT, DP...>::const_value_type &value) {
        using TensorType = Tensor<DT, DP...>;
        using exec_space_type = typename TensorType::execution_space;

// On A64FX memset seems to do the wrong thing with regards to first touch
// leading to the significant performance issues
#ifndef FLARE_ARCH_A64FX
        if (detail::is_zero_byte(value))
            ZeroMemset<exec_space_type, Tensor<DT, DP...>>(dst, value);
        else
#endif
            contiguous_fill(exec_space_type(), dst, value);
    }

    template<class DT, class... DP>
    inline std::enable_if_t<
            !(std::is_trivial<typename TensorTraits<DT, DP...>::value_type>::value &&
              std::is_trivially_copy_assignable<
                      typename TensorTraits<DT, DP...>::value_type>::value)>
    contiguous_fill_or_memset(
            const Tensor<DT, DP...> &dst,
            typename TensorTraits<DT, DP...>::const_value_type &value) {
        using TensorType = Tensor<DT, DP...>;
        using exec_space_type = typename TensorType::execution_space;

        contiguous_fill(exec_space_type(), dst, value);
    }
}  // namespace flare::detail

namespace flare {
    /** \brief  Deep copy a value from Host memory into a tensor.  */
    template<class DT, class... DP>
    inline void deep_copy(
            const Tensor<DT, DP...> &dst,
            typename TensorTraits<DT, DP...>::const_value_type &value,
            std::enable_if_t<std::is_same<typename TensorTraits<DT, DP...>::specialize,
                    void>::value> * = nullptr) {
        using TensorType = Tensor<DT, DP...>;
        using exec_space_type = typename TensorType::execution_space;

        if (flare::Tools::experimental::get_callbacks().begin_deep_copy != nullptr) {
            flare::Profiling::beginDeepCopy(
                    flare::Profiling::make_space_handle(TensorType::memory_space::name()),
                    dst.label(), dst.data(),
                    flare::Profiling::make_space_handle(flare::HostSpace::name()),
                    "Scalar", &value, dst.span() * sizeof(typename TensorType::value_type));
        }

        if (dst.data() == nullptr) {
            flare::fence(
                    "flare::deep_copy: scalar copy, fence because destination is null");
            if (flare::Tools::experimental::get_callbacks().end_deep_copy != nullptr) {
                flare::Profiling::endDeepCopy();
            }
            return;
        }

        flare::fence("flare::deep_copy: scalar copy, pre copy fence");
        static_assert(std::is_same<typename TensorType::non_const_value_type,
                              typename TensorType::value_type>::value,
                      "deep_copy requires non-const type");

        // If contiguous we can simply do a 1D flat loop or use memset
        if (dst.span_is_contiguous()) {
            detail::contiguous_fill_or_memset(dst, value);
            flare::fence("flare::deep_copy: scalar copy, post copy fence");
            if (flare::Tools::experimental::get_callbacks().end_deep_copy != nullptr) {
                flare::Profiling::endDeepCopy();
            }
            return;
        }

        // Figure out iteration order to do the TensorFill
        int64_t strides[TensorType::rank + 1];
        dst.stride(strides);
        flare::Iterate iterate;
        if (std::is_same<typename TensorType::array_layout,
                flare::LayoutRight>::value) {
            iterate = flare::Iterate::Right;
        } else if (std::is_same<typename TensorType::array_layout,
                flare::LayoutLeft>::value) {
            iterate = flare::Iterate::Left;
        } else if (std::is_same<typename TensorType::array_layout,
                flare::LayoutStride>::value) {
            if (strides[0] > strides[TensorType::rank > 0 ? TensorType::rank - 1 : 0])
                iterate = flare::Iterate::Right;
            else
                iterate = flare::Iterate::Left;
        } else {
            if (std::is_same<typename TensorType::execution_space::array_layout,
                    flare::LayoutRight>::value)
                iterate = flare::Iterate::Right;
            else
                iterate = flare::Iterate::Left;
        }

        // Lets call the right TensorFill functor based on integer space needed and
        // iteration type
        using TensorTypeUniform =
                std::conditional_t<TensorType::rank == 0,
                        typename TensorType::uniform_runtime_type,
                        typename TensorType::uniform_runtime_nomemspace_type>;
        if (dst.span() > static_cast<size_t>(std::numeric_limits<int>::max())) {
            if (iterate == flare::Iterate::Right)
                flare::detail::TensorFill<TensorTypeUniform, flare::LayoutRight,
                        exec_space_type, TensorType::rank, int64_t>(
                        dst, value, exec_space_type());
            else
                flare::detail::TensorFill<TensorTypeUniform, flare::LayoutLeft,
                        exec_space_type, TensorType::rank, int64_t>(
                        dst, value, exec_space_type());
        } else {
            if (iterate == flare::Iterate::Right)
                flare::detail::TensorFill<TensorTypeUniform, flare::LayoutRight,
                        exec_space_type, TensorType::rank, int>(
                        dst, value, exec_space_type());
            else
                flare::detail::TensorFill<TensorTypeUniform, flare::LayoutLeft,
                        exec_space_type, TensorType::rank, int>(
                        dst, value, exec_space_type());
        }
        flare::fence("flare::deep_copy: scalar copy, post copy fence");

        if (flare::Tools::experimental::get_callbacks().end_deep_copy != nullptr) {
            flare::Profiling::endDeepCopy();
        }
    }

/** \brief  Deep copy into a value in Host memory from a tensor.  */
    template<class ST, class... SP>
    inline void deep_copy(
            typename TensorTraits<ST, SP...>::non_const_value_type &dst,
            const Tensor<ST, SP...> &src,
            std::enable_if_t<std::is_same<typename TensorTraits<ST, SP...>::specialize,
                    void>::value> * = nullptr) {
        using src_traits = TensorTraits<ST, SP...>;
        using src_memory_space = typename src_traits::memory_space;

        static_assert(src_traits::rank == 0,
                      "ERROR: Non-rank-zero tensor in deep_copy( value , Tensor )");

        if (flare::Tools::experimental::get_callbacks().begin_deep_copy != nullptr) {
            flare::Profiling::beginDeepCopy(
                    flare::Profiling::make_space_handle(flare::HostSpace::name()),
                    "Scalar", &dst,
                    flare::Profiling::make_space_handle(src_memory_space::name()),
                    src.label(), src.data(),
                    src.span() * sizeof(typename src_traits::value_type));
        }

        if (src.data() == nullptr) {
            flare::fence("flare::deep_copy: copy into scalar, src is null");
        } else {
            flare::fence("flare::deep_copy: copy into scalar, pre copy fence");
            flare::detail::DeepCopy<HostSpace, src_memory_space>(&dst, src.data(),
                                                                 sizeof(ST));
            flare::fence("flare::deep_copy: copy into scalar, post copy fence");
        }

        if (flare::Tools::experimental::get_callbacks().end_deep_copy != nullptr) {
            flare::Profiling::endDeepCopy();
        }
    }

//----------------------------------------------------------------------------
/** \brief  A deep copy between tensors of compatible type, and rank zero.  */
    template<class DT, class... DP, class ST, class... SP>
    inline void deep_copy(
            const Tensor<DT, DP...> &dst, const Tensor<ST, SP...> &src,
            std::enable_if_t<
                    (std::is_void<typename TensorTraits<DT, DP...>::specialize>::value &&
                     std::is_void<typename TensorTraits<ST, SP...>::specialize>::value &&
                     (unsigned(TensorTraits<DT, DP...>::rank) == unsigned(0) &&
                      unsigned(TensorTraits<ST, SP...>::rank) == unsigned(0)))> * = nullptr) {
        using dst_type = Tensor<DT, DP...>;
        using src_type = Tensor<ST, SP...>;

        using value_type = typename dst_type::value_type;
        using dst_memory_space = typename dst_type::memory_space;
        using src_memory_space = typename src_type::memory_space;

        static_assert(std::is_same<typename dst_type::value_type,
                              typename src_type::non_const_value_type>::value,
                      "deep_copy requires matching non-const destination type");

        if (flare::Tools::experimental::get_callbacks().begin_deep_copy != nullptr) {
            flare::Profiling::beginDeepCopy(
                    flare::Profiling::make_space_handle(dst_memory_space::name()),
                    dst.label(), dst.data(),
                    flare::Profiling::make_space_handle(src_memory_space::name()),
                    src.label(), src.data(),
                    src.span() * sizeof(typename dst_type::value_type));
        }

        if (dst.data() == nullptr && src.data() == nullptr) {
            flare::fence(
                    "flare::deep_copy: scalar to scalar copy, both pointers null");
            if (flare::Tools::experimental::get_callbacks().end_deep_copy != nullptr) {
                flare::Profiling::endDeepCopy();
            }
            return;
        }

        flare::fence("flare::deep_copy: scalar to scalar copy, pre copy fence");
        if (dst.data() != src.data()) {
            flare::detail::DeepCopy<dst_memory_space, src_memory_space>(
                    dst.data(), src.data(), sizeof(value_type));
            flare::fence("flare::deep_copy: scalar to scalar copy, post copy fence");
        }
        if (flare::Tools::experimental::get_callbacks().end_deep_copy != nullptr) {
            flare::Profiling::endDeepCopy();
        }
    }

//----------------------------------------------------------------------------
/** \brief  A deep copy between tensors of the default specialization, compatible
 * type, same non-zero rank, same contiguous layout.
 */
    template<class DT, class... DP, class ST, class... SP>
    inline void deep_copy(
            const Tensor<DT, DP...> &dst, const Tensor<ST, SP...> &src,
            std::enable_if_t<
                    (std::is_void<typename TensorTraits<DT, DP...>::specialize>::value &&
                     std::is_void<typename TensorTraits<ST, SP...>::specialize>::value &&
                     (unsigned(TensorTraits<DT, DP...>::rank) != 0 ||
                      unsigned(TensorTraits<ST, SP...>::rank) != 0))> * = nullptr) {
        using dst_type = Tensor<DT, DP...>;
        using src_type = Tensor<ST, SP...>;
        using dst_execution_space = typename dst_type::execution_space;
        using src_execution_space = typename src_type::execution_space;
        using dst_memory_space = typename dst_type::memory_space;
        using src_memory_space = typename src_type::memory_space;
        using dst_value_type = typename dst_type::value_type;
        using src_value_type = typename src_type::value_type;

        static_assert(std::is_same<typename dst_type::value_type,
                              typename dst_type::non_const_value_type>::value,
                      "deep_copy requires non-const destination type");

        static_assert((unsigned(dst_type::rank) == unsigned(src_type::rank)),
                      "deep_copy requires Tensors of equal rank");

        if (flare::Tools::experimental::get_callbacks().begin_deep_copy != nullptr) {
            flare::Profiling::beginDeepCopy(
                    flare::Profiling::make_space_handle(dst_memory_space::name()),
                    dst.label(), dst.data(),
                    flare::Profiling::make_space_handle(src_memory_space::name()),
                    src.label(), src.data(),
                    src.span() * sizeof(typename dst_type::value_type));
        }

        if (dst.data() == nullptr || src.data() == nullptr) {
            // throw if dimension mismatch
            if ((src.extent(0) != dst.extent(0)) || (src.extent(1) != dst.extent(1)) ||
                (src.extent(2) != dst.extent(2)) || (src.extent(3) != dst.extent(3)) ||
                (src.extent(4) != dst.extent(4)) || (src.extent(5) != dst.extent(5)) ||
                (src.extent(6) != dst.extent(6)) || (src.extent(7) != dst.extent(7))) {
                std::string message(
                        "Deprecation Error: flare::deep_copy extents of tensors don't "
                        "match: ");
                message += dst.label();
                message += "(";
                message += std::to_string(dst.extent(0));
                for (size_t r = 1; r < dst_type::rank; r++) {
                    message += ",";
                    message += std::to_string(dst.extent(r));
                }
                message += ") ";
                message += src.label();
                message += "(";
                message += std::to_string(src.extent(0));
                for (size_t r = 1; r < src_type::rank; r++) {
                    message += ",";
                    message += std::to_string(src.extent(r));
                }
                message += ") ";

                flare::detail::throw_runtime_exception(message);
            }
            flare::fence(
                    "flare::deep_copy: copy between contiguous tensors, fence due to null "
                    "argument");
            if (flare::Tools::experimental::get_callbacks().end_deep_copy != nullptr) {
                flare::Profiling::endDeepCopy();
            }
            return;
        }

        enum {
            DstExecCanAccessSrc =
            flare::SpaceAccessibility<dst_execution_space,
                    src_memory_space>::accessible
        };

        enum {
            SrcExecCanAccessDst =
            flare::SpaceAccessibility<src_execution_space,
                    dst_memory_space>::accessible
        };

        // Checking for Overlapping Tensors.
        dst_value_type *dst_start = dst.data();
        dst_value_type *dst_end = dst.data() + dst.span();
        src_value_type *src_start = src.data();
        src_value_type *src_end = src.data() + src.span();
        if (((std::ptrdiff_t) dst_start == (std::ptrdiff_t) src_start) &&
            ((std::ptrdiff_t) dst_end == (std::ptrdiff_t) src_end) &&
            (dst.span_is_contiguous() && src.span_is_contiguous())) {
            flare::fence(
                    "flare::deep_copy: copy between contiguous tensors, fence due to same "
                    "spans");
            if (flare::Tools::experimental::get_callbacks().end_deep_copy != nullptr) {
                flare::Profiling::endDeepCopy();
            }
            return;
        }

        if ((((std::ptrdiff_t) dst_start < (std::ptrdiff_t) src_end) &&
             ((std::ptrdiff_t) dst_end > (std::ptrdiff_t) src_start)) &&
            ((dst.span_is_contiguous() && src.span_is_contiguous()))) {
            std::string message("Error: flare::deep_copy of overlapping tensors: ");
            message += dst.label();
            message += "(";
            message += std::to_string((std::ptrdiff_t) dst_start);
            message += ",";
            message += std::to_string((std::ptrdiff_t) dst_end);
            message += ") ";
            message += src.label();
            message += "(";
            message += std::to_string((std::ptrdiff_t) src_start);
            message += ",";
            message += std::to_string((std::ptrdiff_t) src_end);
            message += ") ";
            flare::detail::throw_runtime_exception(message);
        }

        // Check for same extents
        if ((src.extent(0) != dst.extent(0)) || (src.extent(1) != dst.extent(1)) ||
            (src.extent(2) != dst.extent(2)) || (src.extent(3) != dst.extent(3)) ||
            (src.extent(4) != dst.extent(4)) || (src.extent(5) != dst.extent(5)) ||
            (src.extent(6) != dst.extent(6)) || (src.extent(7) != dst.extent(7))) {
            std::string message(
                    "Deprecation Error: flare::deep_copy extents of tensors don't match: ");
            message += dst.label();
            message += "(";
            message += std::to_string(dst.extent(0));
            for (size_t r = 1; r < dst_type::rank; r++) {
                message += ",";
                message += std::to_string(dst.extent(r));
            }
            message += ") ";
            message += src.label();
            message += "(";
            message += std::to_string(src.extent(0));
            for (size_t r = 1; r < src_type::rank; r++) {
                message += ",";
                message += std::to_string(src.extent(r));
            }
            message += ") ";

            flare::detail::throw_runtime_exception(message);
        }

        // If same type, equal layout, equal dimensions, equal span, and contiguous
        // memory then can byte-wise copy

        if (std::is_same<typename dst_type::value_type,
                typename src_type::non_const_value_type>::value &&
            (std::is_same<typename dst_type::array_layout,
                    typename src_type::array_layout>::value ||
             (dst_type::rank == 1 && src_type::rank == 1)) &&
            dst.span_is_contiguous() && src.span_is_contiguous() &&
            ((dst_type::rank < 1) || (dst.stride_0() == src.stride_0())) &&
            ((dst_type::rank < 2) || (dst.stride_1() == src.stride_1())) &&
            ((dst_type::rank < 3) || (dst.stride_2() == src.stride_2())) &&
            ((dst_type::rank < 4) || (dst.stride_3() == src.stride_3())) &&
            ((dst_type::rank < 5) || (dst.stride_4() == src.stride_4())) &&
            ((dst_type::rank < 6) || (dst.stride_5() == src.stride_5())) &&
            ((dst_type::rank < 7) || (dst.stride_6() == src.stride_6())) &&
            ((dst_type::rank < 8) || (dst.stride_7() == src.stride_7()))) {
            const size_t nbytes = sizeof(typename dst_type::value_type) * dst.span();
            flare::fence(
                    "flare::deep_copy: copy between contiguous tensors, pre tensor equality "
                    "check");
            if ((void *) dst.data() != (void *) src.data() && 0 < nbytes) {
                flare::detail::DeepCopy<dst_memory_space, src_memory_space>(
                        dst.data(), src.data(), nbytes);
                flare::fence(
                        "flare::deep_copy: copy between contiguous tensors, post deep copy "
                        "fence");
            }
        } else {
            flare::fence(
                    "flare::deep_copy: copy between contiguous tensors, pre copy fence");
            detail::tensor_copy(dst, src);
            flare::fence(
                    "flare::deep_copy: copy between contiguous tensors, post copy fence");
        }
        if (flare::Tools::experimental::get_callbacks().end_deep_copy != nullptr) {
            flare::Profiling::endDeepCopy();
        }
    }

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
    namespace experimental {
/** \brief  A local deep copy between tensors of the default specialization,
 * compatible type, same non-zero rank.
 */
        template<class TeamType, class DT, class... DP, class ST, class... SP>
        void FLARE_INLINE_FUNCTION
        local_deep_copy_contiguous(const TeamType &team, const Tensor<DT, DP...> &dst,
                                   const Tensor<ST, SP...> &src) {
            flare::parallel_for(flare::TeamVectorRange(team, src.span()),
                                [&](const int &i) { dst.data()[i] = src.data()[i]; });
        }

//----------------------------------------------------------------------------
        template<class DT, class... DP, class ST, class... SP>
        void FLARE_INLINE_FUNCTION local_deep_copy_contiguous(
                const Tensor<DT, DP...> &dst, const Tensor<ST, SP...> &src) {
            for (size_t i = 0; i < src.span(); ++i) {
                dst.data()[i] = src.data()[i];
            }
        }

//----------------------------------------------------------------------------
        template<class TeamType, class DT, class... DP, class ST, class... SP>
        void FLARE_INLINE_FUNCTION local_deep_copy(
                const TeamType &team, const Tensor<DT, DP...> &dst,
                const Tensor<ST, SP...> &src,
                std::enable_if_t<(unsigned(TensorTraits<DT, DP...>::rank) == 1 &&
                                  unsigned(TensorTraits<ST, SP...>::rank) == 1)> * = nullptr) {
            if (dst.data() == nullptr) {
                return;
            }

            const size_t N = dst.extent(0);

            team.team_barrier();
            flare::parallel_for(flare::TeamVectorRange(team, N),
                                [&](const int &i) { dst(i) = src(i); });
            team.team_barrier();
        }

//----------------------------------------------------------------------------
        template<class TeamType, class DT, class... DP, class ST, class... SP>
        void FLARE_INLINE_FUNCTION local_deep_copy(
                const TeamType &team, const Tensor<DT, DP...> &dst,
                const Tensor<ST, SP...> &src,
                std::enable_if_t<(unsigned(TensorTraits<DT, DP...>::rank) == 2 &&
                                  unsigned(TensorTraits<ST, SP...>::rank) == 2)> * = nullptr) {
            if (dst.data() == nullptr) {
                return;
            }

            const size_t N = dst.extent(0) * dst.extent(1);

            if (dst.span_is_contiguous() && src.span_is_contiguous()) {
                team.team_barrier();
                local_deep_copy_contiguous(team, dst, src);
                team.team_barrier();
            } else {
                team.team_barrier();
                flare::parallel_for(flare::TeamVectorRange(team, N), [&](const int &i) {
                    int i0 = i % dst.extent(0);
                    int i1 = i / dst.extent(0);
                    dst(i0, i1) = src(i0, i1);
                });
                team.team_barrier();
            }
        }

//----------------------------------------------------------------------------
        template<class TeamType, class DT, class... DP, class ST, class... SP>
        void FLARE_INLINE_FUNCTION local_deep_copy(
                const TeamType &team, const Tensor<DT, DP...> &dst,
                const Tensor<ST, SP...> &src,
                std::enable_if_t<(unsigned(TensorTraits<DT, DP...>::rank) == 3 &&
                                  unsigned(TensorTraits<ST, SP...>::rank) == 3)> * = nullptr) {
            if (dst.data() == nullptr) {
                return;
            }

            const size_t N = dst.extent(0) * dst.extent(1) * dst.extent(2);

            if (dst.span_is_contiguous() && src.span_is_contiguous()) {
                team.team_barrier();
                local_deep_copy_contiguous(team, dst, src);
                team.team_barrier();
            } else {
                team.team_barrier();
                flare::parallel_for(flare::TeamVectorRange(team, N), [&](const int &i) {
                    int i0 = i % dst.extent(0);
                    int itmp = i / dst.extent(0);
                    int i1 = itmp % dst.extent(1);
                    int i2 = itmp / dst.extent(1);
                    dst(i0, i1, i2) = src(i0, i1, i2);
                });
                team.team_barrier();
            }
        }

//----------------------------------------------------------------------------
        template<class TeamType, class DT, class... DP, class ST, class... SP>
        void FLARE_INLINE_FUNCTION local_deep_copy(
                const TeamType &team, const Tensor<DT, DP...> &dst,
                const Tensor<ST, SP...> &src,
                std::enable_if_t<(unsigned(TensorTraits<DT, DP...>::rank) == 4 &&
                                  unsigned(TensorTraits<ST, SP...>::rank) == 4)> * = nullptr) {
            if (dst.data() == nullptr) {
                return;
            }

            const size_t N =
                    dst.extent(0) * dst.extent(1) * dst.extent(2) * dst.extent(3);

            if (dst.span_is_contiguous() && src.span_is_contiguous()) {
                team.team_barrier();
                local_deep_copy_contiguous(team, dst, src);
                team.team_barrier();
            } else {
                team.team_barrier();
                flare::parallel_for(flare::TeamVectorRange(team, N), [&](const int &i) {
                    int i0 = i % dst.extent(0);
                    int itmp = i / dst.extent(0);
                    int i1 = itmp % dst.extent(1);
                    itmp = itmp / dst.extent(1);
                    int i2 = itmp % dst.extent(2);
                    int i3 = itmp / dst.extent(2);
                    dst(i0, i1, i2, i3) = src(i0, i1, i2, i3);
                });
                team.team_barrier();
            }
        }

//----------------------------------------------------------------------------
        template<class TeamType, class DT, class... DP, class ST, class... SP>
        void FLARE_INLINE_FUNCTION local_deep_copy(
                const TeamType &team, const Tensor<DT, DP...> &dst,
                const Tensor<ST, SP...> &src,
                std::enable_if_t<(unsigned(TensorTraits<DT, DP...>::rank) == 5 &&
                                  unsigned(TensorTraits<ST, SP...>::rank) == 5)> * = nullptr) {
            if (dst.data() == nullptr) {
                return;
            }

            const size_t N = dst.extent(0) * dst.extent(1) * dst.extent(2) *
                             dst.extent(3) * dst.extent(4);

            if (dst.span_is_contiguous() && src.span_is_contiguous()) {
                team.team_barrier();
                local_deep_copy_contiguous(team, dst, src);
                team.team_barrier();
            } else {
                team.team_barrier();
                flare::parallel_for(flare::TeamVectorRange(team, N), [&](const int &i) {
                    int i0 = i % dst.extent(0);
                    int itmp = i / dst.extent(0);
                    int i1 = itmp % dst.extent(1);
                    itmp = itmp / dst.extent(1);
                    int i2 = itmp % dst.extent(2);
                    itmp = itmp / dst.extent(2);
                    int i3 = itmp % dst.extent(3);
                    int i4 = itmp / dst.extent(3);
                    dst(i0, i1, i2, i3, i4) = src(i0, i1, i2, i3, i4);
                });
                team.team_barrier();
            }
        }

//----------------------------------------------------------------------------
        template<class TeamType, class DT, class... DP, class ST, class... SP>
        void FLARE_INLINE_FUNCTION local_deep_copy(
                const TeamType &team, const Tensor<DT, DP...> &dst,
                const Tensor<ST, SP...> &src,
                std::enable_if_t<(unsigned(TensorTraits<DT, DP...>::rank) == 6 &&
                                  unsigned(TensorTraits<ST, SP...>::rank) == 6)> * = nullptr) {
            if (dst.data() == nullptr) {
                return;
            }

            const size_t N = dst.extent(0) * dst.extent(1) * dst.extent(2) *
                             dst.extent(3) * dst.extent(4) * dst.extent(5);

            if (dst.span_is_contiguous() && src.span_is_contiguous()) {
                team.team_barrier();
                local_deep_copy_contiguous(team, dst, src);
                team.team_barrier();
            } else {
                team.team_barrier();
                flare::parallel_for(flare::TeamVectorRange(team, N), [&](const int &i) {
                    int i0 = i % dst.extent(0);
                    int itmp = i / dst.extent(0);
                    int i1 = itmp % dst.extent(1);
                    itmp = itmp / dst.extent(1);
                    int i2 = itmp % dst.extent(2);
                    itmp = itmp / dst.extent(2);
                    int i3 = itmp % dst.extent(3);
                    itmp = itmp / dst.extent(3);
                    int i4 = itmp % dst.extent(4);
                    int i5 = itmp / dst.extent(4);
                    dst(i0, i1, i2, i3, i4, i5) = src(i0, i1, i2, i3, i4, i5);
                });
                team.team_barrier();
            }
        }

//----------------------------------------------------------------------------
        template<class TeamType, class DT, class... DP, class ST, class... SP>
        void FLARE_INLINE_FUNCTION local_deep_copy(
                const TeamType &team, const Tensor<DT, DP...> &dst,
                const Tensor<ST, SP...> &src,
                std::enable_if_t<(unsigned(TensorTraits<DT, DP...>::rank) == 7 &&
                                  unsigned(TensorTraits<ST, SP...>::rank) == 7)> * = nullptr) {
            if (dst.data() == nullptr) {
                return;
            }

            const size_t N = dst.extent(0) * dst.extent(1) * dst.extent(2) *
                             dst.extent(3) * dst.extent(4) * dst.extent(5) *
                             dst.extent(6);

            if (dst.span_is_contiguous() && src.span_is_contiguous()) {
                team.team_barrier();
                local_deep_copy_contiguous(team, dst, src);
                team.team_barrier();
            } else {
                team.team_barrier();
                flare::parallel_for(flare::TeamVectorRange(team, N), [&](const int &i) {
                    int i0 = i % dst.extent(0);
                    int itmp = i / dst.extent(0);
                    int i1 = itmp % dst.extent(1);
                    itmp = itmp / dst.extent(1);
                    int i2 = itmp % dst.extent(2);
                    itmp = itmp / dst.extent(2);
                    int i3 = itmp % dst.extent(3);
                    itmp = itmp / dst.extent(3);
                    int i4 = itmp % dst.extent(4);
                    itmp = itmp / dst.extent(4);
                    int i5 = itmp % dst.extent(5);
                    int i6 = itmp / dst.extent(5);
                    dst(i0, i1, i2, i3, i4, i5, i6) = src(i0, i1, i2, i3, i4, i5, i6);
                });
                team.team_barrier();
            }
        }

//----------------------------------------------------------------------------
        template<class DT, class... DP, class ST, class... SP>
        void FLARE_INLINE_FUNCTION local_deep_copy(
                const Tensor<DT, DP...> &dst, const Tensor<ST, SP...> &src,
                std::enable_if_t<(unsigned(TensorTraits<DT, DP...>::rank) == 1 &&
                                  unsigned(TensorTraits<ST, SP...>::rank) == 1)> * = nullptr) {
            if (dst.data() == nullptr) {
                return;
            }

            const size_t N = dst.extent(0);

            for (size_t i = 0; i < N; ++i) {
                dst(i) = src(i);
            }
        }

//----------------------------------------------------------------------------
        template<class DT, class... DP, class ST, class... SP>
        void FLARE_INLINE_FUNCTION local_deep_copy(
                const Tensor<DT, DP...> &dst, const Tensor<ST, SP...> &src,
                std::enable_if_t<(unsigned(TensorTraits<DT, DP...>::rank) == 2 &&
                                  unsigned(TensorTraits<ST, SP...>::rank) == 2)> * = nullptr) {
            if (dst.data() == nullptr) {
                return;
            }

            if (dst.span_is_contiguous() && src.span_is_contiguous()) {
                local_deep_copy_contiguous(dst, src);
            } else {
                for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
                    for (size_t i1 = 0; i1 < dst.extent(1); ++i1) dst(i0, i1) = src(i0, i1);
            }
        }

//----------------------------------------------------------------------------
        template<class DT, class... DP, class ST, class... SP>
        void FLARE_INLINE_FUNCTION local_deep_copy(
                const Tensor<DT, DP...> &dst, const Tensor<ST, SP...> &src,
                std::enable_if_t<(unsigned(TensorTraits<DT, DP...>::rank) == 3 &&
                                  unsigned(TensorTraits<ST, SP...>::rank) == 3)> * = nullptr) {
            if (dst.data() == nullptr) {
                return;
            }

            if (dst.span_is_contiguous() && src.span_is_contiguous()) {
                local_deep_copy_contiguous(dst, src);
            } else {
                for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
                    for (size_t i1 = 0; i1 < dst.extent(1); ++i1)
                        for (size_t i2 = 0; i2 < dst.extent(2); ++i2)
                            dst(i0, i1, i2) = src(i0, i1, i2);
            }
        }

//----------------------------------------------------------------------------
        template<class DT, class... DP, class ST, class... SP>
        void FLARE_INLINE_FUNCTION local_deep_copy(
                const Tensor<DT, DP...> &dst, const Tensor<ST, SP...> &src,
                std::enable_if_t<(unsigned(TensorTraits<DT, DP...>::rank) == 4 &&
                                  unsigned(TensorTraits<ST, SP...>::rank) == 4)> * = nullptr) {
            if (dst.data() == nullptr) {
                return;
            }

            if (dst.span_is_contiguous() && src.span_is_contiguous()) {
                local_deep_copy_contiguous(dst, src);
            } else {
                for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
                    for (size_t i1 = 0; i1 < dst.extent(1); ++i1)
                        for (size_t i2 = 0; i2 < dst.extent(2); ++i2)
                            for (size_t i3 = 0; i3 < dst.extent(3); ++i3)
                                dst(i0, i1, i2, i3) = src(i0, i1, i2, i3);
            }
        }

//----------------------------------------------------------------------------
        template<class DT, class... DP, class ST, class... SP>
        void FLARE_INLINE_FUNCTION local_deep_copy(
                const Tensor<DT, DP...> &dst, const Tensor<ST, SP...> &src,
                std::enable_if_t<(unsigned(TensorTraits<DT, DP...>::rank) == 5 &&
                                  unsigned(TensorTraits<ST, SP...>::rank) == 5)> * = nullptr) {
            if (dst.data() == nullptr) {
                return;
            }

            if (dst.span_is_contiguous() && src.span_is_contiguous()) {
                local_deep_copy_contiguous(dst, src);
            } else {
                for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
                    for (size_t i1 = 0; i1 < dst.extent(1); ++i1)
                        for (size_t i2 = 0; i2 < dst.extent(2); ++i2)
                            for (size_t i3 = 0; i3 < dst.extent(3); ++i3)
                                for (size_t i4 = 0; i4 < dst.extent(4); ++i4)
                                    dst(i0, i1, i2, i3, i4) = src(i0, i1, i2, i3, i4);
            }
        }

//----------------------------------------------------------------------------
        template<class DT, class... DP, class ST, class... SP>
        void FLARE_INLINE_FUNCTION local_deep_copy(
                const Tensor<DT, DP...> &dst, const Tensor<ST, SP...> &src,
                std::enable_if_t<(unsigned(TensorTraits<DT, DP...>::rank) == 6 &&
                                  unsigned(TensorTraits<ST, SP...>::rank) == 6)> * = nullptr) {
            if (dst.data() == nullptr) {
                return;
            }

            if (dst.span_is_contiguous() && src.span_is_contiguous()) {
                local_deep_copy_contiguous(dst, src);
            } else {
                for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
                    for (size_t i1 = 0; i1 < dst.extent(1); ++i1)
                        for (size_t i2 = 0; i2 < dst.extent(2); ++i2)
                            for (size_t i3 = 0; i3 < dst.extent(3); ++i3)
                                for (size_t i4 = 0; i4 < dst.extent(4); ++i4)
                                    for (size_t i5 = 0; i5 < dst.extent(5); ++i5)
                                        dst(i0, i1, i2, i3, i4, i5) = src(i0, i1, i2, i3, i4, i5);
            }
        }

//----------------------------------------------------------------------------
        template<class DT, class... DP, class ST, class... SP>
        void FLARE_INLINE_FUNCTION local_deep_copy(
                const Tensor<DT, DP...> &dst, const Tensor<ST, SP...> &src,
                std::enable_if_t<(unsigned(TensorTraits<DT, DP...>::rank) == 7 &&
                                  unsigned(TensorTraits<ST, SP...>::rank) == 7)> * = nullptr) {
            if (dst.data() == nullptr) {
                return;
            }

            if (dst.span_is_contiguous() && src.span_is_contiguous()) {
                local_deep_copy_contiguous(dst, src);
            } else {
                for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
                    for (size_t i1 = 0; i1 < dst.extent(1); ++i1)
                        for (size_t i2 = 0; i2 < dst.extent(2); ++i2)
                            for (size_t i3 = 0; i3 < dst.extent(3); ++i3)
                                for (size_t i4 = 0; i4 < dst.extent(4); ++i4)
                                    for (size_t i5 = 0; i5 < dst.extent(5); ++i5)
                                        for (size_t i6 = 0; i6 < dst.extent(6); ++i6)
                                            dst(i0, i1, i2, i3, i4, i5, i6) =
                                                    src(i0, i1, i2, i3, i4, i5, i6);
            }
        }
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
/** \brief  Deep copy a value into a tensor.  */
        template<class TeamType, class DT, class... DP>
        void FLARE_INLINE_FUNCTION local_deep_copy_contiguous(
                const TeamType &team, const Tensor<DT, DP...> &dst,
                typename TensorTraits<DT, DP...>::const_value_type &value,
                std::enable_if_t<std::is_same<typename TensorTraits<DT, DP...>::specialize,
                        void>::value> * = nullptr) {
            flare::parallel_for(flare::TeamVectorRange(team, dst.span()),
                                [&](const int &i) { dst.data()[i] = value; });
        }

//----------------------------------------------------------------------------
        template<class DT, class... DP>
        void FLARE_INLINE_FUNCTION local_deep_copy_contiguous(
                const Tensor<DT, DP...> &dst,
                typename TensorTraits<DT, DP...>::const_value_type &value,
                std::enable_if_t<std::is_same<typename TensorTraits<DT, DP...>::specialize,
                        void>::value> * = nullptr) {
            for (size_t i = 0; i < dst.span(); ++i) {
                dst.data()[i] = value;
            }
        }

//----------------------------------------------------------------------------
        template<class TeamType, class DT, class... DP>
        void FLARE_INLINE_FUNCTION local_deep_copy(
                const TeamType &team, const Tensor<DT, DP...> &dst,
                typename TensorTraits<DT, DP...>::const_value_type &value,
                std::enable_if_t<(unsigned(TensorTraits<DT, DP...>::rank) == 1)> * = nullptr) {
            if (dst.data() == nullptr) {
                return;
            }

            const size_t N = dst.extent(0);

            team.team_barrier();
            flare::parallel_for(flare::TeamVectorRange(team, N),
                                [&](const int &i) { dst(i) = value; });
            team.team_barrier();
        }

//----------------------------------------------------------------------------
        template<class TeamType, class DT, class... DP>
        void FLARE_INLINE_FUNCTION local_deep_copy(
                const TeamType &team, const Tensor<DT, DP...> &dst,
                typename TensorTraits<DT, DP...>::const_value_type &value,
                std::enable_if_t<(unsigned(TensorTraits<DT, DP...>::rank) == 2)> * = nullptr) {
            if (dst.data() == nullptr) {
                return;
            }

            const size_t N = dst.extent(0) * dst.extent(1);

            if (dst.span_is_contiguous()) {
                team.team_barrier();
                local_deep_copy_contiguous(team, dst, value);
                team.team_barrier();
            } else {
                team.team_barrier();
                flare::parallel_for(flare::TeamVectorRange(team, N), [&](const int &i) {
                    int i0 = i % dst.extent(0);
                    int i1 = i / dst.extent(0);
                    dst(i0, i1) = value;
                });
                team.team_barrier();
            }
        }

//----------------------------------------------------------------------------
        template<class TeamType, class DT, class... DP>
        void FLARE_INLINE_FUNCTION local_deep_copy(
                const TeamType &team, const Tensor<DT, DP...> &dst,
                typename TensorTraits<DT, DP...>::const_value_type &value,
                std::enable_if_t<(unsigned(TensorTraits<DT, DP...>::rank) == 3)> * = nullptr) {
            if (dst.data() == nullptr) {
                return;
            }

            const size_t N = dst.extent(0) * dst.extent(1) * dst.extent(2);

            if (dst.span_is_contiguous()) {
                team.team_barrier();
                local_deep_copy_contiguous(team, dst, value);
                team.team_barrier();
            } else {
                team.team_barrier();
                flare::parallel_for(flare::TeamVectorRange(team, N), [&](const int &i) {
                    int i0 = i % dst.extent(0);
                    int itmp = i / dst.extent(0);
                    int i1 = itmp % dst.extent(1);
                    int i2 = itmp / dst.extent(1);
                    dst(i0, i1, i2) = value;
                });
                team.team_barrier();
            }
        }

//----------------------------------------------------------------------------
        template<class TeamType, class DT, class... DP>
        void FLARE_INLINE_FUNCTION local_deep_copy(
                const TeamType &team, const Tensor<DT, DP...> &dst,
                typename TensorTraits<DT, DP...>::const_value_type &value,
                std::enable_if_t<(unsigned(TensorTraits<DT, DP...>::rank) == 4)> * = nullptr) {
            if (dst.data() == nullptr) {
                return;
            }

            const size_t N =
                    dst.extent(0) * dst.extent(1) * dst.extent(2) * dst.extent(3);

            if (dst.span_is_contiguous()) {
                team.team_barrier();
                local_deep_copy_contiguous(team, dst, value);
                team.team_barrier();
            } else {
                team.team_barrier();
                flare::parallel_for(flare::TeamVectorRange(team, N), [&](const int &i) {
                    int i0 = i % dst.extent(0);
                    int itmp = i / dst.extent(0);
                    int i1 = itmp % dst.extent(1);
                    itmp = itmp / dst.extent(1);
                    int i2 = itmp % dst.extent(2);
                    int i3 = itmp / dst.extent(2);
                    dst(i0, i1, i2, i3) = value;
                });
                team.team_barrier();
            }
        }

//----------------------------------------------------------------------------
        template<class TeamType, class DT, class... DP>
        void FLARE_INLINE_FUNCTION local_deep_copy(
                const TeamType &team, const Tensor<DT, DP...> &dst,
                typename TensorTraits<DT, DP...>::const_value_type &value,
                std::enable_if_t<(unsigned(TensorTraits<DT, DP...>::rank) == 5)> * = nullptr) {
            if (dst.data() == nullptr) {
                return;
            }

            const size_t N = dst.extent(0) * dst.extent(1) * dst.extent(2) *
                             dst.extent(3) * dst.extent(4);

            if (dst.span_is_contiguous()) {
                team.team_barrier();
                local_deep_copy_contiguous(team, dst, value);
                team.team_barrier();
            } else {
                team.team_barrier();
                flare::parallel_for(flare::TeamVectorRange(team, N), [&](const int &i) {
                    int i0 = i % dst.extent(0);
                    int itmp = i / dst.extent(0);
                    int i1 = itmp % dst.extent(1);
                    itmp = itmp / dst.extent(1);
                    int i2 = itmp % dst.extent(2);
                    itmp = itmp / dst.extent(2);
                    int i3 = itmp % dst.extent(3);
                    int i4 = itmp / dst.extent(3);
                    dst(i0, i1, i2, i3, i4) = value;
                });
                team.team_barrier();
            }
        }

//----------------------------------------------------------------------------
        template<class TeamType, class DT, class... DP>
        void FLARE_INLINE_FUNCTION local_deep_copy(
                const TeamType &team, const Tensor<DT, DP...> &dst,
                typename TensorTraits<DT, DP...>::const_value_type &value,
                std::enable_if_t<(unsigned(TensorTraits<DT, DP...>::rank) == 6)> * = nullptr) {
            if (dst.data() == nullptr) {
                return;
            }

            const size_t N = dst.extent(0) * dst.extent(1) * dst.extent(2) *
                             dst.extent(3) * dst.extent(4) * dst.extent(5);

            if (dst.span_is_contiguous()) {
                team.team_barrier();
                local_deep_copy_contiguous(team, dst, value);
                team.team_barrier();
            } else {
                team.team_barrier();
                flare::parallel_for(flare::TeamVectorRange(team, N), [&](const int &i) {
                    int i0 = i % dst.extent(0);
                    int itmp = i / dst.extent(0);
                    int i1 = itmp % dst.extent(1);
                    itmp = itmp / dst.extent(1);
                    int i2 = itmp % dst.extent(2);
                    itmp = itmp / dst.extent(2);
                    int i3 = itmp % dst.extent(3);
                    itmp = itmp / dst.extent(3);
                    int i4 = itmp % dst.extent(4);
                    int i5 = itmp / dst.extent(4);
                    dst(i0, i1, i2, i3, i4, i5) = value;
                });
                team.team_barrier();
            }
        }

//----------------------------------------------------------------------------
        template<class TeamType, class DT, class... DP>
        void FLARE_INLINE_FUNCTION local_deep_copy(
                const TeamType &team, const Tensor<DT, DP...> &dst,
                typename TensorTraits<DT, DP...>::const_value_type &value,
                std::enable_if_t<(unsigned(TensorTraits<DT, DP...>::rank) == 7)> * = nullptr) {
            if (dst.data() == nullptr) {
                return;
            }

            const size_t N = dst.extent(0) * dst.extent(1) * dst.extent(2) *
                             dst.extent(3) * dst.extent(4) * dst.extent(5) *
                             dst.extent(6);

            if (dst.span_is_contiguous()) {
                team.team_barrier();
                local_deep_copy_contiguous(team, dst, value);
                team.team_barrier();
            } else {
                team.team_barrier();
                flare::parallel_for(flare::TeamVectorRange(team, N), [&](const int &i) {
                    int i0 = i % dst.extent(0);
                    int itmp = i / dst.extent(0);
                    int i1 = itmp % dst.extent(1);
                    itmp = itmp / dst.extent(1);
                    int i2 = itmp % dst.extent(2);
                    itmp = itmp / dst.extent(2);
                    int i3 = itmp % dst.extent(3);
                    itmp = itmp / dst.extent(3);
                    int i4 = itmp % dst.extent(4);
                    itmp = itmp / dst.extent(4);
                    int i5 = itmp % dst.extent(5);
                    int i6 = itmp / dst.extent(5);
                    dst(i0, i1, i2, i3, i4, i5, i6) = value;
                });
                team.team_barrier();
            }
        }

//----------------------------------------------------------------------------
        template<class DT, class... DP>
        void FLARE_INLINE_FUNCTION local_deep_copy(
                const Tensor<DT, DP...> &dst,
                typename TensorTraits<DT, DP...>::const_value_type &value,
                std::enable_if_t<(unsigned(TensorTraits<DT, DP...>::rank) == 1)> * = nullptr) {
            if (dst.data() == nullptr) {
                return;
            }

            const size_t N = dst.extent(0);

            for (size_t i = 0; i < N; ++i) {
                dst(i) = value;
            }
        }

//----------------------------------------------------------------------------
        template<class DT, class... DP>
        void FLARE_INLINE_FUNCTION local_deep_copy(
                const Tensor<DT, DP...> &dst,
                typename TensorTraits<DT, DP...>::const_value_type &value,
                std::enable_if_t<(unsigned(TensorTraits<DT, DP...>::rank) == 2)> * = nullptr) {
            if (dst.data() == nullptr) {
                return;
            }

            if (dst.span_is_contiguous()) {
                local_deep_copy_contiguous(dst, value);
            } else {
                for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
                    for (size_t i1 = 0; i1 < dst.extent(1); ++i1) dst(i0, i1) = value;
            }
        }

//----------------------------------------------------------------------------
        template<class DT, class... DP>
        void FLARE_INLINE_FUNCTION local_deep_copy(
                const Tensor<DT, DP...> &dst,
                typename TensorTraits<DT, DP...>::const_value_type &value,
                std::enable_if_t<(unsigned(TensorTraits<DT, DP...>::rank) == 3)> * = nullptr) {
            if (dst.data() == nullptr) {
                return;
            }

            if (dst.span_is_contiguous()) {
                local_deep_copy_contiguous(dst, value);
            } else {
                for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
                    for (size_t i1 = 0; i1 < dst.extent(1); ++i1)
                        for (size_t i2 = 0; i2 < dst.extent(2); ++i2) dst(i0, i1, i2) = value;
            }
        }

//----------------------------------------------------------------------------
        template<class DT, class... DP>
        void FLARE_INLINE_FUNCTION local_deep_copy(
                const Tensor<DT, DP...> &dst,
                typename TensorTraits<DT, DP...>::const_value_type &value,
                std::enable_if_t<(unsigned(TensorTraits<DT, DP...>::rank) == 4)> * = nullptr) {
            if (dst.data() == nullptr) {
                return;
            }

            if (dst.span_is_contiguous()) {
                local_deep_copy_contiguous(dst, value);
            } else {
                for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
                    for (size_t i1 = 0; i1 < dst.extent(1); ++i1)
                        for (size_t i2 = 0; i2 < dst.extent(2); ++i2)
                            for (size_t i3 = 0; i3 < dst.extent(3); ++i3)
                                dst(i0, i1, i2, i3) = value;
            }
        }

//----------------------------------------------------------------------------
        template<class DT, class... DP>
        void FLARE_INLINE_FUNCTION local_deep_copy(
                const Tensor<DT, DP...> &dst,
                typename TensorTraits<DT, DP...>::const_value_type &value,
                std::enable_if_t<(unsigned(TensorTraits<DT, DP...>::rank) == 5)> * = nullptr) {
            if (dst.data() == nullptr) {
                return;
            }

            if (dst.span_is_contiguous()) {
                local_deep_copy_contiguous(dst, value);
            } else {
                for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
                    for (size_t i1 = 0; i1 < dst.extent(1); ++i1)
                        for (size_t i2 = 0; i2 < dst.extent(2); ++i2)
                            for (size_t i3 = 0; i3 < dst.extent(3); ++i3)
                                for (size_t i4 = 0; i4 < dst.extent(4); ++i4)
                                    dst(i0, i1, i2, i3, i4) = value;
            }
        }

//----------------------------------------------------------------------------
        template<class DT, class... DP>
        void FLARE_INLINE_FUNCTION local_deep_copy(
                const Tensor<DT, DP...> &dst,
                typename TensorTraits<DT, DP...>::const_value_type &value,
                std::enable_if_t<(unsigned(TensorTraits<DT, DP...>::rank) == 6)> * = nullptr) {
            if (dst.data() == nullptr) {
                return;
            }

            if (dst.span_is_contiguous()) {
                local_deep_copy_contiguous(dst, value);
            } else {
                for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
                    for (size_t i1 = 0; i1 < dst.extent(1); ++i1)
                        for (size_t i2 = 0; i2 < dst.extent(2); ++i2)
                            for (size_t i3 = 0; i3 < dst.extent(3); ++i3)
                                for (size_t i4 = 0; i4 < dst.extent(4); ++i4)
                                    for (size_t i5 = 0; i5 < dst.extent(5); ++i5)
                                        dst(i0, i1, i2, i3, i4, i5) = value;
            }
        }

//----------------------------------------------------------------------------
        template<class DT, class... DP>
        void FLARE_INLINE_FUNCTION local_deep_copy(
                const Tensor<DT, DP...> &dst,
                typename TensorTraits<DT, DP...>::const_value_type &value,
                std::enable_if_t<(unsigned(TensorTraits<DT, DP...>::rank) == 7)> * = nullptr) {
            if (dst.data() == nullptr) {
                return;
            }

            if (dst.span_is_contiguous()) {
                local_deep_copy_contiguous(dst, value);
            } else {
                for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
                    for (size_t i1 = 0; i1 < dst.extent(1); ++i1)
                        for (size_t i2 = 0; i2 < dst.extent(2); ++i2)
                            for (size_t i3 = 0; i3 < dst.extent(3); ++i3)
                                for (size_t i4 = 0; i4 < dst.extent(4); ++i4)
                                    for (size_t i5 = 0; i5 < dst.extent(5); ++i5)
                                        for (size_t i6 = 0; i6 < dst.extent(6); ++i6)
                                            dst(i0, i1, i2, i3, i4, i5, i6) = value;
            }
        }
    } /* namespace experimental */
} /* namespace flare */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace flare {

/** \brief  Deep copy a value from Host memory into a tensor. ExecSpace can access
 * dst */
    template<class ExecSpace, class DT, class... DP>
    inline void deep_copy(
            const ExecSpace &space, const Tensor<DT, DP...> &dst,
            typename TensorTraits<DT, DP...>::const_value_type &value,
            std::enable_if_t<
                    flare::is_execution_space<ExecSpace>::value &&
                    std::is_void<typename TensorTraits<DT, DP...>::specialize>::value &&
                    flare::SpaceAccessibility<ExecSpace, typename TensorTraits<DT, DP...>::
                    memory_space>::accessible> * =
            nullptr) {
        using dst_traits = TensorTraits<DT, DP...>;
        static_assert(std::is_same<typename dst_traits::non_const_value_type,
                              typename dst_traits::value_type>::value,
                      "deep_copy requires non-const type");
        using dst_memory_space = typename dst_traits::memory_space;
        if (flare::Tools::experimental::get_callbacks().begin_deep_copy != nullptr) {
            flare::Profiling::beginDeepCopy(
                    flare::Profiling::make_space_handle(dst_memory_space::name()),
                    dst.label(), dst.data(),
                    flare::Profiling::make_space_handle(flare::HostSpace::name()),
                    "(none)", &value, dst.span() * sizeof(typename dst_traits::value_type));
        }
        if (dst.data() == nullptr) {
            space.fence("flare::deep_copy: scalar copy on space, dst data is null");
        } else if (dst.span_is_contiguous()) {
            detail::contiguous_fill_or_memset(space, dst, value);
        } else {
            using TensorType = Tensor<DT, DP...>;
            // Figure out iteration order to do the TensorFill
            int64_t strides[TensorType::rank + 1];
            dst.stride(strides);
            flare::Iterate iterate;
            if (std::is_same<typename TensorType::array_layout,
                    flare::LayoutRight>::value) {
                iterate = flare::Iterate::Right;
            } else if (std::is_same<typename TensorType::array_layout,
                    flare::LayoutLeft>::value) {
                iterate = flare::Iterate::Left;
            } else if (std::is_same<typename TensorType::array_layout,
                    flare::LayoutStride>::value) {
                if (strides[0] > strides[TensorType::rank > 0 ? TensorType::rank - 1 : 0])
                    iterate = flare::Iterate::Right;
                else
                    iterate = flare::Iterate::Left;
            } else {
                if (std::is_same<typename TensorType::execution_space::array_layout,
                        flare::LayoutRight>::value)
                    iterate = flare::Iterate::Right;
                else
                    iterate = flare::Iterate::Left;
            }

            // Lets call the right TensorFill functor based on integer space needed and
            // iteration type
            using TensorTypeUniform =
                    std::conditional_t<TensorType::rank == 0,
                            typename TensorType::uniform_runtime_type,
                            typename TensorType::uniform_runtime_nomemspace_type>;
            if (dst.span() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
                if (iterate == flare::Iterate::Right)
                    flare::detail::TensorFill<TensorTypeUniform, flare::LayoutRight, ExecSpace,
                            TensorType::rank, int64_t>(dst, value, space);
                else
                    flare::detail::TensorFill<TensorTypeUniform, flare::LayoutLeft, ExecSpace,
                            TensorType::rank, int64_t>(dst, value, space);
            } else {
                if (iterate == flare::Iterate::Right)
                    flare::detail::TensorFill<TensorTypeUniform, flare::LayoutRight, ExecSpace,
                            TensorType::rank, int32_t>(dst, value, space);
                else
                    flare::detail::TensorFill<TensorTypeUniform, flare::LayoutLeft, ExecSpace,
                            TensorType::rank, int32_t>(dst, value, space);
            }
        }
        if (flare::Tools::experimental::get_callbacks().end_deep_copy != nullptr) {
            flare::Profiling::endDeepCopy();
        }
    }

    /** \brief  Deep copy a value from Host memory into a tensor. ExecSpace can not
     * access dst */
    template<class ExecSpace, class DT, class... DP>
    inline void deep_copy(
            const ExecSpace &space, const Tensor<DT, DP...> &dst,
            typename TensorTraits<DT, DP...>::const_value_type &value,
            std::enable_if_t<
                    flare::is_execution_space<ExecSpace>::value &&
                    std::is_void<typename TensorTraits<DT, DP...>::specialize>::value &&
                    !flare::SpaceAccessibility<ExecSpace, typename TensorTraits<DT, DP...>::
                    memory_space>::accessible> * =
            nullptr) {
        using dst_traits = TensorTraits<DT, DP...>;
        static_assert(std::is_same<typename dst_traits::non_const_value_type,
                              typename dst_traits::value_type>::value,
                      "deep_copy requires non-const type");
        using dst_memory_space = typename dst_traits::memory_space;
        if (flare::Tools::experimental::get_callbacks().begin_deep_copy != nullptr) {
            flare::Profiling::beginDeepCopy(
                    flare::Profiling::make_space_handle(dst_memory_space::name()),
                    dst.label(), dst.data(),
                    flare::Profiling::make_space_handle(flare::HostSpace::name()),
                    "(none)", &value, dst.span() * sizeof(typename dst_traits::value_type));
        }
        if (dst.data() == nullptr) {
            space.fence(
                    "flare::deep_copy: scalar-to-tensor copy on space, dst data is null");
        } else {
            space.fence("flare::deep_copy: scalar-to-tensor copy on space, pre copy");
            using fill_exec_space = typename dst_traits::memory_space::execution_space;
            if (dst.span_is_contiguous()) {
                detail::contiguous_fill_or_memset(fill_exec_space(), dst, value);
            } else {
                using TensorTypeUniform = std::conditional_t<
                        Tensor<DT, DP...>::rank == 0,
                        typename Tensor<DT, DP...>::uniform_runtime_type,
                        typename Tensor<DT, DP...>::uniform_runtime_nomemspace_type>;
                flare::detail::TensorFill<TensorTypeUniform, typename dst_traits::array_layout,
                        fill_exec_space>(dst, value, fill_exec_space());
            }
            fill_exec_space().fence(
                    "flare::deep_copy: scalar-to-tensor copy on space, fence after fill");
        }
        if (flare::Tools::experimental::get_callbacks().end_deep_copy != nullptr) {
            flare::Profiling::endDeepCopy();
        }
    }

    /** \brief  Deep copy into a value in Host memory from a tensor.  */
    template<class ExecSpace, class ST, class... SP>
    inline void deep_copy(
            const ExecSpace &exec_space,
            typename TensorTraits<ST, SP...>::non_const_value_type &dst,
            const Tensor<ST, SP...> &src,
            std::enable_if_t<flare::is_execution_space<ExecSpace>::value &&
                             std::is_same<typename TensorTraits<ST, SP...>::specialize,
                                     void>::value> * = nullptr) {
        using src_traits = TensorTraits<ST, SP...>;
        using src_memory_space = typename src_traits::memory_space;
        static_assert(src_traits::rank == 0,
                      "ERROR: Non-rank-zero tensor in deep_copy( value , Tensor )");
        if (flare::Tools::experimental::get_callbacks().begin_deep_copy != nullptr) {
            flare::Profiling::beginDeepCopy(
                    flare::Profiling::make_space_handle(flare::HostSpace::name()),
                    "(none)", &dst,
                    flare::Profiling::make_space_handle(src_memory_space::name()),
                    src.label(), src.data(), sizeof(ST));
        }

        if (src.data() == nullptr) {
            exec_space.fence(
                    "flare::deep_copy: tensor-to-scalar copy on space, src data is null");
            if (flare::Tools::experimental::get_callbacks().end_deep_copy != nullptr) {
                flare::Profiling::endDeepCopy();
            }
            return;
        }

        flare::detail::DeepCopy<HostSpace, src_memory_space, ExecSpace>(
                exec_space, &dst, src.data(), sizeof(ST));
        if (flare::Tools::experimental::get_callbacks().end_deep_copy != nullptr) {
            flare::Profiling::endDeepCopy();
        }
    }

//----------------------------------------------------------------------------
/** \brief  A deep copy between tensors of compatible type, and rank zero.  */
    template<class ExecSpace, class DT, class... DP, class ST, class... SP>
    inline void deep_copy(
            const ExecSpace &exec_space, const Tensor<DT, DP...> &dst,
            const Tensor<ST, SP...> &src,
            std::enable_if_t<
                    (flare::is_execution_space<ExecSpace>::value &&
                     std::is_void<typename TensorTraits<DT, DP...>::specialize>::value &&
                     std::is_void<typename TensorTraits<ST, SP...>::specialize>::value &&
                     (unsigned(TensorTraits<DT, DP...>::rank) == unsigned(0) &&
                      unsigned(TensorTraits<ST, SP...>::rank) == unsigned(0)))> * = nullptr) {
        using src_traits = TensorTraits<ST, SP...>;
        using dst_traits = TensorTraits<DT, DP...>;

        using src_memory_space = typename src_traits::memory_space;
        using dst_memory_space = typename dst_traits::memory_space;
        static_assert(std::is_same<typename dst_traits::value_type,
                              typename src_traits::non_const_value_type>::value,
                      "deep_copy requires matching non-const destination type");

        if (flare::Tools::experimental::get_callbacks().begin_deep_copy != nullptr) {
            flare::Profiling::beginDeepCopy(
                    flare::Profiling::make_space_handle(dst_memory_space::name()),
                    dst.label(), dst.data(),
                    flare::Profiling::make_space_handle(src_memory_space::name()),
                    src.label(), src.data(), sizeof(DT));
        }

        if (dst.data() == nullptr && src.data() == nullptr) {
            exec_space.fence(
                    "flare::deep_copy: tensor-to-tensor copy on space, data is null");
            if (flare::Tools::experimental::get_callbacks().end_deep_copy != nullptr) {
                flare::Profiling::endDeepCopy();
            }
            return;
        }

        if (dst.data() != src.data()) {
            flare::detail::DeepCopy<dst_memory_space, src_memory_space, ExecSpace>(
                    exec_space, dst.data(), src.data(),
                    sizeof(typename dst_traits::value_type));
        }
        if (flare::Tools::experimental::get_callbacks().end_deep_copy != nullptr) {
            flare::Profiling::endDeepCopy();
        }
    }

//----------------------------------------------------------------------------
/** \brief  A deep copy between tensors of the default specialization, compatible
 * type, same non-zero rank
 */
    template<class ExecSpace, class DT, class... DP, class ST, class... SP>
    inline void deep_copy(
            const ExecSpace &exec_space, const Tensor<DT, DP...> &dst,
            const Tensor<ST, SP...> &src,
            std::enable_if_t<
                    (flare::is_execution_space<ExecSpace>::value &&
                     std::is_void<typename TensorTraits<DT, DP...>::specialize>::value &&
                     std::is_void<typename TensorTraits<ST, SP...>::specialize>::value &&
                     (unsigned(TensorTraits<DT, DP...>::rank) != 0 ||
                      unsigned(TensorTraits<ST, SP...>::rank) != 0))> * = nullptr) {
        using dst_type = Tensor<DT, DP...>;
        using src_type = Tensor<ST, SP...>;

        static_assert(std::is_same<typename dst_type::value_type,
                              typename dst_type::non_const_value_type>::value,
                      "deep_copy requires non-const destination type");

        static_assert((unsigned(dst_type::rank) == unsigned(src_type::rank)),
                      "deep_copy requires Tensors of equal rank");

        using dst_execution_space = typename dst_type::execution_space;
        using src_execution_space = typename src_type::execution_space;
        using dst_memory_space = typename dst_type::memory_space;
        using src_memory_space = typename src_type::memory_space;
        using dst_value_type = typename dst_type::value_type;
        using src_value_type = typename src_type::value_type;

        if (flare::Tools::experimental::get_callbacks().begin_deep_copy != nullptr) {
            flare::Profiling::beginDeepCopy(
                    flare::Profiling::make_space_handle(dst_memory_space::name()),
                    dst.label(), dst.data(),
                    flare::Profiling::make_space_handle(src_memory_space::name()),
                    src.label(), src.data(), dst.span() * sizeof(dst_value_type));
        }

        dst_value_type *dst_start = dst.data();
        dst_value_type *dst_end = dst.data() + dst.span();
        src_value_type *src_start = src.data();
        src_value_type *src_end = src.data() + src.span();

        // Early dropout if identical range
        if ((dst_start == nullptr || src_start == nullptr) ||
            ((std::ptrdiff_t(dst_start) == std::ptrdiff_t(src_start)) &&
             (std::ptrdiff_t(dst_end) == std::ptrdiff_t(src_end)))) {
            // throw if dimension mismatch
            if ((src.extent(0) != dst.extent(0)) || (src.extent(1) != dst.extent(1)) ||
                (src.extent(2) != dst.extent(2)) || (src.extent(3) != dst.extent(3)) ||
                (src.extent(4) != dst.extent(4)) || (src.extent(5) != dst.extent(5)) ||
                (src.extent(6) != dst.extent(6)) || (src.extent(7) != dst.extent(7))) {
                std::string message(
                        "Deprecation Error: flare::deep_copy extents of tensors don't "
                        "match: ");
                message += dst.label();
                message += "(";
                message += std::to_string(dst.extent(0));
                for (size_t r = 1; r < dst_type::rank; r++) {
                    message += ",";
                    message += std::to_string(dst.extent(r));
                }
                message += ") ";
                message += src.label();
                message += "(";
                message += std::to_string(src.extent(0));
                for (size_t r = 1; r < src_type::rank; r++) {
                    message += ",";
                    message += std::to_string(src.extent(r));
                }
                message += ") ";

                flare::detail::throw_runtime_exception(message);
            }
            if (flare::Tools::experimental::get_callbacks().end_deep_copy != nullptr) {
                flare::Profiling::endDeepCopy();
            }
            return;
        }

        enum {
            ExecCanAccessSrcDst =
            flare::SpaceAccessibility<ExecSpace, dst_memory_space>::accessible &&
            flare::SpaceAccessibility<ExecSpace, src_memory_space>::accessible
        };
        enum {
            DstExecCanAccessSrc =
            flare::SpaceAccessibility<dst_execution_space,
                    src_memory_space>::accessible
        };

        enum {
            SrcExecCanAccessDst =
            flare::SpaceAccessibility<src_execution_space,
                    dst_memory_space>::accessible
        };

        // Error out for non-identical overlapping tensors.
        if ((((std::ptrdiff_t) dst_start < (std::ptrdiff_t) src_end) &&
             ((std::ptrdiff_t) dst_end > (std::ptrdiff_t) src_start)) &&
            ((dst.span_is_contiguous() && src.span_is_contiguous()))) {
            std::string message("Error: flare::deep_copy of overlapping tensors: ");
            message += dst.label();
            message += "(";
            message += std::to_string((std::ptrdiff_t) dst_start);
            message += ",";
            message += std::to_string((std::ptrdiff_t) dst_end);
            message += ") ";
            message += src.label();
            message += "(";
            message += std::to_string((std::ptrdiff_t) src_start);
            message += ",";
            message += std::to_string((std::ptrdiff_t) src_end);
            message += ") ";
            flare::detail::throw_runtime_exception(message);
        }

        // Check for same extents
        if ((src.extent(0) != dst.extent(0)) || (src.extent(1) != dst.extent(1)) ||
            (src.extent(2) != dst.extent(2)) || (src.extent(3) != dst.extent(3)) ||
            (src.extent(4) != dst.extent(4)) || (src.extent(5) != dst.extent(5)) ||
            (src.extent(6) != dst.extent(6)) || (src.extent(7) != dst.extent(7))) {
            std::string message(
                    "Deprecation Error: flare::deep_copy extents of tensors don't match: ");
            message += dst.label();
            message += "(";
            message += std::to_string(dst.extent(0));
            for (size_t r = 1; r < dst_type::rank; r++) {
                message += ",";
                message += std::to_string(dst.extent(r));
            }
            message += ") ";
            message += src.label();
            message += "(";
            message += std::to_string(src.extent(0));
            for (size_t r = 1; r < src_type::rank; r++) {
                message += ",";
                message += std::to_string(src.extent(r));
            }
            message += ") ";

            flare::detail::throw_runtime_exception(message);
        }

        // If same type, equal layout, equal dimensions, equal span, and contiguous
        // memory then can byte-wise copy

        if (std::is_same<typename dst_type::value_type,
                typename src_type::non_const_value_type>::value &&
            (std::is_same<typename dst_type::array_layout,
                    typename src_type::array_layout>::value ||
             (dst_type::rank == 1 && src_type::rank == 1)) &&
            dst.span_is_contiguous() && src.span_is_contiguous() &&
            ((dst_type::rank < 1) || (dst.stride_0() == src.stride_0())) &&
            ((dst_type::rank < 2) || (dst.stride_1() == src.stride_1())) &&
            ((dst_type::rank < 3) || (dst.stride_2() == src.stride_2())) &&
            ((dst_type::rank < 4) || (dst.stride_3() == src.stride_3())) &&
            ((dst_type::rank < 5) || (dst.stride_4() == src.stride_4())) &&
            ((dst_type::rank < 6) || (dst.stride_5() == src.stride_5())) &&
            ((dst_type::rank < 7) || (dst.stride_6() == src.stride_6())) &&
            ((dst_type::rank < 8) || (dst.stride_7() == src.stride_7()))) {
            const size_t nbytes = sizeof(typename dst_type::value_type) * dst.span();
            if ((void *) dst.data() != (void *) src.data() && 0 < nbytes) {
                flare::detail::DeepCopy<dst_memory_space, src_memory_space, ExecSpace>(
                        exec_space, dst.data(), src.data(), nbytes);
            }
        } else {
            // Copying data between tensors in accessible memory spaces and either
            // non-contiguous or incompatible shape.
            if (ExecCanAccessSrcDst) {
                detail::tensor_copy(exec_space, dst, src);
            } else if (DstExecCanAccessSrc || SrcExecCanAccessDst) {
                using cpy_exec_space =
                        std::conditional_t<DstExecCanAccessSrc, dst_execution_space,
                                src_execution_space>;
                exec_space.fence(
                        "flare::deep_copy: tensor-to-tensor noncontiguous copy on space, pre "
                        "copy");
                detail::tensor_copy(cpy_exec_space(), dst, src);
                cpy_exec_space().fence(
                        "flare::deep_copy: tensor-to-tensor noncontiguous copy on space, post "
                        "copy");
            } else {
                flare::detail::throw_runtime_exception(
                        "deep_copy given tensors that would require a temporary allocation");
            }
        }
        if (flare::Tools::experimental::get_callbacks().end_deep_copy != nullptr) {
            flare::Profiling::endDeepCopy();
        }
    }

} /* namespace flare */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace flare {

    namespace detail {
        template<typename TensorType>
        bool size_mismatch(const TensorType &tensor, unsigned int max_extent,
                           const size_t new_extents[8]) {
            for (unsigned int dim = 0; dim < max_extent; ++dim)
                if (new_extents[dim] != tensor.extent(dim)) {
                    return true;
                }
            for (unsigned int dim = max_extent; dim < 8; ++dim)
                if (new_extents[dim] != FLARE_IMPL_CTOR_DEFAULT_ARG) {
                    return true;
                }
            return false;
        }

    }  // namespace detail

/** \brief  Resize a tensor with copying old data to new data at the corresponding
 * indices. */
    template<class T, class... P, class... TensorCtorArgs>
    inline typename std::enable_if<
            std::is_same<typename flare::Tensor<T, P...>::array_layout,
                    flare::LayoutLeft>::value ||
            std::is_same<typename flare::Tensor<T, P...>::array_layout,
                    flare::LayoutRight>::value>::type
    impl_resize(const detail::TensorCtorProp<TensorCtorArgs...> &arg_prop,
                flare::Tensor<T, P...> &v, const size_t n0, const size_t n1,
                const size_t n2, const size_t n3, const size_t n4, const size_t n5,
                const size_t n6, const size_t n7) {
        using tensor_type = flare::Tensor<T, P...>;
        using alloc_prop_input = detail::TensorCtorProp<TensorCtorArgs...>;

        static_assert(flare::TensorTraits<T, P...>::is_managed,
                      "Can only resize managed tensors");
        static_assert(!alloc_prop_input::has_label,
                      "The tensor constructor arguments passed to flare::resize "
                      "must not include a label!");
        static_assert(!alloc_prop_input::has_pointer,
                      "The tensor constructor arguments passed to flare::resize must "
                      "not include a pointer!");
        static_assert(!alloc_prop_input::has_memory_space,
                      "The tensor constructor arguments passed to flare::resize must "
                      "not include a memory space instance!");

        // TODO (mfh 27 Jun 2017) If the old Tensor has enough space but just
        // different dimensions (e.g., if the product of the dimensions,
        // including extra space for alignment, will not change), then
        // consider just reusing storage.  For now, flare always
        // reallocates if any of the dimensions change, even if the old Tensor
        // has enough space.

        const size_t new_extents[8] = {n0, n1, n2, n3, n4, n5, n6, n7};
        const bool sizeMismatch = detail::size_mismatch(v, v.rank_dynamic, new_extents);

        if (sizeMismatch) {
            auto prop_copy = detail::with_properties_if_unset(
                    arg_prop, typename tensor_type::execution_space{}, v.label());

            tensor_type v_resized(prop_copy, n0, n1, n2, n3, n4, n5, n6, n7);

            if constexpr (alloc_prop_input::has_execution_space)
                flare::detail::TensorRemap<tensor_type, tensor_type>(
                        v_resized, v, detail::get_property<detail::ExecutionSpaceTag>(prop_copy));
            else {
                flare::detail::TensorRemap<tensor_type, tensor_type>(v_resized, v);
                flare::fence("flare::resize(Tensor)");
            }

            v = v_resized;
        }
    }

    template<class T, class... P, class... TensorCtorArgs>
    inline std::enable_if_t<
            std::is_same<typename flare::Tensor<T, P...>::array_layout,
                    flare::LayoutLeft>::value ||
            std::is_same<typename flare::Tensor<T, P...>::array_layout,
                    flare::LayoutRight>::value>
    resize(const detail::TensorCtorProp<TensorCtorArgs...> &arg_prop,
           flare::Tensor<T, P...> &v, const size_t n0 = FLARE_IMPL_CTOR_DEFAULT_ARG,
           const size_t n1 = FLARE_IMPL_CTOR_DEFAULT_ARG,
           const size_t n2 = FLARE_IMPL_CTOR_DEFAULT_ARG,
           const size_t n3 = FLARE_IMPL_CTOR_DEFAULT_ARG,
           const size_t n4 = FLARE_IMPL_CTOR_DEFAULT_ARG,
           const size_t n5 = FLARE_IMPL_CTOR_DEFAULT_ARG,
           const size_t n6 = FLARE_IMPL_CTOR_DEFAULT_ARG,
           const size_t n7 = FLARE_IMPL_CTOR_DEFAULT_ARG) {
        impl_resize(arg_prop, v, n0, n1, n2, n3, n4, n5, n6, n7);
    }

    template<class T, class... P>
    inline std::enable_if_t<
            std::is_same<typename flare::Tensor<T, P...>::array_layout,
                    flare::LayoutLeft>::value ||
            std::is_same<typename flare::Tensor<T, P...>::array_layout,
                    flare::LayoutRight>::value>
    resize(flare::Tensor<T, P...> &v, const size_t n0 = FLARE_IMPL_CTOR_DEFAULT_ARG,
           const size_t n1 = FLARE_IMPL_CTOR_DEFAULT_ARG,
           const size_t n2 = FLARE_IMPL_CTOR_DEFAULT_ARG,
           const size_t n3 = FLARE_IMPL_CTOR_DEFAULT_ARG,
           const size_t n4 = FLARE_IMPL_CTOR_DEFAULT_ARG,
           const size_t n5 = FLARE_IMPL_CTOR_DEFAULT_ARG,
           const size_t n6 = FLARE_IMPL_CTOR_DEFAULT_ARG,
           const size_t n7 = FLARE_IMPL_CTOR_DEFAULT_ARG) {
        impl_resize(detail::TensorCtorProp<>{}, v, n0, n1, n2, n3, n4, n5, n6, n7);
    }

    template<class I, class T, class... P>
    inline std::enable_if_t<
            (detail::is_tensor_ctor_property<I>::value ||
             flare::is_execution_space<I>::value) &&
            (std::is_same<typename flare::Tensor<T, P...>::array_layout,
                    flare::LayoutLeft>::value ||
             std::is_same<typename flare::Tensor<T, P...>::array_layout,
                     flare::LayoutRight>::value)>
    resize(const I &arg_prop, flare::Tensor<T, P...> &v,
           const size_t n0 = FLARE_IMPL_CTOR_DEFAULT_ARG,
           const size_t n1 = FLARE_IMPL_CTOR_DEFAULT_ARG,
           const size_t n2 = FLARE_IMPL_CTOR_DEFAULT_ARG,
           const size_t n3 = FLARE_IMPL_CTOR_DEFAULT_ARG,
           const size_t n4 = FLARE_IMPL_CTOR_DEFAULT_ARG,
           const size_t n5 = FLARE_IMPL_CTOR_DEFAULT_ARG,
           const size_t n6 = FLARE_IMPL_CTOR_DEFAULT_ARG,
           const size_t n7 = FLARE_IMPL_CTOR_DEFAULT_ARG) {
        impl_resize(flare::tensor_alloc(arg_prop), v, n0, n1, n2, n3, n4, n5, n6, n7);
    }

    template<class T, class... P, class... TensorCtorArgs>
    inline std::enable_if_t<
            std::is_same<typename flare::Tensor<T, P...>::array_layout,
                    flare::LayoutLeft>::value ||
            std::is_same<typename flare::Tensor<T, P...>::array_layout,
                    flare::LayoutRight>::value ||
            std::is_same<typename flare::Tensor<T, P...>::array_layout,
                    flare::LayoutStride>::value ||
            is_layouttiled<typename flare::Tensor<T, P...>::array_layout>::value>
    impl_resize(const detail::TensorCtorProp<TensorCtorArgs...> &arg_prop,
                flare::Tensor<T, P...> &v,
                const typename flare::Tensor<T, P...>::array_layout &layout) {
        using tensor_type = flare::Tensor<T, P...>;
        using alloc_prop_input = detail::TensorCtorProp<TensorCtorArgs...>;

        static_assert(flare::TensorTraits<T, P...>::is_managed,
                      "Can only resize managed tensors");
        static_assert(!alloc_prop_input::has_label,
                      "The tensor constructor arguments passed to flare::resize "
                      "must not include a label!");
        static_assert(!alloc_prop_input::has_pointer,
                      "The tensor constructor arguments passed to flare::resize must "
                      "not include a pointer!");
        static_assert(!alloc_prop_input::has_memory_space,
                      "The tensor constructor arguments passed to flare::resize must "
                      "not include a memory space instance!");

        if (v.layout() != layout) {
            auto prop_copy = detail::with_properties_if_unset(arg_prop, v.label());

            tensor_type v_resized(prop_copy, layout);

            if constexpr (alloc_prop_input::has_execution_space)
                flare::detail::TensorRemap<tensor_type, tensor_type>(
                        v_resized, v, detail::get_property<detail::ExecutionSpaceTag>(arg_prop));
            else {
                flare::detail::TensorRemap<tensor_type, tensor_type>(v_resized, v);
                flare::fence("flare::resize(Tensor)");
            }

            v = v_resized;
        }
    }

// FIXME User-provided (custom) layouts are not required to have a comparison
// operator. Hence, there is no way to check if the requested layout is actually
// the same as the existing one.
    template<class T, class... P, class... TensorCtorArgs>
    inline std::enable_if_t<
            !(std::is_same<typename flare::Tensor<T, P...>::array_layout,
                    flare::LayoutLeft>::value ||
              std::is_same<typename flare::Tensor<T, P...>::array_layout,
                      flare::LayoutRight>::value ||
              std::is_same<typename flare::Tensor<T, P...>::array_layout,
                      flare::LayoutStride>::value ||
              is_layouttiled<typename flare::Tensor<T, P...>::array_layout>::value)>
    impl_resize(const detail::TensorCtorProp<TensorCtorArgs...> &arg_prop,
                flare::Tensor<T, P...> &v,
                const typename flare::Tensor<T, P...>::array_layout &layout) {
        using tensor_type = flare::Tensor<T, P...>;
        using alloc_prop_input = detail::TensorCtorProp<TensorCtorArgs...>;

        static_assert(flare::TensorTraits<T, P...>::is_managed,
                      "Can only resize managed tensors");
        static_assert(!alloc_prop_input::has_label,
                      "The tensor constructor arguments passed to flare::resize "
                      "must not include a label!");
        static_assert(!alloc_prop_input::has_pointer,
                      "The tensor constructor arguments passed to flare::resize must "
                      "not include a pointer!");
        static_assert(!alloc_prop_input::has_memory_space,
                      "The tensor constructor arguments passed to flare::resize must "
                      "not include a memory space instance!");

        auto prop_copy = detail::with_properties_if_unset(arg_prop, v.label());

        tensor_type v_resized(prop_copy, layout);

        if constexpr (alloc_prop_input::has_execution_space)
            flare::detail::TensorRemap<tensor_type, tensor_type>(
                    v_resized, v, detail::get_property<detail::ExecutionSpaceTag>(arg_prop));
        else {
            flare::detail::TensorRemap<tensor_type, tensor_type>(v_resized, v);
            flare::fence("flare::resize(Tensor)");
        }

        v = v_resized;
    }

    template<class T, class... P, class... TensorCtorArgs>
    inline void resize(const detail::TensorCtorProp<TensorCtorArgs...> &arg_prop,
                       flare::Tensor<T, P...> &v,
                       const typename flare::Tensor<T, P...>::array_layout &layout) {
        impl_resize(arg_prop, v, layout);
    }

    template<class I, class T, class... P>
    inline std::enable_if_t<detail::is_tensor_ctor_property<I>::value ||
                            flare::is_execution_space<I>::value>
    resize(const I &arg_prop, flare::Tensor<T, P...> &v,
           const typename flare::Tensor<T, P...>::array_layout &layout) {
        impl_resize(arg_prop, v, layout);
    }

    template<class ExecutionSpace, class T, class... P>
    inline void resize(const ExecutionSpace &exec_space, flare::Tensor<T, P...> &v,
                       const typename flare::Tensor<T, P...>::array_layout &layout) {
        impl_resize(detail::TensorCtorProp<>(), exec_space, v, layout);
    }

    template<class T, class... P>
    inline void resize(flare::Tensor<T, P...> &v,
                       const typename flare::Tensor<T, P...>::array_layout &layout) {
        impl_resize(detail::TensorCtorProp<>{}, v, layout);
    }

/** \brief  Resize a tensor with discarding old data. */
    template<class T, class... P, class... TensorCtorArgs>
    inline std::enable_if_t<
            std::is_same<typename flare::Tensor<T, P...>::array_layout,
                    flare::LayoutLeft>::value ||
            std::is_same<typename flare::Tensor<T, P...>::array_layout,
                    flare::LayoutRight>::value>
    impl_realloc(flare::Tensor<T, P...> &v, const size_t n0, const size_t n1,
                 const size_t n2, const size_t n3, const size_t n4, const size_t n5,
                 const size_t n6, const size_t n7,
                 const detail::TensorCtorProp<TensorCtorArgs...> &arg_prop) {
        using tensor_type = flare::Tensor<T, P...>;
        using alloc_prop_input = detail::TensorCtorProp<TensorCtorArgs...>;

        static_assert(flare::TensorTraits<T, P...>::is_managed,
                      "Can only realloc managed tensors");
        static_assert(!alloc_prop_input::has_label,
                      "The tensor constructor arguments passed to flare::realloc must "
                      "not include a label!");
        static_assert(!alloc_prop_input::has_pointer,
                      "The tensor constructor arguments passed to flare::realloc must "
                      "not include a pointer!");
        static_assert(!alloc_prop_input::has_memory_space,
                      "The tensor constructor arguments passed to flare::realloc must "
                      "not include a memory space instance!");

        const size_t new_extents[8] = {n0, n1, n2, n3, n4, n5, n6, n7};
        const bool sizeMismatch = detail::size_mismatch(v, v.rank_dynamic, new_extents);

        if (sizeMismatch) {
            auto arg_prop_copy = detail::with_properties_if_unset(arg_prop, v.label());
            v = tensor_type();  // Best effort to deallocate in case no other tensor refers
            // to the shared allocation
            v = tensor_type(arg_prop_copy, n0, n1, n2, n3, n4, n5, n6, n7);
        } else if (alloc_prop_input::initialize) {
            if constexpr (alloc_prop_input::has_execution_space) {
                const auto &exec_space =
                        detail::get_property<detail::ExecutionSpaceTag>(arg_prop);
                flare::deep_copy(exec_space, v, typename tensor_type::value_type{});
            } else
                flare::deep_copy(v, typename tensor_type::value_type{});
        }
    }

    template<class T, class... P, class... TensorCtorArgs>
    inline std::enable_if_t<
            std::is_same<typename flare::Tensor<T, P...>::array_layout,
                    flare::LayoutLeft>::value ||
            std::is_same<typename flare::Tensor<T, P...>::array_layout,
                    flare::LayoutRight>::value>
    realloc(const detail::TensorCtorProp<TensorCtorArgs...> &arg_prop,
            flare::Tensor<T, P...> &v,
            const size_t n0 = FLARE_IMPL_CTOR_DEFAULT_ARG,
            const size_t n1 = FLARE_IMPL_CTOR_DEFAULT_ARG,
            const size_t n2 = FLARE_IMPL_CTOR_DEFAULT_ARG,
            const size_t n3 = FLARE_IMPL_CTOR_DEFAULT_ARG,
            const size_t n4 = FLARE_IMPL_CTOR_DEFAULT_ARG,
            const size_t n5 = FLARE_IMPL_CTOR_DEFAULT_ARG,
            const size_t n6 = FLARE_IMPL_CTOR_DEFAULT_ARG,
            const size_t n7 = FLARE_IMPL_CTOR_DEFAULT_ARG) {
        impl_realloc(v, n0, n1, n2, n3, n4, n5, n6, n7, arg_prop);
    }

    template<class T, class... P>
    inline std::enable_if_t<
            std::is_same<typename flare::Tensor<T, P...>::array_layout,
                    flare::LayoutLeft>::value ||
            std::is_same<typename flare::Tensor<T, P...>::array_layout,
                    flare::LayoutRight>::value>
    realloc(flare::Tensor<T, P...> &v,
            const size_t n0 = FLARE_IMPL_CTOR_DEFAULT_ARG,
            const size_t n1 = FLARE_IMPL_CTOR_DEFAULT_ARG,
            const size_t n2 = FLARE_IMPL_CTOR_DEFAULT_ARG,
            const size_t n3 = FLARE_IMPL_CTOR_DEFAULT_ARG,
            const size_t n4 = FLARE_IMPL_CTOR_DEFAULT_ARG,
            const size_t n5 = FLARE_IMPL_CTOR_DEFAULT_ARG,
            const size_t n6 = FLARE_IMPL_CTOR_DEFAULT_ARG,
            const size_t n7 = FLARE_IMPL_CTOR_DEFAULT_ARG) {
        impl_realloc(v, n0, n1, n2, n3, n4, n5, n6, n7, detail::TensorCtorProp<>{});
    }

    template<class I, class T, class... P>
    inline std::enable_if_t<
            detail::is_tensor_ctor_property<I>::value &&
            (std::is_same<typename flare::Tensor<T, P...>::array_layout,
                    flare::LayoutLeft>::value ||
             std::is_same<typename flare::Tensor<T, P...>::array_layout,
                     flare::LayoutRight>::value)>
    realloc(const I &arg_prop, flare::Tensor<T, P...> &v,
            const size_t n0 = FLARE_IMPL_CTOR_DEFAULT_ARG,
            const size_t n1 = FLARE_IMPL_CTOR_DEFAULT_ARG,
            const size_t n2 = FLARE_IMPL_CTOR_DEFAULT_ARG,
            const size_t n3 = FLARE_IMPL_CTOR_DEFAULT_ARG,
            const size_t n4 = FLARE_IMPL_CTOR_DEFAULT_ARG,
            const size_t n5 = FLARE_IMPL_CTOR_DEFAULT_ARG,
            const size_t n6 = FLARE_IMPL_CTOR_DEFAULT_ARG,
            const size_t n7 = FLARE_IMPL_CTOR_DEFAULT_ARG) {
        impl_realloc(v, n0, n1, n2, n3, n4, n5, n6, n7, flare::tensor_alloc(arg_prop));
    }

    template<class T, class... P, class... TensorCtorArgs>
    inline std::enable_if_t<
            std::is_same<typename flare::Tensor<T, P...>::array_layout,
                    flare::LayoutLeft>::value ||
            std::is_same<typename flare::Tensor<T, P...>::array_layout,
                    flare::LayoutRight>::value ||
            std::is_same<typename flare::Tensor<T, P...>::array_layout,
                    flare::LayoutStride>::value ||
            is_layouttiled<typename flare::Tensor<T, P...>::array_layout>::value>
    impl_realloc(flare::Tensor<T, P...> &v,
                 const typename flare::Tensor<T, P...>::array_layout &layout,
                 const detail::TensorCtorProp<TensorCtorArgs...> &arg_prop) {
        using tensor_type = flare::Tensor<T, P...>;
        using alloc_prop_input = detail::TensorCtorProp<TensorCtorArgs...>;

        static_assert(flare::TensorTraits<T, P...>::is_managed,
                      "Can only realloc managed tensors");
        static_assert(!alloc_prop_input::has_label,
                      "The tensor constructor arguments passed to flare::realloc must "
                      "not include a label!");
        static_assert(!alloc_prop_input::has_pointer,
                      "The tensor constructor arguments passed to flare::realloc must "
                      "not include a pointer!");
        static_assert(!alloc_prop_input::has_memory_space,
                      "The tensor constructor arguments passed to flare::realloc must "
                      "not include a memory space instance!");

        if (v.layout() != layout) {
            v = tensor_type();  // Deallocate first, if the only tensor to allocation
            v = tensor_type(arg_prop, layout);
        } else if (alloc_prop_input::initialize) {
            if constexpr (alloc_prop_input::has_execution_space) {
                const auto &exec_space =
                        detail::get_property<detail::ExecutionSpaceTag>(arg_prop);
                flare::deep_copy(exec_space, v, typename tensor_type::value_type{});
            } else
                flare::deep_copy(v, typename tensor_type::value_type{});
        }
    }

// FIXME User-provided (custom) layouts are not required to have a comparison
// operator. Hence, there is no way to check if the requested layout is actually
// the same as the existing one.
    template<class T, class... P, class... TensorCtorArgs>
    inline std::enable_if_t<
            !(std::is_same<typename flare::Tensor<T, P...>::array_layout,
                    flare::LayoutLeft>::value ||
              std::is_same<typename flare::Tensor<T, P...>::array_layout,
                      flare::LayoutRight>::value ||
              std::is_same<typename flare::Tensor<T, P...>::array_layout,
                      flare::LayoutStride>::value ||
              is_layouttiled<typename flare::Tensor<T, P...>::array_layout>::value)>
    impl_realloc(flare::Tensor<T, P...> &v,
                 const typename flare::Tensor<T, P...>::array_layout &layout,
                 const detail::TensorCtorProp<TensorCtorArgs...> &arg_prop) {
        using tensor_type = flare::Tensor<T, P...>;
        using alloc_prop_input = detail::TensorCtorProp<TensorCtorArgs...>;

        static_assert(flare::TensorTraits<T, P...>::is_managed,
                      "Can only realloc managed tensors");
        static_assert(!alloc_prop_input::has_label,
                      "The tensor constructor arguments passed to flare::realloc must "
                      "not include a label!");
        static_assert(!alloc_prop_input::has_pointer,
                      "The tensor constructor arguments passed to flare::realloc must "
                      "not include a pointer!");
        static_assert(!alloc_prop_input::has_memory_space,
                      "The tensor constructor arguments passed to flare::realloc must "
                      "not include a memory space instance!");

        auto arg_prop_copy = detail::with_properties_if_unset(arg_prop, v.label());

        v = tensor_type();  // Deallocate first, if the only tensor to allocation
        v = tensor_type(arg_prop_copy, layout);
    }

    template<class T, class... P, class... TensorCtorArgs>
    inline void realloc(
            const detail::TensorCtorProp<TensorCtorArgs...> &arg_prop,
            flare::Tensor<T, P...> &v,
            const typename flare::Tensor<T, P...>::array_layout &layout) {
        impl_realloc(v, layout, arg_prop);
    }

    template<class I, class T, class... P>
    inline std::enable_if_t<detail::is_tensor_ctor_property<I>::value> realloc(
            const I &arg_prop, flare::Tensor<T, P...> &v,
            const typename flare::Tensor<T, P...>::array_layout &layout) {
        impl_realloc(v, layout, flare::tensor_alloc(arg_prop));
    }

    template<class T, class... P>
    inline void realloc(
            flare::Tensor<T, P...> &v,
            const typename flare::Tensor<T, P...>::array_layout &layout) {
        impl_realloc(v, layout, detail::TensorCtorProp<>{});
    }

} /* namespace flare */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace flare::detail {

    // Deduce Mirror Types
    template<class Space, class T, class... P>
    struct MirrorTensorType {
        // The incoming tensor_type
        using src_tensor_type = typename flare::Tensor<T, P...>;
        // The memory space for the mirror tensor
        using memory_space = typename Space::memory_space;
        // Check whether it is the same memory space
        enum {
            is_same_memspace =
            std::is_same<memory_space, typename src_tensor_type::memory_space>::value
        };
        // The array_layout
        using array_layout = typename src_tensor_type::array_layout;
        // The data type (we probably want it non-const since otherwise we can't even
        // deep_copy to it.
        using data_type = typename src_tensor_type::non_const_data_type;
        // The destination tensor type if it is not the same memory space
        using dest_tensor_type = flare::Tensor<data_type, array_layout, Space>;
        // If it is the same memory_space return the existsing tensor_type
        // This will also keep the unmanaged trait if necessary
        using tensor_type =
                std::conditional_t<is_same_memspace, src_tensor_type, dest_tensor_type>;
    };

    template<class Space, class T, class... P>
    struct MirrorType {
        // The incoming tensor_type
        using src_tensor_type = typename flare::Tensor<T, P...>;
        // The memory space for the mirror tensor
        using memory_space = typename Space::memory_space;
        // Check whether it is the same memory space
        enum {
            is_same_memspace =
            std::is_same<memory_space, typename src_tensor_type::memory_space>::value
        };
        // The array_layout
        using array_layout = typename src_tensor_type::array_layout;
        // The data type (we probably want it non-const since otherwise we can't even
        // deep_copy to it.
        using data_type = typename src_tensor_type::non_const_data_type;
        // The destination tensor type if it is not the same memory space
        using tensor_type = flare::Tensor<data_type, array_layout, Space>;
    };

    template<class... TensorCtorArgs>
    void check_tensor_ctor_args_create_mirror() {
        using alloc_prop_input = detail::TensorCtorProp<TensorCtorArgs...>;

        static_assert(
                !alloc_prop_input::has_label,
                "The tensor constructor arguments passed to flare::create_mirror[_tensor] "
                "must not include a label!");
        static_assert(!alloc_prop_input::has_pointer,
                      "The tensor constructor arguments passed to "
                      "flare::create_mirror[_tensor] must "
                      "not include a pointer!");
        static_assert(!alloc_prop_input::allow_padding,
                      "The tensor constructor arguments passed to "
                      "flare::create_mirror[_tensor] must "
                      "not explicitly allow padding!");
    }

    template<class T, class... P, class... TensorCtorArgs>
    inline std::enable_if_t<!detail::TensorCtorProp<TensorCtorArgs...>::has_memory_space,
            typename flare::Tensor<T, P...>::HostMirror>
    create_mirror(const flare::Tensor<T, P...> &src,
                  const detail::TensorCtorProp<TensorCtorArgs...> &arg_prop) {
        using src_type = Tensor<T, P...>;
        using dst_type = typename src_type::HostMirror;

        check_tensor_ctor_args_create_mirror<TensorCtorArgs...>();

        auto prop_copy = detail::with_properties_if_unset(
                arg_prop, std::string(src.label()).append("_mirror"));

        return dst_type(prop_copy, src.layout());
    }

// Create a mirror in a new space (specialization for different space)
    template<class T, class... P, class... TensorCtorArgs,
            class Enable = std::enable_if_t<
                    detail::TensorCtorProp<TensorCtorArgs...>::has_memory_space>>
    auto create_mirror(const flare::Tensor<T, P...> &src,
                       const detail::TensorCtorProp<TensorCtorArgs...> &arg_prop) {
        check_tensor_ctor_args_create_mirror<TensorCtorArgs...>();

        auto prop_copy = detail::with_properties_if_unset(
                arg_prop, std::string(src.label()).append("_mirror"));
        using alloc_prop = decltype(prop_copy);

        return typename detail::MirrorType<typename alloc_prop::memory_space, T,
                P...>::tensor_type(prop_copy, src.layout());
    }
}  // namespace flare::detail
namespace flare {
    template<class T, class... P>
    std::enable_if_t<std::is_void<typename TensorTraits<T, P...>::specialize>::value,
            typename flare::Tensor<T, P...>::HostMirror>
    create_mirror(flare::Tensor<T, P...> const &v) {
        return detail::create_mirror(v, detail::TensorCtorProp<>{});
    }

    template<class T, class... P>
    std::enable_if_t<std::is_void<typename TensorTraits<T, P...>::specialize>::value,
            typename flare::Tensor<T, P...>::HostMirror>
    create_mirror(flare::detail::WithoutInitializing_t wi,
                  flare::Tensor<T, P...> const &v) {
        return detail::create_mirror(v, tensor_alloc(wi));
    }

    template<class Space, class T, class... P,
            typename Enable = std::enable_if_t<flare::is_space<Space>::value>>
    std::enable_if_t<std::is_void<typename TensorTraits<T, P...>::specialize>::value,
            typename detail::MirrorType<Space, T, P...>::tensor_type>
    create_mirror(Space const &, flare::Tensor<T, P...> const &v) {
        return detail::create_mirror(v, tensor_alloc(typename Space::memory_space{}));
    }

    template<class T, class... P, class... TensorCtorArgs,
            typename Enable = std::enable_if_t<
                    std::is_void<typename TensorTraits<T, P...>::specialize>::value &&
                    detail::TensorCtorProp<TensorCtorArgs...>::has_memory_space>>
    auto create_mirror(detail::TensorCtorProp<TensorCtorArgs...> const &arg_prop,
                       flare::Tensor<T, P...> const &v) {
        return detail::create_mirror(v, arg_prop);
    }

    template<class T, class... P, class... TensorCtorArgs>
    std::enable_if_t<
            std::is_void<typename TensorTraits<T, P...>::specialize>::value &&
            !detail::TensorCtorProp<TensorCtorArgs...>::has_memory_space,
            typename flare::Tensor<T, P...>::HostMirror>
    create_mirror(detail::TensorCtorProp<TensorCtorArgs...> const &arg_prop,
                  flare::Tensor<T, P...> const &v) {
        return detail::create_mirror(v, arg_prop);
    }

    template<class Space, class T, class... P,
            typename Enable = std::enable_if_t<flare::is_space<Space>::value>>
    std::enable_if_t<std::is_void<typename TensorTraits<T, P...>::specialize>::value,
            typename detail::MirrorType<Space, T, P...>::tensor_type>
    create_mirror(flare::detail::WithoutInitializing_t wi, Space const &,
                  flare::Tensor<T, P...> const &v) {
        return detail::create_mirror(v, tensor_alloc(typename Space::memory_space{}, wi));
    }
}  // namespace flare
namespace flare::detail {

    template<class T, class... P, class... TensorCtorArgs>
    inline std::enable_if_t<
            !detail::TensorCtorProp<TensorCtorArgs...>::has_memory_space &&
            (std::is_same<
                    typename flare::Tensor<T, P...>::memory_space,
                    typename flare::Tensor<T, P...>::HostMirror::memory_space>::value &&
             std::is_same<
                     typename flare::Tensor<T, P...>::data_type,
                     typename flare::Tensor<T, P...>::HostMirror::data_type>::value),
            typename flare::Tensor<T, P...>::HostMirror>
    create_mirror_tensor(const flare::Tensor<T, P...> &src,
                       const detail::TensorCtorProp<TensorCtorArgs...> &) {
        check_tensor_ctor_args_create_mirror<TensorCtorArgs...>();
        return src;
    }

    template<class T, class... P, class... TensorCtorArgs>
    inline std::enable_if_t<
            !detail::TensorCtorProp<TensorCtorArgs...>::has_memory_space &&
            !(std::is_same<typename flare::Tensor<T, P...>::memory_space,
                    typename flare::Tensor<
                            T, P...>::HostMirror::memory_space>::value &&
              std::is_same<
                      typename flare::Tensor<T, P...>::data_type,
                      typename flare::Tensor<T, P...>::HostMirror::data_type>::value),
            typename flare::Tensor<T, P...>::HostMirror>
    create_mirror_tensor(const flare::Tensor<T, P...> &src,
                       const detail::TensorCtorProp<TensorCtorArgs...> &arg_prop) {
        return flare::detail::create_mirror(src, arg_prop);
    }

    // Create a mirror tensor in a new space (specialization for same space)
    template<class T, class... P, class... TensorCtorArgs,
            class = std::enable_if_t<
                    detail::TensorCtorProp<TensorCtorArgs...>::has_memory_space>>
    std::enable_if_t<detail::MirrorTensorType<
            typename detail::TensorCtorProp<TensorCtorArgs...>::memory_space,
            T, P...>::is_same_memspace,
            typename detail::MirrorTensorType<
                    typename detail::TensorCtorProp<TensorCtorArgs...>::memory_space,
                    T, P...>::tensor_type>
    create_mirror_tensor(const flare::Tensor<T, P...> &src,
                       const detail::TensorCtorProp<TensorCtorArgs...> &) {
        check_tensor_ctor_args_create_mirror<TensorCtorArgs...>();
        return src;
    }

    // Create a mirror tensor in a new space (specialization for different space)
    template<class T, class... P, class... TensorCtorArgs,
            class = std::enable_if_t<
                    detail::TensorCtorProp<TensorCtorArgs...>::has_memory_space>>
    std::enable_if_t<!detail::MirrorTensorType<
            typename detail::TensorCtorProp<TensorCtorArgs...>::memory_space,
            T, P...>::is_same_memspace,
            typename detail::MirrorTensorType<
                    typename detail::TensorCtorProp<TensorCtorArgs...>::memory_space,
                    T, P...>::tensor_type>
    create_mirror_tensor(const flare::Tensor<T, P...> &src,
                       const detail::TensorCtorProp<TensorCtorArgs...> &arg_prop) {
        return flare::detail::create_mirror(src, arg_prop);
    }
}  // namespace flare::detail

namespace flare {
    template<class T, class... P>
    std::enable_if_t<
            std::is_same<
                    typename flare::Tensor<T, P...>::memory_space,
                    typename flare::Tensor<T, P...>::HostMirror::memory_space>::value &&
            std::is_same<
                    typename flare::Tensor<T, P...>::data_type,
                    typename flare::Tensor<T, P...>::HostMirror::data_type>::value,
            typename flare::Tensor<T, P...>::HostMirror>
    create_mirror_tensor(const flare::Tensor<T, P...> &src) {
        return src;
    }

    template<class T, class... P>
    std::enable_if_t<
            !(std::is_same<
                    typename flare::Tensor<T, P...>::memory_space,
                    typename flare::Tensor<T, P...>::HostMirror::memory_space>::value &&
              std::is_same<
                      typename flare::Tensor<T, P...>::data_type,
                      typename flare::Tensor<T, P...>::HostMirror::data_type>::value),
            typename flare::Tensor<T, P...>::HostMirror>
    create_mirror_tensor(const flare::Tensor<T, P...> &src) {
        return flare::create_mirror(src);
    }

    template<class T, class... P>
    typename flare::Tensor<T, P...>::HostMirror create_mirror_tensor(
            flare::detail::WithoutInitializing_t wi, flare::Tensor<T, P...> const &v) {
        return detail::create_mirror_tensor(v, tensor_alloc(wi));
    }

    // FIXME_C++17 Improve SFINAE here.
    template<class Space, class T, class... P,
            class Enable = std::enable_if_t<flare::is_space<Space>::value>>
    typename detail::MirrorTensorType<Space, T, P...>::tensor_type create_mirror_tensor(
            const Space &, const flare::Tensor<T, P...> &src,
            std::enable_if_t<detail::MirrorTensorType<Space, T, P...>::is_same_memspace> * =
            nullptr) {
        return src;
    }

    // FIXME_C++17 Improve SFINAE here.
    template<class Space, class T, class... P,
            class Enable = std::enable_if_t<flare::is_space<Space>::value>>
    typename detail::MirrorTensorType<Space, T, P...>::tensor_type create_mirror_tensor(
            const Space &space, const flare::Tensor<T, P...> &src,
            std::enable_if_t<!detail::MirrorTensorType<Space, T, P...>::is_same_memspace> * =
            nullptr) {
        return flare::create_mirror(space, src);
    }

    template<class Space, class T, class... P,
            typename Enable = std::enable_if_t<flare::is_space<Space>::value>>
    typename detail::MirrorTensorType<Space, T, P...>::tensor_type create_mirror_tensor(
            flare::detail::WithoutInitializing_t wi, Space const &,
            flare::Tensor<T, P...> const &v) {
        return detail::create_mirror_tensor(
                v, tensor_alloc(typename Space::memory_space{}, wi));
    }

    template<class T, class... P, class... TensorCtorArgs>
    auto create_mirror_tensor(const detail::TensorCtorProp<TensorCtorArgs...> &arg_prop,
                            const flare::Tensor<T, P...> &v) {
        return detail::create_mirror_tensor(v, arg_prop);
    }

    template<class... TensorCtorArgs, class T, class... P>
    auto create_mirror_tensor_and_copy(
            const detail::TensorCtorProp<TensorCtorArgs...> &,
            const flare::Tensor<T, P...> &src,
            std::enable_if_t<
                    std::is_void<typename TensorTraits<T, P...>::specialize>::value &&
                    detail::MirrorTensorType<
                            typename detail::TensorCtorProp<TensorCtorArgs...>::memory_space, T,
                            P...>::is_same_memspace> * = nullptr) {
        using alloc_prop_input = detail::TensorCtorProp<TensorCtorArgs...>;
        static_assert(
                alloc_prop_input::has_memory_space,
                "The tensor constructor arguments passed to "
                "flare::create_mirror_tensor_and_copy must include a memory space!");
        static_assert(!alloc_prop_input::has_pointer,
                      "The tensor constructor arguments passed to "
                      "flare::create_mirror_tensor_and_copy must "
                      "not include a pointer!");
        static_assert(!alloc_prop_input::allow_padding,
                      "The tensor constructor arguments passed to "
                      "flare::create_mirror_tensor_and_copy must "
                      "not explicitly allow padding!");

        // same behavior as deep_copy(src, src)
        if (!alloc_prop_input::has_execution_space)
            fence(
                    "flare::create_mirror_tensor_and_copy: fence before returning src tensor");
        return src;
    }

    template<class... TensorCtorArgs, class T, class... P>
    auto create_mirror_tensor_and_copy(
            const detail::TensorCtorProp<TensorCtorArgs...> &arg_prop,
            const ::flare::Tensor<T, P...> &src,
            std::enable_if_t<
                    std::is_void<typename TensorTraits<T, P...>::specialize>::value &&
                    !detail::MirrorTensorType<
                            typename detail::TensorCtorProp<TensorCtorArgs...>::memory_space, T,
                            P...>::is_same_memspace> * = nullptr) {
        using alloc_prop_input = detail::TensorCtorProp<TensorCtorArgs...>;
        static_assert(
                alloc_prop_input::has_memory_space,
                "The tensor constructor arguments passed to "
                "flare::create_mirror_tensor_and_copy must include a memory space!");
        static_assert(!alloc_prop_input::has_pointer,
                      "The tensor constructor arguments passed to "
                      "flare::create_mirror_tensor_and_copy must "
                      "not include a pointer!");
        static_assert(!alloc_prop_input::allow_padding,
                      "The tensor constructor arguments passed to "
                      "flare::create_mirror_tensor_and_copy must "
                      "not explicitly allow padding!");
        using Space = typename alloc_prop_input::memory_space;
        using Mirror = typename detail::MirrorTensorType<Space, T, P...>::tensor_type;

        auto arg_prop_copy = detail::with_properties_if_unset(
                arg_prop, std::string{}, WithoutInitializing,
                typename Space::execution_space{});

        std::string &label = detail::get_property<detail::LabelTag>(arg_prop_copy);
        if (label.empty()) label = src.label();
        auto mirror = typename Mirror::non_const_type{arg_prop_copy, src.layout()};
        if constexpr (alloc_prop_input::has_execution_space) {
            deep_copy(detail::get_property<detail::ExecutionSpaceTag>(arg_prop_copy),
                      mirror, src);
        } else
            deep_copy(mirror, src);
        return mirror;
    }

    // Previously when using auto here, the intel compiler 19.3 would
    // sometimes not create a symbol, guessing that it somehow is a combination
    // of auto and just forwarding arguments (see issue #5196)
    template<class Space, class T, class... P,
            typename Enable = std::enable_if_t<flare::is_space<Space>::value>>
    typename detail::MirrorTensorType<Space, T, P...>::tensor_type
    create_mirror_tensor_and_copy(
            const Space &, const flare::Tensor<T, P...> &src,
            std::string const &name = "",
            std::enable_if_t<
                    std::is_void<typename TensorTraits<T, P...>::specialize>::value> * =
            nullptr) {
        return create_mirror_tensor_and_copy(
                flare::tensor_alloc(typename Space::memory_space{}, name), src);
    }

}  // namespace flare

#endif  // FLARE_CORE_TENSOR_COPY_TENSORS_H_
