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

#ifndef FLARE_CORE_TENSOR_TENSOR_CTOR_PROP_H_
#define FLARE_CORE_TENSOR_TENSOR_CTOR_PROP_H_

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace flare::detail {

    struct WithoutInitializing_t {
    };
    struct AllowPadding_t {
    };

    template<typename>
    struct is_tensor_ctor_property : public std::false_type {
    };

    template<>
    struct is_tensor_ctor_property<WithoutInitializing_t> : public std::true_type {
    };

    template<>
    struct is_tensor_ctor_property<AllowPadding_t> : public std::true_type {
    };

    //----------------------------------------------------------------------------
    /**\brief Whether a type can be used for a tensor label */

    template<typename>
    struct is_tensor_label : public std::false_type {
    };

    template<>
    struct is_tensor_label<std::string> : public std::true_type {
    };

    template<unsigned N>
    struct is_tensor_label<char[N]> : public std::true_type {
    };

    template<unsigned N>
    struct is_tensor_label<const char[N]> : public std::true_type {
    };

    //----------------------------------------------------------------------------

    template<typename... P>
    struct TensorCtorProp;

    // Forward declare
    template<typename Specialize, typename T>
    struct CommonTensorAllocProp;

    /* Dummy to allow for empty TensorCtorProp object
     */
    template<>
    struct TensorCtorProp<void> {
    };

    /* Common value_type stored as TensorCtorProp
     */
    template<typename Specialize, typename T>
    struct TensorCtorProp<void, CommonTensorAllocProp<Specialize, T>> {
        TensorCtorProp() = default;

        TensorCtorProp(const TensorCtorProp &) = default;

        TensorCtorProp &operator=(const TensorCtorProp &) = default;

        using type = CommonTensorAllocProp<Specialize, T>;

        FLARE_FUNCTION
        TensorCtorProp(const type &arg) : value(arg) {}

        FLARE_FUNCTION
        TensorCtorProp(type &&arg) : value(arg) {}

        type value;
    };

    /* Property flags have constexpr value */
    template<typename P>
    struct TensorCtorProp<
            std::enable_if_t<std::is_same<P, AllowPadding_t>::value ||
                             std::is_same<P, WithoutInitializing_t>::value>,
            P> {
        TensorCtorProp() = default;

        TensorCtorProp(const TensorCtorProp &) = default;

        TensorCtorProp &operator=(const TensorCtorProp &) = default;

        using type = P;

        TensorCtorProp(const type &) {}

        type value = type();
    };

    /* Map input label type to std::string */
    template<typename Label>
    struct TensorCtorProp<std::enable_if_t<is_tensor_label<Label>::value>, Label> {
        TensorCtorProp() = default;

        TensorCtorProp(const TensorCtorProp &) = default;

        TensorCtorProp &operator=(const TensorCtorProp &) = default;

        using type = std::string;

        TensorCtorProp(const type &arg) : value(arg) {}

        TensorCtorProp(type &&arg) : value(arg) {}

        type value;
    };

    template<typename Space>
    struct TensorCtorProp<std::enable_if_t<flare::is_memory_space<Space>::value ||
                                           flare::is_execution_space<Space>::value>,
            Space> {
        TensorCtorProp() = default;

        TensorCtorProp(const TensorCtorProp &) = default;

        TensorCtorProp &operator=(const TensorCtorProp &) = default;

        using type = Space;

        TensorCtorProp(const type &arg) : value(arg) {}

        type value;
    };

    template<typename T>
    struct TensorCtorProp<void, T *> {
        TensorCtorProp() = default;

        TensorCtorProp(const TensorCtorProp &) = default;

        TensorCtorProp &operator=(const TensorCtorProp &) = default;

        using type = T *;

        FLARE_FUNCTION
        TensorCtorProp(const type arg) : value(arg) {}

        type value;
    };

    // For some reason I don't understand I needed this specialization explicitly
    // for NVCC/MSVC
    template<typename T>
    struct TensorCtorProp<T *> : public TensorCtorProp<void, T *> {
        static constexpr bool has_memory_space = false;
        static constexpr bool has_execution_space = false;
        static constexpr bool has_pointer = true;
        static constexpr bool has_label = false;
        static constexpr bool allow_padding = false;
        static constexpr bool initialize = true;

        using memory_space = void;
        using execution_space = void;
        using pointer_type = T *;

        FLARE_FUNCTION TensorCtorProp(const pointer_type arg)
                : TensorCtorProp<void, pointer_type>(arg) {}
    };

    // If we use `TensorCtorProp<Args...>` and `TensorCtorProp<void, Args>...` directly
    // in the parameter lists and base class initializers, respectively, as far as
    // we can tell MSVC 16.5.5+CUDA 10.2 thinks that `TensorCtorProp` refers to the
    // current instantiation, not the template itself, and gets all kinds of
    // confused. To work around this, we just use a couple of alias templates that
    // amount to the same thing.
    template<typename... Args>
    using tensor_ctor_prop_args = TensorCtorProp<Args...>;

    template<typename Arg>
    using tensor_ctor_prop_base = TensorCtorProp<void, Arg>;

    template<typename... P>
    struct TensorCtorProp : public TensorCtorProp<void, P> ... {
    private:
        using var_memory_space =
                flare::detail::has_condition<void, flare::is_memory_space, P...>;

        using var_execution_space =
                flare::detail::has_condition<void, flare::is_execution_space, P...>;

        struct VOIDDUMMY {
        };

        using var_pointer =
                flare::detail::has_condition<VOIDDUMMY, std::is_pointer, P...>;

    public:
        /* Flags for the common properties */
        static constexpr bool has_memory_space = var_memory_space::value;
        static constexpr bool has_execution_space = var_execution_space::value;
        static constexpr bool has_pointer = var_pointer::value;
        static constexpr bool has_label =
                flare::detail::has_type<std::string, P...>::value;
        static constexpr bool allow_padding =
                flare::detail::has_type<AllowPadding_t, P...>::value;
        static constexpr bool initialize =
                !flare::detail::has_type<WithoutInitializing_t, P...>::value;

        using memory_space = typename var_memory_space::type;
        using execution_space = typename var_execution_space::type;
        using pointer_type = typename var_pointer::type;

        /*  Copy from a matching argument list.
         *  Requires  std::is_same< P , TensorCtorProp< void , Args >::value ...
         */
        template<typename... Args>
        inline TensorCtorProp(Args const &... args) : TensorCtorProp<void, P>(args)... {}

        template<typename... Args>
        FLARE_FUNCTION TensorCtorProp(pointer_type arg0, Args const &... args)
                : TensorCtorProp<void, pointer_type>(arg0),
                  TensorCtorProp<void, typename TensorCtorProp<void, Args>::type>(args)... {}

        /* Copy from a matching property subset */
        FLARE_FUNCTION TensorCtorProp(pointer_type arg0)
                : TensorCtorProp<void, pointer_type>(arg0) {}

        // If we use `TensorCtorProp<Args...>` and `TensorCtorProp<void, Args>...` here
        // directly, MSVC 16.5.5+CUDA 10.2 appears to think that `TensorCtorProp` refers
        // to the current instantiation, not the template itself, and gets all kinds
        // of confused. To work around this, we just use a couple of alias templates
        // that amount to the same thing.
        template<typename... Args>
        TensorCtorProp(tensor_ctor_prop_args<Args...> const &arg)
                : tensor_ctor_prop_base<Args>(
                static_cast<tensor_ctor_prop_base<Args> const &>(arg))... {
            // Suppress an unused argument warning that (at least at one point) would
            // show up if sizeof...(Args) == 0
            (void) arg;
        }
    };

#if !defined(FLARE_COMPILER_MSVC) || !defined(FLARE_COMPILER_NVCC)

    template<typename... P>
    auto with_properties_if_unset(const TensorCtorProp<P...> &tensor_ctor_prop) {
        return tensor_ctor_prop;
    }

    template<typename... P, typename Property, typename... Properties>
    auto with_properties_if_unset(const TensorCtorProp<P...> &tensor_ctor_prop,
                                  [[maybe_unused]] const Property &property,
                                  const Properties &... properties) {
        if constexpr ((is_execution_space<Property>::value &&
                       !TensorCtorProp<P...>::has_execution_space) ||
                      (is_memory_space<Property>::value &&
                       !TensorCtorProp<P...>::has_memory_space) ||
                      (is_tensor_label<Property>::value &&
                       !TensorCtorProp<P...>::has_label) ||
                      (std::is_same_v<Property, WithoutInitializing_t> &&
                       TensorCtorProp<P...>::initialize)) {
            using NewTensorCtorProp = TensorCtorProp<P..., Property>;
            NewTensorCtorProp new_tensor_ctor_prop(tensor_ctor_prop);
            static_cast<TensorCtorProp<void, Property> &>(new_tensor_ctor_prop).value =
                    property;
            return with_properties_if_unset(new_tensor_ctor_prop, properties...);
        } else
            return with_properties_if_unset(tensor_ctor_prop, properties...);

// A workaround placed to prevent spurious "missing return statement at the
// end of non-void function" warnings from CUDA builds (issue #5470). Because
// FLARE_ENABLE_DEBUG_BOUNDS_CHECK removes [[noreturn]] attribute from
// cuda_abort(), an unreachable while(true); is placed as a fallback method
#if (defined(FLARE_COMPILER_NVCC) && (FLARE_COMPILER_NVCC < 1150)) || \
    (defined(FLARE_COMPILER_INTEL) && (FLARE_COMPILER_INTEL <= 2100))
        flare::abort(
            "Prevents an incorrect warning: missing return statement at end of "
            "non-void function");
#ifdef FLARE_ENABLE_DEBUG_BOUNDS_CHECK
        while (true)
          ;
#endif
#endif
    }

#else

    template <class TensorCtorP, class... Properties>
    struct WithPropertiesIfUnset;

    template <class TensorCtorP>
    struct WithPropertiesIfUnset<TensorCtorP> {
      static constexpr auto apply_prop(const TensorCtorP &tensor_ctor_prop) {
        return tensor_ctor_prop;
      }
    };

    template <class... P, class Property, class... Properties>
    struct WithPropertiesIfUnset<TensorCtorProp<P...>, Property, Properties...> {
      static constexpr auto apply_prop(const TensorCtorProp<P...> &tensor_ctor_prop,
                                       const Property &prop,
                                       const Properties &... properties) {
        if constexpr ((is_execution_space<Property>::value &&
                       !TensorCtorProp<P...>::has_execution_space) ||
                      (is_memory_space<Property>::value &&
                       !TensorCtorProp<P...>::has_memory_space) ||
                      (is_tensor_label<Property>::value &&
                       !TensorCtorProp<P...>::has_label) ||
                      (std::is_same_v<Property, WithoutInitializing_t> &&
                       TensorCtorProp<P...>::initialize)) {
          using NewTensorCtorProp = TensorCtorProp<P..., Property>;
          NewTensorCtorProp new_tensor_ctor_prop(tensor_ctor_prop);
          static_cast<TensorCtorProp<void, Property> &>(new_tensor_ctor_prop).value =
              prop;
          return WithPropertiesIfUnset<NewTensorCtorProp, Properties...>::apply_prop(
              new_tensor_ctor_prop, properties...);
        } else
          return WithPropertiesIfUnset<TensorCtorProp<P...>,
                                       Properties...>::apply_prop(tensor_ctor_prop,
                                                                  properties...);
      }
    };

    template <typename... P, class... Properties>
    auto with_properties_if_unset(const TensorCtorProp<P...> &tensor_ctor_prop,
                                  const Properties &... properties) {
      return WithPropertiesIfUnset<TensorCtorProp<P...>, Properties...>::apply_prop(
          tensor_ctor_prop, properties...);
    }

#endif

    struct ExecutionSpaceTag {
    };
    struct MemorySpaceTag {
    };
    struct LabelTag {
    };
    struct PointerTag {
    };

    template<typename Tag, typename... P>
    FLARE_FUNCTION const auto &get_property(
            const TensorCtorProp<P...> &tensor_ctor_prop) {
        if constexpr (std::is_same_v<Tag, ExecutionSpaceTag>) {
            static_assert(TensorCtorProp<P...>::has_execution_space);
            using execution_space_type = typename TensorCtorProp<P...>::execution_space;
            return static_cast<const TensorCtorProp<void, execution_space_type> &>(
                    tensor_ctor_prop)
                    .value;
        } else if constexpr (std::is_same_v<Tag, MemorySpaceTag>) {
            static_assert(TensorCtorProp<P...>::has_memory_space);
            using memory_space_type = typename TensorCtorProp<P...>::memory_space;
            return static_cast<const TensorCtorProp<void, memory_space_type> &>(
                    tensor_ctor_prop)
                    .value;
        } else if constexpr (std::is_same_v<Tag, LabelTag>) {
            static_assert(TensorCtorProp<P...>::has_label);
            return static_cast<const TensorCtorProp<void, std::string> &>(tensor_ctor_prop)
                    .value;
        } else if constexpr (std::is_same_v<Tag, PointerTag>) {
            static_assert(TensorCtorProp<P...>::has_pointer);
            using pointer_type = typename TensorCtorProp<P...>::pointer_type;
            return static_cast<const TensorCtorProp<void, pointer_type> &>(tensor_ctor_prop)
                    .value;
        } else {
            static_assert(std::is_same_v<Tag, void>, "Invalid property tag!");
            return tensor_ctor_prop;
        }

// A workaround placed to prevent spurious "missing return statement at the
// end of non-void function" warnings from CUDA builds (issue #5470). Because
// FLARE_ENABLE_DEBUG_BOUNDS_CHECK removes [[noreturn]] attribute from
// cuda_abort(), an unreachable while(true); is placed as a fallback method
#if (defined(FLARE_COMPILER_NVCC) && (FLARE_COMPILER_NVCC < 1150)) || \
    (defined(FLARE_COMPILER_INTEL) && (FLARE_COMPILER_INTEL <= 2100))
        flare::abort(
            "Prevents an incorrect warning: missing return statement at end of "
            "non-void function");
#ifdef FLARE_ENABLE_DEBUG_BOUNDS_CHECK
        while (true)
          ;
#endif
#endif
    }

#if defined(FLARE_COMPILER_NVCC) && (FLARE_COMPILER_NVCC < 1150)
    // pragma pop is getting a warning from the underlying GCC
    // for unknown pragma if -pedantic is used
#ifdef __CUDA_ARCH__
#pragma pop
#endif
#endif
#ifdef FLARE_IMPL_INTEL_BOGUS_MISSING_RETURN_STATEMENT_AT_END_OF_NON_VOID_FUNCTION
#pragma warning(pop)
#undef FLARE_IMPL_INTEL_BOGUS_MISSING_RETURN_STATEMENT_AT_END_OF_NON_VOID_FUNCTION
#endif

    template<typename Tag, typename... P>
    FLARE_FUNCTION auto &get_property(TensorCtorProp<P...> &tensor_ctor_prop) {
        // Avoid code duplication by deferring to the const-qualified overload and
        // casting the const away from the return type
        const auto &tmp = get_property<Tag>(
                static_cast<const TensorCtorProp<P...> &>(tensor_ctor_prop));
        return const_cast<std::decay_t<decltype(tmp)> &>(tmp);
    }

    struct TensorAllocateWithoutInitializingBackwardCompat {
    };

    template<>
    struct TensorCtorProp<void, TensorAllocateWithoutInitializingBackwardCompat> {
    };

    // NOTE This specialization is meant to be used as the
    // TensorAllocateWithoutInitializing alias below. All it does is add a
    // constructor that takes the label as single argument.
    template<>
    struct TensorCtorProp<WithoutInitializing_t, std::string,
            TensorAllocateWithoutInitializingBackwardCompat>
            : TensorCtorProp<WithoutInitializing_t, std::string>,
              TensorCtorProp<void, TensorAllocateWithoutInitializingBackwardCompat> {
        TensorCtorProp(std::string label)
                : TensorCtorProp<WithoutInitializing_t, std::string>(
                WithoutInitializing_t(), std::move(label)) {}
    };
}  // namespace flare::detail

namespace flare {
    using TensorAllocateWithoutInitializing =
            detail::TensorCtorProp<detail::WithoutInitializing_t, std::string,
                    detail::TensorAllocateWithoutInitializingBackwardCompat>;

} /* namespace flare */

#endif  // FLARE_CORE_TENSOR_TENSOR_CTOR_PROP_H_
