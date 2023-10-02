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

#ifndef FLARE_CORE_TENSOR_VIEW_HOOKS_H_
#define FLARE_CORE_TENSOR_VIEW_HOOKS_H_

namespace flare {
namespace experimental {

namespace detail {
template <typename View>
using copy_subscription_function_type = void (*)(View &, const View &);

template <template <typename> class Invoker, typename... Subscribers>
struct invoke_subscriber_impl;

template <template <typename> class Invoker>
struct invoke_subscriber_impl<Invoker> {
  template <typename ViewType>
  static void invoke(ViewType &, const ViewType &) {}
};

template <template <typename> class Invoker, typename Subscriber,
          typename... RemSubscribers>
struct invoke_subscriber_impl<Invoker, Subscriber, RemSubscribers...> {
  template <typename ViewType>
  static void invoke(ViewType &self, const ViewType &other) {
    Invoker<Subscriber>::call(self, other);
    invoke_subscriber_impl<Invoker, RemSubscribers...>::invoke(self, other);
  }
};

template <typename Subscriber>
struct copy_constructor_invoker {
  template <typename View>
  static void call(View &self, const View &other) {
    Subscriber::copy_constructed(self, other);
  }
};

template <typename Subscriber>
struct move_constructor_invoker {
  template <typename View>
  static void call(View &self, const View &other) {
    Subscriber::move_constructed(self, other);
  }
};

template <typename Subscriber>
struct copy_assignment_operator_invoker {
  template <typename View>
  static void call(View &self, const View &other) {
    Subscriber::copy_assigned(self, other);
  }
};

template <typename Subscriber>
struct move_assignment_operator_invoker {
  template <typename View>
  static void call(View &self, const View &other) {
    Subscriber::move_assigned(self, other);
  }
};
}  // namespace detail

struct EmptyViewHooks {
  using hooks_policy = EmptyViewHooks;

  template <typename View>
  static void copy_construct(View &, const View &) {}
  template <typename View>
  static void copy_assign(View &, const View &) {}
  template <typename View>
  static void move_construct(View &, const View &) {}
  template <typename View>
  static void move_assign(View &, const View &) {}
};

template <class... Subscribers>
struct SubscribableViewHooks {
  using hooks_policy = SubscribableViewHooks<Subscribers...>;

  template <typename View>
  static void copy_construct(View &self, const View &other) {
    detail::invoke_subscriber_impl<detail::copy_constructor_invoker,
                                 Subscribers...>::invoke(self, other);
  }
  template <typename View>
  static void copy_assign(View &self, const View &other) {
    detail::invoke_subscriber_impl<detail::copy_assignment_operator_invoker,
                                 Subscribers...>::invoke(self, other);
  }
  template <typename View>
  static void move_construct(View &self, const View &other) {
    detail::invoke_subscriber_impl<detail::move_constructor_invoker,
                                 Subscribers...>::invoke(self, other);
  }
  template <typename View>
  static void move_assign(View &self, const View &other) {
    detail::invoke_subscriber_impl<detail::move_assignment_operator_invoker,
                                 Subscribers...>::invoke(self, other);
  }
};

using DefaultViewHooks = EmptyViewHooks;

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_CORE_TENSOR_VIEW_HOOKS_H_
