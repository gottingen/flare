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

#ifndef FLARE_CORE_TENSOR_TENSOR_TRACKER_H_
#define FLARE_CORE_TENSOR_TENSOR_TRACKER_H_

namespace flare {

    template<class DataType, class... Properties>
    class Tensor;
}  // namespace flare
namespace flare::detail {

    /*
     * \class TensorTracker
     * \brief template class to wrap the shared allocation tracker
     *
     * \section This class is templated on the Tensor and provides
     * constructors that match the tensor.  The constructors and assignments
     * from tensor will externalize the logic needed to enable/disable
     * ref counting to provide a single gate to enable further developments
     * which may hinge on the same logic.
     *
     */
    template<class ParentTensor>
    struct TensorTracker {
        using track_type = flare::detail::SharedAllocationTracker;
        using tensor_traits = typename ParentTensor::traits;

        track_type m_tracker;

        FLARE_INLINE_FUNCTION
        TensorTracker() : m_tracker() {}

        FLARE_INLINE_FUNCTION
        TensorTracker(const TensorTracker &vt) noexcept
                : m_tracker(vt.m_tracker, tensor_traits::is_managed) {}

        FLARE_INLINE_FUNCTION
        explicit TensorTracker(const ParentTensor &vt) noexcept: m_tracker() {
            assign(vt);
        }

        template<class RT, class... RP>
        FLARE_INLINE_FUNCTION explicit TensorTracker(
                const Tensor<RT, RP...> &vt) noexcept
                : m_tracker() {
            assign(vt);
        }

        template<class RT, class... RP>
        FLARE_INLINE_FUNCTION void assign(const Tensor<RT, RP...> &vt) noexcept {
            if (this == reinterpret_cast<const TensorTracker *>(&vt.m_track)) return;
            FLARE_IF_ON_HOST((
                                     if (tensor_traits::is_managed && flare::detail::SharedAllocationRecord<
                                     void, void>::tracking_enabled()) {
                                     m_tracker.assign_direct(vt.m_track.m_tracker);
                             } else { m_tracker.assign_force_disable(vt.m_track.m_tracker); }))

            FLARE_IF_ON_DEVICE((m_tracker.assign_force_disable(vt.m_track.m_tracker);))
        }

        FLARE_INLINE_FUNCTION TensorTracker &operator=(
                const TensorTracker &rhs) noexcept {
            if (this == &rhs) return *this;
            FLARE_IF_ON_HOST((
                                     if (tensor_traits::is_managed && flare::detail::SharedAllocationRecord<
                                     void, void>::tracking_enabled()) {
                                     m_tracker.assign_direct(rhs.m_tracker);
                             } else { m_tracker.assign_force_disable(rhs.m_tracker); }))

            FLARE_IF_ON_DEVICE((m_tracker.assign_force_disable(rhs.m_tracker);))
            return *this;
        }

        FLARE_INLINE_FUNCTION
        explicit TensorTracker(const track_type &tt) noexcept
                : m_tracker(tt, tensor_traits::is_managed) {}
    };

}  // namespace flare::detail

#endif  // FLARE_CORE_TENSOR_TENSOR_TRACKER_H_
