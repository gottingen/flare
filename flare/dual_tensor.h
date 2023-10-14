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

/// \file dual_tensor.h
/// \brief Declaration and definition of flare::DualTensor.
///
/// This header file declares and defines flare::DualTensor and its
/// related nonmember functions.

#ifndef FLARE_DUAL_TENSOR_H_
#define FLARE_DUAL_TENSOR_H_

#include <flare/core.h>
#include <flare/core/common/error.h>

namespace flare {

/* \class DualTensor
 * \brief Container to manage mirroring a flare::Tensor that lives
 *   in device memory with a flare::Tensor that lives in host memory.
 *
 * This class provides capabilities to manage data which exists in two
 * memory spaces at the same time.  It keeps tensors of the same layout
 * on two memory spaces as well as modified flags for both
 * allocations.  Users are responsible for setting the modified flags
 * manually if they change the data in either memory space, by calling
 * the sync() method templated on the device where they modified the
 * data.  Users may synchronize data by calling the modify() function,
 * templated on the device towards which they want to synchronize
 * (i.e., the target of the one-way copy operation).
 *
 * The DualTensor class also provides convenience methods such as
 * realloc, resize and capacity which call the appropriate methods of
 * the underlying flare::Tensor objects.
 *
 * The four template arguments are the same as those of flare::Tensor.
 * (Please refer to that class' documentation for a detailed
 * description.)
 *
 *   \tparam DataType The type of the entries stored in the container.
 *
 *   \tparam Layout The array's layout in memory.
 *
 *   \tparam Device The flare Device type.  If its memory space is
 *     not the same as the host's memory space, then DualTensor will
 *     contain two separate Tensors: one in device memory, and one in
 *     host memory.  Otherwise, DualTensor will only store one Tensor.
 *
 *   \tparam MemoryTraits (optional) The user's intended memory access
 *     behavior.  Please see the documentation of flare::Tensor for
 *     examples.  The default suffices for most users.
 */

namespace detail {

#ifdef FLARE_ON_CUDA_DEVICE

inline const flare::Cuda& get_cuda_space(const flare::Cuda& in) { return in; }

inline const flare::Cuda& get_cuda_space() {
  return *flare::detail::cuda_get_deep_copy_space();
}

template <typename NonCudaExecSpace>
inline const flare::Cuda& get_cuda_space(const NonCudaExecSpace&) {
  return get_cuda_space();
}

#endif  // FLARE_ON_CUDA_DEVICE

}  // namespace detail

template <class DataType, class... Properties>
class DualTensor;

template <class>
struct is_dual_tensor : public std::false_type {};

template <class DT, class... DP>
struct is_dual_tensor<DualTensor<DT, DP...>> : public std::true_type {};

template <class DT, class... DP>
struct is_dual_tensor<const DualTensor<DT, DP...>> : public std::true_type {};

template <class T>
inline constexpr bool is_dual_tensor_v = is_dual_tensor<T>::value;

template <class DataType, class... Properties>
class DualTensor : public TensorTraits<DataType, Properties...> {
  template <class, class...>
  friend class DualTensor;

 public:
  //! \name Typedefs for device types and various flare::Tensor specializations.
  //@{

  using traits      = TensorTraits<DataType, Properties...>;

  //! The flare Host Device type;
  using host_mirror_space = typename traits::host_mirror_space;

  //! The type of a flare::Tensor on the device.
  using t_dev       = Tensor<typename traits::data_type, Properties...>;

  /// \typedef t_host
  /// \brief The type of a flare::Tensor host mirror of \c t_dev.
  using t_host = typename t_dev::HostMirror;

  //! The type of a const Tensor on the device.
  //! The type of a flare::Tensor on the device.
  using t_dev_const = Tensor<typename traits::const_data_type, Properties...>;

  /// \typedef t_host_const
  /// \brief The type of a const Tensor host mirror of \c t_dev_const.
  using t_host_const = typename t_dev_const::HostMirror;

  //! The type of a const, random-access Tensor on the device.
  using t_dev_const_randomread =
      Tensor<typename traits::const_data_type, typename traits::array_layout,
           typename traits::device_type,
           flare::MemoryTraits<flare::RandomAccess>>;

  /// \typedef t_host_const_randomread
  /// \brief The type of a const, random-access Tensor host mirror of
  ///   \c t_dev_const_randomread.
  using t_host_const_randomread = typename t_dev_const_randomread::HostMirror;

  //! The type of an unmanaged Tensor on the device.
  using t_dev_um =
      Tensor<typename traits::data_type, typename traits::array_layout,
           typename traits::device_type, MemoryUnmanaged>;

  //! The type of an unmanaged Tensor host mirror of \c t_dev_um.
  using t_host_um =
      Tensor<typename t_host::data_type, typename t_host::array_layout,
           typename t_host::device_type, MemoryUnmanaged>;

  //! The type of a const unmanaged Tensor on the device.
  using t_dev_const_um =
      Tensor<typename traits::const_data_type, typename traits::array_layout,
           typename traits::device_type, MemoryUnmanaged>;

  //! The type of a const unmanaged Tensor host mirror of \c t_dev_const_um.
  using t_host_const_um =
      Tensor<typename t_host::const_data_type, typename t_host::array_layout,
           typename t_host::device_type, MemoryUnmanaged>;

  //! The type of a const, random-access Tensor on the device.
  using t_dev_const_randomread_um =
      Tensor<typename t_host::const_data_type, typename t_host::array_layout,
           typename t_host::device_type,
           flare::MemoryTraits<flare::Unmanaged | flare::RandomAccess>>;

  /// \typedef t_host_const_randomread
  /// \brief The type of a const, random-access Tensor host mirror of
  ///   \c t_dev_const_randomread.
  using t_host_const_randomread_um =
      typename t_dev_const_randomread_um::HostMirror;

  //@}
  //! \name Counters to keep track of changes ("modified" flags)
  //@{

 protected:
  // modified_flags[0] -> host
  // modified_flags[1] -> device
  using t_modified_flags = Tensor<unsigned int[2], LayoutLeft, flare::HostSpace>;
  t_modified_flags modified_flags;

 public:
  //@}

  // Moved this specifically after modified_flags to resolve an alignment issue
  // on MSVC/NVCC
  //! \name The two Tensor instances.
  //@{
  t_dev d_tensor;
  t_host h_tensor;
  //@}

  //! \name Constructors
  //@{

  /// \brief Empty constructor.
  ///
  /// Both device and host Tensor objects are constructed using their
  /// default constructors.  The "modified" flags are both initialized
  /// to "unmodified."
  DualTensor() = default;

  /// \brief Constructor that allocates Tensor objects on both host and device.
  ///
  /// This constructor works like the analogous constructor of Tensor.
  /// The first argument is a string label, which is entirely for your
  /// benefit.  (Different DualTensor objects may have the same label if
  /// you like.)  The arguments that follow are the dimensions of the
  /// Tensor objects.  For example, if the Tensor has three dimensions,
  /// the first three integer arguments will be nonzero, and you may
  /// omit the integer arguments that follow.
  DualTensor(const std::string& label,
           const size_t n0 = FLARE_IMPL_CTOR_DEFAULT_ARG,
           const size_t n1 = FLARE_IMPL_CTOR_DEFAULT_ARG,
           const size_t n2 = FLARE_IMPL_CTOR_DEFAULT_ARG,
           const size_t n3 = FLARE_IMPL_CTOR_DEFAULT_ARG,
           const size_t n4 = FLARE_IMPL_CTOR_DEFAULT_ARG,
           const size_t n5 = FLARE_IMPL_CTOR_DEFAULT_ARG,
           const size_t n6 = FLARE_IMPL_CTOR_DEFAULT_ARG,
           const size_t n7 = FLARE_IMPL_CTOR_DEFAULT_ARG)
      : modified_flags(
            flare::tensor_alloc(typename t_modified_flags::execution_space{},
                               "DualTensor::modified_flags")),
        d_tensor(label, n0, n1, n2, n3, n4, n5, n6, n7),
        h_tensor(create_mirror_tensor(d_tensor))  // without UVM, host Tensor mirrors
  {}

  /// \brief Constructor that allocates Tensor objects on both host and device.
  ///
  /// This constructor works like the analogous constructor of Tensor.
  /// The first arguments are wrapped up in a Tensor Ctor class, this allows
  /// for a label, without initializing, and all of the other things that can
  /// be wrapped up in a Ctor class.
  /// The arguments that follow are the dimensions of the
  /// Tensor objects.  For example, if the Tensor has three dimensions,
  /// the first three integer arguments will be nonzero, and you may
  /// omit the integer arguments that follow.
  template <class... P>
  DualTensor(const detail::TensorCtorProp<P...>& arg_prop,
           std::enable_if_t<!detail::TensorCtorProp<P...>::has_pointer,
                            size_t> const n0 = FLARE_IMPL_CTOR_DEFAULT_ARG,
           const size_t n1                   = FLARE_IMPL_CTOR_DEFAULT_ARG,
           const size_t n2                   = FLARE_IMPL_CTOR_DEFAULT_ARG,
           const size_t n3                   = FLARE_IMPL_CTOR_DEFAULT_ARG,
           const size_t n4                   = FLARE_IMPL_CTOR_DEFAULT_ARG,
           const size_t n5                   = FLARE_IMPL_CTOR_DEFAULT_ARG,
           const size_t n6                   = FLARE_IMPL_CTOR_DEFAULT_ARG,
           const size_t n7                   = FLARE_IMPL_CTOR_DEFAULT_ARG)
      : modified_flags(t_modified_flags("DualTensor::modified_flags")),
        d_tensor(arg_prop, n0, n1, n2, n3, n4, n5, n6, n7) {
    // without UVM, host Tensor mirrors
    if constexpr (flare::detail::has_type<detail::WithoutInitializing_t,
                                         P...>::value)
      h_tensor = flare::create_mirror_tensor(flare::WithoutInitializing, d_tensor);
    else
      h_tensor = flare::create_mirror_tensor(d_tensor);
  }

  //! Copy constructor (shallow copy)
  template <typename DT, typename... DP>
  DualTensor(const DualTensor<DT, DP...>& src)
      : modified_flags(src.modified_flags),
        d_tensor(src.d_tensor),
        h_tensor(src.h_tensor) {}

  //! Subtensor constructor
  template <class DT, class... DP, class Arg0, class... Args>
  DualTensor(const DualTensor<DT, DP...>& src, const Arg0& arg0, Args... args)
      : modified_flags(src.modified_flags),
        d_tensor(flare::subtensor(src.d_tensor, arg0, args...)),
        h_tensor(flare::subtensor(src.h_tensor, arg0, args...)) {}

  /// \brief Create DualTensor from existing device and host Tensor objects.
  ///
  /// This constructor assumes that the device and host Tensor objects
  /// are synchronized.  You, the caller, are responsible for making
  /// sure this is the case before calling this constructor.  After
  /// this constructor returns, you may use DualTensor's sync() and
  /// modify() methods to ensure synchronization of the Tensor objects.
  ///
  /// \param d_tensor_ Device Tensor
  /// \param h_tensor_ Host Tensor (must have type t_host = t_dev::HostMirror)
  DualTensor(const t_dev& d_tensor_, const t_host& h_tensor_)
      : modified_flags(t_modified_flags("DualTensor::modified_flags")),
        d_tensor(d_tensor_),
        h_tensor(h_tensor_) {
    if (int(d_tensor.rank) != int(h_tensor.rank) ||
        d_tensor.extent(0) != h_tensor.extent(0) ||
        d_tensor.extent(1) != h_tensor.extent(1) ||
        d_tensor.extent(2) != h_tensor.extent(2) ||
        d_tensor.extent(3) != h_tensor.extent(3) ||
        d_tensor.extent(4) != h_tensor.extent(4) ||
        d_tensor.extent(5) != h_tensor.extent(5) ||
        d_tensor.extent(6) != h_tensor.extent(6) ||
        d_tensor.extent(7) != h_tensor.extent(7) ||
        d_tensor.stride_0() != h_tensor.stride_0() ||
        d_tensor.stride_1() != h_tensor.stride_1() ||
        d_tensor.stride_2() != h_tensor.stride_2() ||
        d_tensor.stride_3() != h_tensor.stride_3() ||
        d_tensor.stride_4() != h_tensor.stride_4() ||
        d_tensor.stride_5() != h_tensor.stride_5() ||
        d_tensor.stride_6() != h_tensor.stride_6() ||
        d_tensor.stride_7() != h_tensor.stride_7() ||
        d_tensor.span() != h_tensor.span()) {
      flare::detail::throw_runtime_exception(
          "DualTensor constructed with incompatible tensors");
    }
  }
  // does the DualTensor have only one device
  struct impl_dualtensor_is_single_device {
    enum : bool {
      value = std::is_same<typename t_dev::device_type,
                           typename t_host::device_type>::value
    };
  };

  // does the given device match the device of t_dev?
  template <typename Device>
  struct impl_device_matches_tdev_device {
    enum : bool {
      value = std::is_same<typename t_dev::device_type, Device>::value
    };
  };
  // does the given device match the device of t_host?
  template <typename Device>
  struct impl_device_matches_thost_device {
    enum : bool {
      value = std::is_same<typename t_host::device_type, Device>::value
    };
  };

  // does the given device match the execution space of t_host?
  template <typename Device>
  struct impl_device_matches_thost_exec {
    enum : bool {
      value = std::is_same<typename t_host::execution_space, Device>::value
    };
  };

  // does the given device match the execution space of t_dev?
  template <typename Device>
  struct impl_device_matches_tdev_exec {
    enum : bool {
      value = std::is_same<typename t_dev::execution_space, Device>::value
    };
  };

  // does the given device's memory space match the memory space of t_dev?
  template <typename Device>
  struct impl_device_matches_tdev_memory_space {
    enum : bool {
      value = std::is_same<typename t_dev::memory_space,
                           typename Device::memory_space>::value
    };
  };

  //@}
  //! \name Methods for synchronizing, marking as modified, and getting Tensors.
  //@{

  /// \brief Return a Tensor on a specific device \c Device.
  ///
  /// Please don't be afraid of the nested if_c expressions in the return
  /// value's type.  That just tells the method what the return type
  /// should be: t_dev if the \c Device template parameter matches
  /// this DualTensor's device type, else t_host.
  ///
  /// For example, suppose you create a DualTensor on Cuda, like this:
  /// \code
  ///   using dual_tensor_type =
  ///       flare::DualTensor<float, flare::LayoutRight, flare::Cuda>;
  ///   dual_tensor_type DV ("my dual tensor", 100);
  /// \endcode
  /// If you want to get the CUDA device Tensor, do this:
  /// \code
  ///   typename dual_tensor_type::t_dev cudaTensor = DV.tensor<flare::Cuda> ();
  /// \endcode
  /// and if you want to get the host mirror of that Tensor, do this:
  /// \code
  ///   using host_device_type = typename flare::HostSpace::execution_space;
  ///   typename dual_tensor_type::t_host hostTensor = DV.tensor<host_device_type> ();
  /// \endcode
  template <class Device>
  FLARE_INLINE_FUNCTION const typename std::conditional_t<
      impl_device_matches_tdev_device<Device>::value, t_dev,
      typename std::conditional_t<
          impl_device_matches_thost_device<Device>::value, t_host,
          typename std::conditional_t<
              impl_device_matches_thost_exec<Device>::value, t_host,
              typename std::conditional_t<
                  impl_device_matches_tdev_exec<Device>::value, t_dev,
                  typename std::conditional_t<
                      impl_device_matches_tdev_memory_space<Device>::value,
                      t_dev, t_host>>>>>
  tensor() const {
    constexpr bool device_is_memspace =
        std::is_same<Device, typename Device::memory_space>::value;
    constexpr bool device_is_execspace =
        std::is_same<Device, typename Device::execution_space>::value;
    constexpr bool device_exec_is_t_dev_exec =
        std::is_same<typename Device::execution_space,
                     typename t_dev::execution_space>::value;
    constexpr bool device_mem_is_t_dev_mem =
        std::is_same<typename Device::memory_space,
                     typename t_dev::memory_space>::value;
    constexpr bool device_exec_is_t_host_exec =
        std::is_same<typename Device::execution_space,
                     typename t_host::execution_space>::value;
    constexpr bool device_mem_is_t_host_mem =
        std::is_same<typename Device::memory_space,
                     typename t_host::memory_space>::value;
    constexpr bool device_is_t_host_device =
        std::is_same<typename Device::execution_space,
                     typename t_host::device_type>::value;
    constexpr bool device_is_t_dev_device =
        std::is_same<typename Device::memory_space,
                     typename t_host::device_type>::value;

    static_assert(
        device_is_t_dev_device || device_is_t_host_device ||
            (device_is_memspace &&
             (device_mem_is_t_dev_mem || device_mem_is_t_host_mem)) ||
            (device_is_execspace &&
             (device_exec_is_t_dev_exec || device_exec_is_t_host_exec)) ||
            ((!device_is_execspace && !device_is_memspace) &&
             ((device_mem_is_t_dev_mem || device_mem_is_t_host_mem) ||
              (device_exec_is_t_dev_exec || device_exec_is_t_host_exec))),
        "Template parameter to .tensor() must exactly match one of the "
        "DualTensor's device types or one of the execution or memory spaces");

    return detail::if_c<std::is_same<typename t_dev::memory_space,
                                   typename Device::memory_space>::value,
                      t_dev, t_host>::select(d_tensor, h_tensor);
  }

  FLARE_INLINE_FUNCTION
  t_host tensor_host() const { return h_tensor; }

  FLARE_INLINE_FUNCTION
  t_dev tensor_device() const { return d_tensor; }

  FLARE_INLINE_FUNCTION constexpr bool is_allocated() const {
    return (d_tensor.is_allocated() && h_tensor.is_allocated());
  }

  template <class Device>
  static int get_device_side() {
    constexpr bool device_is_memspace =
        std::is_same<Device, typename Device::memory_space>::value;
    constexpr bool device_is_execspace =
        std::is_same<Device, typename Device::execution_space>::value;
    constexpr bool device_exec_is_t_dev_exec =
        std::is_same<typename Device::execution_space,
                     typename t_dev::execution_space>::value;
    constexpr bool device_mem_is_t_dev_mem =
        std::is_same<typename Device::memory_space,
                     typename t_dev::memory_space>::value;
    constexpr bool device_exec_is_t_host_exec =
        std::is_same<typename Device::execution_space,
                     typename t_host::execution_space>::value;
    constexpr bool device_mem_is_t_host_mem =
        std::is_same<typename Device::memory_space,
                     typename t_host::memory_space>::value;
    constexpr bool device_is_t_host_device =
        std::is_same<typename Device::execution_space,
                     typename t_host::device_type>::value;
    constexpr bool device_is_t_dev_device =
        std::is_same<typename Device::memory_space,
                     typename t_host::device_type>::value;

    static_assert(
        device_is_t_dev_device || device_is_t_host_device ||
            (device_is_memspace &&
             (device_mem_is_t_dev_mem || device_mem_is_t_host_mem)) ||
            (device_is_execspace &&
             (device_exec_is_t_dev_exec || device_exec_is_t_host_exec)) ||
            ((!device_is_execspace && !device_is_memspace) &&
             ((device_mem_is_t_dev_mem || device_mem_is_t_host_mem) ||
              (device_exec_is_t_dev_exec || device_exec_is_t_host_exec))),
        "Template parameter to .sync() must exactly match one of the "
        "DualTensor's device types or one of the execution or memory spaces");

    int dev = -1;
    if (device_is_t_dev_device)
      dev = 1;
    else if (device_is_t_host_device)
      dev = 0;
    else {
      if (device_is_memspace) {
        if (device_mem_is_t_dev_mem) dev = 1;
        if (device_mem_is_t_host_mem) dev = 0;
        if (device_mem_is_t_host_mem && device_mem_is_t_dev_mem) dev = -1;
      }
      if (device_is_execspace) {
        if (device_exec_is_t_dev_exec) dev = 1;
        if (device_exec_is_t_host_exec) dev = 0;
        if (device_exec_is_t_host_exec && device_exec_is_t_dev_exec) dev = -1;
      }
      if (!device_is_execspace && !device_is_memspace) {
        if (device_mem_is_t_dev_mem) dev = 1;
        if (device_mem_is_t_host_mem) dev = 0;
        if (device_mem_is_t_host_mem && device_mem_is_t_dev_mem) dev = -1;
        if (device_exec_is_t_dev_exec) dev = 1;
        if (device_exec_is_t_host_exec) dev = 0;
        if (device_exec_is_t_host_exec && device_exec_is_t_dev_exec) dev = -1;
      }
    }
    return dev;
  }
  static constexpr const int tensor_header_size = 128;
  void impl_report_host_sync() const noexcept {
    if (flare::Tools::experimental::get_callbacks().sync_dual_tensor !=
        nullptr) {
      flare::Tools::syncDualTensor(
          h_tensor.label(),
          reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(h_tensor.data()) -
                                  tensor_header_size),
          false);
    }
  }
  void impl_report_device_sync() const noexcept {
    if (flare::Tools::experimental::get_callbacks().sync_dual_tensor !=
        nullptr) {
      flare::Tools::syncDualTensor(
          d_tensor.label(),
          reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(d_tensor.data()) -
                                  tensor_header_size),
          true);
    }
  }

  /// \brief Update data on device or host only if data in the other
  ///   space has been marked as modified.
  ///
  /// If \c Device is the same as this DualTensor's device type, then
  /// copy data from host to device.  Otherwise, copy data from device
  /// to host.  In either case, only copy if the source of the copy
  /// has been modified.
  ///
  /// This is a one-way synchronization only.  If the target of the
  /// copy has been modified, this operation will discard those
  /// modifications.  It will also reset both device and host modified
  /// flags.
  ///
  /// \note This method doesn't know on its own whether you modified
  ///   the data in either Tensor.  You must manually mark modified data
  ///   as modified, by calling the modify() method with the
  ///   appropriate template parameter.
  // deliberately passing args by cref as they're used multiple times
  template <class Device, class... Args>
  void sync_impl(std::true_type, Args const&... args) {
    if (modified_flags.data() == nullptr) return;

    int dev = get_device_side<Device>();

    if (dev == 1) {  // if Device is the same as DualTensor's device type
      if ((modified_flags(0) > 0) && (modified_flags(0) >= modified_flags(1))) {
#ifdef FLARE_ON_CUDA_DEVICE
        if (std::is_same<typename t_dev::memory_space,
                         flare::CudaUVMSpace>::value) {
          if (d_tensor.data() == h_tensor.data())
            flare::detail::cuda_prefetch_pointer(
                detail::get_cuda_space(args...), d_tensor.data(),
                sizeof(typename t_dev::value_type) * d_tensor.span(), true);
        }
#endif

        deep_copy(args..., d_tensor, h_tensor);
        modified_flags(0) = modified_flags(1) = 0;
        impl_report_device_sync();
      }
    }
    if (dev == 0) {  // hopefully Device is the same as DualTensor's host type
      if ((modified_flags(1) > 0) && (modified_flags(1) >= modified_flags(0))) {
#ifdef FLARE_ON_CUDA_DEVICE
        if (std::is_same<typename t_dev::memory_space,
                         flare::CudaUVMSpace>::value) {
          if (d_tensor.data() == h_tensor.data())
            flare::detail::cuda_prefetch_pointer(
                detail::get_cuda_space(args...), d_tensor.data(),
                sizeof(typename t_dev::value_type) * d_tensor.span(), false);
        }
#endif

        deep_copy(args..., h_tensor, d_tensor);
        modified_flags(0) = modified_flags(1) = 0;
        impl_report_host_sync();
      }
    }
    if constexpr (std::is_same<typename t_host::memory_space,
                               typename t_dev::memory_space>::value) {
      typename t_dev::execution_space().fence(
          "flare::DualTensor<>::sync: fence after syncing DualTensor");
      typename t_host::execution_space().fence(
          "flare::DualTensor<>::sync: fence after syncing DualTensor");
    }
  }

  template <class Device>
  void sync(const std::enable_if_t<
                (std::is_same<typename traits::data_type,
                              typename traits::non_const_data_type>::value) ||
                    (std::is_same<Device, int>::value),
                int>& = 0) {
    sync_impl<Device>(std::true_type{});
  }

  template <class Device, class ExecutionSpace>
  void sync(const ExecutionSpace& exec,
            const std::enable_if_t<
                (std::is_same<typename traits::data_type,
                              typename traits::non_const_data_type>::value) ||
                    (std::is_same<Device, int>::value),
                int>& = 0) {
    sync_impl<Device>(std::true_type{}, exec);
  }

  // deliberately passing args by cref as they're used multiple times
  template <class Device, class... Args>
  void sync_impl(std::false_type, Args const&...) {
    if (modified_flags.data() == nullptr) return;

    int dev = get_device_side<Device>();

    if (dev == 1) {  // if Device is the same as DualTensor's device type
      if ((modified_flags(0) > 0) && (modified_flags(0) >= modified_flags(1))) {
        detail::throw_runtime_exception(
            "Calling sync on a DualTensor with a const datatype.");
      }
      impl_report_device_sync();
    }
    if (dev == 0) {  // hopefully Device is the same as DualTensor's host type
      if ((modified_flags(1) > 0) && (modified_flags(1) >= modified_flags(0))) {
        detail::throw_runtime_exception(
            "Calling sync on a DualTensor with a const datatype.");
      }
      impl_report_host_sync();
    }
  }

  template <class Device>
  void sync(const std::enable_if_t<
                (!std::is_same<typename traits::data_type,
                               typename traits::non_const_data_type>::value) ||
                    (std::is_same<Device, int>::value),
                int>& = 0) {
    sync_impl<Device>(std::false_type{});
  }
  template <class Device, class ExecutionSpace>
  void sync(const ExecutionSpace& exec,
            const std::enable_if_t<
                (!std::is_same<typename traits::data_type,
                               typename traits::non_const_data_type>::value) ||
                    (std::is_same<Device, int>::value),
                int>& = 0) {
    sync_impl<Device>(std::false_type{}, exec);
  }

  // deliberately passing args by cref as they're used multiple times
  template <typename... Args>
  void sync_host_impl(Args const&... args) {
    if (!std::is_same<typename traits::data_type,
                      typename traits::non_const_data_type>::value)
      detail::throw_runtime_exception(
          "Calling sync_host on a DualTensor with a const datatype.");
    if (modified_flags.data() == nullptr) return;
    if (modified_flags(1) > modified_flags(0)) {
#ifdef FLARE_ON_CUDA_DEVICE
      if (std::is_same<typename t_dev::memory_space,
                       flare::CudaUVMSpace>::value) {
        if (d_tensor.data() == h_tensor.data())
          flare::detail::cuda_prefetch_pointer(
              detail::get_cuda_space(args...), d_tensor.data(),
              sizeof(typename t_dev::value_type) * d_tensor.span(), false);
      }
#endif

      deep_copy(args..., h_tensor, d_tensor);
      modified_flags(1) = modified_flags(0) = 0;
      impl_report_host_sync();
    }
  }

  template <class ExecSpace>
  void sync_host(const ExecSpace& exec) {
    sync_host_impl(exec);
  }
  void sync_host() { sync_host_impl(); }

  // deliberately passing args by cref as they're used multiple times
  template <typename... Args>
  void sync_device_impl(Args const&... args) {
    if (!std::is_same<typename traits::data_type,
                      typename traits::non_const_data_type>::value)
      detail::throw_runtime_exception(
          "Calling sync_device on a DualTensor with a const datatype.");
    if (modified_flags.data() == nullptr) return;
    if (modified_flags(0) > modified_flags(1)) {
#ifdef FLARE_ON_CUDA_DEVICE
      if (std::is_same<typename t_dev::memory_space,
                       flare::CudaUVMSpace>::value) {
        if (d_tensor.data() == h_tensor.data())
          flare::detail::cuda_prefetch_pointer(
              detail::get_cuda_space(args...), d_tensor.data(),
              sizeof(typename t_dev::value_type) * d_tensor.span(), true);
      }
#endif

      deep_copy(args..., d_tensor, h_tensor);
      modified_flags(1) = modified_flags(0) = 0;
      impl_report_device_sync();
    }
  }

  template <class ExecSpace>
  void sync_device(const ExecSpace& exec) {
    sync_device_impl(exec);
  }
  void sync_device() { sync_device_impl(); }

  template <class Device>
  bool need_sync() const {
    if (modified_flags.data() == nullptr) return false;
    int dev = get_device_side<Device>();

    if (dev == 1) {  // if Device is the same as DualTensor's device type
      if ((modified_flags(0) > 0) && (modified_flags(0) >= modified_flags(1))) {
        return true;
      }
    }
    if (dev == 0) {  // hopefully Device is the same as DualTensor's host type
      if ((modified_flags(1) > 0) && (modified_flags(1) >= modified_flags(0))) {
        return true;
      }
    }
    return false;
  }

  inline bool need_sync_host() const {
    if (modified_flags.data() == nullptr) return false;
    return modified_flags(0) < modified_flags(1);
  }

  inline bool need_sync_device() const {
    if (modified_flags.data() == nullptr) return false;
    return modified_flags(1) < modified_flags(0);
  }
  void impl_report_device_modification() {
    if (flare::Tools::experimental::get_callbacks().modify_dual_tensor !=
        nullptr) {
      flare::Tools::modifyDualTensor(
          d_tensor.label(),
          reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(d_tensor.data()) -
                                  tensor_header_size),
          true);
    }
  }
  void impl_report_host_modification() {
    if (flare::Tools::experimental::get_callbacks().modify_dual_tensor !=
        nullptr) {
      flare::Tools::modifyDualTensor(
          h_tensor.label(),
          reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(h_tensor.data()) -
                                  tensor_header_size),
          false);
    }
  }
  /// \brief Mark data as modified on the given device \c Device.
  ///
  /// If \c Device is the same as this DualTensor's device type, then
  /// mark the device's data as modified.  Otherwise, mark the host's
  /// data as modified.
  template <class Device, class Dummy = DualTensor,
            std::enable_if_t<!Dummy::impl_dualtensor_is_single_device::value>* =
                nullptr>
  void modify() {
    if (modified_flags.data() == nullptr) {
      modified_flags = t_modified_flags("DualTensor::modified_flags");
    }

    int dev = get_device_side<Device>();

    if (dev == 1) {  // if Device is the same as DualTensor's device type
      // Increment the device's modified count.
      modified_flags(1) =
          (modified_flags(1) > modified_flags(0) ? modified_flags(1)
                                                 : modified_flags(0)) +
          1;
      impl_report_device_modification();
    }
    if (dev == 0) {  // hopefully Device is the same as DualTensor's host type
      // Increment the host's modified count.
      modified_flags(0) =
          (modified_flags(1) > modified_flags(0) ? modified_flags(1)
                                                 : modified_flags(0)) +
          1;
      impl_report_host_modification();
    }

#ifdef FLARE_ENABLE_DEBUG_DUALTENSOR_MODIFY_CHECK
    if (modified_flags(0) && modified_flags(1)) {
      std::string msg = "flare::DualTensor::modify ERROR: ";
      msg += "Concurrent modification of host and device tensors ";
      msg += "in DualTensor \"";
      msg += d_tensor.label();
      msg += "\"\n";
      flare::abort(msg.c_str());
    }
#endif
  }

  template <
      class Device, class Dummy = DualTensor,
      std::enable_if_t<Dummy::impl_dualtensor_is_single_device::value>* = nullptr>
  void modify() {
    return;
  }

  template <class Dummy = DualTensor,
            std::enable_if_t<!Dummy::impl_dualtensor_is_single_device::value>* =
                nullptr>
  inline void modify_host() {
    if (modified_flags.data() != nullptr) {
      modified_flags(0) =
          (modified_flags(1) > modified_flags(0) ? modified_flags(1)
                                                 : modified_flags(0)) +
          1;
      impl_report_host_modification();
#ifdef FLARE_ENABLE_DEBUG_DUALTENSOR_MODIFY_CHECK
      if (modified_flags(0) && modified_flags(1)) {
        std::string msg = "flare::DualTensor::modify_host ERROR: ";
        msg += "Concurrent modification of host and device tensors ";
        msg += "in DualTensor \"";
        msg += d_tensor.label();
        msg += "\"\n";
        flare::abort(msg.c_str());
      }
#endif
    }
  }

  template <
      class Dummy = DualTensor,
      std::enable_if_t<Dummy::impl_dualtensor_is_single_device::value>* = nullptr>
  inline void modify_host() {
    return;
  }

  template <class Dummy = DualTensor,
            std::enable_if_t<!Dummy::impl_dualtensor_is_single_device::value>* =
                nullptr>
  inline void modify_device() {
    if (modified_flags.data() != nullptr) {
      modified_flags(1) =
          (modified_flags(1) > modified_flags(0) ? modified_flags(1)
                                                 : modified_flags(0)) +
          1;
      impl_report_device_modification();
#ifdef FLARE_ENABLE_DEBUG_DUALTENSOR_MODIFY_CHECK
      if (modified_flags(0) && modified_flags(1)) {
        std::string msg = "flare::DualTensor::modify_device ERROR: ";
        msg += "Concurrent modification of host and device tensors ";
        msg += "in DualTensor \"";
        msg += d_tensor.label();
        msg += "\"\n";
        flare::abort(msg.c_str());
      }
#endif
    }
  }

  template <
      class Dummy = DualTensor,
      std::enable_if_t<Dummy::impl_dualtensor_is_single_device::value>* = nullptr>
  inline void modify_device() {
    return;
  }

  inline void clear_sync_state() {
    if (modified_flags.data() != nullptr)
      modified_flags(1) = modified_flags(0) = 0;
  }

  //@}
  //! \name Methods for reallocating or resizing the Tensor objects.
  //@{

  /// \brief Reallocate both Tensor objects.
  ///
  /// This discards any existing contents of the objects, and resets
  /// their modified flags.  It does <i>not</i> copy the old contents
  /// of either Tensor into the new Tensor objects.
  template <class... TensorCtorArgs>
  void impl_realloc(const size_t n0, const size_t n1, const size_t n2,
                    const size_t n3, const size_t n4, const size_t n5,
                    const size_t n6, const size_t n7,
                    const detail::TensorCtorProp<TensorCtorArgs...>& arg_prop) {
    using alloc_prop_input = detail::TensorCtorProp<TensorCtorArgs...>;

    static_assert(!alloc_prop_input::has_label,
                  "The tensor constructor arguments passed to flare::realloc "
                  "must not include a label!");
    static_assert(
        !alloc_prop_input::has_pointer,
        "The tensor constructor arguments passed to flare::realloc must "
        "not include a pointer!");
    static_assert(
        !alloc_prop_input::has_memory_space,
        "The tensor constructor arguments passed to flare::realloc must "
        "not include a memory space instance!");

    const size_t new_extents[8] = {n0, n1, n2, n3, n4, n5, n6, n7};
    const bool sizeMismatch =
        detail::size_mismatch(h_tensor, h_tensor.rank_dynamic, new_extents);

    if (sizeMismatch) {
      ::flare::realloc(arg_prop, d_tensor, n0, n1, n2, n3, n4, n5, n6, n7);
      if (alloc_prop_input::initialize) {
        h_tensor = create_mirror_tensor(typename t_host::memory_space(), d_tensor);
      } else {
        h_tensor = create_mirror_tensor(flare::WithoutInitializing,
                                    typename t_host::memory_space(), d_tensor);
      }
    } else if (alloc_prop_input::initialize) {
      if constexpr (alloc_prop_input::has_execution_space) {
        const auto& exec_space =
            detail::get_property<detail::ExecutionSpaceTag>(arg_prop);
        ::flare::deep_copy(exec_space, d_tensor, typename t_dev::value_type{});
      } else
        ::flare::deep_copy(d_tensor, typename t_dev::value_type{});
    }

    /* Reset dirty flags */
    if (modified_flags.data() == nullptr) {
      modified_flags = t_modified_flags("DualTensor::modified_flags");
    } else
      modified_flags(1) = modified_flags(0) = 0;
  }

  template <class... TensorCtorArgs>
  void realloc(const detail::TensorCtorProp<TensorCtorArgs...>& arg_prop,
               const size_t n0 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n1 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n2 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n3 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n4 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n5 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n6 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n7 = FLARE_IMPL_CTOR_DEFAULT_ARG) {
    impl_realloc(n0, n1, n2, n3, n4, n5, n6, n7, arg_prop);
  }

  void realloc(const size_t n0 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n1 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n2 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n3 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n4 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n5 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n6 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n7 = FLARE_IMPL_CTOR_DEFAULT_ARG) {
    impl_realloc(n0, n1, n2, n3, n4, n5, n6, n7, detail::TensorCtorProp<>{});
  }

  template <typename I>
  std::enable_if_t<detail::is_tensor_ctor_property<I>::value> realloc(
      const I& arg_prop, const size_t n0 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t n1 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t n2 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t n3 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t n4 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t n5 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t n6 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t n7 = FLARE_IMPL_CTOR_DEFAULT_ARG) {
    impl_realloc(n0, n1, n2, n3, n4, n5, n6, n7, flare::tensor_alloc(arg_prop));
  }

  /// \brief Resize both tensors, copying old contents into new if necessary.
  ///
  /// This method only copies the old contents into the new Tensor
  /// objects for the device which was last marked as modified.
  template <class... TensorCtorArgs>
  void impl_resize(const detail::TensorCtorProp<TensorCtorArgs...>& arg_prop,
                   const size_t n0, const size_t n1, const size_t n2,
                   const size_t n3, const size_t n4, const size_t n5,
                   const size_t n6, const size_t n7) {
    using alloc_prop_input = detail::TensorCtorProp<TensorCtorArgs...>;

    static_assert(!alloc_prop_input::has_label,
                  "The tensor constructor arguments passed to flare::resize "
                  "must not include a label!");
    static_assert(
        !alloc_prop_input::has_pointer,
        "The tensor constructor arguments passed to flare::resize must "
        "not include a pointer!");
    static_assert(
        !alloc_prop_input::has_memory_space,
        "The tensor constructor arguments passed to flare::resize must "
        "not include a memory space instance!");

    const size_t new_extents[8] = {n0, n1, n2, n3, n4, n5, n6, n7};
    const bool sizeMismatch =
        detail::size_mismatch(h_tensor, h_tensor.rank_dynamic, new_extents);

    if (modified_flags.data() == nullptr) {
      modified_flags = t_modified_flags("DualTensor::modified_flags");
    }

    [[maybe_unused]] auto resize_on_device = [&](const auto& properties) {
      /* Resize on Device */
      if (sizeMismatch) {
        ::flare::resize(properties, d_tensor, n0, n1, n2, n3, n4, n5, n6, n7);
        if (alloc_prop_input::initialize) {
          h_tensor = create_mirror_tensor(typename t_host::memory_space(), d_tensor);
        } else {
          h_tensor = create_mirror_tensor(flare::WithoutInitializing,
                                      typename t_host::memory_space(), d_tensor);
        }

        /* Mark Device copy as modified */
        ++modified_flags(1);
      }
    };

    [[maybe_unused]] auto resize_on_host = [&](const auto& properties) {
      /* Resize on Host */
      if (sizeMismatch) {
        ::flare::resize(properties, h_tensor, n0, n1, n2, n3, n4, n5, n6, n7);
        if (alloc_prop_input::initialize) {
          d_tensor = create_mirror_tensor(typename t_dev::memory_space(), h_tensor);

        } else {
          d_tensor = create_mirror_tensor(flare::WithoutInitializing,
                                      typename t_dev::memory_space(), h_tensor);
        }

        /* Mark Host copy as modified */
        ++modified_flags(0);
      }
    };

    constexpr bool has_execution_space = alloc_prop_input::has_execution_space;

    if constexpr (has_execution_space) {
      using ExecSpace = typename alloc_prop_input::execution_space;
      const auto& exec_space =
          detail::get_property<detail::ExecutionSpaceTag>(arg_prop);
      constexpr bool exec_space_can_access_device =
          SpaceAccessibility<ExecSpace,
                             typename t_dev::memory_space>::accessible;
      constexpr bool exec_space_can_access_host =
          SpaceAccessibility<ExecSpace,
                             typename t_host::memory_space>::accessible;
      static_assert(exec_space_can_access_device || exec_space_can_access_host);
      if constexpr (exec_space_can_access_device) {
        sync<typename t_dev::memory_space>(exec_space);
        resize_on_device(arg_prop);
        return;
      }
      if constexpr (exec_space_can_access_host) {
        sync<typename t_host::memory_space>(exec_space);
        resize_on_host(arg_prop);
        return;
      }
    } else {
      if (modified_flags(1) >= modified_flags(0)) {
        resize_on_device(arg_prop);
      } else {
        resize_on_host(arg_prop);
      }
    }
  }

  void resize(const size_t n0 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n1 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n2 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n3 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n4 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n5 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n6 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n7 = FLARE_IMPL_CTOR_DEFAULT_ARG) {
    impl_resize(detail::TensorCtorProp<>{}, n0, n1, n2, n3, n4, n5, n6, n7);
  }

  template <class... TensorCtorArgs>
  void resize(const detail::TensorCtorProp<TensorCtorArgs...>& arg_prop,
              const size_t n0 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n1 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n2 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n3 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n4 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n5 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n6 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n7 = FLARE_IMPL_CTOR_DEFAULT_ARG) {
    impl_resize(arg_prop, n0, n1, n2, n3, n4, n5, n6, n7);
  }

  template <class I>
  std::enable_if_t<detail::is_tensor_ctor_property<I>::value> resize(
      const I& arg_prop, const size_t n0 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t n1 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t n2 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t n3 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t n4 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t n5 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t n6 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t n7 = FLARE_IMPL_CTOR_DEFAULT_ARG) {
    impl_resize(flare::tensor_alloc(arg_prop), n0, n1, n2, n3, n4, n5, n6, n7);
  }

  //@}
  //! \name Methods for getting capacity, stride, or dimension(s).
  //@{

  //! The allocation size (same as flare::Tensor::span).
  FLARE_INLINE_FUNCTION constexpr size_t span() const { return d_tensor.span(); }

  FLARE_INLINE_FUNCTION bool span_is_contiguous() const {
    return d_tensor.span_is_contiguous();
  }

  //! Get stride(s) for each dimension.
  template <typename iType>
  void stride(iType* stride_) const {
    d_tensor.stride(stride_);
  }

  template <typename iType>
  FLARE_INLINE_FUNCTION constexpr std::enable_if_t<
      std::is_integral<iType>::value, size_t>
  extent(const iType& r) const {
    return d_tensor.extent(r);
  }

  template <typename iType>
  FLARE_INLINE_FUNCTION constexpr std::enable_if_t<
      std::is_integral<iType>::value, int>
  extent_int(const iType& r) const {
    return static_cast<int>(d_tensor.extent(r));
  }

  //@}
};

}  // namespace flare

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
//
// Partial specializations of flare::subtensor() for DualTensor objects.
//

namespace flare {
namespace detail {

template <class V>
struct V2DV;

template <class D, class... P>
struct V2DV<Tensor<D, P...>> {
  using type = DualTensor<D, P...>;
};
} /* namespace detail */

template <class DataType, class... Properties, class... Args>
auto subtensor(const DualTensor<DataType, Properties...>& src, Args&&... args) {
  // leverage flare::Tensor facilities to deduce the properties of the subtensor
  using deduce_subtensor_type =
      decltype(subtensor(std::declval<Tensor<DataType, Properties...>>(),
                       std::forward<Args>(args)...));
  // map it back to dual tensor
  return typename detail::V2DV<deduce_subtensor_type>::type(
      src, std::forward<Args>(args)...);
}

} /* namespace flare */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace flare {

//
// Partial specialization of flare::deep_copy() for DualTensor objects.
//

template <class DT, class... DP, class ST, class... SP>
void deep_copy(DualTensor<DT, DP...>& dst, const DualTensor<ST, SP...>& src) {
  if (src.need_sync_device()) {
    deep_copy(dst.h_tensor, src.h_tensor);
    dst.modify_host();
  } else {
    deep_copy(dst.d_tensor, src.d_tensor);
    dst.modify_device();
  }
}

template <class ExecutionSpace, class DT, class... DP, class ST, class... SP>
void deep_copy(const ExecutionSpace& exec, DualTensor<DT, DP...>& dst,
               const DualTensor<ST, SP...>& src) {
  if (src.need_sync_device()) {
    deep_copy(exec, dst.h_tensor, src.h_tensor);
    dst.modify_host();
  } else {
    deep_copy(exec, dst.d_tensor, src.d_tensor);
    dst.modify_device();
  }
}

}  // namespace flare

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace flare {

//
// Non-member resize and realloc
//

template <class... Properties, class... Args>
void resize(DualTensor<Properties...>& dv, Args&&... args) noexcept(
    noexcept(dv.resize(std::forward<Args>(args)...))) {
  dv.resize(std::forward<Args>(args)...);
}

template <class... TensorCtorArgs, class... Properties, class... Args>
void resize(
    const detail::TensorCtorProp<TensorCtorArgs...>& arg_prop,
    DualTensor<Properties...>& dv,
    Args&&... args) noexcept(noexcept(dv.resize(arg_prop,
                                                std::forward<Args>(args)...))) {
  dv.resize(arg_prop, std::forward<Args>(args)...);
}

template <class I, class... Properties, class... Args>
std::enable_if_t<detail::is_tensor_ctor_property<I>::value> resize(
    const I& arg_prop, DualTensor<Properties...>& dv,
    Args&&... args) noexcept(noexcept(dv.resize(arg_prop,
                                                std::forward<Args>(args)...))) {
  dv.resize(arg_prop, std::forward<Args>(args)...);
}

template <class... TensorCtorArgs, class... Properties, class... Args>
void realloc(const detail::TensorCtorProp<TensorCtorArgs...>& arg_prop,
             DualTensor<Properties...>& dv,
             Args&&... args) noexcept(noexcept(dv
                                                   .realloc(std::forward<Args>(
                                                       args)...))) {
  dv.realloc(arg_prop, std::forward<Args>(args)...);
}

template <class... Properties, class... Args>
void realloc(DualTensor<Properties...>& dv, Args&&... args) noexcept(
    noexcept(dv.realloc(std::forward<Args>(args)...))) {
  dv.realloc(std::forward<Args>(args)...);
}

template <class I, class... Properties, class... Args>
std::enable_if_t<detail::is_tensor_ctor_property<I>::value> realloc(
    const I& arg_prop, DualTensor<Properties...>& dv,
    Args&&... args) noexcept(noexcept(dv.realloc(arg_prop,
                                                 std::forward<Args>(
                                                     args)...))) {
  dv.realloc(arg_prop, std::forward<Args>(args)...);
}

}  // end namespace flare

#endif  // FLARE_DUAL_TENSOR_H_
