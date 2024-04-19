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

#ifdef __cplusplus
namespace fly
{
class array;

/// \ingroup device_func_count
/// \copydoc getDeviceCount()
/// \deprecated Use getDeviceCount() instead
FLY_DEPRECATED("Use getDeviceCount instead")
FLY_API int devicecount();

/// \ingroup device_func_get
/// \copydoc getDevice()
/// \deprecated Use getDevice() instead
FLY_DEPRECATED("Use getDevice instead")
FLY_API int deviceget();

/// \ingroup device_func_set
/// \copydoc setDevice()
/// \deprecated Use setDevice() instead
FLY_DEPRECATED("Use setDevice instead")
FLY_API void deviceset(const int device);

/// \ingroup imageio_func_load
/// \copydoc loadImage
/// \deprecated Use \ref loadImage instead
FLY_DEPRECATED("Use loadImage instead")
FLY_API array loadimage(const char* filename, const bool is_color=false);

/// \ingroup imageio_func_save
/// \copydoc saveImage
/// \deprecated Use \ref saveImage instead
FLY_DEPRECATED("Use saveImage instead")
FLY_API void saveimage(const char* filename, const array& in);

/// \ingroup image_func_gauss
/// \copydoc image_func_gauss
/// \deprecated Use \ref gaussianKernel instead
FLY_DEPRECATED("Use gaussianKernel instead")
FLY_API array gaussiankernel(const int rows, const int cols, const double sig_r = 0, const double sig_c = 0);

/// \ingroup reduce_func_all_true
/// \copydoc fly::allTrue(const array&)
/// \deprecated Use \ref fly::allTrue(const array&) instead
template<typename T>
FLY_DEPRECATED("Use allTrue instead")
T alltrue(const array &in);

/// \ingroup reduce_func_any_true
/// \copydoc fly::allTrue(const array&)
/// \deprecated Use \ref fly::anyTrue(const array&) instead
template<typename T>
FLY_DEPRECATED("Use anyTrue instead")
T anytrue(const array &in);

/// \ingroup reduce_func_all_true
/// \copydoc allTrue
/// \deprecated Use \ref fly::allTrue instead
FLY_DEPRECATED("Use allTrue instead")
FLY_API array alltrue(const array &in, const int dim = -1);

/// \ingroup reduce_func_any_true
/// \copydoc anyTrue
/// \deprecated Use \ref fly::anyTrue instead
FLY_DEPRECATED("Use anyTrue instead")
FLY_API array anytrue(const array &in, const int dim = -1);

/// \ingroup set_func_unique
/// \copydoc setUnique
/// \deprecated Use \ref setUnique instead
FLY_DEPRECATED("Use setUnique instead")
FLY_API array setunique(const array &in, const bool is_sorted=false);

/// \ingroup set_func_union
/// \copydoc setUnion
/// \deprecated Use \ref setUnion instead
FLY_DEPRECATED("Use setUnion instead")
FLY_API array setunion(const array &first, const array &second, const bool is_unique=false);

/// \ingroup set_func_intersect
/// \copydoc setIntersect
/// \deprecated Use \ref setIntersect instead
FLY_DEPRECATED("Use setIntersect instead")
FLY_API array setintersect(const array &first, const array &second, const bool is_unique=false);

/// \ingroup image_func_histequal
/// \copydoc histEqual
/// \deprecated Use \ref histEqual instead
FLY_DEPRECATED("Use histEqual instead")
FLY_API array histequal(const array& in, const array& hist);

/// \ingroup image_func_colorspace
/// \copydoc colorSpace
/// \deprecated Use \ref colorSpace instead
FLY_DEPRECATED("Use colorSpace instead")
FLY_API array colorspace(const array& image, const CSpace to, const CSpace from);

/// Image Filtering
/// \code
/// // filter (convolve) an image with a 3x3 sobel kernel
/// const float h_kernel[] = { -2.0, -1.0,  0.0,
///                            -1.0,  0.0,  1.0,
///                             0.0,  1.0,  2.0 };
/// array kernel = array(3,3,h_kernel);
/// array img_out = filter(img_in, kernel);
/// \endcode
///
/// \param[in] image
/// \param[in] kernel coefficient matrix
/// \returns filtered image (same size as input)
///
/// \note Filtering done using correlation. Array values outside bounds are assumed to have zero value (0).
/// \ingroup image_func_filter
/// \deprecated Use \ref fly::convolve instead
FLY_DEPRECATED("Use fly::convolve instead")
FLY_API array filter(const array& image, const array& kernel);

/// \ingroup reduce_func_product
/// \copydoc product(const array&, const int);
/// \deprecated Use \ref product instead
FLY_DEPRECATED("Use fly::product instead")
FLY_API array mul(const array& in, const int dim = -1);

/// \ingroup reduce_func_product
/// \copydoc product(const array&)
/// \deprecated Use \ref product instead
template<typename T>
FLY_DEPRECATED("Use fly::product instead")
T mul(const array& in);

/// \ingroup device_func_prop
/// \copydoc deviceInfo
/// \deprecated Use \ref deviceInfo instead
FLY_DEPRECATED("Use deviceInfo instead")
FLY_API void deviceprop(char* d_name, char* d_platform, char *d_toolkit, char* d_compute);

}
#endif
