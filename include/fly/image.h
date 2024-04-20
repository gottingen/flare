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
#include <fly/features.h>

#ifdef __cplusplus
namespace fly
{
class array;

/**
   C++ Interface for calculating the gradients

   \param[out] dx the gradient along first dimension
   \param[out] dy the gradient along second dimension
   \param[in] in is the input array

   \ingroup calc_func_grad
*/
FLY_API void grad(array& dx, array& dy, const array& in);

/**
    C++ Interface for loading an image

    \param[in] filename is name of file to be loaded
    \param[in] is_color boolean denoting if the image should be loaded as 1 channel or 3 channel
    \return image loaded as \ref fly::array()

    \ingroup imageio_func_load
*/
FLY_API array loadImage(const char* filename, const bool is_color=false);

/**
    C++ Interface for saving an image

    \param[in] filename is name of file to be loaded
    \param[in] in is the flare array to be saved as an image

    \ingroup imageio_func_save
*/
FLY_API void saveImage(const char* filename, const array& in);

/**
    C++ Interface for loading an image from memory

    \param[in] ptr is the location of the image data in memory. This is the pointer
    created by saveImage.
    \return image loaded as \ref fly::array()

    \note The pointer used is a void* cast of the FreeImage type FIMEMORY which is
    created using the FreeImage_OpenMemory API. If the user is opening a FreeImage
    stream external to Flare, that pointer can be passed to this function as well.

    \ingroup imagemem_func_load
*/
FLY_API array loadImageMem(const void *ptr);

/**
    C++ Interface for saving an image to memory

    \param[in] in is the flare array to be saved as an image
    \param[in] format is the type of image to create in memory. The enum borrows from
    the FREE_IMAGE_FORMAT enum of FreeImage. Other values not included in imageFormat
    but included in FREE_IMAGE_FORMAT can also be passed to this function.

    \return a void* pointer which is a type cast of the FreeImage type FIMEMORY* pointer.

    \note Ensure that \ref deleteImageMem is called on this pointer. Otherwise there will
    be memory leaks

    \ingroup imagemem_func_save
*/
FLY_API void* saveImageMem(const array& in, const imageFormat format = FLY_FIF_PNG);

/**
    C++ Interface for deleting memory created by \ref saveImageMem or
    \ref fly_save_image_memory

    \param[in] ptr is the pointer to the FreeImage stream created by saveImageMem.

    \ingroup imagemem_func_delete
*/
FLY_API void deleteImageMem(void *ptr);


/**
    C++ Interface for loading an image as its original type

    This load image function allows you to load images as u8, u16 or f32
    depending on the type of input image as shown by the table below.

     Bits per Color (Gray/RGB/RGBA Bits Per Pixel) | Array Type  | Range
    -----------------------------------------------|-------------|---------------
      8 ( 8/24/32  BPP)                            | u8          | 0 - 255
     16 (16/48/64  BPP)                            | u16         | 0 - 65535
     32 (32/96/128 BPP)                            | f32         | 0 - 1

    \param[in] filename is name of file to be loaded
    \return image loaded as \ref fly::array()

    \ingroup imageio_func_load
*/
FLY_API array loadImageNative(const char* filename);


/**
    C++ Interface for saving an image without modifications

    This function only accepts u8, u16, f32 arrays. These arrays are saved to
    images without any modifications.

    You must also note that note all image type support 16 or 32 bit images.

    The best options for 16 bit images are PNG, PPM and TIFF.
    The best option for 32 bit images is TIFF.
    These allow lossless storage.

    The images stored have the following properties:

     Array Type  | Bits per Color (Gray/RGB/RGBA Bits Per Pixel) | Range
    -------------|-----------------------------------------------|---------------
     u8          |  8 ( 8/24/32  BPP)                            | 0 - 255
     u16         | 16 (16/48/64  BPP)                            | 0 - 65535
     f32         | 32 (32/96/128 BPP)                            | 0 - 1

    \param[in] filename is name of file to be saved
    \param[in] in is the array to be saved. Should be u8 for saving 8-bit image,
    u16 for 16-bit image, and f32 for 32-bit image.

    \ingroup imageio_func_save
*/
FLY_API void saveImageNative(const char* filename, const array& in);


/**
    Function to check if Image IO is available

    \returns true if Flare was commpiled with ImageIO support, false otherwise.
    \ingroup imageio_func_available
*/
FLY_API bool isImageIOAvailable();

/**
    C++ Interface for resizing an image to specified dimensions

    \param[in] in is input image
    \param[in] odim0 is the size for the first output dimension
    \param[in] odim1 is the size for the second output dimension
    \param[in] method is the interpolation type (Nearest by default)
    \return the resized image of specified by \p odim0 and \p odim1

    \ingroup transform_func_resize
*/
FLY_API array resize(const array& in, const dim_t odim0, const dim_t odim1, const interpType method=FLY_INTERP_NEAREST);

/**
    C++ Interface for resizing an image to specified scales

    \param[in] scale0 is scale used for first input dimension
    \param[in] scale1 is scale used for second input dimension
    \param[in] in is input image
    \param[in] method is the interpolation type (Nearest by default)
    \return the image scaled by the specified by \p scale0 and \p scale1

    \ingroup transform_func_resize
*/
FLY_API array resize(const float scale0, const float scale1, const array& in, const interpType method=FLY_INTERP_NEAREST);

/**
    C++ Interface for resizing an image to specified scale

    \param[in] scale is scale used for both input dimensions
    \param[in] in is input image
    \param[in] method is the interpolation type (Nearest by default)
    \return the image scaled by the specified by \p scale

    \ingroup transform_func_resize
*/
FLY_API array resize(const float scale, const array& in, const interpType method=FLY_INTERP_NEAREST);

/**
    C++ Interface for rotating an image

    \param[in] in is input image
    \param[in] theta is the degree (in radians) by which the input is rotated
    \param[in] crop if true the output is cropped original dimensions. If false the output dimensions scale based on \p theta
    \param[in] method is the interpolation type (Nearest by default)
    \return the image rotated by \p theta

    \ingroup transform_func_rotate
*/
FLY_API array rotate(const array& in, const float theta, const bool crop=true, const interpType method=FLY_INTERP_NEAREST);

/**
    C++ Interface for transforming an image

    \param[in] in is input image
    \param[in] transform is transformation matrix
    \param[in] odim0 is the first output dimension
    \param[in] odim1 is the second output dimension
    \param[in] method is the interpolation type (Nearest by default)
    \param[in] inverse if true applies inverse transform, if false applies forward transoform
    \return the transformed image

    \ingroup transform_func_transform
*/
FLY_API array transform(const array& in, const array& transform, const dim_t odim0 = 0, const dim_t odim1 = 0,
                      const interpType method=FLY_INTERP_NEAREST, const bool inverse=true);

/**
    C++ Interface for transforming coordinates

    \param[in] tf is transformation matrix
    \param[in] d0 is the first input dimension
    \param[in] d1 is the second input dimension
    \return the transformed coordinates

    \ingroup transform_func_coordinates
*/
FLY_API array transformCoordinates(const array& tf, const float d0, const float d1);

/**
    C++ Interface for translating an image

    \param[in] in is input image
    \param[in] trans0 is amount by which the first dimension is translated
    \param[in] trans1 is amount by which the second dimension is translated
    \param[in] odim0 is the first output dimension
    \param[in] odim1 is the second output dimension
    \param[in] method is the interpolation type (Nearest by default)
    \return the translated image

    \ingroup transform_func_translate
*/
FLY_API array translate(const array& in, const float trans0, const float trans1, const dim_t odim0 = 0, const dim_t odim1 = 0, const interpType method=FLY_INTERP_NEAREST);

/**
    C++ Interface for scaling an image

    \param[in] in is input image
    \param[in] scale0 is amount by which the first dimension is scaled
    \param[in] scale1 is amount by which the second dimension is scaled
    \param[in] odim0 is the first output dimension
    \param[in] odim1 is the second output dimension
    \param[in] method is the interpolation type (Nearest by default)
    \return the scaled image

    \ingroup transform_func_scale
*/
FLY_API array scale(const array& in, const float scale0, const float scale1, const dim_t odim0 = 0, const dim_t odim1 = 0, const interpType method=FLY_INTERP_NEAREST);

/**
    C++ Interface for skewing an image

    \param[in] in is input image
    \param[in] skew0 is amount by which the first dimension is skewed
    \param[in] skew1 is amount by which the second dimension is skewed
    \param[in] odim0 is the first output dimension
    \param[in] odim1 is the second output dimension
    \param[in] inverse if true applies inverse transform, if false applies forward transoform
    \param[in] method is the interpolation type (Nearest by default)
    \return the skewed image

    \ingroup transform_func_skew
*/
FLY_API array skew(const array& in, const float skew0, const float skew1, const dim_t odim0 = 0, const dim_t odim1 = 0, const bool inverse=true, const interpType method=FLY_INTERP_NEAREST);

/**
    C++ Interface for bilateral filter

    \param[in]  in array is the input image
    \param[in]  spatial_sigma is the spatial variance parameter that decides the filter window
    \param[in]  chromatic_sigma is the chromatic variance parameter
    \param[in]  is_color indicates if the input \p in is color image or grayscale
    \return     the processed image

    \ingroup image_func_bilateral
*/
FLY_API array bilateral(const array &in, const float spatial_sigma, const float chromatic_sigma, const bool is_color=false);

/**
   C++ Interface for histogram

   \snippet test/histogram.cpp ex_image_hist_minmax

   \param[in]  in is the input array
   \param[in]  nbins  Number of bins to populate between min and max
   \param[in]  minval minimum bin value (accumulates -inf to min)
   \param[in]  maxval minimum bin value (accumulates max to +inf)
   \return     histogram array of type u32

   \ingroup image_func_histogram
 */
FLY_API array histogram(const array &in, const unsigned nbins, const double minval, const double maxval);

/**
   C++ Interface for histogram

   \snippet test/histogram.cpp ex_image_hist_nominmax

   \param[in]  in is the input array
   \param[in]  nbins  Number of bins to populate between min and max
   \return     histogram array of type u32

   \ingroup image_func_histogram
 */
FLY_API array histogram(const array &in, const unsigned nbins);

/**
    C++ Interface for mean shift

    \param[in]  in array is the input image
    \param[in]  spatial_sigma is the spatial variance parameter that decides the filter window
    \param[in]  chromatic_sigma is the chromatic variance parameter
    \param[in]  iter is the number of iterations filter operation is performed
    \param[in]  is_color indicates if the input \p in is color image or grayscale
    \return     the processed image

    \ingroup image_func_mean_shift
*/
FLY_API array meanShift(const array& in, const float spatial_sigma, const float chromatic_sigma, const unsigned iter, const bool is_color=false);

/**
    C++ Interface for minimum filter

    \param[in]  in array is the input image
    \param[in]  wind_length is the kernel height
    \param[in]  wind_width is the kernel width
    \param[in]  edge_pad value will decide what happens to border when running
                filter in their neighborhood. It takes one of the values [\ref FLY_PAD_ZERO | \ref FLY_PAD_SYM]
    \return     the processed image

    \ingroup image_func_minfilt
*/
FLY_API array minfilt(const array& in, const dim_t wind_length = 3, const dim_t wind_width = 3, const borderType edge_pad = FLY_PAD_ZERO);

/**
    C++ Interface for maximum filter

    \param[in]  in array is the input image
    \param[in]  wind_length is the kernel height
    \param[in]  wind_width is the kernel width
    \param[in]  edge_pad value will decide what happens to border when running
                filter in their neighborhood. It takes one of the values [\ref FLY_PAD_ZERO | \ref FLY_PAD_SYM]
    \return     the processed image

    \ingroup image_func_maxfilt
*/
FLY_API array maxfilt(const array& in, const dim_t wind_length = 3, const dim_t wind_width = 3, const borderType edge_pad = FLY_PAD_ZERO);

/**
    C++ Interface for image dilation (max filter)

    \param[in]  in array is the input image
    \param[in]  mask is the neighborhood window
    \return     the dilated image

    \note if \p mask is all ones, this function behaves like max filter

    \ingroup image_func_dilate
*/
FLY_API array dilate(const array& in, const array& mask);

/**
    C++ Interface for 3D image dilation

    \param[in]  in array is the input volume
    \param[in]  mask is the neighborhood delta volume
    \return     the dilated volume

    \ingroup image_func_dilate3d
*/
FLY_API array dilate3(const array& in, const array& mask);

/**
    C++ Interface for image erosion (min filter)

    \param[in]  in array is the input image
    \param[in]  mask is the neighborhood window
    \return     the eroded image

    \note This function can be used as min filter by using a mask of all ones

    \ingroup image_func_erode
*/
FLY_API array erode(const array& in, const array& mask);

/**
    C++ Interface for 3d for image erosion

    \param[in]  in array is the input volume
    \param[in]  mask is the neighborhood delta volume
    \return     the eroded volume

    \ingroup image_func_erode3d
*/
FLY_API array erode3(const array& in, const array& mask);

/**
    C++ Interface for getting regions in an image

    Below given are sample input and output for each type of connectivity value for \p type

    <table border="0">
    <tr>
    <td> Example for \p type == \ref FLY_CONNECTIVITY_8_4 </td>
    <td> Example for \p type == \ref FLY_CONNECTIVITY_4 </td>
    </tr>
    <tr>
    <td>
        \snippet test/regions.cpp ex_image_regions
    </td>
    <td>
        \snippet test/regions.cpp ex_image_regions_4conn
    </td>
    </tr>
    </table>

    \param[in]  in array should be binary image of type \ref b8
    \param[in]  connectivity can take one of the following [\ref FLY_CONNECTIVITY_4 | \ref FLY_CONNECTIVITY_8_4]
    \param[in]  type is type of output array
    \return     returns array with labels indicating different regions. Throws exceptions if any issue occur.

    \ingroup image_func_regions
*/
FLY_API array regions(const array& in, const fly::connectivity connectivity=FLY_CONNECTIVITY_4, const dtype type=f32);

/**
   C++ Interface for extracting sobel gradients

   \param[out] dx is derivative along horizontal direction
   \param[out] dy is derivative along vertical direction
   \param[in]  img is an array with image data
   \param[in]  ker_size sobel kernel size or window size

   \note If \p img is 3d array, a batch operation will be performed.

   \ingroup image_func_sobel
 */
FLY_API void sobel(array &dx, array &dy, const array &img, const unsigned ker_size=3);

/**
   C++ Interface for sobel filtering

   \param[in]  img is an array with image data
   \param[in]  ker_size sobel kernel size or window size
   \param[in]  isFast = true uses \f$G=G_x+G_y\f$, otherwise \f$G=\sqrt (G_x^2+G_y^2)\f$
   \return     an array with sobel gradient values

   \note If \p img is 3d array, a batch operation will be performed.

   \ingroup image_func_sobel
 */
FLY_API array sobel(const array &img, const unsigned ker_size=3, const bool isFast=false);

/**
   C++ Interface for RGB to gray conversion

   \param[in]  in is an array in the RGB colorspace
   \param[in]  rPercent is percentage of red channel value contributing to grayscale intensity
   \param[in]  gPercent is percentage of green channel value contributing to grayscale intensity
   \param[in]  bPercent is percentage of blue channel value contributing to grayscale intensity
   \return     array in Grayscale colorspace

   \note \p in must be three dimensional for RGB to Grayscale conversion.

   \ingroup image_func_rgb2gray
 */
FLY_API array rgb2gray(const array& in, const float rPercent=0.2126f, const float gPercent=0.7152f, const float bPercent=0.0722f);

/**
   C++ Interface for gray to RGB conversion

   \param[in]  in is an array in the Grayscale colorspace
   \param[in]  rFactor is percentage of intensity value contributing to red channel
   \param[in]  gFactor is percentage of intensity value contributing to green channel
   \param[in]  bFactor is percentage of intensity value contributing to blue channel
   \return     array in RGB colorspace

   \note \p in must be two dimensional for Grayscale to RGB conversion.

   \ingroup image_func_gray2rgb
 */
FLY_API array gray2rgb(const array& in, const float rFactor=1.0, const float gFactor=1.0, const float bFactor=1.0);

/**
   C++ Interface for histogram equalization

   \snippet test/histogram.cpp ex_image_histequal

   \param[in]  in is the input array, non-normalized input (!! assumes values [0-255] !!)
   \param[in]  hist target histogram to approximate in output (based on number of bins)
   \return     data with histogram approximately equal to histogram

   \note \p in must be two dimensional.

   \ingroup image_func_histequal
 */
FLY_API array histEqual(const array& in, const array& hist);

/**
   C++ Interface for generating gausian kernels

   \param[in]  rows number of rows of the kernel
   \param[in]  cols number of columns of the kernel
   \param[in]  sig_r (default 0) (calculated internally as 0.25 * rows + 0.75)
   \param[in]  sig_c (default 0) (calculated internally as 0.25 * cols + 0.75)
   \return     an array with values generated using gaussian function

   \ingroup image_func_gauss
 */
FLY_API array gaussianKernel(const int rows, const int cols, const double sig_r = 0, const double sig_c = 0);

/**
   C++ Interface for converting HSV to RGB

   \param[in]  in is an array in the HSV colorspace
   \return     array in RGB colorspace

   \note \p in must be three dimensional

   \ingroup image_func_hsv2rgb
 */
FLY_API array hsv2rgb(const array& in);

/**
   C++ Interface for converting RGB to HSV

   \param[in]  in is an array in the RGB colorspace
   \return     array in HSV colorspace

   \note \p in must be three dimensional

   \ingroup image_func_rgb2hsv
 */
FLY_API array rgb2hsv(const array& in);

/**
   C++ Interface wrapper for colorspace conversion

   \param[in]  image is the input array
   \param[in]  to is the target array colorspace
   \param[in]  from is the input array colorspace
   \return     array in target colorspace

   \note  \p image must be 3 dimensional for \ref FLY_HSV to \ref FLY_RGB, \ref FLY_RGB to
   \ref FLY_HSV, & \ref FLY_RGB to \ref FLY_GRAY transformations. For \ref FLY_GRAY to \ref FLY_RGB
   transformation, 2D array is expected.

   \ingroup image_func_colorspace
 */
FLY_API array colorSpace(const array& image, const CSpace to, const CSpace from);

/**
   C++ Interface for rearranging windowed sections of an input into columns
   (or rows)

   \param[in]  in is the input array
   \param[in]  wx is the window size along dimension 0
   \param[in]  wy is the window size along dimension 1
   \param[in]  sx is the stride along dimension 0
   \param[in]  sy is the stride along dimension 1
   \param[in]  px is the padding along dimension 0
   \param[in]  py is the padding along dimension 1
   \param[in]  is_column determines whether the section becomes a column (if
               true) or a row (if false)
   \returns    an array with the input's sections rearraged as columns (or rows)

   \note \p in can hold multiple images for processing if it is three or
         four-dimensional
   \note \p wx and \p wy must be between [1, input.dims(0 (1)) + px (py)]
   \note \p sx and \p sy must be greater than 1
   \note \p px and \p py must be between [0, wx (wy) - 1]. Padding becomes part of
         the input image prior to the windowing

   \ingroup image_func_unwrap
*/
FLY_API array unwrap(const array& in, const dim_t wx, const dim_t wy,
                   const dim_t sx, const dim_t sy, const dim_t px=0, const dim_t py=0,
                   const bool is_column = true);

/**
   C++ Interface for performing the opposite of \ref unwrap

   \param[in]  in is the input array
   \param[in]  ox is the output's dimension 0 size
   \param[in]  oy is the output's dimension 1 size
   \param[in]  wx is the window size along dimension 0
   \param[in]  wy is the window size along dimension 1
   \param[in]  sx is the stride along dimension 0
   \param[in]  sy is the stride along dimension 1
   \param[in]  px is the padding along dimension 0
   \param[in]  py is the padding along dimension 1
   \param[in]  is_column determines whether an output patch is formed from a
               column (if true) or a row (if false)
   \returns    an array with the input's columns (or rows) reshaped as patches

   \note Wrap is typically used to recompose an unwrapped image. If this is the
         case, use the same parameters that were used in \ref unwrap(). Also
         use the original image size (before unwrap) for \p ox and \p oy.
   \note The window/patch size, \p wx \f$\times\f$ \p wy, must equal
         `input.dims(0)` (or `input.dims(1)` if \p is_column is false).
   \note \p sx and \p sy must be at least 1
   \note \p px and \p py must be between [0, wx) and [0, wy), respectively
   \note The number of patches, `input.dims(1)` (or `input.dims(0)` if
         \p is_column is false), must equal \f$nx \times\ ny\f$, where
         \f$\displaystyle nx = \frac{ox + 2px - wx}{sx} + 1\f$ and
         \f$\displaystyle ny = \frac{oy + 2py - wy}{sy} + 1\f$
   \note Batched wrap can be performed on multiple 2D slices at once if \p in
         is three or four-dimensional

   \ingroup image_func_wrap
*/
FLY_API array wrap(const array& in,
                 const dim_t ox, const dim_t oy,
                 const dim_t wx, const dim_t wy,
                 const dim_t sx, const dim_t sy,
                 const dim_t px = 0, const dim_t py = 0,
                 const bool is_column = true);

/**
   C++ Interface wrapper for summed area tables

   \param[in]  in is the input array
   \returns the summed area table of input image

   \ingroup image_func_sat
*/
FLY_API array sat(const array& in);

/**
   C++ Interface for converting YCbCr to RGB

   \param[in]  in is an array in the YCbCr colorspace
   \param[in]  standard specifies the ITU-R BT "xyz" standard which determines the Kb, Kr values
   used in colorspace conversion equation
   \return     array in RGB colorspace

   \note \p in must be three dimensional and values should lie in the range [0,1]

   \ingroup image_func_ycbcr2rgb
 */
FLY_API array ycbcr2rgb(const array& in, const YCCStd standard=FLY_YCC_601);

/**
   C++ Interface for converting RGB to YCbCr

   \param[in]  in is an array in the RGB colorspace
   \param[in]  standard specifies the ITU-R BT "xyz" standard which determines the Kb, Kr values
   used in colorspace conversion equation
   \return     array in YCbCr colorspace

   \note \p in must be three dimensional and values should lie in the range [0,1]

   \ingroup image_func_rgb2ycbcr
 */
FLY_API array rgb2ycbcr(const array& in, const YCCStd standard=FLY_YCC_601);

/**
   C++ Interface for calculating an image moment

   \param[out] out is a pointer to a pre-allocated array where the calculated moment(s) will be placed.
   User is responsible for ensuring enough space to hold all requested moments
   \param[in]  in is the input image
   \param[in] moment is moment(s) to calculate

   \ingroup image_func_moments
 */
FLY_API void moments(double* out, const array& in, const momentType moment=FLY_MOMENT_FIRST_ORDER);

/**
   C++ Interface for calculating image moments

   \param[in]  in contains the input image(s)
   \param[in] moment is moment(s) to calculate
   \return array containing the requested moment of each image

   \ingroup image_func_moments
 */
FLY_API array moments(const array& in, const momentType moment=FLY_MOMENT_FIRST_ORDER);

/**
   C++ Interface for canny edge detector

   \param[in] in                  is the input image
   \param[in] thresholdType       determines if user set high threshold is to be used or not. It
                                  can take values defined by the enum \ref fly_canny_threshold
   \param[in] lowThresholdRatio   is the lower threshold % of maximum or auto-derived high threshold
   \param[in] highThresholdRatio  is the higher threshold % of maximum value in gradient image used
                                  in hysteresis procedure. This value is ignored if
                                  \ref FLY_CANNY_THRESHOLD_AUTO_OTSU is chosen as
                                  \ref fly_canny_threshold
   \param[in] sobelWindow     is the window size of sobel kernel for computing gradient direction and
                              magnitude
   \param[in] isFast     indicates if L<SUB>1</SUB> norm(faster but less accurate) is used to compute
                         image gradient magnitude instead of L<SUB>2</SUB> norm.
   \return binary array containing edges

   \ingroup image_func_canny
*/
FLY_API array canny(const array& in, const cannyThreshold thresholdType,
                  const float lowThresholdRatio, const float highThresholdRatio,
                  const unsigned sobelWindow = 3, const bool isFast = false);

/**
   C++ Interface for gradient anisotropic(non-linear diffusion) smoothing

   \param[in] in is the input image, expects non-integral (float/double) typed fly::array
   \param[in] timestep is the time step used in solving the diffusion equation.
   \param[in] conductance parameter controls the sensitivity of conductance in diffusion equation.
   \param[in] iterations is the number of times the diffusion step is performed.
   \param[in] fftype indicates whether quadratic or exponential flux function is used by algorithm.
    \param[in] diffusionKind will let the user choose what kind of diffusion method to perform. It will take
               any value of enum \ref diffusionEq
   \return A filtered image that is of same size as the input.

   \ingroup image_func_anisotropic_diffusion
*/
FLY_API array anisotropicDiffusion(const fly::array& in, const float timestep,
                                 const float conductance, const unsigned iterations,
                                 const fluxFunction fftype=FLY_FLUX_EXPONENTIAL,
                                 const diffusionEq diffusionKind=FLY_DIFFUSION_GRAD);

/**
  C++ Interface for Iterative deconvolution algorithm

  \param[in] in is the blurred input image
  \param[in] ker is the kernel(point spread function) known to have caused
             the blur in the system
  \param[in] iterations is the number of iterations the algorithm will run
  \param[in] relaxFactor is the relaxation factor multiplied with distance
             of estimate from observed image.
  \param[in] algo takes value of type enum \ref fly_iterative_deconv_algo
             indicating the iterative deconvolution algorithm to be used
  \return sharp image estimate generated from the blurred input

  \note \p relax_factor argument is ignore when it
  \ref FLY_ITERATIVE_DECONV_RICHARDSONLUCY algorithm is used.

  \ingroup image_func_iterative_deconv
 */
FLY_API array iterativeDeconv(const array& in, const array& ker,
                            const unsigned iterations, const float relaxFactor,
                            const iterativeDeconvAlgo algo);

/**
   C++ Interface for Tikhonov deconvolution algorithm

   \param[in] in is the blurred input image
   \param[in] psf is the kernel(point spread function) known to have caused
              the blur in the system
   \param[in] gamma is a user defined regularization constant
   \param[in] algo takes different meaning depending on the algorithm chosen.
              If \p algo is FLY_INVERSE_DECONV_TIKHONOV, then \p gamma is
              a user defined regularization constant.
   \return sharp image estimate generated from the blurred input

   \ingroup image_func_inverse_deconv
 */
FLY_API array inverseDeconv(const array& in, const array& psf,
                          const float gamma, const inverseDeconvAlgo algo);

/**
   C++ Interface for confidence connected components

   \param[in] in is the input image, expects non-integral (float/double)
              typed fly_array
   \param[in] seeds is an fly::array of x & y coordinates of the seed points
              with coordinate values along columns of this fly::array i.e. they
              are not stored in interleaved fashion.
   \param[in] radius is the neighborhood region to be considered around
              each seed point
   \param[in] multiplier controls the threshold range computed from
              the mean and variance of seed point neighborhoods
   \param[in] iter is number of iterations
   \param[in] segmentedValue is the value to which output array valid
              pixels are set to.
   \return out is the output fly_array having the connected components

   \ingroup image_func_confidence_cc
*/
FLY_API array confidenceCC(const array &in, const array &seeds,
                         const unsigned radius,
                         const unsigned multiplier, const int iter,
                         const double segmentedValue);

/**
   C++ Interface for confidence connected components

   \param[in] in is the input image, expects non-integral (float/double)
              typed fly_array
   \param[in] seedx is an fly::array of x coordinates of the seed points
   \param[in] seedy is an fly::array of y coordinates of the seed points
   \param[in] radius is the neighborhood region to be considered around
              each seed point
   \param[in] multiplier controls the threshold range computed from
              the mean and variance of seed point neighborhoods
   \param[in] iter is number of iterations
   \param[in] segmentedValue is the value to which output array valid
              pixels are set to.
   \return out is the output fly_array having the connected components

   \ingroup image_func_confidence_cc
*/
FLY_API array confidenceCC(const array &in, const array &seedx,
                         const array &seedy, const unsigned radius,
                         const unsigned multiplier, const int iter,
                         const double segmentedValue);

/**
   C++ Interface for confidence connected components

   \param[in] in is the input image, expects non-integral (float/double)
              typed fly_array
   \param[in] num_seeds is the total number of seeds
   \param[in] seedx is an array of x coordinates of the seed points
   \param[in] seedy is an array of y coordinates of the seed points
   \param[in] radius is the neighborhood region to be considered around
              each seed point
   \param[in] multiplier controls the threshold range computed from
              the mean and variance of seed point neighborhoods
   \param[in] iter is number of iterations
   \param[in] segmentedValue is the value to which output array valid
              pixels are set to.
   \return out is the output fly_array having the connected components

   \ingroup image_func_confidence_cc
*/
FLY_API array confidenceCC(const array &in, const size_t num_seeds,
                         const unsigned *seedx, const unsigned *seedy,
                         const unsigned radius, const unsigned multiplier,
                         const int iter, const double segmentedValue);

}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    /**
        C Interface for calculating the gradients

        \param[out] dx the gradient along first dimension
        \param[out] dy the gradient along second dimension
        \param[in]  in is the input array
        \return     \ref FLY_SUCCESS if the color transformation is successful,
        otherwise an appropriate error code is returned.

        \ingroup calc_func_grad
    */
    FLY_API fly_err fly_gradient(fly_array *dx, fly_array *dy, const fly_array in);

    /**
        C Interface for loading an image

        \param[out] out will contain the image
        \param[in] filename is name of file to be loaded
        \param[in] isColor boolean denoting if the image should be loaded as 1 channel or 3 channel
        \return     \ref FLY_SUCCESS if the color transformation is successful,
        otherwise an appropriate error code is returned.

        \ingroup imageio_func_load
    */
    FLY_API fly_err fly_load_image(fly_array *out, const char* filename, const bool isColor);

    /**
        C Interface for saving an image

        \param[in] filename is name of file to be loaded
        \param[in] in is the flare array to be saved as an image
        \return     \ref FLY_SUCCESS if the color transformation is successful,
        otherwise an appropriate error code is returned.

        \ingroup imageio_func_save
    */
    FLY_API fly_err fly_save_image(const char* filename, const fly_array in);

    /**
        C Interface for loading an image from memory

        \param[out] out is an array that will contain the image
        \param[in] ptr is the FIMEMORY pointer created by either saveImageMem function, the
        fly_save_image_memory function, or the FreeImage_OpenMemory API.
        \return     \ref FLY_SUCCESS if successful

        \ingroup imagemem_func_load
    */
    FLY_API fly_err fly_load_image_memory(fly_array *out, const void* ptr);

    /**
        C Interface for saving an image to memory using FreeImage

        \param[out] ptr is the FIMEMORY pointer created by FreeImage.
        \param[in] in is the flare array to be saved as an image
        \param[in] format is the type of image to create in memory. The enum borrows from
        the FREE_IMAGE_FORMAT enum of FreeImage. Other values not included in fly_image_format
        but included in FREE_IMAGE_FORMAT can also be passed to this function.
        \return     \ref FLY_SUCCESS if successful.

        \ingroup imagemem_func_save
    */
    FLY_API fly_err fly_save_image_memory(void** ptr, const fly_array in, const fly_image_format format);

    /**
        C Interface for deleting an image from memory

        \param[in] ptr is the FIMEMORY pointer created by either saveImageMem function, the
        fly_save_image_memory function, or the FreeImage_OpenMemory API.
        \return     \ref FLY_SUCCESS if successful

        \ingroup imagemem_func_delete
    */
    FLY_API fly_err fly_delete_image_memory(void* ptr);

    /**
        C Interface for loading an image as is original type

        This load image function allows you to load images as u8, u16 or f32
        depending on the type of input image as shown by the table below.

         Bits per Color (Gray/RGB/RGBA Bits Per Pixel) | Array Type  | Range
        -----------------------------------------------|-------------|---------------
          8 ( 8/24/32  BPP)                            | u8          | 0 - 255
         16 (16/48/64  BPP)                            | u16         | 0 - 65535
         32 (32/96/128 BPP)                            | f32         | 0 - 1

        \param[out] out contains them image
        \param[in] filename is name of file to be loaded
        \return     \ref FLY_SUCCESS if successful

        \ingroup imageio_func_load
    */
    FLY_API fly_err fly_load_image_native(fly_array *out, const char* filename);

    /**
        C Interface for saving an image without modifications

        This function only accepts u8, u16, f32 arrays. These arrays are saved to
        images without any modifications.

        You must also note that note all image type support 16 or 32 bit images.

        The best options for 16 bit images are PNG, PPM and TIFF.
        The best option for 32 bit images is TIFF.
        These allow lossless storage.

        The images stored have the following properties:

         Array Type  | Bits per Color (Gray/RGB/RGBA Bits Per Pixel) | Range
        -------------|-----------------------------------------------|---------------
         u8          |  8 ( 8/24/32  BPP)                            | 0 - 255
         u16         | 16 (16/48/64  BPP)                            | 0 - 65535
         f32         | 32 (32/96/128 BPP)                            | 0 - 1

        \param[in] filename is name of file to be saved
        \param[in] in is the array to be saved. Should be u8 for saving 8-bit image,
        u16 for 16-bit image, and f32 for 32-bit image.

        \return     \ref FLY_SUCCESS if successful

        \ingroup imageio_func_save
    */
    FLY_API fly_err fly_save_image_native(const char* filename, const fly_array in);

    /**
        Function to check if Image IO is available

        \param[out] out is true if Flare was commpiled with ImageIO support,
        false otherwise.

        \return     \ref FLY_SUCCESS if successful

        \ingroup imageio_func_available
    */
    FLY_API fly_err fly_is_image_io_available(bool *out);

    /**
       C Interface for resizing an image to specified dimensions

       \param[out] out will contain the resized image of specified by \p odim0 and \p odim1
       \param[in] in is input image
       \param[in] odim0 is the size for the first output dimension
       \param[in] odim1 is the size for the second output dimension
       \param[in] method is the interpolation type (Nearest by default)

       \return     \ref FLY_SUCCESS if the color transformation is successful,
       otherwise an appropriate error code is returned.

       \ingroup transform_func_resize
    */
    FLY_API fly_err fly_resize(fly_array *out, const fly_array in, const dim_t odim0, const dim_t odim1, const fly_interp_type method);

    /**
       C Interface for transforming an image

       \param[out] out       will contain the transformed image
       \param[in]  in        is input image
       \param[in]  transform is transformation matrix
       \param[in]  odim0     is the first output dimension
       \param[in]  odim1     is the second output dimension
       \param[in]  method    is the interpolation type (Nearest by default)
       \param[in]  inverse   if true applies inverse transform, if false applies forward transoform

       \return \ref FLY_SUCCESS if the color transformation is successful,
       otherwise an appropriate error code is returned.

       \ingroup transform_func_transform
    */
    FLY_API fly_err fly_transform(fly_array *out, const fly_array in, const fly_array transform,
                              const dim_t odim0, const dim_t odim1,
                              const fly_interp_type method, const bool inverse);

    /**
       C Interface for the version of \ref fly_transform that accepts a
       preallocated output array

       \param[out] out       will contain the transformed image
       \param[in]  in        is input image
       \param[in]  transform is transformation matrix
       \param[in]  odim0     is the first output dimension
       \param[in]  odim1     is the second output dimension
       \param[in]  method    is the interpolation type (Nearest by default)
       \param[in]  inverse   if true applies inverse transform, if false applies forward transoform

       \return \ref FLY_SUCCESS if the color transformation is successful,
       otherwise an appropriate error code is returned.

       \note \p out can either be a null or existing `fly_array` object. If it is a
             sub-array of an existing `fly_array`, only the corresponding portion of
             the `fly_array` will be overwritten
       \note Passing an `fly_array` that has not been initialized to \p out will
             cause undefined behavior.

       \ingroup transform_func_transform
    */
    FLY_API fly_err fly_transform_v2(fly_array *out, const fly_array in, const fly_array transform,
                                 const dim_t odim0, const dim_t odim1,
                                 const fly_interp_type method, const bool inverse);

    /**
       C Interface for transforming an image
       C++ Interface for transforming coordinates

       \param[out] out the transformed coordinates
       \param[in] tf is transformation matrix
       \param[in] d0 is the first input dimension
       \param[in] d1 is the second input dimension

       \ingroup transform_func_coordinates
    */
    FLY_API fly_err fly_transform_coordinates(fly_array *out, const fly_array tf, const float d0, const float d1);

    /**
       C Interface for rotating an image

       \param[out] out will contain the image \p in rotated by \p theta
       \param[in] in is input image
       \param[in] theta is the degree (in radians) by which the input is rotated
       \param[in] crop if true the output is cropped original dimensions. If false the output dimensions scale based on \p theta
       \param[in] method is the interpolation type (Nearest by default)
       \return     \ref FLY_SUCCESS if the color transformation is successful,
       otherwise an appropriate error code is returned.

       \ingroup transform_func_rotate
    */
    FLY_API fly_err fly_rotate(fly_array *out, const fly_array in, const float theta,
                           const bool crop, const fly_interp_type method);
   /**
      C Interface for translate an image

      \param[out] out will contain the translated image
      \param[in] in is input image
      \param[in] trans0 is amount by which the first dimension is translated
      \param[in] trans1 is amount by which the second dimension is translated
      \param[in] odim0 is the first output dimension
      \param[in] odim1 is the second output dimension
      \param[in] method is the interpolation type (Nearest by default)
      \return     \ref FLY_SUCCESS if the color transformation is successful,
      otherwise an appropriate error code is returned.

      \ingroup transform_func_translate
   */
    FLY_API fly_err fly_translate(fly_array *out, const fly_array in, const float trans0, const float trans1,
                              const dim_t odim0, const dim_t odim1, const fly_interp_type method);
    /**
       C Interface for scaling an image

       \param[out] out will contain the scaled image
       \param[in] in is input image
       \param[in] scale0 is amount by which the first dimension is scaled
       \param[in] scale1 is amount by which the second dimension is scaled
       \param[in] odim0 is the first output dimension
       \param[in] odim1 is the second output dimension
       \param[in] method is the interpolation type (Nearest by default)
       \return     \ref FLY_SUCCESS if the color transformation is successful,
       otherwise an appropriate error code is returned.

       \ingroup transform_func_scale
    */
    FLY_API fly_err fly_scale(fly_array *out, const fly_array in, const float scale0, const float scale1,
                          const dim_t odim0, const dim_t odim1, const fly_interp_type method);
    /**
       C Interface for skewing an image

       \param[out] out will contain the skewed image
       \param[in] in is input image
       \param[in] skew0 is amount by which the first dimension is skewed
       \param[in] skew1 is amount by which the second dimension is skewed
       \param[in] odim0 is the first output dimension
       \param[in] odim1 is the second output dimension
       \param[in] inverse if true applies inverse transform, if false applies forward transoform
       \param[in] method is the interpolation type (Nearest by default)
       \return     \ref FLY_SUCCESS if the color transformation is successful,
       otherwise an appropriate error code is returned.

       \ingroup transform_func_skew
    */
    FLY_API fly_err fly_skew(fly_array *out, const fly_array in, const float skew0, const float skew1,
                         const dim_t odim0, const dim_t odim1, const fly_interp_type method,
                         const bool inverse);

    /**
       C Interface for histogram

       \param[out] out (type u32) is the histogram for input array in
       \param[in]  in is the input array
       \param[in]  nbins  Number of bins to populate between min and max
       \param[in]  minval minimum bin value (accumulates -inf to min)
       \param[in]  maxval minimum bin value (accumulates max to +inf)
       \return     \ref FLY_SUCCESS if the histogram is successfully created,
       otherwise an appropriate error code is returned.

       \ingroup image_func_histogram
     */
    FLY_API fly_err fly_histogram(fly_array *out, const fly_array in, const unsigned nbins, const double minval, const double maxval);

    /**
        C Interface for image dilation (max filter)

        \param[out] out array is the dilated image
        \param[in]  in array is the input image
        \param[in]  mask is the neighborhood window
        \return     \ref FLY_SUCCESS if the dilated successfully,
        otherwise an appropriate error code is returned.

        \note if \p mask is all ones, this function behaves like max filter

        \ingroup image_func_dilate
    */
    FLY_API fly_err fly_dilate(fly_array *out, const fly_array in, const fly_array mask);

    /**
        C Interface for 3d image dilation

        \param[out] out array is the dilated volume
        \param[in]  in array is the input volume
        \param[in]  mask is the neighborhood delta volume
        \return     \ref FLY_SUCCESS if the dilated successfully,
        otherwise an appropriate error code is returned.

        \ingroup image_func_dilate3d
    */
    FLY_API fly_err fly_dilate3(fly_array *out, const fly_array in, const fly_array mask);

    /**
        C Interface for image erosion (min filter)

        \param[out] out array is the eroded image
        \param[in]  in array is the input image
        \param[in]  mask is the neighborhood window
        \return     \ref FLY_SUCCESS if the eroded successfully,
        otherwise an appropriate error code is returned.

        \note if \p mask is all ones, this function behaves like min filter

        \ingroup image_func_erode
    */
    FLY_API fly_err fly_erode(fly_array *out, const fly_array in, const fly_array mask);

    /**
        C Interface for 3D image erosion

        \param[out] out array is the eroded volume
        \param[in]  in array is the input volume
        \param[in]  mask is the neighborhood delta volume
        \return     \ref FLY_SUCCESS if the eroded successfully,
        otherwise an appropriate error code is returned.

        \ingroup image_func_erode3d
    */
    FLY_API fly_err fly_erode3(fly_array *out, const fly_array in, const fly_array mask);

    /**
        C Interface for bilateral filter

        \param[out] out array is the processed image
        \param[in]  in array is the input image
        \param[in]  spatial_sigma is the spatial variance parameter that decides the filter window
        \param[in]  chromatic_sigma is the chromatic variance parameter
        \param[in]  isColor indicates if the input \p in is color image or grayscale
        \return     \ref FLY_SUCCESS if the filter is applied successfully,
        otherwise an appropriate error code is returned.

        \ingroup image_func_bilateral
    */
    FLY_API fly_err fly_bilateral(fly_array *out, const fly_array in, const float spatial_sigma, const float chromatic_sigma, const bool isColor);

    /**
        C Interface for mean shift

        \param[out] out array is the processed image
        \param[in]  in array is the input image
        \param[in]  spatial_sigma is the spatial variance parameter that decides the filter window
        \param[in]  chromatic_sigma is the chromatic variance parameter
        \param[in]  iter is the number of iterations filter operation is performed
        \param[in]  is_color indicates if the input \p in is color image or grayscale
        \return     \ref FLY_SUCCESS if the filter is applied successfully,
        otherwise an appropriate error code is returned.

        \ingroup image_func_mean_shift
    */
    FLY_API fly_err fly_mean_shift(fly_array *out, const fly_array in, const float spatial_sigma, const float chromatic_sigma, const unsigned iter, const bool is_color);

    /**
        C Interface for minimum filter

        \param[out] out array is the processed image
        \param[in]  in array is the input image
        \param[in]  wind_length is the kernel height
        \param[in]  wind_width is the kernel width
        \param[in]  edge_pad value will decide what happens to border when running
                    filter in their neighborhood. It takes one of the values [\ref FLY_PAD_ZERO | \ref FLY_PAD_SYM]
        \return     \ref FLY_SUCCESS if the minimum filter is applied successfully,
        otherwise an appropriate error code is returned.

        \ingroup image_func_minfilt
    */
    FLY_API fly_err fly_minfilt(fly_array *out, const fly_array in, const dim_t wind_length, const dim_t wind_width, const fly_border_type edge_pad);

    /**
       C Interface for maximum filter

       \param[out] out array is the processed image
       \param[in]  in array is the input image
       \param[in]  wind_length is the kernel height
       \param[in]  wind_width is the kernel width
       \param[in]  edge_pad value will decide what happens to border when running
       filter in their neighborhood. It takes one of the values [\ref FLY_PAD_ZERO | \ref FLY_PAD_SYM]
       \return     \ref FLY_SUCCESS if the maximum filter is applied successfully,
       otherwise an appropriate error code is returned.

       \ingroup image_func_maxfilt
    */
    FLY_API fly_err fly_maxfilt(fly_array *out, const fly_array in, const dim_t wind_length, const dim_t wind_width, const fly_border_type edge_pad);

    /**
        C Interface for regions in an image

        \param[out] out array will have labels indicating different regions
        \param[in]  in array should be binary image of type \ref b8
        \param[in]  connectivity can take one of the following [\ref FLY_CONNECTIVITY_4 | \ref FLY_CONNECTIVITY_8_4]
        \param[in]  ty is type of output array
        \return     \ref FLY_SUCCESS if the regions are identified successfully,
        otherwise an appropriate error code is returned.

        \ingroup image_func_regions
    */
    FLY_API fly_err fly_regions(fly_array *out, const fly_array in, const fly_connectivity connectivity, const fly_dtype ty);

    /**
       C Interface for getting sobel gradients

       \param[out] dx is derivative along horizontal direction
       \param[out] dy is derivative along vertical direction
       \param[in]  img is an array with image data
       \param[in]  ker_size sobel kernel size or window size
       \return     \ref FLY_SUCCESS if sobel derivatives are computed successfully,
       otherwise an appropriate error code is returned.

       \note If \p img is 3d array, a batch operation will be performed.

       \ingroup image_func_sobel
    */
    FLY_API fly_err fly_sobel_operator(fly_array *dx, fly_array *dy, const fly_array img, const unsigned ker_size);

    /**
       C Interface for converting RGB to gray

       \param[out] out is an array in target color space
       \param[in]  in is an array in the RGB color space
       \param[in]  rPercent is percentage of red channel value contributing to grayscale intensity
       \param[in]  gPercent is percentage of green channel value contributing to grayscale intensity
       \param[in]  bPercent is percentage of blue channel value contributing to grayscale intensity
       \return     \ref FLY_SUCCESS if the color transformation is successful,
       otherwise an appropriate error code is returned.

       \note \p in must be three dimensional for RGB to Grayscale conversion.

       \ingroup image_func_rgb2gray
    */
    FLY_API fly_err fly_rgb2gray(fly_array* out, const fly_array in, const float rPercent, const float gPercent, const float bPercent);

    /**
       C Interface for converting gray to RGB

       \param[out] out is an array in target color space
       \param[in]  in is an array in the Grayscale color space
       \param[in]  rFactor is percentage of intensity value contributing to red channel
       \param[in]  gFactor is percentage of intensity value contributing to green channel
       \param[in]  bFactor is percentage of intensity value contributing to blue channel
       \return     \ref FLY_SUCCESS if the color transformation is successful,
       otherwise an appropriate error code is returned.

       \note \p in must be two dimensional for Grayscale to RGB conversion.

       \ingroup image_func_gray2rgb
    */
    FLY_API fly_err fly_gray2rgb(fly_array* out, const fly_array in, const float rFactor, const float gFactor, const float bFactor);

    /**
       C Interface for histogram equalization

       \param[out] out is an array with data that has histogram approximately equal to histogram
       \param[in]  in is the input array, non-normalized input (!! assumes values [0-255] !!)
       \param[in]  hist target histogram to approximate in output (based on number of bins)
       \return     \ref FLY_SUCCESS if the color transformation is successful,
       otherwise an appropriate error code is returned.

       \note \p in must be two dimensional.

       \ingroup image_func_histequal
    */
    FLY_API fly_err fly_hist_equal(fly_array *out, const fly_array in, const fly_array hist);

    /**
       C Interface generating gaussian kernels

       \param[out] out is an array with values generated using gaussian function
       \param[in]  rows number of rows of the gaussian kernel
       \param[in]  cols number of columns of the gaussian kernel
       \param[in]  sigma_r (default 0) (calculated internally as 0.25 * rows + 0.75)
       \param[in]  sigma_c (default 0) (calculated internally as 0.25 * cols + 0.75)
       \return     \ref FLY_SUCCESS if gaussian distribution values are generated successfully,
       otherwise an appropriate error code is returned.

       \ingroup image_func_gauss
    */
    FLY_API fly_err fly_gaussian_kernel(fly_array *out,
                                    const int rows, const int cols,
                                    const double sigma_r, const double sigma_c);

    /**
       C Interface for converting HSV to RGB

       \param[out] out is an array in the RGB color space
       \param[in]  in is an array in the HSV color space
       \return     \ref FLY_SUCCESS if the color transformation is successful,
       otherwise an appropriate error code is returned.

       \note \p in must be three dimensional

       \ingroup image_func_hsv2rgb
    */
    FLY_API fly_err fly_hsv2rgb(fly_array* out, const fly_array in);

    /**
       C Interface for converting RGB to HSV

       \param[out] out is an array in the HSV color space
       \param[in]  in is an array in the RGB color space
       \return     \ref FLY_SUCCESS if the color transformation is successful,
       otherwise an appropriate error code is returned.

       \note \p in must be three dimensional

       \ingroup image_func_rgb2hsv
    */
    FLY_API fly_err fly_rgb2hsv(fly_array* out, const fly_array in);

    /**
       C Interface wrapper for color space conversion

       \param[out] out is an array in target color space
       \param[in]  image is the input array
       \param[in]  to is the target array color space \param[in]
       from is the input array color space
       \return     \ref FLY_SUCCESS if the color transformation is successful,
       otherwise an appropriate error code
       is returned.

       \note  \p image must be 3 dimensional for \ref FLY_HSV to \ref FLY_RGB, \ref
       FLY_RGB to \ref FLY_HSV, & \ref FLY_RGB to \ref FLY_GRAY transformations.
       For \ref FLY_GRAY to \ref FLY_RGB transformation, 2D array is expected.

       \ingroup image_func_colorspace
    */
    FLY_API fly_err fly_color_space(fly_array *out, const fly_array image, const fly_cspace_t to, const fly_cspace_t from);

    /**
       C Interface for rearranging windowed sections of an input into columns
       (or rows)

       \param[out] out is an array with the input's sections rearraged as columns
                   (or rows)
       \param[in]  in is the input array
       \param[in]  wx is the window size along dimension 0
       \param[in]  wy is the window size along dimension 1
       \param[in]  sx is the stride along dimension 0
       \param[in]  sy is the stride along dimension 1
       \param[in]  px is the padding along dimension 0
       \param[in]  py is the padding along dimension 1
       \param[in]  is_column determines whether the section becomes a column (if
                   true) or a row (if false)
       \return     \ref FLY_SUCCESS if unwrap is successful,
                   otherwise an appropriate error code is returned.

       \note \p in can hold multiple images for processing if it is three or
             four-dimensional
       \note \p wx and \p wy must be between [1, input.dims(0 (1)) + px (py)]
       \note \p sx and \p sy must be greater than 1
       \note \p px and \p py must be between [0, wx (wy) - 1]. Padding becomes
             part of the input image prior to the windowing

       \ingroup image_func_unwrap
    */
    FLY_API fly_err fly_unwrap(fly_array *out, const fly_array in, const dim_t wx, const dim_t wy,
                           const dim_t sx, const dim_t sy, const dim_t px, const dim_t py,
                           const bool is_column);

    /**
       C Interface for performing the opposite of \ref fly::unwrap()

       \param[out] out is an array with the input's columns (or rows) reshaped as
                   patches
       \param[in]  in is the input array
       \param[in]  ox is the output's dimension 0 size
       \param[in]  oy is the output's dimension 1 size
       \param[in]  wx is the window size along dimension 0
       \param[in]  wy is the window size along dimension 1
       \param[in]  sx is the stride along dimension 0
       \param[in]  sy is the stride along dimension 1
       \param[in]  px is the padding along dimension 0
       \param[in]  py is the padding along dimension 1
       \param[in]  is_column determines whether an output patch is formed from a
                   column (if true) or a row (if false)
       \return     \ref FLY_SUCCESS if the color transformation is successful,
       otherwise an appropriate error code is returned.

       \note Wrap is typically used to recompose an unwrapped image. If this is the
             case, use the same parameters that were used in \ref fly::unwrap(). Also
             use the original image size (before unwrap) for \p ox and \p oy.
       \note The window/patch size, \p wx \f$\times\f$ \p wy, must equal
             `input.dims(0)` (or `input.dims(1)` if \p is_column is false).
       \note \p sx and \p sy must be at least 1
       \note \p px and \p py must be between [0, wx) and [0, wy), respectively
       \note The number of patches, `input.dims(1)` (or `input.dims(0)` if
             \p is_column is false), must equal \f$nx \times\ ny\f$, where
             \f$\displaystyle nx = \frac{ox + 2px - wx}{sx} + 1\f$ and
             \f$\displaystyle ny = \frac{oy + 2py - wy}{sy} + 1\f$
       \note Batched wrap can be performed on multiple 2D slices at once if \p in
             is three or four-dimensional

       \ingroup image_func_wrap
    */
    FLY_API fly_err fly_wrap(fly_array *out,
                         const fly_array in,
                         const dim_t ox, const dim_t oy,
                         const dim_t wx, const dim_t wy,
                         const dim_t sx, const dim_t sy,
                         const dim_t px, const dim_t py,
                         const bool is_column);

    /**
       C Interface for the version of \ref fly_wrap that accepts a
       preallocated output array

       \param[out] out is an array with the input's columns (or rows) reshaped as
                   patches
       \param[in]  in is the input array
       \param[in]  ox is the output's dimension 0 size
       \param[in]  oy is the output's dimension 1 size
       \param[in]  wx is the window size along dimension 0
       \param[in]  wy is the window size along dimension 1
       \param[in]  sx is the stride along dimension 0
       \param[in]  sy is the stride along dimension 1
       \param[in]  px is the padding along dimension 0
       \param[in]  py is the padding along dimension 1
       \param[in]  is_column determines whether an output patch is formed from a
                   column (if true) or a row (if false)
       \return     \ref FLY_SUCCESS if the color transformation is successful,
       otherwise an appropriate error code is returned.

       \note Wrap is typically used to recompose an unwrapped image. If this is the
             case, use the same parameters that were used in \ref fly::unwrap(). Also
             use the original image size (before unwrap) for \p ox and \p oy.
       \note The window/patch size, \p wx \f$\times\f$ \p wy, must equal
             `input.dims(0)` (or `input.dims(1)` if \p is_column is false).
       \note \p sx and \p sy must be at least 1
       \note \p px and \p py must be between [0, wx) and [0, wy), respectively
       \note The number of patches, `input.dims(1)` (or `input.dims(0)` if
             \p is_column is false), must equal \f$nx \times\ ny\f$, where
             \f$\displaystyle nx = \frac{ox + 2px - wx}{sx} + 1\f$ and
             \f$\displaystyle ny = \frac{oy + 2py - wy}{sy} + 1\f$
       \note Batched wrap can be performed on multiple 2D slices at once if \p in
             is three or four-dimensional

       \ingroup image_func_wrap
    */
    FLY_API fly_err fly_wrap_v2(fly_array *out,
                            const fly_array in,
                            const dim_t ox, const dim_t oy,
                            const dim_t wx, const dim_t wy,
                            const dim_t sx, const dim_t sy,
                            const dim_t px, const dim_t py,
                            const bool is_column);

    /**
       C Interface wrapper for summed area tables

       \param[out] out is the summed area table on input image(s)
       \param[in]  in is the input array
       \return \ref FLY_SUCCESS if the sat computation is successful,
       otherwise an appropriate error code is returned.

       \ingroup image_func_sat
    */
    FLY_API fly_err fly_sat(fly_array *out, const fly_array in);

    /**
       C Interface for converting YCbCr to RGB

       \param[out] out is an array in the RGB color space
       \param[in]  in is an array in the YCbCr color space
       \param[in]  standard specifies the ITU-R BT "xyz" standard which determines the Kb, Kr values
       used in colorspace conversion equation
       \return     \ref FLY_SUCCESS if the color transformation is successful,
       otherwise an appropriate error code is returned.

       \note \p in must be three dimensional and values should lie in the range [0,1]

       \ingroup image_func_ycbcr2rgb
    */
    FLY_API fly_err fly_ycbcr2rgb(fly_array* out, const fly_array in, const fly_ycc_std standard);

    /**
       C Interface for converting RGB to YCbCr

       \param[out] out is an array in the YCbCr color space
       \param[in]  in is an array in the RGB color space
       \param[in]  standard specifies the ITU-R BT "xyz" standard which determines the Kb, Kr values
       used in colorspace conversion equation
       \return     \ref FLY_SUCCESS if the color transformation is successful,
       otherwise an appropriate error code is returned.

       \note \p in must be three dimensional and values should lie in the range [0,1]

       \ingroup image_func_rgb2ycbcr
    */
    FLY_API fly_err fly_rgb2ycbcr(fly_array* out, const fly_array in, const fly_ycc_std standard);

    /**
       C Interface for finding image moments

       \param[out] out is an array containing the calculated moments
       \param[in]  in is an array of image(s)
       \param[in] moment is moment(s) to calculate
       \return     ref FLY_SUCCESS if the moment calculation is successful,
       otherwise an appropriate error code is returned.

       \ingroup image_func_moments
    */
    FLY_API fly_err fly_moments(fly_array *out, const fly_array in, const fly_moment_type moment);

    /**
       C Interface for calculating image moment(s) of a single image

       \param[out] out is a pointer to a pre-allocated array where the calculated moment(s) will be placed.
       User is responsible for ensuring enough space to hold all requested moments
       \param[in] in is the input image
       \param[in] moment is moment(s) to calculate
       \return     ref FLY_SUCCESS if the moment calculation is successful,
       otherwise an appropriate error code is returned.

       \ingroup image_func_moments
    */
    FLY_API fly_err fly_moments_all(double* out, const fly_array in, const fly_moment_type moment);

    /**
       C Interface for canny edge detector

       \param[out] out is an binary array containing edges
       \param[in] in is the input image
       \param[in] threshold_type     determines if user set high threshold is to be used or not. It
                                     can take values defined by the enum \ref fly_canny_threshold
       \param[in] low_threshold_ratio   is the lower threshold % of the maximum or auto-derived high
                                        threshold
       \param[in] high_threshold_ratio  is the higher threshold % of maximum value in gradient image
                                        used in hysteresis procedure. This value is ignored if
                                        \ref FLY_CANNY_THRESHOLD_AUTO_OTSU is chosen as
                                        \ref fly_canny_threshold
       \param[in] sobel_window      is the window size of sobel kernel for computing gradient direction
                                    and magnitude
       \param[in] is_fast indicates   if L<SUB>1</SUB> norm(faster but less accurate) is used to
                                      compute image gradient magnitude instead of L<SUB>2</SUB> norm.
       \return    \ref FLY_SUCCESS if the moment calculation is successful,
       otherwise an appropriate error code is returned.

       \ingroup image_func_canny
    */
    FLY_API fly_err fly_canny(fly_array* out, const fly_array in,
                          const fly_canny_threshold threshold_type,
                          const float low_threshold_ratio,
                          const float high_threshold_ratio,
                          const unsigned sobel_window, const bool is_fast);

    /**
       C Interface for anisotropic diffusion

       It can do both gradient and curvature based anisotropic smoothing.

       \param[out] out is an fly_array containing anisotropically smoothed image pixel values
       \param[in] in is the input image, expects non-integral (float/double) typed fly_array
       \param[in] timestep is the time step used in solving the diffusion equation.
       \param[in] conductance parameter controls the sensitivity of conductance in diffusion equation.
       \param[in] iterations is the number of times the diffusion step is performed.
       \param[in] fftype indicates whether quadratic or exponential flux function is used by algorithm.
       \param[in] diffusion_kind will let the user choose what kind of diffusion method to perform. It will take
                  any value of enum \ref fly_diffusion_eq
       \return \ref FLY_SUCCESS if the moment calculation is successful,
       otherwise an appropriate error code is returned.

       \ingroup image_func_anisotropic_diffusion
    */
    FLY_API fly_err fly_anisotropic_diffusion(fly_array* out, const fly_array in,
                                          const float timestep,
                                          const float conductance,
                                          const unsigned iterations,
                                          const fly_flux_function fftype,
                                          const fly_diffusion_eq diffusion_kind);

    /**
       C Interface for Iterative deconvolution algorithm

       \param[out] out is the sharp estimate generated from the blurred input
       \param[in] in is the blurred input image
       \param[in] ker is the kernel(point spread function) known to have caused
                  the blur in the system
       \param[in] iterations is the number of iterations the algorithm will run
       \param[in] relax_factor is the relaxation factor multiplied with
                  distance of estimate from observed image.
       \param[in] algo takes value of type enum \ref fly_iterative_deconv_algo
                  indicating the iterative deconvolution algorithm to be used
       \return \ref FLY_SUCCESS if the deconvolution is successful,
       otherwise an appropriate error code is returned.

       \note \p relax_factor argument is ignore when it
       \ref FLY_ITERATIVE_DECONV_RICHARDSONLUCY algorithm is used.

       \ingroup image_func_iterative_deconv
     */
    FLY_API fly_err fly_iterative_deconv(fly_array* out,
                                     const fly_array in, const fly_array ker,
                                     const unsigned iterations,
                                     const float relax_factor,
                                     const fly_iterative_deconv_algo algo);

    /**
       C Interface for Tikhonov deconvolution algorithm

       \param[out] out is the sharp estimate generated from the blurred input
       \param[in] in is the blurred input image
       \param[in] psf is the kernel(point spread function) known to have caused
                  the blur in the system
       \param[in] gamma takes different meaning depending on the algorithm
                  chosen. If \p algo is FLY_INVERSE_DECONV_TIKHONOV, then
                  \p gamma is a user defined regularization constant.
       \param[in] algo takes value of type enum \ref fly_inverse_deconv_algo
                  indicating the inverse deconvolution algorithm to be used
       \return \ref FLY_SUCCESS if the deconvolution is successful,
       otherwise an appropriate error code is returned.

       \ingroup image_func_inverse_deconv
     */
    FLY_API fly_err fly_inverse_deconv(fly_array* out, const fly_array in,
                                   const fly_array psf, const float gamma,
                                   const fly_inverse_deconv_algo algo);

    /**
       C Interface for confidence connected components

       \param[out] out is the output fly_array having the connected components
       \param[in] in is the input image, expects non-integral (float/double)
                  typed fly_array
       \param[in] seedx is an fly_array of x coordinates of the seed points
       \param[in] seedy is an fly_array of y coordinates of the seed points
       \param[in] radius is the neighborhood region to be considered around
                  each seed point
       \param[in] multiplier controls the threshold range computed from
                  the mean and variance of seed point neighborhoods
       \param[in] iter is number of iterations
       \param[in] segmented_value is the value to which output array valid
                  pixels are set to.
       \return \ref FLY_SUCCESS if the execution is successful, otherwise an
       appropriate error code is returned.

       \ingroup image_func_confidence_cc
    */
    FLY_API fly_err fly_confidence_cc(fly_array *out, const fly_array in,
                                  const fly_array seedx, const fly_array seedy,
                                  const unsigned radius,
                                  const unsigned multiplier, const int iter,
                                  const double segmented_value);


#ifdef __cplusplus
}
#endif
