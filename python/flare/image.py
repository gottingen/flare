##########################################################################
# Copyright 2023 The EA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################

"""
Image processing functions.
"""

from .library import *
from .array import *
from .data import constant
from .signal import medfilt
import os

def gradient(image):
    """
    Find the horizontal and vertical gradients.

    Parameters
    ----------
    image : fly.Array
          - A 2 D flare array representing an image, or
          - A multi dimensional array representing batch of images.

    Returns
    ---------
    (dx, dy) : Tuple of fly.Array.
             - `dx` containing the horizontal gradients of `image`.
             - `dy` containing the vertical gradients of `image`.

    """
    dx = Array()
    dy = Array()
    safe_call(backend.get().fly_gradient(c_pointer(dx.arr), c_pointer(dy.arr), image.arr))
    return dx, dy

def load_image(file_name, is_color=False):
    """
    Load an image on the disk as an array.

    Parameters
    ----------
    file_name: str
          - Full path of the file name on disk.

    is_color : optional: bool. default: False.
          - Specifies if the image is loaded as 1 channel (if False) or 3 channel image (if True).

    Returns
    -------
    image - fly.Array
            A 2 dimensional (1 channel) or 3 dimensional (3 channel) array containing the image.

    """
    assert(os.path.isfile(file_name))
    image = Array()
    safe_call(backend.get().fly_load_image(c_pointer(image.arr),
                                          c_char_ptr_t(file_name.encode('ascii')), is_color))
    return image

def save_image(image, file_name):
    """
    Save an array as an image on the disk.

    Parameters
    ----------
    image : fly.Array
          - A 2 D flare array representing an image.

    file_name: str
          - Full path of the file name on the disk.
    """
    assert(isinstance(file_name, str))
    safe_call(backend.get().fly_save_image(c_char_ptr_t(file_name.encode('ascii')), image.arr))
    return image


def load_image_native(file_name):
    """
    Load an image on the disk as an array in native format.

    Parameters
    ----------
    file_name: str
          - Full path of the file name on disk.

    Returns
    -------
    image - fly.Array
            A 2 dimensional (1 channel) or 3 dimensional (3 or 4 channel) array containing the image.

    """
    assert(os.path.isfile(file_name))
    image = Array()
    safe_call(backend.get().fly_load_image_native(c_pointer(image.arr),
                                                 c_char_ptr_t(file_name.encode('ascii'))))
    return image

def save_image_native(image, file_name):
    """
    Save an array as an image on the disk in native format.

    Parameters
    ----------
    image : fly.Array
          - A 2 or 3 dimensional flare array representing an image.

    file_name: str
          - Full path of the file name on the disk.
    """
    assert(isinstance(file_name, str))
    safe_call(backend.get().fly_save_image_native(c_char_ptr_t(file_name.encode('ascii')), image.arr))
    return image

def resize(image, scale=None, odim0=None, odim1=None, method=INTERP.NEAREST):
    """
    Resize an image.

    Parameters
    ----------
    image : fly.Array
          - A 2 D flare array representing an image, or
          - A multi dimensional array representing batch of images.

    scale : optional: scalar. default: None.
          - Scale factor for the image resizing.

    odim0 : optional: int. default: None.
          - Size of the first dimension of the output.

    odim1 : optional: int. default: None.
          - Size of the second dimension of the output.

    method : optional: fly.INTERP. default: fly.INTERP.NEAREST.
          - Interpolation method used for resizing.

    Returns
    ---------
    out  : fly.Array
          - Output image after resizing.

    Note
    -----

    - If `scale` is None, `odim0` and `odim1` need to be specified.
    - If `scale` is not None, `odim0` and `odim1` are ignored.

    """
    if (scale is None):
        assert(odim0 is not None)
        assert(odim1 is not None)
    else:
        idims = image.dims()
        odim0 = int(scale * idims[0])
        odim1 = int(scale * idims[1])

    output = Array()
    safe_call(backend.get().fly_resize(c_pointer(output.arr),
                                      image.arr, c_dim_t(odim0),
                                      c_dim_t(odim1), method.value))

    return output

def transform(image, trans_mat, odim0 = 0, odim1 = 0, method=INTERP.NEAREST, is_inverse=True):
    """
    Transform an image using a transformation matrix.

    Parameters
    ----------
    image : fly.Array
          - A 2 D flare array representing an image, or
          - A multi dimensional array representing batch of images.

    trans_mat : fly.Array
          - A 2 D floating point flare array of size [3, 2].

    odim0 : optional: int. default: 0.
          - Size of the first dimension of the output.

    odim1 : optional: int. default: 0.
          - Size of the second dimension of the output.

    method : optional: fly.INTERP. default: fly.INTERP.NEAREST.
          - Interpolation method used for transformation.

    is_inverse : optional: bool. default: True.
          - Specifies if the inverse transform is applied.

    Returns
    ---------
    out  : fly.Array
          - Output image after transformation.

    Note
    -----

    - If `odim0` and `odim` are 0, the output dimensions are automatically calculated by the function.

    """
    output = Array()
    safe_call(backend.get().fly_transform(c_pointer(output.arr),
                                         image.arr, trans_mat.arr,
                                         c_dim_t(odim0), c_dim_t(odim1),
                                         method.value, is_inverse))
    return output


def rotate(image, theta, is_crop = True, method = INTERP.NEAREST):
    """
    Rotate an image.

    Parameters
    ----------
    image : fly.Array
          - A 2 D flare array representing an image, or
          - A multi dimensional array representing batch of images.

    theta : scalar
          - The angle to rotate in radians.

    is_crop : optional: bool. default: True.
          - Specifies if the output should be cropped to the input size.

    method : optional: fly.INTERP. default: fly.INTERP.NEAREST.
          - Interpolation method used for rotating.

    Returns
    ---------
    out  : fly.Array
          - Output image after rotating.
    """
    output = Array()
    safe_call(backend.get().fly_rotate(c_pointer(output.arr), image.arr,
                                      c_float_t(theta), is_crop, method.value))
    return output

def translate(image, trans0, trans1, odim0 = 0, odim1 = 0, method = INTERP.NEAREST):
    """
    Translate an image.

    Parameters
    ----------
    image : fly.Array
          - A 2 D flare array representing an image, or
          - A multi dimensional array representing batch of images.

    trans0: int.
          - Translation along first dimension in pixels.

    trans1: int.
          - Translation along second dimension in pixels.

    odim0 : optional: int. default: 0.
          - Size of the first dimension of the output.

    odim1 : optional: int. default: 0.
          - Size of the second dimension of the output.

    method : optional: fly.INTERP. default: fly.INTERP.NEAREST.
          - Interpolation method used for translation.

    Returns
    ---------
    out  : fly.Array
          - Output image after translation.

    Note
    -----

    - If `odim0` and `odim` are 0, the output dimensions are automatically calculated by the function.

    """
    output = Array()
    safe_call(backend.get().fly_translate(c_pointer(output.arr),
                                         image.arr, trans0, trans1,
                                         c_dim_t(odim0), c_dim_t(odim1), method.value))
    return output

def scale(image, scale0, scale1, odim0 = 0, odim1 = 0, method = INTERP.NEAREST):
    """
    Scale an image.

    Parameters
    ----------
    image : fly.Array
          - A 2 D flare array representing an image, or
          - A multi dimensional array representing batch of images.

    scale0 : scalar.
          - Scale factor for the first dimension.

    scale1 : scalar.
          - Scale factor for the second dimension.

    odim0 : optional: int. default: None.
          - Size of the first dimension of the output.

    odim1 : optional: int. default: None.
          - Size of the second dimension of the output.

    method : optional: fly.INTERP. default: fly.INTERP.NEAREST.
          - Interpolation method used for resizing.

    Returns
    ---------
    out  : fly.Array
          - Output image after scaling.

    Note
    -----

    - If `odim0` and `odim` are 0, the output dimensions are automatically calculated by the function.

    """
    output = Array()
    safe_call(backend.get().fly_scale(c_pointer(output.arr),
                                     image.arr, c_float_t(scale0), c_float_t(scale1),
                                     c_dim_t(odim0), c_dim_t(odim1), method.value))
    return output

def skew(image, skew0, skew1, odim0 = 0, odim1 = 0, method = INTERP.NEAREST, is_inverse=True):
    """
    Skew an image.

    Parameters
    ----------
    image : fly.Array
          - A 2 D flare array representing an image, or
          - A multi dimensional array representing batch of images.

    skew0 : scalar.
          - Skew factor for the first dimension.

    skew1 : scalar.
          - Skew factor for the second dimension.

    odim0 : optional: int. default: None.
          - Size of the first dimension of the output.

    odim1 : optional: int. default: None.
          - Size of the second dimension of the output.

    method : optional: fly.INTERP. default: fly.INTERP.NEAREST.
          - Interpolation method used for resizing.

    is_inverse : optional: bool. default: True.
          - Specifies if the inverse skew  is applied.

    Returns
    ---------
    out  : fly.Array
          - Output image after skewing.

    Note
    -----

    - If `odim0` and `odim` are 0, the output dimensions are automatically calculated by the function.

    """
    output = Array()
    safe_call(backend.get().fly_skew(c_pointer(output.arr),
                                    image.arr, c_float_t(skew0), c_float_t(skew1),
                                    c_dim_t(odim0), c_dim_t(odim1),
                                    method.value, is_inverse))

    return output

def histogram(image, nbins, min_val = None, max_val = None):
    """
    Find the histogram of an image.

    Parameters
    ----------
    image : fly.Array
          - A 2 D flare array representing an image, or
          - A multi dimensional array representing batch of images.

    nbins : int.
          - Number of bins in the histogram.

    min_val : optional: scalar. default: None.
          - The lower bound for the bin values.
          - If None, `fly.min(image)` is used.

    max_val : optional: scalar. default: None.
          - The upper bound for the bin values.
          - If None, `fly.max(image)` is used.

    Returns
    ---------
    hist : fly.Array
          - Containing the histogram of the image.

    """
    from .algorithm import min as fly_min
    from .algorithm import max as fly_max

    if min_val is None:
        min_val = fly_min(image)

    if max_val is None:
        max_val = fly_max(image)

    output = Array()
    safe_call(backend.get().fly_histogram(c_pointer(output.arr),
                                         image.arr, c_uint_t(nbins),
                                         c_double_t(min_val), c_double_t(max_val)))
    return output

def hist_equal(image, hist):
    """
    Equalize an image based on a histogram.

    Parameters
    ----------
    image : fly.Array
          - A 2 D flare array representing an image, or
          - A multi dimensional array representing batch of images.

    hist : fly.Array
          - Containing the histogram of an image.

    Returns
    ---------

    output : fly.Array
           - The equalized image.

    """
    output = Array()
    safe_call(backend.get().fly_hist_equal(c_pointer(output.arr), image.arr, hist.arr))
    return output

def dilate(image, mask = None):
    """
    Run image dilate on the image.

    Parameters
    ----------
    image : fly.Array
          - A 2 D flare array representing an image, or
          - A multi dimensional array representing batch of images.

    mask  : optional: fly.Array. default: None.
          - Specifies the neighborhood of a pixel.
          - When None, a [3, 3] array of all ones is used.

    Returns
    ---------

    output : fly.Array
           - The dilated image.

    """
    if mask is None:
        mask = constant(1, 3, 3, dtype=Dtype.f32)

    output = Array()
    safe_call(backend.get().fly_dilate(c_pointer(output.arr), image.arr, mask.arr))

    return output

def dilate3(volume, mask = None):
    """
    Run volume dilate on a volume.

    Parameters
    ----------
    volume : fly.Array
          - A 3 D flare array representing a volume, or
          - A multi dimensional array representing batch of volumes.

    mask  : optional: fly.Array. default: None.
          - Specifies the neighborhood of a pixel.
          - When None, a [3, 3, 3] array of all ones is used.

    Returns
    ---------

    output : fly.Array
           - The dilated volume.

    """
    if mask is None:
        mask = constant(1, 3, 3, 3, dtype=Dtype.f32)

    output = Array()
    safe_call(backend.get().fly_dilate3(c_pointer(output.arr), volume.arr, mask.arr))

    return output

def erode(image, mask = None):
    """
    Run image erode on the image.

    Parameters
    ----------
    image : fly.Array
          - A 2 D flare array representing an image, or
          - A multi dimensional array representing batch of images.

    mask  : optional: fly.Array. default: None.
          - Specifies the neighborhood of a pixel.
          - When None, a [3, 3] array of all ones is used.

    Returns
    ---------

    output : fly.Array
           - The eroded image.

    """
    if mask is None:
        mask = constant(1, 3, 3, dtype=Dtype.f32)

    output = Array()
    safe_call(backend.get().fly_erode(c_pointer(output.arr), image.arr, mask.arr))

    return output

def erode3(volume, mask = None):
    """
    Run volume erode on the volume.

    Parameters
    ----------
    volume : fly.Array
          - A 3 D flare array representing an volume, or
          - A multi dimensional array representing batch of volumes.

    mask  : optional: fly.Array. default: None.
          - Specifies the neighborhood of a pixel.
          - When None, a [3, 3, 3] array of all ones is used.

    Returns
    ---------

    output : fly.Array
           - The eroded volume.

    """

    if mask is None:
        mask = constant(1, 3, 3, 3, dtype=Dtype.f32)

    output = Array()
    safe_call(backend.get().fly_erode3(c_pointer(output.arr), volume.arr, mask.arr))

    return output

def bilateral(image, s_sigma, c_sigma, is_color = False):
    """
    Apply bilateral filter to the image.

    Parameters
    ----------
    image : fly.Array
          - A 2 D flare array representing an image, or
          - A multi dimensional array representing batch of images.

    s_sigma : scalar.
          - Sigma value for the co-ordinate space.

    c_sigma : scalar.
          - Sigma value for the color space.

    is_color : optional: bool. default: False.
          - Specifies if the third dimension is 3rd channel (if True) or a batch (if False).

    Returns
    ---------

    output : fly.Array
           - The image after the application of the bilateral filter.

    """
    output = Array()
    safe_call(backend.get().fly_bilateral(c_pointer(output.arr),
                                         image.arr, c_float_t(s_sigma),
                                         c_float_t(c_sigma), is_color))
    return output

def mean_shift(image, s_sigma, c_sigma, n_iter, is_color = False):
    """
    Apply mean shift to the image.

    Parameters
    ----------
    image : fly.Array
          - A 2 D flare array representing an image, or
          - A multi dimensional array representing batch of images.

    s_sigma : scalar.
          - Sigma value for the co-ordinate space.

    c_sigma : scalar.
          - Sigma value for the color space.

    n_iter  : int.
          - Number of mean shift iterations.

    is_color : optional: bool. default: False.
          - Specifies if the third dimension is 3rd channel (if True) or a batch (if False).

    Returns
    ---------

    output : fly.Array
           - The image after the application of the meanshift.

    """
    output = Array()
    safe_call(backend.get().fly_mean_shift(c_pointer(output.arr),
                                          image.arr, c_float_t(s_sigma), c_float_t(c_sigma),
                                          c_uint_t(n_iter), is_color))
    return output

def minfilt(image, w_len = 3, w_wid = 3, edge_pad = PAD.ZERO):
    """
    Apply min filter for the image.

    Parameters
    ----------
    image : fly.Array
          - A 2 D flare array representing an image, or
          - A multi dimensional array representing batch of images.

    w0 : optional: int. default: 3.
          - The length of the filter along the first dimension.

    w1 : optional: int. default: 3.
          - The length of the filter along the second dimension.

    edge_pad : optional: fly.PAD. default: fly.PAD.ZERO
          - Flag specifying how the min at the edge should be treated.

    Returns
    ---------

    output : fly.Array
           - The image after min filter is applied.

    """
    output = Array()
    safe_call(backend.get().fly_minfilt(c_pointer(output.arr),
                                       image.arr, c_dim_t(w_len),
                                       c_dim_t(w_wid), edge_pad.value))
    return output

def maxfilt(image, w_len = 3, w_wid = 3, edge_pad = PAD.ZERO):
    """
    Apply max filter for the image.

    Parameters
    ----------
    image : fly.Array
          - A 2 D flare array representing an image, or
          - A multi dimensional array representing batch of images.

    w0 : optional: int. default: 3.
          - The length of the filter along the first dimension.

    w1 : optional: int. default: 3.
          - The length of the filter along the second dimension.

    edge_pad : optional: fly.PAD. default: fly.PAD.ZERO
          - Flag specifying how the max at the edge should be treated.

    Returns
    ---------

    output : fly.Array
           - The image after max filter is applied.

    """
    output = Array()
    safe_call(backend.get().fly_maxfilt(c_pointer(output.arr),
                                       image.arr, c_dim_t(w_len),
                                       c_dim_t(w_wid), edge_pad.value))
    return output

def regions(image, conn = CONNECTIVITY.FOUR, out_type = Dtype.f32):
    """
    Find the connected components in the image.

    Parameters
    ----------
    image : fly.Array
          - A 2 D flare array representing an image.

    conn : optional: fly.CONNECTIVITY. default: fly.CONNECTIVITY.FOUR.
          - Specifies the connectivity of the pixels.

    out_type : optional: fly.Dtype. default: fly.Dtype.f32.
          - Specifies the type for the output.

    Returns
    ---------

    output : fly.Array
           - An array where each pixel is labeled with its component number.

    """
    output = Array()
    safe_call(backend.get().fly_regions(c_pointer(output.arr), image.arr,
                                       conn.value, out_type.value))
    return output

def confidenceCC(image, seedx, seedy, radius, multiplier, iters, segmented_value):
    """
    Find the confidence connected components in the image.

    Parameters
    ----------
    image : fly.Array
          - A 2 D flare array representing an image.
            Expects non-integral type

    seedx : fly.Array
          - An array with x-coordinates of seed points

    seedy : fly.Array
          - An array with y-coordinates of seed points

    radius : scalar
          - The neighborhood region to be considered around
            each seed point

    multiplier : scalar
          - Controls the threshold range computed from
            the mean and variance of seed point neighborhoods

    iters : scalar
          - is number of iterations

    segmented_value : scalar
          - the value to which output array valid
            pixels are set to.

    Returns
    ---------

    output : fly.Array
           - Output array with resulting connected components

    """
    output = Array()
    safe_call(backend.get().fly_confidence_cc(c_pointer(output.arr), image.arr, seedx.arr, seedy.arr,
                c_uint_t(radius), c_uint_t(multiplier), c_int_t(iters), c_double_t(segmented_value)))
    return output

def sobel_derivatives(image, w_len=3):
    """
    Find the sobel derivatives of the image.

    Parameters
    ----------
    image : fly.Array
          - A 2 D flare array representing an image, or
          - A multi dimensional array representing batch of images.

    w_len : optional: int. default: 3.
          - The size of the sobel operator.

    Returns
    ---------

    (dx, dy) : tuple of fly.Arrays.
           - `dx` is the sobel derivative along the horizontal direction.
           - `dy` is the sobel derivative along the vertical direction.

    """
    dx = Array()
    dy = Array()
    safe_call(backend.get().fly_sobel_operator(c_pointer(dx.arr), c_pointer(dy.arr),
                                              image.arr, c_uint_t(w_len)))
    return dx,dy

def gaussian_kernel(rows, cols, sigma_r = None, sigma_c = None):
    """
    Create a gaussian kernel with the given parameters.

    Parameters
    ----------
    image : fly.Array
          - A 2 D flare array representing an image, or
          - A multi dimensional array representing batch of images.

    rows : int
         - The number of rows in the gaussian kernel.

    cols : int
         - The number of columns in the gaussian kernel.

    sigma_r : optional: number. default: None.
         - The sigma value along rows
         - If None, calculated as (0.25 * rows + 0.75)

    sigma_c : optional: number. default: None.
         - The sigma value along columns
         - If None, calculated as (0.25 * cols + 0.75)

    Returns
    -------
    out   : fly.Array
          - A gaussian kernel of size (rows, cols)
    """
    out = Array()

    if (sigma_r is None):
        sigma_r = 0.25 * rows + 0.75

    if (sigma_c is None):
        sigma_c = 0.25 * cols + 0.75

    safe_call(backend.get().fly_gaussian_kernel(c_pointer(out.arr),
                                               c_int_t(rows), c_int_t(cols),
                                               c_double_t(sigma_r), c_double_t(sigma_c)))
    return out

def sobel_filter(image, w_len = 3, is_fast = False):
    """
    Apply sobel filter to the image.

    Parameters
    ----------
    image : fly.Array
          - A 2 D flare array representing an image, or
          - A multi dimensional array representing batch of images.

    w_len : optional: int. default: 3.
          - The size of the sobel operator.

    is_fast : optional: bool. default: False.
          - Specifies if the magnitude is generated using SAD (if True) or SSD (if False).

    Returns
    ---------

    output : fly.Array
           - Image containing the magnitude of the sobel derivatives.

    """
    from .arith import abs as fly_abs
    from .arith import hypot as fly_hypot

    dx,dy = sobel_derivatives(image, w_len)
    if (is_fast):
        return fly_abs(dx) + fly_abs(dy)
    else:
        return fly_hypot(dx, dy)

def rgb2gray(image, r_factor = 0.2126, g_factor = 0.7152, b_factor = 0.0722):
    """
    Convert RGB image to Grayscale.

    Parameters
    ----------
    image : fly.Array
          - A 3 D flare array representing an 3 channel image, or
          - A multi dimensional array representing batch of images.

    r_factor : optional: scalar. default: 0.2126.
          - Weight for the red channel.

    g_factor : optional: scalar. default: 0.7152.
          - Weight for the green channel.

    b_factor : optional: scalar. default: 0.0722.
          - Weight for the blue channel.

    Returns
    --------

    output : fly.Array
          - A grayscale image.

    """
    output=Array()
    safe_call(backend.get().fly_rgb2gray(c_pointer(output.arr),
                                        image.arr, c_float_t(r_factor), c_float_t(g_factor), c_float_t(b_factor)))
    return output

def gray2rgb(image, r_factor = 1.0, g_factor = 1.0, b_factor = 1.0):
    """
    Convert Grayscale image to an RGB image.

    Parameters
    ----------
    image : fly.Array
          - A 2 D flare array representing an image, or
          - A multi dimensional array representing batch of images.

    r_factor : optional: scalar. default: 1.0.
          - Scale factor for the red channel.

    g_factor : optional: scalar. default: 1.0.
          - Scale factor for the green channel.

    b_factor : optional: scalar. default: 1.0
          - Scale factor for the blue channel.

    Returns
    --------

    output : fly.Array
          - An RGB image.
          - The channels are not coalesced, i.e. they appear along the third dimension.

    """
    output=Array()
    safe_call(backend.get().fly_gray2rgb(c_pointer(output.arr),
                                        image.arr, c_float_t(r_factor), c_float_t(g_factor), c_float_t(b_factor)))
    return output

def hsv2rgb(image):
    """
    Convert HSV image to RGB.

    Parameters
    ----------
    image : fly.Array
          - A 3 D flare array representing an 3 channel image, or
          - A multi dimensional array representing batch of images.

    Returns
    --------

    output : fly.Array
          - A HSV image.

    """
    output = Array()
    safe_call(backend.get().fly_hsv2rgb(c_pointer(output.arr), image.arr))
    return output

def rgb2hsv(image):
    """
    Convert RGB image to HSV.

    Parameters
    ----------
    image : fly.Array
          - A 3 D flare array representing an 3 channel image, or
          - A multi dimensional array representing batch of images.

    Returns
    --------

    output : fly.Array
          - A RGB image.

    """
    output = Array()
    safe_call(backend.get().fly_rgb2hsv(c_pointer(output.arr), image.arr))
    return output

def color_space(image, to_type, from_type):
    """
    Convert an image from one color space to another.

    Parameters
    ----------
    image : fly.Array
          - A multi dimensional array representing batch of images in `from_type` color space.

    to_type : fly.CSPACE
          - An enum for the destination color space.

    from_type : fly.CSPACE
          - An enum for the source color space.

    Returns
    --------

    output : fly.Array
          - An image in the `to_type` color space.

    """
    output = Array()
    safe_call(backend.get().fly_color_space(c_pointer(output.arr), image.arr,
                                           to_type.value, from_type.value))
    return output

def unwrap(image, wx, wy, sx, sy, px=0, py=0, is_column=True):
    """
    Unrwap an image into an array.

    Parameters
    ----------

    image  : fly.Array
           A multi dimensional array specifying an image or batch of images.

    wx     : Integer.
           Block window size along the first dimension.

    wy     : Integer.
           Block window size along the second dimension.

    sx     : Integer.
           Stride along the first dimension.

    sy     : Integer.
           Stride along the second dimension.

    px     : Integer. Optional. Default: 0
           Padding along the first dimension.

    py     : Integer. Optional. Default: 0
           Padding along the second dimension.

    is_column : Boolean. Optional. Default: True.
           Specifies if the patch should be laid along row or columns.

    Returns
    -------

    out   : fly.Array
          A multi dimensional array contianing the image patches along specified dimension.

    Examples
    --------
    >>> import flare as fly
    >>> a = fly.randu(6, 6)
    >>> fly.display(a)

    [6 6 1 1]
        0.4107     0.3775     0.0901     0.8060     0.0012     0.9250
        0.8224     0.3027     0.5933     0.5938     0.8703     0.3063
        0.9518     0.6456     0.1098     0.8395     0.5259     0.9313
        0.1794     0.5591     0.1046     0.1933     0.1443     0.8684
        0.4198     0.6600     0.8827     0.7270     0.3253     0.6592
        0.0081     0.0764     0.1647     0.0322     0.5081     0.4387

    >>> b = fly.unwrap(a, 2, 2, 2, 2)
    >>> fly.display(b)

    [4 9 1 1]
        0.4107     0.9518     0.4198     0.0901     0.1098     0.8827     0.0012     0.5259     0.3253
        0.8224     0.1794     0.0081     0.5933     0.1046     0.1647     0.8703     0.1443     0.5081
        0.3775     0.6456     0.6600     0.8060     0.8395     0.7270     0.9250     0.9313     0.6592
        0.3027     0.5591     0.0764     0.5938     0.1933     0.0322     0.3063     0.8684     0.4387
    """

    out = Array()
    safe_call(backend.get().fly_unwrap(c_pointer(out.arr), image.arr,
                                      c_dim_t(wx), c_dim_t(wy),
                                      c_dim_t(sx), c_dim_t(sy),
                                      c_dim_t(px), c_dim_t(py),
                                      is_column))
    return out

def wrap(a, ox, oy, wx, wy, sx, sy, px=0, py=0, is_column=True):
    """
    Wrap an array into an image.

    Parameters
    ----------

    a      : fly.Array
           A multi dimensional array containing patches of images.

    wx     : Integer.
           Block window size along the first dimension.

    wy     : Integer.
           Block window size along the second dimension.

    sx     : Integer.
           Stride along the first dimension.

    sy     : Integer.
           Stride along the second dimension.

    px     : Integer. Optional. Default: 0
           Padding along the first dimension.

    py     : Integer. Optional. Default: 0
           Padding along the second dimension.

    is_column : Boolean. Optional. Default: True.
           Specifies if the patch should be laid along row or columns.

    Returns
    -------

    out   : fly.Array
          A multi dimensional array contianing the images.


    Examples
    --------
    >>> import flare as fly
    >>> a = fly.randu(6, 6)
    >>> fly.display(a)

    [6 6 1 1]
        0.4107     0.3775     0.0901     0.8060     0.0012     0.9250
        0.8224     0.3027     0.5933     0.5938     0.8703     0.3063
        0.9518     0.6456     0.1098     0.8395     0.5259     0.9313
        0.1794     0.5591     0.1046     0.1933     0.1443     0.8684
        0.4198     0.6600     0.8827     0.7270     0.3253     0.6592
        0.0081     0.0764     0.1647     0.0322     0.5081     0.4387

    >>> b = fly.unwrap(a, 2, 2, 2, 2)
    >>> fly.display(b)

    [4 9 1 1]
        0.4107     0.9518     0.4198     0.0901     0.1098     0.8827     0.0012     0.5259     0.3253
        0.8224     0.1794     0.0081     0.5933     0.1046     0.1647     0.8703     0.1443     0.5081
        0.3775     0.6456     0.6600     0.8060     0.8395     0.7270     0.9250     0.9313     0.6592
        0.3027     0.5591     0.0764     0.5938     0.1933     0.0322     0.3063     0.8684     0.4387

    >>> fly.display(c)

    [6 6 1 1]
        0.4107     0.3775     0.0901     0.8060     0.0012     0.9250
        0.8224     0.3027     0.5933     0.5938     0.8703     0.3063
        0.9518     0.6456     0.1098     0.8395     0.5259     0.9313
        0.1794     0.5591     0.1046     0.1933     0.1443     0.8684
        0.4198     0.6600     0.8827     0.7270     0.3253     0.6592
        0.0081     0.0764     0.1647     0.0322     0.5081     0.4387


    """

    out = Array()
    safe_call(backend.get().fly_wrap(c_pointer(out.arr), a.arr,
                                    c_dim_t(ox), c_dim_t(oy),
                                    c_dim_t(wx), c_dim_t(wy),
                                    c_dim_t(sx), c_dim_t(sy),
                                    c_dim_t(px), c_dim_t(py),
                                    is_column))
    return out

def sat(image):
    """
    Summed Area Tables

    Parameters
    ----------
    image : fly.Array
          A multi dimensional array specifying image or batch of images

    Returns
    -------
    out  : fly.Array
         A multi dimensional array containing the summed area table of input image
    """

    out = Array()
    safe_call(backend.get().fly_sat(c_pointer(out.arr), image.arr))
    return out

def ycbcr2rgb(image, standard=YCC_STD.BT_601):
    """
    YCbCr to RGB colorspace conversion.

    Parameters
    ----------

    image   : fly.Array
              A multi dimensional array containing an image or batch of images in YCbCr format.

    standard: YCC_STD. optional. default: YCC_STD.BT_601
            - Specifies the YCbCr format.
            - Can be one of YCC_STD.BT_601, YCC_STD.BT_709, and YCC_STD.BT_2020.

    Returns
    --------

    out     : fly.Array
            A multi dimensional array containing an image or batch of images in RGB format

    """

    out = Array()
    safe_call(backend.get().fly_ycbcr2rgb(c_pointer(out.arr), image.arr, standard.value))
    return out

def rgb2ycbcr(image, standard=YCC_STD.BT_601):
    """
    RGB to YCbCr colorspace conversion.

    Parameters
    ----------

    image   : fly.Array
              A multi dimensional array containing an image or batch of images in RGB format.

    standard: YCC_STD. optional. default: YCC_STD.BT_601
            - Specifies the YCbCr format.
            - Can be one of YCC_STD.BT_601, YCC_STD.BT_709, and YCC_STD.BT_2020.

    Returns
    --------

    out     : fly.Array
            A multi dimensional array containing an image or batch of images in YCbCr format

    """

    out = Array()
    safe_call(backend.get().fly_rgb2ycbcr(c_pointer(out.arr), image.arr, standard.value))
    return out

def moments(image, moment = MOMENT.FIRST_ORDER):
    """
    Calculate image moments.

    Parameters
    ----------
    image : fly.Array
          - A 2 D flare array representing an image, or
          - A multi dimensional array representing batch of images.

    moment : optional: fly.MOMENT. default: fly.MOMENT.FIRST_ORDER.
          Moment(s) to calculate. Can be one of:
          - fly.MOMENT.M00
          - fly.MOMENT.M01
          - fly.MOMENT.M10
          - fly.MOMENT.M11
          - fly.MOMENT.FIRST_ORDER

    Returns
    ---------
    out  : fly.Array
          - array containing requested moment(s) of each image
    """
    output = Array()
    safe_call(backend.get().fly_moments(c_pointer(output.arr), image.arr, moment.value))
    return output

def canny(image,
          low_threshold, high_threshold = None,
          threshold_type = CANNY_THRESHOLD.MANUAL,
          sobel_window = 3, is_fast = False):
    """
    Canny edge detector.

    Parameters
    ----------
    image : fly.Array
          - A 2 D flare array representing an image

    threshold_type : optional: fly.CANNY_THRESHOLD. default: fly.CANNY_THRESHOLD.MANUAL.
          Can be one of:
          - fly.CANNY_THRESHOLD.MANUAL
          - fly.CANNY_THRESHOLD.AUTO_OTSU

    low_threshold :  required: float.
          Specifies the % of maximum in gradient image if threshold_type is MANUAL.
          Specifies the % of auto dervied high value if threshold_type is AUTO_OTSU.

    high_threshold : optional: float. default: None
          Specifies the % of maximum in gradient image if threshold_type is MANUAL.
          Ignored if threshold_type is AUTO_OTSU

    sobel_window : optional: int. default: 3
          Specifies the size of sobel kernel when computing the gradient image.

    Returns
    --------

    out : fly.Array
        - A binary image containing the edges

    """
    output = Array()
    if threshold_type.value == CANNY_THRESHOLD.MANUAL.value:
        assert(high_threshold is not None)

    high_threshold = high_threshold if high_threshold else 0
    safe_call(backend.get().fly_canny(c_pointer(output.arr), image.arr,
                                     c_int_t(threshold_type.value),
                                     c_float_t(low_threshold), c_float_t(high_threshold),
                                     c_uint_t(sobel_window), c_bool_t(is_fast)))
    return output

def anisotropic_diffusion(image, time_step, conductance, iterations, flux_function_type = FLUX.QUADRATIC, diffusion_kind = DIFFUSION.GRAD):
    """
    Anisotropic smoothing filter.

    Parameters
    ----------
    image: fly.Array
        The input image.

    time_step: scalar.
        The time step used in solving the diffusion equation.

    conductance:
        Controls conductance sensitivity in diffusion equation.

    iterations:
        Number of times the diffusion step is performed.

    flux_function_type:
        Type of flux function to be used. Available flux functions:
          - Quadratic (fly.FLUX.QUADRATIC)
          - Exponential (fly.FLUX.EXPONENTIAL)

    diffusion_kind:
        Type of diffusion equatoin to be used. Available diffusion equations:
          - Gradient diffusion equation (fly.DIFFUSION.GRAD)
          - Modified curvature diffusion equation (fly.DIFFUSION.MCDE)

    Returns
    -------
    out: fly.Array
        Anisotropically-smoothed output image.

    """
    out = Array()
    safe_call(backend.get().
              fly_anisotropic_diffusion(c_pointer(out.arr), image.arr,
                                       c_float_t(time_step), c_float_t(conductance), c_uint_t(iterations),
                                       flux_function_type.value, diffusion_kind.value))
    return out

def iterativeDeconv(image, psf, iterations, relax_factor, algo = ITERATIVE_DECONV.DEFAULT):
    """
    Iterative deconvolution algorithm.

    Parameters
    ----------
    image: fly.Array
        The blurred input image.

    psf: fly.Array
        The kernel(point spread function) known to have caused
        the blur in the system.

    iterations:
        Number of times the algorithm will run.

    relax_factor: scalar.
        is the relaxation factor multiplied with distance
        of estimate from observed image.

    algo:
        takes enum value of type fly.ITERATIVE_DECONV
        indicating the iterative deconvolution algorithm to be used

    Returns
    -------
    out: fly.Array
        sharp image estimate generated from the blurred input

    Note
    -------
    relax_factor argument is ignored when the RICHARDSONLUCY algorithm is used.

    """
    out = Array()
    safe_call(backend.get().
              fly_iterative_deconv(c_pointer(out.arr), image.arr, psf.arr,
                                  c_uint_t(iterations), c_float_t(relax_factor), algo.value))
    return out

def inverseDeconv(image, psf, gamma, algo = ITERATIVE_DECONV.DEFAULT):
    """
    Inverse deconvolution algorithm.

    Parameters
    ----------
    image: fly.Array
        The blurred input image.

    psf: fly.Array
        The kernel(point spread function) known to have caused
        the blur in the system.

    gamma: scalar.
        is a user defined regularization constant

    algo:
        takes enum value of type fly.INVERSE_DECONV
        indicating the inverse deconvolution algorithm to be used

    Returns
    -------
    out: fly.Array
        sharp image estimate generated from the blurred input

    """
    out = Array()
    safe_call(backend.get().
              fly_inverse_deconv(c_pointer(out.arr), image.arr, psf.arr,
                                  c_float_t(gamma), algo.value))
    return out

def is_image_io_available():
    """
    Function to check if the flare library was built with Image IO support.
    """
    res = c_bool_t(False)
    safe_call(backend.get().fly_is_image_io_available(c_pointer(res)))
    return res.value
