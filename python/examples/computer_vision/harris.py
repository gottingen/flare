#!/usr/bin/env python

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

from time import time
import flare as fly
import os
import sys

def get_assert_path():

    cwd_path = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(cwd_path + "/../../assets"):
        return cwd_path + "/../../assets"

    if os.path.exists(cwd_path + "/../../../share/flare/asserts"):
        return cwd_path + "/../../../share/flare/asserts"
    return ""


def draw_corners(img, x, y, draw_len):
    # Draw vertical line of (draw_len * 2 + 1) pixels centered on  the corner
    # Set only the first channel to 1 (green lines)
    xmin = max(0, x - draw_len)
    xmax = min(img.dims()[1], x + draw_len)

    img[y, xmin : xmax, 0] = 0.0
    img[y, xmin : xmax, 1] = 1.0
    img[y, xmin : xmax, 2] = 0.0

    # Draw vertical line of (draw_len * 2 + 1) pixels centered on  the corner
    # Set only the first channel to 1 (green lines)
    ymin = max(0, y - draw_len)
    ymax = min(img.dims()[0], y + draw_len)

    img[ymin : ymax, x, 0] = 0.0
    img[ymin : ymax, x, 1] = 1.0
    img[ymin : ymax, x, 2] = 0.0
    return img

def harris_demo(console):

    apath = get_assert_path()
    if apath == "":
        print("can not get asserts path")
        return

    file_path = apath
    if console:
        file_path += "/examples/images/square.png"
    else:
        file_path += "/examples/images/man.jpg"
    img_color = fly.load_image(file_path, True);

    img = fly.color_space(img_color, fly.CSPACE.GRAY, fly.CSPACE.RGB)
    img_color /= 255.0

    ix, iy = fly.gradient(img)
    ixx = ix * ix
    ixy = ix * iy
    iyy = iy * iy

    # Compute a Gaussian kernel with standard deviation of 1.0 and length of 5 pixels
    # These values can be changed to use a smaller or larger window
    gauss_filt = fly.gaussian_kernel(5, 5, 1.0, 1.0)

    # Filter second order derivatives
    ixx = fly.convolve(ixx, gauss_filt)
    ixy = fly.convolve(ixy, gauss_filt)
    iyy = fly.convolve(iyy, gauss_filt)

    # Calculate trace
    itr = ixx + iyy

    # Calculate determinant
    idet = ixx * iyy - ixy * ixy

    # Calculate Harris response
    response = idet - 0.04 * (itr * itr)

    # Get maximum response for each 3x3 neighborhood
    mask = fly.constant(1, 3, 3)
    max_resp = fly.dilate(response, mask)

    # Discard responses that are not greater than threshold
    corners = response > 1e5
    corners = corners * response

    # Discard responses that are not equal to maximum neighborhood response,
    # scale them to original value
    corners = (corners == max_resp) * corners

    # Copy device array to python list on host
    corners_list = corners.to_list()

    draw_len = 3
    good_corners = 0
    for x in range(img_color.dims()[1]):
        for y in range(img_color.dims()[0]):
            if corners_list[x][y] > 1e5:
                img_color = draw_corners(img_color, x, y, draw_len)
                good_corners += 1


    print("Corners found: {}".format(good_corners))
    if not console:
        # Previews color image with green crosshairs
        wnd = fly.Window(512, 512, "Harris Feature Detector")

        while not wnd.close():
            wnd.image(img_color)
    else:
        idx = fly.where(corners)

        corners_x = idx / float(corners.dims()[0])
        corners_y = idx % float(corners.dims()[0])

        print(corners_x)
        print(corners_y)


if __name__ == "__main__":
    if (len(sys.argv) > 1):
        fly.set_device(int(sys.argv[1]))
    console = (sys.argv[2] == '-') if len(sys.argv) > 2 else False

    fly.info()
    print("** flare Harris Corner Detector Demo **\n")

    harris_demo(console)

