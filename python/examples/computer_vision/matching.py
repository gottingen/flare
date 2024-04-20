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


def get_assert_path():

    cwd_path = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(cwd_path + "/../../assets"):
        return cwd_path + "/../../assets"

    if os.path.exists(cwd_path + "/../../../share/flare/asserts"):
        return cwd_path + "/../../../share/flare/asserts"
    return ""
def normalize(a):
    max_ = float(fly.max(a))
    min_ = float(fly.min(a))
    return  (a - min_) /  (max_ - min_)

def draw_rectangle(img, x, y, wx, wy):
    print("\nMatching patch origin = ({}, {})\n".format(x, y))

    # top edge
    img[y, x : x + wx, 0] = 0.0
    img[y, x : x + wx, 1] = 0.0
    img[y, x : x + wx, 2] = 1.0

    # bottom edge
    img[y + wy, x : x + wx, 0] = 0.0
    img[y + wy, x : x + wx, 1] = 0.0
    img[y + wy, x : x + wx, 2] = 1.0

    # left edge
    img[y : y + wy, x, 0] = 0.0
    img[y : y + wy, x, 1] = 0.0
    img[y : y + wy, x, 2] = 1.0

    # left edge
    img[y : y + wy, x + wx, 0] = 0.0
    img[y : y + wy, x + wx, 1] = 0.0
    img[y : y + wy, x + wx, 2] = 1.0

    return img

def templateMatchingDemo(console):

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

    # Convert the image from RGB to gray-scale
    img = fly.color_space(img_color, fly.CSPACE.GRAY, fly.CSPACE.RGB)
    iDims = img.dims()
    print("Input image dimensions: ", iDims)

    # Extract a patch from the input image
    patch_size = 100
    tmp_img = img[100 : 100+patch_size, 100 : 100+patch_size]

    result = fly.match_template(img, tmp_img) # Default disparity metric is
                                             # Sum of Absolute differences (SAD)
                                             # Currently supported metrics are
                                             # FLY_SAD, FLY_ZSAD, FLY_LSAD, FLY_SSD,
                                             # FLY_ZSSD, FLY_LSSD

    disp_img = img / 255.0
    disp_tmp = tmp_img / 255.0
    disp_res = normalize(result)

    minval, minloc = fly.imin(disp_res)
    print("Location(linear index) of minimum disparity value = {}".format(minloc))

    if not console:
        marked_res = fly.tile(disp_img, 1, 1, 3)
        marked_res = draw_rectangle(marked_res, minloc%iDims[0], minloc/iDims[0],\
                                    patch_size, patch_size)

        print("Note: Based on the disparity metric option provided to matchTemplate function")
        print("either minimum or maximum disparity location is the starting corner")
        print("of our best matching patch to template image in the search image")

        wnd = fly.Window(512, 512, "Template Matching Demo")

        while not wnd.close():
            wnd.set_colormap(fly.COLORMAP.DEFAULT)
            wnd.grid(2, 2)
            wnd[0, 0].image(disp_img, "Search Image" )
            wnd[0, 1].image(disp_tmp, "Template Patch" )
            wnd[1, 0].image(marked_res, "Best Match" )
            wnd.set_colormap(fly.COLORMAP.HEAT)
            wnd[1, 1].image(disp_res, "Disparity Values")
            wnd.show()


if __name__ == "__main__":
    if (len(sys.argv) > 1):
        fly.set_device(int(sys.argv[1]))
    console = (sys.argv[2] == '-') if len(sys.argv) > 2 else False

    fly.info()
    print("** flare template matching Demo **\n")
    templateMatchingDemo(console)

