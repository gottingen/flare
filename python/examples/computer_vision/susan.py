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

def susan_demo(console):

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

    features = fly.susan(img)

    xs = features.get_xpos().to_list()
    ys = features.get_ypos().to_list()

    draw_len = 3;
    num_features = features.num_features().value
    for f in range(num_features):
        print(f)
        x = xs[f]
        y = ys[f]

        # TODO fix coord order to x,y after upstream fix
        img_color = draw_corners(img_color, y, x, draw_len)


    print("Features found: {}".format(num_features))
    if not console:
        # Previews color image with green crosshairs
        wnd = fly.Window(512, 512, "SUSAN Feature Detector")

        while not wnd.close():
            wnd.image(img_color)
    else:
        print(xs);
        print(ys);


if __name__ == "__main__":
    if (len(sys.argv) > 1):
        fly.set_device(int(sys.argv[1]))
    console = (sys.argv[2] == '-') if len(sys.argv) > 2 else False

    fly.info()
    print("** flare SUSAN Feature Detector Demo **\n")
    susan_demo(console)

