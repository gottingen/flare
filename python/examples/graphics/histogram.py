#!/usr/bin/python

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

import flare as fly
import sys
import os

if __name__ == "__main__":

    if (len(sys.argv) == 1):
        raise RuntimeError("Expected to the image as the first argument")

    if not os.path.isfile(sys.argv[1]):
        raise RuntimeError("File %s not found" % sys.argv[1])

    if (len(sys.argv) >  2):
        fly.set_device(int(sys.argv[2]))

    fly.info()

    hist_win = fly.Window(512, 512, "3D Plot example using flare")
    img_win  = fly.Window(480, 640, "Input Image")

    img = fly.load_image(sys.argv[1]).as_type(fly.Dtype.u8)
    hist = fly.histogram(img, 256, 0, 255)

    while (not hist_win.close()) and (not img_win.close()):
        hist_win.hist(hist, 0, 255)
        img_win.image(img)
