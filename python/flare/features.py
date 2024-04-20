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
Features class used for Computer Vision algorithms.
"""

from .library import *
from .array import *
import numbers

class Features(object):
    """
    A container class used for various feature detectors.

    Parameters
    ----------

    num: optional: int. default: 0.
         Specifies the number of features.
    """

    def __init__(self, num=0):
        self.feat = c_void_ptr_t(0)
        if num is not None:
            assert(isinstance(num, numbers.Number))
            safe_call(backend.get().fly_create_features(c_pointer(self.feat), c_dim_t(num)))

    def __del__(self):
        """
        Release features' memory
        """
        if self.feat:
            backend.get().fly_release_features(self.feat)
            self.feat = None

    def num_features(self):
        """
        Returns the number of features detected.
        """
        num = c_dim_t(0)
        safe_call(backend.get().fly_get_features_num(c_pointer(num), self.feat))
        return num

    def get_xpos(self):
        """
        Returns the x-positions of the features detected.
        """
        out = Array()
        safe_call(backend.get().fly_get_features_xpos(c_pointer(out.arr), self.feat))
        return out

    def get_ypos(self):
        """
        Returns the y-positions of the features detected.
        """
        out = Array()
        safe_call(backend.get().fly_get_features_ypos(c_pointer(out.arr), self.feat))
        return out

    def get_score(self):
        """
        Returns the scores of the features detected.
        """
        out = Array()
        safe_call(backend.get().fly_get_features_score(c_pointer(out.arr), self.feat))
        return out

    def get_orientation(self):
        """
        Returns the orientations of the features detected.
        """
        out = Array()
        safe_call(backend.get().fly_get_features_orientation(c_pointer(out.arr), self.feat))
        return out

    def get_size(self):
        """
        Returns the sizes of the features detected.
        """
        out = Array()
        safe_call(backend.get().fly_get_features_size(c_pointer(out.arr), self.feat))
        return out
