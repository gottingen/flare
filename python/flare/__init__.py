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
flare is a high performance scientific computing library with an easy to use API.


    >>> # Monte Carlo estimation of pi
    >>> def calc_pi_device(samples):
            # Simple, array based API
            # Generate uniformly distributed random numers
            x = af.randu(samples)
            y = af.randu(samples)
            # Supports Just In Time Compilation
            # The following line generates a single kernel
            within_unit_circle = (x * x + y * y) < 1
            # Intuitive function names
            return 4 * af.count(within_unit_circle) / samples

Programs written using flare are portable across CUDA, CPU devices.

The default backend is chosen in the following order of preference based on the available libraries:

    1. CUDA
    3. CPU

The backend can be chosen at the beginning of the program by using the following function

    >>> af.set_backend(name)

where name is one of 'cuda' or 'cpu'.

The functionality provided by flare spans the following domains:

    1. Vector Algorithms
    2. Image Processing
    3. Signal Processing
    4. Computer Vision
    5. Linear Algebra
    6. Statistics

"""

try:
    import pycuda.autoinit
except ImportError:
    pass

from .library    import *
from .array      import *
from .data       import *
from .util       import *
from .algorithm  import *
from .device     import *
from .blas       import *
from .arith      import *
from .statistics import *
from .lapack     import *
from .signal     import *
from .image      import *
from .features   import *
from .vision     import *
from .graphics   import *
from .bcast      import *
from .index      import *
from .interop    import *
from .timer      import *
from .random     import *
from .sparse     import *
from .ml         import *

# do not export default modules as part of flare
del ct
del inspect
del numbers
del os

if (FLY_NUMPY_FOUND):
    del np
