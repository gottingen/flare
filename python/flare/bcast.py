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
Function to perform broadcasting operations.
"""

class _bcast(object):
    _flag = False
    def get(self):
        return _bcast._flag

    def set(self, flag):
        _bcast._flag = flag

    def toggle(self):
        _bcast._flag ^= True

_bcast_var = _bcast()

def broadcast(func, *args):
    """
    Function to perform broadcast operations.

    This function can be used directly or as an annotation in the following manner.

    Example
    -------

    Using broadcast as an annotation

    >>> import flare as fly
    >>> @fly.broadcast
    ... def add(a, b):
    ...     return a + b
    ...
    >>> a = fly.randu(2,3)
    >>> b = fly.randu(2,1) # b is a different size
    >>> # Trying to add arrays of different sizes raises an exceptions
    >>> c = add(a, b) # This call does not raise an exception because of the annotation
    >>> fly.display(a)
    [2 3 1 1]
        0.4107     0.9518     0.4198
        0.8224     0.1794     0.0081

    >>> fly.display(b)
    [2 1 1 1]
        0.7269
        0.7104

    >>> fly.display(c)
    [2 3 1 1]
        1.1377     1.6787     1.1467
        1.5328     0.8898     0.7185

    Using broadcast as function

    >>> import flare as fly
    >>> add = lambda a,b: a + b
    >>> a = fly.randu(2,3)
    >>> b = fly.randu(2,1) # b is a different size
    >>> # Trying to add arrays of different sizes raises an exceptions
    >>> c = fly.broadcast(add, a, b) # This call does not raise an exception
    >>> fly.display(a)
    [2 3 1 1]
        0.4107     0.9518     0.4198
        0.8224     0.1794     0.0081

    >>> fly.display(b)
    [2 1 1 1]
        0.7269
        0.7104

    >>> fly.display(c)
    [2 3 1 1]
        1.1377     1.6787     1.1467
        1.5328     0.8898     0.7185

    """

    def wrapper(*func_args):
        _bcast_var.toggle()
        res = func(*func_args)
        _bcast_var.toggle()
        return res

    if len(args) == 0:
        return wrapper
    else:
        return wrapper(*args)
