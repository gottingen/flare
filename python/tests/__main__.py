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

from __future__ import absolute_import

import sys

from . import simple

tests = {}
tests['simple'] = simple.tests


def assert_valid(name, name_list, name_str):
    is_valid = any([name == val for val in name_list])
    if is_valid:
        return
    err_str = "The first argument needs to be a %s name\n" % name_str
    err_str += "List of supported %ss: %s" % (name_str, str(list(name_list)))
    raise RuntimeError(err_str)


if __name__ == "__main__":
    module_name = None
    num_args = len(sys.argv)

    if num_args > 1:
        module_name = sys.argv[1].lower()
        assert_valid(sys.argv[1].lower(), tests.keys(), "module")

    if module_name is None:
        for name in tests:
            tests[name].run()
    else:
        test = tests[module_name]
        test_list = None

        if num_args > 2:
            test_list = sys.argv[2:]
            for test_name in test_list:
                assert_valid(test_name.lower(), test.keys(), "test")

        test.run(test_list)
