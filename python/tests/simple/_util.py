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

import logging
import sys
import traceback


class _simple_test_dict(dict):
    def __init__(self):
        self.print_str = "Simple %16s: %s"
        self.failed = False
        super(_simple_test_dict, self).__init__()

    def run(self, name_list=None, verbose=False):
        test_list = name_list if name_list is not None else self.keys()
        for key in test_list:
            self.print_log = ""
            try:
                test = self[key]
            except KeyError:
                print(self.print_str % (key, "NOTFOUND"))
                continue

            try:
                test(verbose)
                print(self.print_str % (key, "PASSED"))
            except Exception:
                print(self.print_str % (key, "FAILED"))
                self.failed = True
                if not verbose:
                    print(tests.print_log)
                logging.error(traceback.format_exc())

        if self.failed:
            sys.exit(1)


tests = _simple_test_dict()


def print_func(verbose):
    def print_func_impl(*args):
        _print_log = ""
        for arg in args:
            _print_log += str(arg) + '\n'
        if verbose:
            print(_print_log)
        tests.print_log += _print_log
    return print_func_impl


def display_func(verbose):
    return print_func(verbose)