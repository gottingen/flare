// Copyright 2023 The Elastic-AI Authors.
// part of Elastic AI Search
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#ifndef SERIAL_CATEGORY_TEST_H_
#define SERIAL_CATEGORY_TEST_H_

#define TEST_CATEGORY serial
#define TEST_CATEGORY_NUMBER 0
#define TEST_CATEGORY_DEATH serial_DeathTest
#define TEST_EXECSPACE flare::Serial
#define TEST_CATEGORY_FIXTURE(name) serial_##name

#endif  // SERIAL_CATEGORY_TEST_H_
