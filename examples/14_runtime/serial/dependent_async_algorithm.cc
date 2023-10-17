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
/**
  This program demonstrates how to use dependent async tasks to create
  dependent algorithm tasks.
*/

#include <flare/runtime/taskflow.h>
#include <flare/runtime/algorithm/for_each.h>
#include <flare/runtime/algorithm/transform.h>
#include <flare/runtime/algorithm/reduce.h>

int main(){

  const size_t N = 65536;

  flare::rt::Executor executor;
  
  int sum{1};
  std::vector<int> data(N);

  // for-each
  flare::rt::AsyncTask A = executor.silent_dependent_async(flare::rt::make_for_each_task(
    data.begin(), data.end(), [](int& i){ i = 1; }
  ));

  // transform
  flare::rt::AsyncTask B = executor.silent_dependent_async(flare::rt::make_transform_task(
    data.begin(), data.end(), data.begin(), [](int& i) { return i*2; }
  ), A);

  // reduce
  flare::rt::AsyncTask C = executor.silent_dependent_async(flare::rt::make_reduce_task(
    data.begin(), data.end(), sum, std::plus<int>{}
  ), B);
  
  // wait for all async task to complete
  executor.wait_for_all();
  
  // verify the result
  if(sum != N*2 + 1) {
    throw std::runtime_error("INCORRECT RESULT");
  }
  else {
    std::cout << "CORRECT RESULT\n";
  }
  
  return 0;
}




