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
#pragma once
#include <runtime/cuda/details/graph_base.h>

/*
   o
 / | \
o  o  o
 \ | /
   o
 / | \
o  o  o
 \ | /
   o
*/
class Diamond: public Graph {

  public:

    Diamond(int num_partitions, int num_iterations);

    ~Diamond();

  private:

    int _num_partitions;
    int _num_iterations;
};

Diamond::Diamond(int num_partitions, int num_iterations):
Graph{3 + 2 * (num_iterations - 1)},
_num_partitions{num_partitions},
_num_iterations{num_iterations}
{
  _graph.resize(3 + 2 * (num_iterations - 1));

  std::vector<size_t> map_out_nodes(num_partitions);
  std::iota(map_out_nodes.begin(), map_out_nodes.end(), 0);

  _graph[0].emplace_back(0, 0, map_out_nodes);

  for(int iter = 0; iter < _num_iterations; ++iter) {
    std::vector<size_t> map_out_nodes(num_partitions);
    std::iota(map_out_nodes.begin(), map_out_nodes.end(), 0);
    int cur_level = 2 * iter + 1;
    int next_level = 2 * iter + 2;

    std::vector<Node> map_nodes;
    for(int i = 0; i < _num_partitions; ++i) {
      std::vector<size_t> reduce_out_nodes(1, 0);
      map_nodes.emplace_back(cur_level, i, reduce_out_nodes);
    }

    _graph[cur_level] = std::move(map_nodes);
    _graph[next_level].emplace_back(next_level, 0, map_out_nodes);
  }

  _num_nodes = 1 + 4 * (num_iterations);
  allocate_nodes();
}

Diamond::~Diamond() {
  free_nodes();
}
