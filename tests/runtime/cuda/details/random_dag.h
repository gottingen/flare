
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

#include <random>
#include <algorithm>
#include <vector>
#include <cstring>

//-------------------------------------------------------------------------
//RandomDAG
//-------------------------------------------------------------------------

class RandomDAG: public Graph {

  public:

    RandomDAG(
      int level, size_t max_nodes_per_level, size_t max_edges_per_node
    );

    ~RandomDAG();

    inline size_t num_nodes() { return _num_nodes; }

    void draw_task_graph(flare::rt::Taskflow& taskflow, int dev_id);

    //inline void print_graph(std::ostream& os);

    //inline bool traversed();

  private:

    size_t _max_nodes_per_level;
    size_t _max_edges_per_node;

};

RandomDAG::RandomDAG(
  int level, size_t max_nodes_per_level, size_t max_edges_per_node
)
: Graph{level}, _max_nodes_per_level{max_nodes_per_level},
  _max_edges_per_node{max_edges_per_node}
{

  std::random_device device;
  std::mt19937 gen(device());
  std::srand(0);
  std::uniform_int_distribution<int> dist(1, _max_nodes_per_level);

  size_t cur_num_nodes = 1; // root
  for(int l = 0; l < _level; ++l) {
    std::vector<Node> cur_nodes;
    cur_nodes.reserve(cur_num_nodes); // number of nodes at current level

    size_t next_num_nodes = dist(gen); //number of nodes at next level

    std::vector<int> next_level_nodes(next_num_nodes);
    std::iota(next_level_nodes.begin(), next_level_nodes.end(), 0);

    //create edges for each node
    for(size_t i = 0; i < cur_num_nodes; ++i) {
      if(l != _level - 1) {
        std::shuffle(next_level_nodes.begin(), next_level_nodes.end(), gen);
        size_t edges = std::rand() % _max_edges_per_node + 1;
        if(edges > next_num_nodes) {
          edges = next_num_nodes;
        }

        std::vector<size_t> out_nodes(
          next_level_nodes.begin(),
          next_level_nodes.begin() + edges
        );

        cur_nodes.emplace_back(l, i, out_nodes);
      }
      else {
        std::vector<size_t> empty;
        cur_nodes.emplace_back(l, i, empty);
      }
    }

    _graph.emplace_back(std::move(cur_nodes));

    _num_nodes += cur_num_nodes;

    cur_num_nodes = next_num_nodes;
  }

  allocate_nodes();
}

RandomDAG::~RandomDAG() {
  free_nodes();
}

