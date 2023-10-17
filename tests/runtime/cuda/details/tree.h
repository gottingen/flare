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

#include <vector>
#include <cassert>
#include <cmath>

//full tree
// child nodes' index = degree * idx ~ degree * (idx + 1) - 1
class Tree : public Graph {

  public:

    Tree(int degree, int level);

    ~Tree();

  private:

    int _degree;
};


Tree::Tree(int degree, int level): _degree{degree}, Graph{level}
{
  assert(_level != 0 && _degree != 0);
  _graph.resize(_level);

  for(int l = 0; l < _level; ++l) {
    size_t id{0};

    std::vector<Node> cur_level_nodes;
    size_t cur_level_num_nodes = std::pow(_degree, l);
    cur_level_nodes.reserve(cur_level_num_nodes);

    for(size_t n = 0; n < cur_level_num_nodes; ++n) {
      std::vector<size_t> out_nodes(_degree);
      std::iota(out_nodes.begin(), out_nodes.end(), id * _degree);
      cur_level_nodes.emplace_back(l, id++, out_nodes);
    }

    _graph[l] = std::move(cur_level_nodes);

    _num_nodes += cur_level_num_nodes;
  }

  allocate_nodes();

}

Tree::~Tree() {
  free_nodes();
}
