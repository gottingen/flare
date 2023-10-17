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
#include <flare/runtime/cuda/cudaflow.h>
#include <cassert>

template <typename OPT>
class GraphExecutor {

  public:

    GraphExecutor(Graph& graph, int dev_id = 0);

    template <typename... OPT_Args>
    void traversal(OPT_Args&&... args);

  private:

    int _dev_id;

    Graph& _g;

};

template <typename OPT>
GraphExecutor<OPT>::GraphExecutor(Graph& graph, int dev_id): _g{graph}, _dev_id{dev_id} {
  //TODO: why we cannot put cuda lambda function here?
}

template <typename OPT>
template <typename... OPT_Args>
void GraphExecutor<OPT>::traversal(OPT_Args&&... args) {

  flare::rt::Taskflow taskflow;
  flare::rt::Executor executor;

  taskflow.emplace([this, args...]() {

    flare::rt::cudaFlowCapturer cf;

    cf.make_optimizer<OPT>(args...);

    std::vector<std::vector<flare::rt::cudaTask>> tasks;
    tasks.resize(_g.get_graph().size());

    for(size_t l = 0; l < _g.get_graph().size(); ++l) {
      tasks[l].resize((_g.get_graph())[l].size());
      for(size_t i = 0; i < (_g.get_graph())[l].size(); ++i) {
        bool* v = _g.at(l, i).visited;
        tasks[l][i] = cf.single_task([v] __device__ () {
          *v = true;
        });
      }
    }

    for(size_t l = 0; l < _g.get_graph().size() - 1; ++l) {
      for(size_t i = 0; i < (_g.get_graph())[l].size(); ++i) {
        for(auto&& out_node: _g.at(l, i).out_nodes) {
          tasks[l][i].precede(tasks[l + 1][out_node]);
        }
      }
    }

    flare::rt::cudaStream stream;
    cf.run(stream);
    stream.synchronize();

  }).name("traverse");

  //auto check_t = taskflow.emplace([this](){
    //assert(_g.traversed());
  //});

  //trav_t.precede(check_t);

  executor.run(taskflow).wait();
}

