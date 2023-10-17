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
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <flare/runtime/taskflow.h>
#include <flare/runtime/cuda/cudaflow.h>
#include <flare/runtime/cuda/algorithm/for_each.h>

#include <runtime/cuda/details/graph_executor.h>
#include <runtime/cuda/details/tree.h>
#include <runtime/cuda/details/random_dag.h>
#include <runtime/cuda/details/tree.h>
#include <runtime/cuda/details/diamond.h>

// ----------------------------------------------------------------------------
// Graph traversal
// ----------------------------------------------------------------------------
template <typename GRAPH, typename OPT, typename... OPT_Args>
void traversal(OPT_Args&&... args) {
  for(int i = 0; i < 13; ++i) {
    Graph* g;
    if constexpr(std::is_same_v<GRAPH, Tree>) {
      g = new Tree(::rand() % 3 + 1, ::rand() % 5 + 1);
    }
    else if constexpr(std::is_same_v<GRAPH, RandomDAG>) {
      g = new RandomDAG(::rand() % 10 + 1, ::rand() % 10 + 1, ::rand() % 10 + 1);
    }
    else if constexpr(std::is_same_v<GRAPH, Diamond>) {
      g = new Diamond(::rand() % 10 + 1, ::rand() % 10 + 1);
    }
    GraphExecutor<OPT> executor(*g, 0); 
    executor.traversal(std::forward<OPT_Args>(args)...);

    REQUIRE(g->traversed());
    delete g;
  }

}

TEST_CASE("cudaFlowCapturer.tree.Sequential") {
  traversal<Tree, flare::rt::cudaFlowSequentialOptimizer>();
}

TEST_CASE("cudaFlowCapturer.tree.RoundRobin.1") {
  traversal<Tree, flare::rt::cudaFlowRoundRobinOptimizer>(1);
}

TEST_CASE("cudaFlowCapturer.tree.RoundRobin.2") {
  traversal<Tree, flare::rt::cudaFlowRoundRobinOptimizer>(2);
}

TEST_CASE("cudaFlowCapturer.tree.RoundRobin.3") {
  traversal<Tree, flare::rt::cudaFlowRoundRobinOptimizer>(3);
}

TEST_CASE("cudaFlowCapturer.tree.RoundRobin.4") {
  traversal<Tree, flare::rt::cudaFlowRoundRobinOptimizer>(4);
}

//TEST_CASE("cudaFlowCapturer.tree.Greedy.1") {
//  traversal<Tree, flare::rt::cudaGreedyCapturing>(1);
//}
//
//TEST_CASE("cudaFlowCapturer.tree.Greedy.2") {
//  traversal<Tree, flare::rt::cudaGreedyCapturing>(2);
//}
//
//TEST_CASE("cudaFlowCapturer.tree.Greedy.3") {
//  traversal<Tree, flare::rt::cudaGreedyCapturing>(3);
//}
//
//TEST_CASE("cudaFlowCapturer.tree.Greedy.4") {
//  traversal<RandomDAG, flare::rt::cudaGreedyCapturing>(4);
//}

TEST_CASE("cudaFlowCapturer.randomDAG.Sequential") {
  traversal<RandomDAG,flare::rt::cudaFlowSequentialOptimizer>();
}

TEST_CASE("cudaFlowCapturer.randomDAG.RoundRobin.1") {
  traversal<RandomDAG, flare::rt::cudaFlowRoundRobinOptimizer>(1);
}

TEST_CASE("cudaFlowCapturer.randomDAG.RoundRobin.2") {
  traversal<RandomDAG, flare::rt::cudaFlowRoundRobinOptimizer>(2);
}

TEST_CASE("cudaFlowCapturer.randomDAG.RoundRobin.3") {
  traversal<RandomDAG, flare::rt::cudaFlowRoundRobinOptimizer>(3);
}

TEST_CASE("cudaFlowCapturer.randomDAG.RoundRobin.4") {
  traversal<RandomDAG, flare::rt::cudaFlowRoundRobinOptimizer>(4);
}

//TEST_CASE("cudaFlowCapturer.randomDAG.Greedy.1") {
//  traversal<RandomDAG, flare::rt::cudaGreedyCapturing>(1);
//}
//
//TEST_CASE("cudaFlowCapturer.randomDAG.Greedy.2") {
//  traversal<RandomDAG, flare::rt::cudaGreedyCapturing>(2);
//}
//
//TEST_CASE("cudaFlowCapturer.randomDAG.Greedy.3") {
//  traversal<RandomDAG, flare::rt::cudaGreedyCapturing>(3);
//}
//
//TEST_CASE("cudaFlowCapturer.randomDAG.Greedy.4") {
//  traversal<RandomDAG, flare::rt::cudaGreedyCapturing>(4);
//}

TEST_CASE("cudaFlowCapturer.diamond.Sequential") {
  traversal<Diamond, flare::rt::cudaFlowSequentialOptimizer>();
}

TEST_CASE("cudaFlowCapturer.diamond.RoundRobin.1") {
  traversal<Diamond, flare::rt::cudaFlowRoundRobinOptimizer>(1);
}

TEST_CASE("cudaFlowCapturer.diamond.RoundRobin.2") {
  traversal<Diamond, flare::rt::cudaFlowRoundRobinOptimizer>(2);
}

TEST_CASE("cudaFlowCapturer.diamond.RoundRobin.3") {
  traversal<Diamond, flare::rt::cudaFlowRoundRobinOptimizer>(3);
}

TEST_CASE("cudaFlowCapturer.diamond.RoundRobin.4") {
  traversal<Diamond, flare::rt::cudaFlowRoundRobinOptimizer>(4);
}

//TEST_CASE("cudaFlowCapturer.diamond.Greedy.1") {
//  traversal<Diamond, flare::rt::cudaGreedyCapturing>(1);
//}
//
//TEST_CASE("cudaFlowCapturer.diamond.Greedy.2") {
//  traversal<Diamond, flare::rt::cudaGreedyCapturing>(2);
//}
//
//TEST_CASE("cudaFlowCapturer.diamond.Greedy.3") {
//  traversal<Diamond, flare::rt::cudaGreedyCapturing>(3);
//}
//
//TEST_CASE("cudaFlowCapturer.diamond.Greedy.4") {
//  traversal<Diamond, flare::rt::cudaGreedyCapturing>(4);
//}

//------------------------------------------------------
// dependencies
//------------------------------------------------------

template <typename OPT, typename... OPT_Args>
void dependencies(OPT_Args ...args) {
  
  for(int t = 0; t < 17; ++t) {
    int num_partitions = ::rand() % 5 + 1;
    int num_iterations = ::rand() % 7 + 1;

    Diamond g(num_partitions, num_iterations);

    flare::rt::cudaFlowCapturer cf;
    cf.make_optimizer<OPT>(std::forward<OPT_Args>(args)...);

    int* inputs{nullptr};
    REQUIRE(cudaMallocManaged(&inputs, num_partitions * sizeof(int)) == cudaSuccess);
    REQUIRE(cudaMemset(inputs, 0, num_partitions * sizeof(int)) == cudaSuccess);

    std::vector<std::vector<flare::rt::cudaTask>> tasks;
    tasks.resize(g.get_size());

    for(size_t l = 0; l < g.get_size(); ++l) {
      tasks[l].resize((g.get_graph())[l].size());
      for(size_t i = 0; i < (g.get_graph())[l].size(); ++i) {
        
        if(l % 2 == 1) {
          tasks[l][i] = cf.single_task([inputs, i] __device__ () {
            inputs[i]++;
          });
        }
        else {
          tasks[l][i] = cf.on([=](cudaStream_t stream){
            cuda_for_each(
              flare::rt::cudaDefaultExecutionPolicy(stream), inputs, inputs + num_partitions,
              [] __device__ (int& v) { v*=2; }
            );
          });
        }
      }
    }

    for(size_t l = 0; l < g.get_size() - 1; ++l) {
      for(size_t i = 0; i < (g.get_graph())[l].size(); ++i) {
        for(auto&& out_node: g.at(l, i).out_nodes) {
          tasks[l][i].precede(tasks[l + 1][out_node]);
        }
      }
    }

    flare::rt::cudaStream stream;
    cf.run(stream);
    stream.synchronize();
    
    int result = 2;
    for(int i = 1; i < num_iterations; ++i) {
      result = result * 2 + 2;
    }

    for(int i = 0; i < num_partitions; ++i) {
      REQUIRE(inputs[i] == result);
    }

    REQUIRE(cudaFree(inputs) == cudaSuccess);
  }
}

TEST_CASE("cudaFlowCapturer.dependencies.diamond.Sequential") {
  dependencies<flare::rt::cudaFlowSequentialOptimizer>();
}

TEST_CASE("cudaFlowCapturer.dependencies.diamond.RoundRobin.1") {
  dependencies<flare::rt::cudaFlowRoundRobinOptimizer>(1);
}

TEST_CASE("cudaFlowCapturer.dependencies.diamond.RoundRobin.2") {
  dependencies<flare::rt::cudaFlowRoundRobinOptimizer>(2);
}

TEST_CASE("cudaFlowCapturer.dependencies.diamond.RoundRobin.3") {
  dependencies<flare::rt::cudaFlowRoundRobinOptimizer>(3);
}

TEST_CASE("cudaFlowCapturer.dependencies.diamond.RoundRobin.4") {
  dependencies<flare::rt::cudaFlowRoundRobinOptimizer>(4);
}

//TEST_CASE("cudaFlowCapturer.dependencies.diamond.Greedy.1") {
//  dependencies<flare::rt::cudaGreedyCapturing>(1);
//}
//
//TEST_CASE("cudaFlowCapturer.dependencies.diamond.Greedy.2") {
//  dependencies<flare::rt::cudaGreedyCapturing>(2);
//}
//
//TEST_CASE("cudaFlowCapturer.dependencies.diamond.Greedy.3") {
//  dependencies<flare::rt::cudaGreedyCapturing>(3);
//}
//
//TEST_CASE("cudaFlowCapturer.dependencies.diamond.Greedy.4") {
//  dependencies<flare::rt::cudaGreedyCapturing>(4);
//}
