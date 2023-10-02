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

#include <flare/core/common/cpu_discovery.h>

#include <cstdlib>  // getenv
#include <string>

int flare::detail::mpi_ranks_per_node() {
  for (char const* env_var : {
           "OMPI_COMM_WORLD_LOCAL_SIZE",  // OpenMPI
           "MV2_COMM_WORLD_LOCAL_SIZE",   // MVAPICH2
           "MPI_LOCALNRANKS",             // MPICH
                                          // SLURM???
           "PMI_LOCAL_SIZE",              // PMI
       }) {
    char const* str = std::getenv(env_var);
    if (str) {
      return std::stoi(str);
    }
  }
  return -1;
}

int flare::detail::mpi_local_rank_on_node() {
  for (char const* env_var : {
           "OMPI_COMM_WORLD_LOCAL_RANK",  // OpenMPI
           "MV2_COMM_WORLD_LOCAL_RANK",   // MVAPICH2
           "MPI_LOCALRANKID",             // MPICH
           "SLURM_LOCALID",               // SLURM
           "PMI_LOCAL_RANK",              // PMI
       }) {
    char const* str = std::getenv(env_var);
    if (str) {
      return std::stoi(str);
    }
  }
  return -1;
}

bool flare::detail::mpi_detected() { return mpi_local_rank_on_node() != -1; }
