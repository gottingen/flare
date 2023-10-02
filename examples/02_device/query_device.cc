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

#include <iostream>
#include <sstream>

#include <flare/core/defines.h>

//#define USE_MPI
#if defined(USE_MPI)
#include <mpi.h>
#endif

#include <flare/core.h>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

int main(int argc, char** argv) {
  std::ostringstream msg;

  (void)argc;
  (void)argv;
#if defined(USE_MPI)

  MPI_Init(&argc, &argv);

  int mpi_rank = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  msg << "MPI rank(" << mpi_rank << ") ";

#endif
  flare::initialize(argc, argv);
  msg << "{" << std::endl;

  if (flare::hwloc::available()) {
    msg << "hwloc( NUMA[" << flare::hwloc::get_available_numa_count()
        << "] x CORE[" << flare::hwloc::get_available_cores_per_numa()
        << "] x HT[" << flare::hwloc::get_available_threads_per_core() << "] )"
        << std::endl;
  }

  flare::print_configuration(msg);

  msg << "}" << std::endl;

  std::cout << msg.str();
  flare::finalize();
#if defined(USE_MPI)

  MPI_Finalize();

#endif

  return 0;
}
