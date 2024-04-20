#
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
#

set(URL "https://sparse.tamu.edu")

function(mtxDownload name group)
  set(root_dir ${flare_BINARY_DIR}/extern/matrixmarket)
  set(target_dir ${root_dir}/${group}/${name})
  set(mtx_name mtxDownload_${group}_${name})
  string(TOLOWER ${mtx_name} mtx_name)

  set_and_mark_depnames_advncd(mtx_prefix ${mtx_name})
  fly_dep_check_and_populate(${mtx_name}
    URI ${URL}/MM/${group}/${name}.tar.gz
  )

  if(NOT EXISTS "${target_dir}/${name}.mtx")
    file(MAKE_DIRECTORY ${target_dir})
    file(COPY ${${mtx_name}_SOURCE_DIR}/${name}.mtx DESTINATION ${target_dir})
  endif()
endfunction()

# Following files are used for testing mtx read fn
# integer data
mtxDownload("Trec4" "JGD_Kocay")
# real data
mtxDownload("bcsstm02" "HB")
# complex data
mtxDownload("young4c" "HB")

#Following files are used for sparse-sparse arith
# real data
#linear programming problem
mtxDownload("lpi_vol1" "LPnetlib")
mtxDownload("lpi_qual" "LPnetlib")
#Subsequent Circuit Simulation problem
mtxDownload("oscil_dcop_12" "Sandia")
mtxDownload("oscil_dcop_42" "Sandia")

# complex data
#Quantum Chemistry problem
mtxDownload("conf6_0-4x4-20" "QCD")
mtxDownload("conf6_0-4x4-30" "QCD")
