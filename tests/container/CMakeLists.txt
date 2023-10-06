#
# Copyright 2023 The titan-search Authors.
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

foreach(Tag Threads;Serial;OpenMP;Cuda)
  string(TOUPPER ${Tag} DEVICE)
  string(TOLOWER ${Tag} dir)
  string(TOLOWER ${Tag} lc_tag)
  set(TAG_EXT cc)
  if (Tag STREQUAL "Cuda")
    set(TAG_EXT cu)
  endif ()
  # Add test for that backend if it is enabled
  if(FLARE_BUILD_${DEVICE})
    set(UnitTestSources)
    set(dir ${CMAKE_CURRENT_BINARY_DIR}/${dir})
    file(MAKE_DIRECTORY ${dir})
    foreach(Name
        bitset
        dual_view
        dynamic_view
        dyn_view_api_generic
        dyn_view_api_rank12345
        dyn_view_api_rank67
        error_reporter
        offset_view
        scatter_view
        static_crs_graph
        container_without_initializing
        unordered_map
        container_view_ctor_prop_embedded_dim
        )
      # Write to a temporary intermediate file and call configure_file to avoid
      # updating timestamps triggering unnecessary rebuilds on subsequent cmake runs.
      set(file ${dir}/${lc_tag}_${Name}_test.${TAG_EXT})
      file(WRITE ${dir}/dummy.${TAG_EXT}
          "#include <${lc_tag}_category_test.h>\n"
          "#include <${Name}_test.h>\n"
      )
      configure_file(${dir}/dummy.${TAG_EXT} ${file})
      list(APPEND UnitTestSources ${file})
    endforeach()
    #fatal error C1128: number of sections exceeded object file format limit: compile with /bigobj
    if(FLARE_BUILD_CUDA AND WIN32)
     LIST(REMOVE_ITEM UnitTestSources ${dir}/cuda_dyn_view_api_generic_test.cu)
    endif()
    if (Tag STREQUAL "Cuda")
      carbin_cc_test(
              NAME containers_test_${lc_tag}
              SOURCES
              ${UnitTestSources}
              ${TEST_MAIN_SRC}
              COPTS ${CARBIN_CXX_OPTIONS}
              CUOPTS ${CARBIN_CU_OPTIONS}
              DEPS
              ${CUDA_CUDA_LIBRARY}
              flare::flare
      )
    else ()
      carbin_cc_test(
              NAME containers_test_${lc_tag}
              SOURCES
              ${UnitTestSources}
              ${TEST_MAIN_SRC}
              COPTS ${CARBIN_CXX_OPTIONS}
              DEPS
              ${CUDA_CUDA_LIBRARY}
              flare::flare
      )
    endif ()

  endif()
endforeach()


SET(COMPILE_ONLY_SOURCES
        create_mirror_test.cc
        dual_view_parameter_pack_test.cc
        is_view_trait_test.cc
        compile_main_test.cc
)

carbin_cc_binary(
        NAME container_compile_only_test
        SOURCES
        ${COMPILE_ONLY_SOURCES}
        COPTS ${CARBIN_CXX_OPTIONS}
        DEPS
        flare::flare
)