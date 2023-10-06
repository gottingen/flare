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


SET(ALGORITHM algo_test_main.cc)

foreach (Tag Threads;Serial;OpenMP;Cuda)
    string(TOUPPER ${Tag} DEVICE)
    string(TOLOWER ${Tag} dir)
    string(TOLOWER ${Tag} lc_tag)

    set(TAG_EXT cc)
    if (Tag STREQUAL "Cuda")
        set(TAG_EXT cu)
    endif ()

    if (FLARE_BUILD_${DEVICE})
        set(dir ${CMAKE_CURRENT_BINARY_DIR}/${dir})
        file(MAKE_DIRECTORY ${dir})

        # ------------------------------------------
        # Sort
        # ------------------------------------------
        # Each of these inputs is an .hpp file.
        # Generate a .cpp file for each one that runs it on the current backend (Tag),
        # and add this .cpp file to the sources for UnitTest_RandomAndSort.
        set(ALGO_SORT_SOURCES_${DEVICE} std_algorithms_common_test.cc)
        foreach (SOURCE_Input
                algo_sort
                algo_sort_custom_comp
                bin_sort_a
                bin_sort_b
                nested_sort
        )
            set(file ${dir}/${SOURCE_Input}_test.${TAG_EXT})
            # Write to a temporary intermediate file and call configure_file to avoid
            # updating timestamps triggering unnecessary rebuilds on subsequent cmake runs.
            file(WRITE ${dir}/dummy.${TAG_EXT}
                    "#include <${lc_tag}_category_test.h>\n"
                    "#include <${SOURCE_Input}_test.h>\n"
            )
            configure_file(${dir}/dummy.${TAG_EXT} ${file})
            list(APPEND ALGO_SORT_SOURCES_${DEVICE} ${file})
        endforeach ()
        if (Tag STREQUAL "Cuda")
            #[[carbin_cc_test(
                    NAME algorithm_sorting_test_${lc_tag}
                    SOURCES
                    ${ALGO_SORT_SOURCES_${DEVICE}}
                    ${TEST_MAIN_SRC}
                    COPTS ${CARBIN_CXX_OPTIONS}
                    CUOPTS ${CARBIN_CU_OPTIONS}
                    DEPS
                    ${CUDA_CUDA_LIBRARY}
                    flare::flare
            )]]
        else ()
            carbin_cc_test(
                    NAME algorithm_sorting_test_${lc_tag}
                    SOURCES
                    ${ALGO_SORT_SOURCES_${DEVICE}}
                    ${TEST_MAIN_SRC}
                    COPTS ${CARBIN_CXX_OPTIONS}
                    DEPS
                    ${CUDA_CUDA_LIBRARY}
                    flare::flare
            )
        endif ()

        # ------------------------------------------
        # Sort
        # ------------------------------------------
        # Each of these inputs is an .hpp file.
        # Generate a .cpp file for each one that runs it on the current backend (Tag),
        # and add this .cpp file to the sources for UnitTest_RandomAndSort.
        set(ALGO_RANDOM_SOURCES_${DEVICE} std_algorithms_common_test.cc)
        foreach (SOURCE_Input
               algo_random
        )
            set(file ${dir}/${SOURCE_Input}_test.${TAG_EXT})
            # Write to a temporary intermediate file and call configure_file to avoid
            # updating timestamps triggering unnecessary rebuilds on subsequent cmake runs.
            file(WRITE ${dir}/dummy.${TAG_EXT}
                    "#include <${lc_tag}_category_test.h>\n"
                    "#include <${SOURCE_Input}_test.h>\n"
            )
            configure_file(${dir}/dummy.${TAG_EXT} ${file})
            list(APPEND ALGO_RANDOM_SOURCES_${DEVICE} ${file})
        endforeach ()
        if (Tag STREQUAL "Cuda")
            carbin_cc_test(
                    NAME algorithm_random_test_${lc_tag}
                    SOURCES
                    ${ALGO_RANDOM_SOURCES_${DEVICE}}
                    ${TEST_MAIN_SRC}
                    COPTS ${CARBIN_CXX_OPTIONS}
                    CUOPTS ${CARBIN_CU_OPTIONS}
                    DEPS
                    ${CUDA_CUDA_LIBRARY}
                    flare::flare
            )
        else ()
            carbin_cc_test(
                    NAME algorithm_random_test_${lc_tag}
                    SOURCES
                    ${ALGO_RANDOM_SOURCES_${DEVICE}}
                    ${TEST_MAIN_SRC}
                    COPTS ${CARBIN_CXX_OPTIONS}
                    DEPS
                    ${CUDA_CUDA_LIBRARY}
                    flare::flare
            )
        endif ()

        set(ALGO_ALL_SOURCES_${DEVICE} std_algorithms_common_test.cc)
        foreach (Name
                std_algorithms_find
                std_algorithms_remove_copy_if
                std_algorithms_shift_right
                std_algorithms_team_for_each_n
                std_algorithms_team_reverse_copy
                std_algorithms_random_access_iterator
                std_algorithms_for_each
                std_algorithms_remove_copy
                std_algorithms_team_adjacent_find
                std_algorithms_team_for_each
                std_algorithms_team_reverse
                std_algorithms_adjacent_difference
                std_algorithms_inclusive_scan
                std_algorithms_remove_if
                std_algorithms_team_all_of
                std_algorithms_team_lexicographical_compare
                std_algorithms_team_rotate_copy
                std_algorithms_adjacent_find
                std_algorithms_is_sorted
                std_algorithms_remove
                std_algorithms_team_any_of
                std_algorithms_team_max_element
                std_algorithms_team_rotate
                std_algorithms_all_any_none_of
                std_algorithms_is_sorted_until
                std_algorithms_replace_copy_if
                std_algorithms_team_count_if
                std_algorithms_team_min_element
                std_algorithms_team_search_n
                std_algorithms_lexicographical_compare
                std_algorithms_replace_copy
                std_algorithms_team_count
                std_algorithms_team_min_max_element
                std_algorithms_team_search
                std_algorithms_min_max_element_ops
                std_algorithms_replace_if
                std_algorithms_team_equal
                std_algorithms_team_mismatch
                std_algorithms_team_shift_left
                std_algorithms_constraints
                std_algorithms_mismatch
                std_algorithms_replace
                std_algorithms_team_fill_n
                std_algorithms_team_move_backward
                std_algorithms_team_shift_right
                std_algorithms_copy_if
                std_algorithms_mod_ops
                std_algorithms_reverse
                std_algorithms_team_fill
                std_algorithms_team_move
                std_algorithms_transform_exclusive_scan
                std_algorithms_count
                std_algorithms_mod_seq_ops
                std_algorithms_rotate_copy
                std_algorithms_team_find_end
                std_algorithms_team_none_of
                std_algorithms_transform_inclusive_scan
                std_algorithms_equal
                std_algorithms_move_backward
                std_algorithms_rotate
                std_algorithms_team_find_first_of
                std_algorithms_team_replace_copy_if
                std_algorithms_transform_unary_op
                std_algorithms_exclusive_scan
                std_algorithms_numerics
                std_algorithms_search_n
                std_algorithms_team_find_if_not
                std_algorithms_team_replace_copy
                std_algorithms_unique_copy
                std_algorithms_find_end
                std_algorithms_partition_copy
                std_algorithms_search
                std_algorithms_team_find_if
                std_algorithms_team_replace_if
                std_algorithms_unique
                std_algorithms_find_first_of
                std_algorithms_partitioning_ops
                std_algorithms_shift_left
                std_algorithms_team_find
                std_algorithms_team_replace
                std_reducers

        )
            set(file ${Name}_test.cc)
            if (Tag STREQUAL "Cuda")
                set(file ${dir}/${Name}_test.${TAG_EXT})
                configure_file(${Name}_test.cc ${file})
            endif ()
            list(APPEND ALGO_ALL_SOURCES_${DEVICE} ${file})
        endforeach ()
        if (Tag STREQUAL "Cuda")
            carbin_cc_test(
                    NAME algorithm_all_test_${lc_tag}
                    SOURCES
                    ${ALGO_ALL_SOURCES_${DEVICE}}
                    ${TEST_MAIN_SRC}
                    COPTS ${CARBIN_CXX_OPTIONS}
                    CUOPTS ${CARBIN_CU_OPTIONS}
                    DEPS
                    ${CUDA_CUDA_LIBRARY}
                    flare::flare
            )
        else ()
            carbin_cc_test(
                    NAME algorithm_all_test_${lc_tag}
                    SOURCES
                    ${ALGO_ALL_SOURCES_${DEVICE}}
                    ${TEST_MAIN_SRC}
                    COPTS ${CARBIN_CXX_OPTIONS}
                    DEPS
                    ${CUDA_CUDA_LIBRARY}
                    flare::flare
            )
        endif ()
    endif ()
endforeach ()
