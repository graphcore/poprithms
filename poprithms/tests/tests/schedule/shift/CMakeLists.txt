function(add_shift_test_executable name source)
  add_executable(${name} ${source})
  target_link_libraries(${name} poprithms-schedule-shift schedule-test-util ${POPRITHMS_COMMON_LIBRARIES})
  target_include_directories (${name} PRIVATE ${poprithms_src_dir}/poprithms)
  target_include_directories (${name} SYSTEM PRIVATE ${POPRITHMS_COMMON_INCLUDES})

endfunction()

function(add_shift_test name source)
  add_shift_test_executable(${name} ${source})
  set(executable_name ${name})
  add_test(NAME ${name}
           COMMAND ${name} ${ARGN}
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
endfunction()

function(add_shift_test_from_params name exe)
  add_test(NAME ${name}
           COMMAND ${exe} ${ARGN}
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
endfunction()

add_shift_test(schedule_shift_bifurcate bifurcate.cpp D 5)
add_shift_test(schedule_shift_grid grid.cpp N 5)
add_shift_test(schedule_shift_test_random test_random.cpp)

add_shift_test(schedule_shift_kahn_decider_0 kahn_decider_0.cpp)

add_shift_test_executable(recompute recompute.cpp)
add_shift_test_from_params(schedule_shift_recompute_sqrt 
                                    recompute N 50 type sqrt)
add_shift_test_from_params(schedule_shift_recompute_log 
                                    recompute N 50 type log)

add_shift_test_executable(branch_doubling branch_doubling.cpp)
add_shift_test_from_params(schedule_shift_branch_doubling_pos
                                    branch_doubling nBranches 4 offset 1)
add_shift_test_from_params(schedule_shift_branch_doubling_neg
                                    branch_doubling nBranches 4 offset -1)

add_shift_test(schedule_shift_random random.cpp N 100 E 5 D 10 graphSeed 1011)
add_shift_test(schedule_shift_attractions attractions.cpp)
add_shift_test(schedule_shift_tight_pairs tight_pairs.cpp)
add_shift_test(schedule_shift_op_0 op_0.cpp)
add_shift_test(schedule_shift_empty_0 empty_0.cpp)
add_shift_test(schedule_shift_graph_basics_0 graph_basics_0.cpp)
add_shift_test(schedule_shift_graph_comparisons_0 graph_comparisons_0.cpp)
add_shift_test(schedule_shift_graph_hash graph_hash.cpp)
add_shift_test(schedule_shift_inputs inputs.cpp)
add_shift_test(schedule_shift_allocweight_0 allocweight_0.cpp)
add_shift_test(schedule_shift_links_0 links_0.cpp)
add_shift_test(schedule_shift_links_1 links_1.cpp)
add_shift_test(schedule_shift_links_2 links_2.cpp)
add_shift_test(schedule_shift_get_merged_0 get_merged_0.cpp)
add_shift_test(schedule_shift_get_schedule get_schedule_0.cpp)
add_shift_test(schedule_shift_serialization_0 serialization_0.cpp)
add_shift_test(schedule_shift_serialization_errors serialization_errors.cpp)
add_shift_test(schedule_shift_is_schedulable schedulable.cpp)
add_shift_test(schedule_shift_is_search_limits searchlimits.cpp)
add_shift_test(schedule_shift_diamond_0 diamond_0.cpp N 19)
add_shift_test(schedule_shift_bin_constraints bin_constraints.cpp)
add_shift_test(schedule_shift_bin_cycle cycle_0.cpp)

add_shift_test(schedule_shift_tco_link_close_tight_pairs_0
                                        tco_link_close_tight_pairs_0.cpp)
add_shift_test(schedule_shift_tco_link_tight_drops_0
                                        tco_link_tight_drops_0.cpp)
add_shift_test(schedule_shift_tco_link_tight_drops_1
                                        tco_link_tight_drops_1.cpp)
add_shift_test(schedule_shift_tco_constrain_weight_separated_0
                                        tco_constrain_weight_separated_0.cpp)
add_shift_test(schedule_shift_tco_constrain_weight_separated_1
                                        tco_constrain_weight_separated_1.cpp)
add_shift_test(schedule_shift_tco_constrain_weight_separated_2
                                        tco_constrain_weight_separated_2.cpp)
add_shift_test(schedule_shift_tco_constrain_weight_separated_3
                                        tco_constrain_weight_separated_3.cpp)
add_shift_test(schedule_shift_tco_constrain_weight_separated_4
                                        tco_constrain_weight_separated_4.cpp)

add_shift_test(schedule_shift_tco_constrain_parallel_chains_0
                                        tco_constrain_parallel_chains_0.cpp)

add_shift_test(schedule_shift_logging_0 logging_0.cpp)

add_shift_test(schedule_shift_filtered_schedule_0 filtered_schedule_0.cpp)
add_shift_test(schedule_shift_constraint_diff_0 constraint_diff_0.cpp)


add_shift_test(schedule_shift_solutioncache_0 solutioncache_0.cpp)

add_shift_test(schedule_shift_alloc_simplifier_0 alloc_simplifier_0.cpp)

add_shift_test(schedule_shift_alloc_partitioned_bins_0 
                                 alloc_partitioned_bins_0.cpp)

add_shift_test(schedule_shift_summary_writer_0
                                 summary_writer_0.cpp)

add_shift_test(schedule_shift_summary_writer_1
                              summary_writer_1.cpp)

configure_file(testgraph0.json.in testgraph0.json COPYONLY)
add_shift_test(schedule_shift_serialization_fromfile0 
                                 serialization_fromfile0.cpp)




