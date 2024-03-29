include(EnableCompilerWarnings)

function(add_memory_inplace_test_executable name source)
  add_executable(${name} ${source})
  target_link_libraries(${name} poprithms-memory-inplace)
  target_include_directories (${name} PRIVATE ${poprithms_src_dir}/poprithms)
endfunction()

function(add_memory_inplace_test name source)
  add_memory_inplace_test_executable(${name} ${source})
  set(executable_name ${name})
  add_test(NAME ${name}
           COMMAND ${name} ${ARGN}
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
endfunction()

add_custom_target(memory_inplace_tests ${inplace_tests})

add_memory_inplace_test(memory_inplace_basic_0  
                                       basic_0.cpp)

add_memory_inplace_test(memory_inplace_staggered_inplacing_0  
                                       staggered_inplacing_0.cpp)

add_memory_inplace_test(memory_inplace_graph_0  
                                       graph_0.cpp)

add_memory_inplace_test(memory_inplace_graphs_comparison_0  
                                       graphs_comparison_0.cpp)

add_memory_inplace_test(memory_inplace_constant_0  
                                       constant_0.cpp)

add_memory_inplace_test(memory_inplace_constraints_0  
                                       constraints_0.cpp)

add_memory_inplace_test(memory_inplace_aliased_0  
                                       aliased_0.cpp)

add_memory_inplace_test(memory_inplace_concat_0  
                                       concat_0.cpp)

add_memory_inplace_test(memory_inplace_settsample_0  
                                       settsample_0.cpp)

add_memory_inplace_test(memory_inplace_demo_0  
                                       demo_0.cpp)

add_memory_inplace_test(memory_inplace_reverse_reshape_0  
                                       reverse_reshape_0.cpp)

add_memory_inplace_test(memory_inplace_dimshuffle_noalias_0  
                                       dimshuffle_noalias_0.cpp)

add_memory_inplace_test(memory_inplace_binary_0  
                                       binary_0.cpp)

add_memory_inplace_test(memory_inplace_pad_0  
                                       pad_0.cpp)

add_memory_inplace_test(memory_inplace_ambiguity_0  
                                       ambiguity_0.cpp)

add_memory_inplace_test(memory_inplace_crosslink_0  
                                       crosslink_0.cpp)

add_memory_inplace_test(memory_inplace_traversal_0  
                                       traversal_0.cpp)

add_memory_inplace_test(memory_inplace_allowmultigatealias_0  
                                       allowmultigatealias_0.cpp)

add_memory_inplace_test(memory_inplace_contains_0  
                                       contains_0.cpp)
