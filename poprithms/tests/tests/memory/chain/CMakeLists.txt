include(EnableCompilerWarnings)

function(add_memory_chain_test_executable name source)
  add_executable(${name} ${source})
  target_link_libraries(${name} poprithms-memory-chain)
  target_include_directories (${name} PRIVATE ${poprithms_src_dir}/poprithms)
endfunction()

function(add_memory_chain_test name source)
  add_memory_chain_test_executable(${name} ${source})
  set(executable_name ${name})
  add_test(NAME ${name}
           COMMAND ${name} ${ARGN}
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
endfunction()

add_custom_target(memory_chain_tests ${chain_tests})

add_memory_chain_test(memory_chain_basic_0  
                                   basic_0.cpp)

add_memory_chain_test(memory_chain_mergecontiguoussametype_0  
                                   mergecontiguoussametype_0.cpp)

add_memory_chain_test(memory_chain_removeidentity_0  
                                   removeidentity_0.cpp)

add_memory_chain_test(memory_chain_reverse_chain_0  
                                   reverse_chain_0.cpp)

add_memory_chain_test(memory_chain_canonicalize_0  
                                   canonicalize_0.cpp)

