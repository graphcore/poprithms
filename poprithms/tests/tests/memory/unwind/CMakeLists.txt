include(EnableCompilerWarnings)

function(add_memory_unwind_test_executable name source)
  add_executable(${name} ${source})
  target_link_libraries(${name} poprithms-memory-unwind memory-unwind-test-util)
  target_include_directories (${name} PRIVATE ${poprithms_src_dir}/poprithms)
endfunction()

function(add_memory_unwind_test name source)
  add_memory_unwind_test_executable(${name} ${source})
  set(executable_name ${name})
  add_test(NAME ${name}
           COMMAND ${name} ${ARGN}
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
endfunction()

add_custom_target(memory_unwind_tests ${unwind_tests})

add_subdirectory(greedy0)
add_subdirectory(mappings)
add_subdirectory(score)


