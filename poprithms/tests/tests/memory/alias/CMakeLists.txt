include(EnableCompilerWarnings)

function(add_memory_alias_test_executable name source)
  add_executable(${name} ${source})
  target_link_libraries(${name} poprithms-memory-alias)
endfunction()

function(add_memory_alias_test name source)
  add_memory_alias_test_executable(${name} ${source})
  set(executable_name ${name})
  add_test(NAME ${name}
           COMMAND ${name} ${ARGN}
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
endfunction()

add_subdirectory(graph)
add_subdirectory(origins)
add_subdirectory(tensor)

