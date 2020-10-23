function(add_memory_nest_test_executable name source)
  add_executable(${name} ${source})
  target_link_libraries(${name} poprithms-memory-nest memory-nest-test-util)
endfunction()

function(add_memory_nest_test name source)
  add_memory_nest_test_executable(${name} ${source})
  set(executable_name ${name})
  add_test(NAME ${name}
           COMMAND ${name} ${ARGN}
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
endfunction()

add_custom_target(memory_nest_tests ${nest_tests})

add_subdirectory(stripe)
add_subdirectory(sett)
add_subdirectory(region)
add_subdirectory(disjointregions)

