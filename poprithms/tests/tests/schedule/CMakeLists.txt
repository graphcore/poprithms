function(add_schedule_test_executable name source)
  add_executable(${name} ${source})
  target_link_libraries(${name} poprithms-schedule schedule-test-util)
endfunction()

function(add_schedule_test name source)
  add_schedule_test_executable(${name} ${source})
  set(executable_name ${name})
  add_test(NAME ${name}
           COMMAND ${name} ${ARGN}
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
endfunction()

function(add_schedule_test_from_params name exe)
  add_test(NAME ${name}
           COMMAND ${exe} ${ARGN}
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
endfunction()


add_subdirectory(shift)
add_subdirectory(transitiveclosure)
add_subdirectory(supercon)
add_subdirectory(dfs)
add_subdirectory(scc)
add_subdirectory(connectedcomponents)
add_subdirectory(vanilla)
