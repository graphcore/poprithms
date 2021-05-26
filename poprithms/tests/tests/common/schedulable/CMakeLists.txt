function(add_common_test_executable name source)
  add_executable(${name} ${source})
  target_link_libraries(${name} poprithms-common-schedulable common-schedulable-test-util)
endfunction()

function(add_common_test name source)
  add_common_test_executable(${name} ${source})
  set(executable_name ${name})
  add_test(NAME ${name}
           COMMAND ${name} ${ARGN}
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
endfunction()


add_common_test(util_common_schedulable_0 schedulable_0.cpp)
