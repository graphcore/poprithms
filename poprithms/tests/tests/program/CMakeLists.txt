include(EnableCompilerWarnings)

function(add_program_test_executable name source)
  add_executable(${name} ${source})
  target_link_libraries(${name} poprithms-program program-test-util)
endfunction()

function(add_program_test name source)
  add_program_test_executable(${name} ${source})
  set(executable_name ${name})
  add_test(NAME ${name}
           COMMAND ${name} ${ARGN}
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
endfunction()

add_subdirectory(callstack)
add_subdirectory(distributed)
add_subdirectory(prune)

