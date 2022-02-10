function(add_program_test_executable name source)
  add_executable(${name} ${source})
  target_link_libraries(${name} poprithms-program)
endfunction()

function(add_program_test name source)
  add_program_test_executable(${name} ${source})
  set(executable_name ${name})
  add_test(NAME ${name}
           COMMAND ${name} ${ARGN}
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
endfunction()

add_program_test(program_callstack_copy_classes_0 copy_classes_0.cpp)
