include(EnableCompilerWarnings)

function(add_common_test_executable name source)
  add_executable(${name} ${source})
  target_link_libraries(${name} poprithms-common-multiout)
endfunction()

function(add_common_test name source)
  add_common_test_executable(${name} ${source})
  set(executable_name ${name})
  add_test(NAME ${name}
           COMMAND ${name} ${ARGN}
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
endfunction()

add_subdirectory(multiout)

