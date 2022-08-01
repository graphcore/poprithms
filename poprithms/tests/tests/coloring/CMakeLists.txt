function(add_coloring_test_executable name source)
  add_executable(${name} ${source})
  target_include_directories(${name}
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR})
      target_link_libraries(${name} poprithms-util)
endfunction()

function(add_coloring_test name source)
  add_coloring_test_executable(${name} ${source})
  set(executable_name ${name})
  add_test(NAME ${name}
           COMMAND ${name} ${ARGN}
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
endfunction()

add_coloring_test(coloring_propagation_0 coloring_0.cpp)
