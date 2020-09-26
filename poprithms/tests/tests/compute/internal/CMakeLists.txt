include(EnableCompilerWarnings)

function(add_compute_host_internal_test_executable name source)
  add_executable(${name} ${source})
  set(LIB_DEPS  poprithms-compute-host poprithms-util )
  target_link_libraries(${name} PRIVATE ${LIB_DEPS})

  # set the path to internal files being tested:
  target_include_directories (${name} PRIVATE ${internal_compute_include_dir})
endfunction()

function(add_compute_host_internal_test name source)
  add_compute_host_internal_test_executable(${name} ${source})
  set(executable_name ${name})
  add_test(NAME ${name}
           COMMAND ${name} ${ARGN}
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
endfunction()

add_compute_host_internal_test(compute_host_internal_base_operators_0 base_operators_0.cpp)
