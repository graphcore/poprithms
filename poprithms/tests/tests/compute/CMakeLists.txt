function(add_compute_host_test_executable name source)
  add_executable(${name} ${source})
  set(LIB_DEPS  poprithms-compute-host poprithms-util )
  if (${POPRITHMS_USE_STACKTRACE})
    list(APPEND LIB_DEPS ${CMAKE_DL_LIBS})
  endif()
  target_link_libraries(${name} ${LIB_DEPS})
  # set the path to internal files being tested:
  target_include_directories (${name} PRIVATE ${poprithms_src_dir}/poprithms)
endfunction()

function(add_compute_host_test name source)
  add_compute_host_test_executable(${name} ${source})
  set(executable_name ${name})
  add_test(NAME ${name}
           COMMAND ${name} ${ARGN}
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
endfunction()

add_subdirectory(viewchange)
add_subdirectory(internal)
add_subdirectory(tensor)

