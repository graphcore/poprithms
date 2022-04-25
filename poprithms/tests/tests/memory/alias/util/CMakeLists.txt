# We have new cmake functions here, as this is the 
# only part of the alias project which needs scheduling. 

function(add_memory_alias_util_test_executable name source)
  add_executable(${name} ${source})
  target_link_libraries(${name} poprithms-memory-alias poprithms-schedule-other)
endfunction()

function(add_memory_alias_util_test name source)
  add_memory_alias_util_test_executable(${name} ${source})
  set(executable_name ${name})
  add_test(NAME ${name}
           COMMAND ${name} ${ARGN}
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
endfunction()

add_memory_alias_util_test(memory_alias_util_jitter_0 jitter_0.cpp)
