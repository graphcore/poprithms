function(add_memory_test_executable name source)
  add_executable(${name} ${source})
  target_include_directories(${name}
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      $<TARGET_PROPERTY:poprithms-memory,INCLUDE_DIRECTORIES>)
  target_link_libraries(${name} poprithms-memory)
endfunction()

function(add_memory_test name source)
  add_memory_test_executable(${name} ${source})
  set(executable_name ${name})
  add_test(NAME ${name}
           COMMAND ${name} ${ARGN}
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
endfunction()

add_subdirectory(nest)

