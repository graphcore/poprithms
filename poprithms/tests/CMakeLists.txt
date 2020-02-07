#
# some common code used by tests, but which 
# does not belong in the main library:
#
file(GLOB_RECURSE test-util-sources testutil/*cpp testutil/*hpp)
add_library(poprithms-test-util SHARED ${test-util-sources})
target_include_directories (poprithms-test-util 
    SYSTEM PUBLIC
     "testutil/include"
      $<TARGET_PROPERTY:poprithms,INCLUDE_DIRECTORIES>)
target_link_libraries(poprithms-test-util poprithms-shared)

function(add_test_executable name source)
  add_executable(${name} ${source})
  target_include_directories(${name}
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      $<TARGET_PROPERTY:poprithms,INCLUDE_DIRECTORIES>
      $<TARGET_PROPERTY:poprithms-test-util,INCLUDE_DIRECTORIES>)
  target_link_libraries(${name} poprithms-shared poprithms-test-util)
endfunction()

function(add_poprithms_unit_test name source)
  add_test_executable(${name} ${source})
  set(executable_name ${name})
  add_test(NAME ${name}
           COMMAND ${name} ${ARGN}
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
endfunction()

add_subdirectory(schedule)

