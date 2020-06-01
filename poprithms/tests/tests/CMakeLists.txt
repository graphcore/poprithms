function(add_test_executable name source)
  add_executable(${name} ${source})
  target_include_directories(${name}
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      $<TARGET_PROPERTY:poprithms,INCLUDE_DIRECTORIES>
      $<TARGET_PROPERTY:poprithms-test-util,INCLUDE_DIRECTORIES>)
  target_link_libraries(${name} poprithms poprithms-test-util)
endfunction()

function(add_poprithms_unit_test name source)
  add_test_executable(${name} ${source})
  set(executable_name ${name})
  add_test(NAME ${name}
           COMMAND ${name} ${ARGN}
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
endfunction()


function(add_poprithms_unit_test_from_params name exe)
  add_test(NAME ${name}
           COMMAND ${exe} ${ARGN}
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
endfunction()

add_subdirectory(schedule)

add_subdirectory(memory)

add_subdirectory(outline)

add_subdirectory(logging)

add_subdirectory(util)

add_subdirectory(verify_cxx_11_interface)

#Test(s) where poprithms is linked statically
add_subdirectory(staticlib)

#Test that tests can use boost test framework
# Currently disabled due to failing external build, see T19253 to re-enable
# add_subdirectory(boostsmoke)
