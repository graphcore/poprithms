function(add_util_test_executable name source)
  add_executable(${name} ${source})
  target_include_directories(${name}
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      $<TARGET_PROPERTY:poprithms-util,INCLUDE_DIRECTORIES>
      )
    target_link_libraries(${name} poprithms-compute-host ${Boost_LIBRARIES})
endfunction()

function(add_util_test name source)
  add_util_test_executable(${name} ${source})
  set(executable_name ${name})
  add_test(NAME ${name}
           COMMAND ${name} ${ARGN}
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
endfunction()

function(add_util_test_from_params name exe)
  add_test(NAME ${name}
           COMMAND ${exe} ${ARGN}
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
endfunction()

add_subdirectory(misc)
add_subdirectory(shape)
add_subdirectory(dtype)
add_subdirectory(logging)
add_subdirectory(interval)
add_subdirectory(ndarray)
add_subdirectory(serialization)
