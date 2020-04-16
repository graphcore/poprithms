function(add_regression_executable name source)
  add_executable(${name} ${source})
  target_include_directories(${name}
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      $<TARGET_PROPERTY:poprithms,INCLUDE_DIRECTORIES>
      $<TARGET_PROPERTY:poprithms-test-util,INCLUDE_DIRECTORIES>)
  target_link_libraries(${name} poprithms-shared poprithms-test-util)
endfunction()

add_regression_executable(regression regression.cpp)
add_regression_executable(fromserial fromserial.cpp)
