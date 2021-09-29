function(add_basic_test name source)
  add_executable(${name} ${source})
  target_link_libraries(${name} poprithms-util ${POPRITHMS_COMMON_LIBRARIES})
  target_include_directories (${name} SYSTEM PRIVATE ${POPRITHMS_COMMON_INCLUDES})
  set(executable_name ${name})
  add_test(NAME ${name}
           COMMAND ${name} ${ARGN}
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
endfunction()


add_basic_test(poprithms_boost_basics boost_basics.cpp)

