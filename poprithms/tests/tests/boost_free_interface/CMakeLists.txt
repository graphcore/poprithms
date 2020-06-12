find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(POPRITHMS_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/poprithms/poprithms/include)

add_test(NAME "boost_free_interface_test"
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/boost_free_interface_test.py ${POPRITHMS_INCLUDE_DIR})
