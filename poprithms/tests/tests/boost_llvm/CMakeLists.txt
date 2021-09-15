find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(POPRITHMS_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/poprithms/poprithms/include)

add_test(NAME "boost_llvm_free_interface_test"
  COMMAND 
  ${Python3_EXECUTABLE} 
  ${CMAKE_CURRENT_SOURCE_DIR}/boost_and_llvm_free_interface_test.py 
  ${POPRITHMS_INCLUDE_DIR})

function(add_basic_test_executable name source)
  add_executable(${name} ${source})
  target_link_libraries(${name} poprithms-util ${POPRITHMS_COMMON_LIBRARIES} ${llvm_libs})
  target_include_directories (${name} SYSTEM PRIVATE ${POPRITHMS_COMMON_INCLUDES})
endfunction()

function(add_basic_test name source)
  add_basic_test_executable(${name} ${source})
  set(executable_name ${name})
  add_test(NAME ${name}
           COMMAND ${name} ${ARGN}
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
endfunction()


add_basic_test(poprithms_llvm_and_boost_basics llvm_and_boost_basics.cpp)

