function(add_basic_test name source)
  add_executable(${name} ${source})

  # llvm_libs is set in the root CMakeLists.txt file
  target_link_libraries(${name} poprithms-util 
          ${POPRITHMS_COMMON_LIBRARIES} ${poprithms_llvm_libs})
  
  target_include_directories (${name} SYSTEM PRIVATE 
           ${POPRITHMS_COMMON_INCLUDES} ${LLVM_INCLUDE_DIRS})
  set(executable_name ${name})
  add_test(NAME ${name}
           COMMAND ${name} ${ARGN}
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
endfunction()


add_basic_test(poprithms_llvm_mlir_basics llvm_mlir_basics.cpp)

