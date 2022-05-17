include(EnableCompilerWarnings)

function(add_autodiff_test_executable name source)
  add_executable(${name} ${source})
  target_link_libraries(${name} 
       poprithms-autodiff-base 
       poprithms-autodiff-automatic 
       autodiff-test-util 
       poprithms-common-multiout)
endfunction()

function(add_autodiff_test name source)
  add_autodiff_test_executable(${name} ${source})
  set(executable_name ${name})
  add_test(NAME ${name}
           COMMAND ${name} ${ARGN}
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
endfunction()

add_custom_target(autodiff_tests ${autodiff_tests})

add_autodiff_test(autodiff_guide_0  
                           guide_0.cpp)

add_autodiff_test(autodiff_core_0
                           core_0.cpp)

add_autodiff_test(autodiff_automatic_finite_difference_0
                           finite_difference_0.cpp)
