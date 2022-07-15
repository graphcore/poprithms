function(add_common_test_executable name source)
  add_executable(${name} ${source})
  target_link_libraries(${name} poprithms-common-compute)
endfunction()

function(add_common_test name source)
  add_common_test_executable(${name} ${source})
  set(executable_name ${name})
  add_test(NAME ${name}
           COMMAND ${name} ${ARGN}
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
endfunction()


add_common_test(poprithms_common_compute_basics_0 
                                         basics_0.cpp)

add_common_test(poprithms_common_compute_differentiator_0
                                         differentiator_0.cpp)

add_common_test(poprithms_common_compute_error_checking_0
                                         error_checking_0.cpp)

add_common_test(poprithms_common_compute_miscellaneous_sim_train_0
                                         miscellaneous_sim_train_0.cpp)

add_common_test(poprithms_common_compute_logging_0
                                         logging_0.cpp)

add_common_test(poprithms_common_compute_prune_call_0 
                                         prune_call_0.cpp)

add_common_test(poprithms_common_compute_transform_sin_example_0 
                                         transform_sin_example_0.cpp)

add_common_test(poprithms_common_compute_train_through_call_0 
                                         train_through_call_0.cpp)

