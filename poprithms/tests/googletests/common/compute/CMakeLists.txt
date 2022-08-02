find_package(GTest REQUIRED CONFIG)

include(GoogleTest)
function(add_gtest name source)
  add_executable(${name} ${source})
  target_link_libraries(
    ${name}
    # link to the whole of poprithms (common::compute will probably use most of 
    # poprithms, so no advantage to linking to the subset used like done in other 
    # projects):
    poprithms
    GTest::gtest
    GTest::gmock
    GTest::gtest_main)
  # Note that this is not needed as the test is added with ctest
  #   gtest_discover_tests(${name})
  add_test(${name} ${name})
  target_include_directories(${name} PRIVATE ${POPRITHMS_MOCKS_DIR}/include)

endfunction()

add_gtest(common_compute_basic_functionality_0 basic_functionality_0.cpp)
add_gtest(common_compute_call_0 call_0.cpp)
add_gtest(common_compute_simexecutable_0 simexecutable_0.cpp)
add_gtest(common_compute_interdevicecopy_0 interdevicecopy_0.cpp)
add_gtest(common_compute_train_misc_0 train_misc_0.cpp)


