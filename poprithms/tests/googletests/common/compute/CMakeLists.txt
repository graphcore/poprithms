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
    common-compute-test-util
    GTest::gtest
    GTest::gmock
    GTest::gtest_main)
  gtest_discover_tests(${name})
  add_test(${name} ${name})
  target_include_directories(${name} PRIVATE ${POPRITHMS_MOCKS_DIR}/include)

endfunction()

add_gtest(common_compute_basic_functionality_0 basic_functionality_0.cpp)


