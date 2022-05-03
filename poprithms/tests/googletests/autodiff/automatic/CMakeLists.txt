
find_package(GTest REQUIRED CONFIG)

include(GoogleTest)
function(add_gtest name source)
  add_executable(${name} ${source})
  target_link_libraries(
    ${name}
    poprithms-autodiff-automatic
    GTest::gtest
    GTest::gmock
    GTest::gtest_main)
  gtest_discover_tests(${name})
  add_test(${name} ${name})
  target_include_directories(${name} PRIVATE ${POPRITHMS_MOCKS_DIR}/include)

endfunction()

add_gtest(autodiff_automatic_gradinfos_0 gradinfos_0.cpp)


