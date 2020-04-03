
function(add_poprithms_cxx_11_unit_test name source)
  add_test_executable(${name} ${source})
  set(executable_name ${name})
  add_test(NAME ${name}
           COMMAND ${name} ${ARGN}
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
  set_property(TARGET ${name} PROPERTY CXX_STANDARD 11)
endfunction()

# Add a test is compiled with the c++11 standard to check that the poprithms
# interface is c++11 compatible. If not, the build will fail when compiling
# the test.

# This test will include all headers in poprithms. Generate a file to include
# all headers for the test to use:
execute_process (
  COMMAND bash -c "echo \"$(for header in $(find ${PROJECT_SOURCE_DIR}/poprithms/poprithms/include -name '*hpp' | sort);
                              do printf \"#include <$header>\n\";
                            done)\""
  OUTPUT_VARIABLE allHeaders
)
# Write to build dir so as not to pollute the poprithms repo during build
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/all_headers.hpp" "${allHeaders}")
# So the test can include 'all_headers.hpp'
target_include_directories(poprithms PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

add_poprithms_cxx_11_unit_test(verify_cxx_11_interface verify_cxx_11_interface.cpp)
