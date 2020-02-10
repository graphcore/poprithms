# Code which is common to tests, but
# does not belong in the main library.

file(GLOB_RECURSE test-util-sources *cpp *hpp)
add_library(poprithms-test-util SHARED ${test-util-sources})
target_include_directories (poprithms-test-util 
    SYSTEM PUBLIC
     "include"
      $<TARGET_PROPERTY:poprithms,INCLUDE_DIRECTORIES>)
target_link_libraries(poprithms-test-util poprithms-shared)
