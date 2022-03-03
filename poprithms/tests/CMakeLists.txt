add_subdirectory(testutil)

add_subdirectory(tests)

add_subdirectory(regression)

if(${POPRITHMS_BUILD_MOCKS})
  add_subdirectory(mocks)
endif()

