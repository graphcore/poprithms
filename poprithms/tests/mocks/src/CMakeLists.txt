# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
add_library(poprithms_mock SHARED
  memory/alias/mocktensor.cpp
  memory/alias/mockgraph.cpp
  util/mockpermutation.cpp
)

target_link_libraries(poprithms_mock
  PRIVATE
    poprithms
    GTest::gtest
    GTest::gmock
    Boost::boost
)

set(MOCK_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/poprithms/tests/mocks/include)

target_include_directories(poprithms_mock
  PRIVATE
    ${MOCK_INCLUDE_DIR}
  PUBLIC
    $<TARGET_PROPERTY:poprithms,INTERFACE_INCLUDE_DIRECTORIES>
)

install(TARGETS poprithms_mock
        COMPONENT poprithms_mock
        EXPORT poprithms_mock
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT poprithms_mock
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/poprithms_mock
        FILE poprithms_mock-targets.cmake
        COMPONENT poprithms_mock)

configure_file(poprithms_mock-config.cmake.in poprithms_mock-config.cmake @ONLY)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/poprithms_mock-config.cmake
        COMPONENT poprithms_mock
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/poprithms_mock)

install(DIRECTORY ${MOCK_INCLUDE_DIR}/mock
        COMPONENT poprithms_mock
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
