find_package(Boost REQUIRED COMPONENTS unit_test_framework)

add_test_executable(smoke_test_boost_unit_test_framework 
                    smoke_test_boost_unit_test_framework.cpp)

target_include_directories(smoke_test_boost_unit_test_framework
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:poprithms,INCLUDE_DIRECTORIES>)

target_link_libraries(smoke_test_boost_unit_test_framework # target
                      poprithms-shared                     
                      ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} 
                      )

add_test(NAME smoke_test_boost_unit_test_framework
         COMMAND smoke_test_boost_unit_test_framework
         WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
