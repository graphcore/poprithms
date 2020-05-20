add_executable(test_static_linkage test_static_linkage.cpp)

target_include_directories(test_static_linkage
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:poprithms,INCLUDE_DIRECTORIES>)

target_link_libraries(test_static_linkage poprithms-static)

#Using lld (otool -L) should not show any poprithms .so (.dylib) files.
#This is contrast to the core tests in sister directories, which should.

add_test(NAME test_static_linkage
         COMMAND test_static_linkage
         WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
