include(GNUInstallDirs)

include(EnableCompilerWarnings)

file(GLOB_RECURSE all_sources src/*cpp)

set(sources
  ${all_sources}
  )

find_package (Threads REQUIRED)

# currently all boost components used are header only, 
# and so no "COMPONENTS foo" is needed when finding boost. 
find_package(Boost REQUIRED)

# Moreover, these lines are not needed while all boost deps are header only:
# set(Boost_USE_STATIC_LIBS ON)
# set(Boost_USE_STATIC_RUNTIME ON)

add_library(poprithms-objlib OBJECT ${sources})
add_library(poprithms STATIC $<TARGET_OBJECTS:poprithms-objlib>)
add_library(poprithms-shared SHARED $<TARGET_OBJECTS:poprithms-objlib>)
message(STATUS "Boost_INCLUDE_DIRS is " ${Boost_INCLUDE_DIRS})

list(APPEND POPRITHMS_COMMON_INCLUDES ${Boost_INCLUDE_DIR} "include" )

target_include_directories (poprithms-objlib SYSTEM PUBLIC ${POPRITHMS_COMMON_INCLUDES})
target_include_directories (poprithms SYSTEM PUBLIC ${POPRITHMS_COMMON_INCLUDES})

set(POPRITHMS_COMMON_LIBRARIES
  ${Boost_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
)

target_link_libraries(poprithms 
  ${POPRITHMS_COMMON_LIBRARIES}
)

target_link_libraries(poprithms-shared
  ${POPRITHMS_COMMON_LIBRARIES}
)

install(TARGETS poprithms
        EXPORT poprithms
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT poprithms)

install(DIRECTORY "include/poprithms"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT poprithms)


