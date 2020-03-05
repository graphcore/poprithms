include(GNUInstallDirs)

include(EnableCompilerWarnings)

file(GLOB_RECURSE all_sources src/*cpp)

set(sources
  ${all_sources}
  )

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_STATIC_RUNTIME ON)
find_package(Boost REQUIRED COMPONENTS graph)


add_library(poprithms-objlib OBJECT ${sources})
add_library(poprithms STATIC $<TARGET_OBJECTS:poprithms-objlib>)
add_library(poprithms-shared SHARED $<TARGET_OBJECTS:poprithms-objlib>)

message(STATUS "Boost include dirs  are " ${Boost_INCLUDE_DIRS})


target_include_directories (poprithms-objlib SYSTEM PUBLIC "include" ${Boost_INCLUDE_DIRS})
target_include_directories (poprithms SYSTEM PUBLIC "include" ${Boost_INCLUDE_DIRS})

set(POPRITHMS_COMMON_LIBRARIES
  ${Boost_LIBRARIES}
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


