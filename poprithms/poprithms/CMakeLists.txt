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
add_library(poprithms SHARED $<TARGET_OBJECTS:poprithms-objlib>)
add_library(poprithms-static STATIC $<TARGET_OBJECTS:poprithms-objlib>)
message(STATUS "Boost_INCLUDE_DIRS is " ${Boost_INCLUDE_DIRS})

list(APPEND POPRITHMS_COMMON_INCLUDES ${Boost_INCLUDE_DIR})

set(POPRITHMS_COMMON_LIBRARIES
  ${Boost_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
)

foreach(tgt IN ITEMS poprithms-objlib poprithms poprithms-static)
  target_include_directories (${tgt} SYSTEM PRIVATE ${POPRITHMS_COMMON_INCLUDES})
  target_link_libraries(${tgt} PRIVATE ${POPRITHMS_COMMON_LIBRARIES})

  target_include_directories(${tgt}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${CMAKE_INSTALL_INCLUDEDIR}>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
endforeach()

install(TARGETS poprithms
        EXPORT poprithmsTargets
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT poprithms)

install(EXPORT poprithmsTargets
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/poprithms
        FILE poprithms-targets.cmake
        COMPONENT poprithms)

install(DIRECTORY "include/poprithms"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT poprithms)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/poprithms-config.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/poprithms
        COMPONENT poprithms)


install(TARGETS poprithms-static
        EXPORT poprithmsStaticTargets
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT poprithms-static)

install(EXPORT poprithmsStaticTargets
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/poprithms-static
        FILE poprithms-static-targets.cmake
        COMPONENT poprithms-static)

install(DIRECTORY "include/poprithms"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT poprithms-static)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/poprithms-static-config.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/poprithms-static
        COMPONENT poprithms-static)
