include(GNUInstallDirs)

include(EnableCompilerWarnings)

set(sources
  ${CMAKE_CURRENT_SOURCE_DIR}/include/poprithms/logging/error.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/poprithms/logging/logging.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/poprithms/memory/nest/error.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/poprithms/memory/nest/stripe.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/poprithms/outline/linear/error.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/poprithms/outline/linear/graph.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/poprithms/outline/linear/linearusings.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/poprithms/outline/linear/logging.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/poprithms/outline/linear/op.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/poprithms/outline/linear/tensor.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/poprithms/schedule/anneal/alloc.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/poprithms/schedule/anneal/allocweight.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/poprithms/schedule/anneal/annealusings.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/poprithms/schedule/anneal/error.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/poprithms/schedule/anneal/filteredschedule.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/poprithms/schedule/anneal/graph.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/poprithms/schedule/anneal/graphserialization.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/poprithms/schedule/anneal/logging.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/poprithms/schedule/anneal/opalloc.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/poprithms/schedule/anneal/op.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/poprithms/schedule/anneal/schedulechange.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/poprithms/schedule/anneal/shiftandcost.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/poprithms/schedule/anneal/trackentry.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/poprithms/schedule/anneal/transitiveclosureoptimizations.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/poprithms/schedule/dfs/dfs.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/poprithms/schedule/dfs/error.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/poprithms/schedule/supercon/error.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/poprithms/schedule/supercon/graph.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/poprithms/schedule/supercon/logging.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/poprithms/schedule/transitiveclosure/error.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/poprithms/schedule/transitiveclosure/logging.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/poprithms/schedule/transitiveclosure/transitiveclosure.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/poprithms/util/error.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/poprithms/util/printiter.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/poprithms/util/stringutil.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/poprithms/util/typedinteger.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/poprithms/util/unisort.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/logging/error.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/logging/logging.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/memory/nest/error.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/memory/nest/stripe.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/outline/linear/error.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/outline/linear/graph.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/outline/linear/linearusings.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/outline/linear/logging.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/outline/linear/op.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/outline/linear/tensor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/schedule/anneal/alloc.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/schedule/anneal/allocweight.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/schedule/anneal/error.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/schedule/anneal/graph.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/schedule/anneal/graphserialization.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/schedule/anneal/logging.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/schedule/anneal/op.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/schedule/anneal/trackentry.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/schedule/anneal/transitiveclosureoptimizations.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/schedule/dfs/dfs.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/schedule/dfs/error.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/schedule/supercon/error.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/schedule/supercon/graph.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/schedule/supercon/logging.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/schedule/transitiveclosure/error.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/schedule/transitiveclosure/logging.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/schedule/transitiveclosure/transitiveclosure.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/util/error.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/util/printiter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/util/stringutil.cpp
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

if (${POPRITHMS_USE_STACKTRACE})
  # Boost stacktrace requires linking with libdl
  list(APPEND POPRITHMS_COMMON_LIBRARIES ${CMAKE_DL_LIBS})
endif()

foreach(tgt IN ITEMS poprithms-objlib poprithms poprithms-static)
  target_include_directories (${tgt} SYSTEM PRIVATE ${POPRITHMS_COMMON_INCLUDES})
  target_link_libraries(${tgt} PRIVATE ${POPRITHMS_COMMON_LIBRARIES})

  target_include_directories(${tgt}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
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
