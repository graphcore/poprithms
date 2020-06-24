include(GNUInstallDirs)

include(EnableCompilerWarnings)



# Currently all Boost components used are header only,
# so no "COMPONENTS foo" is needed when finding Boost.
find_package(Boost REQUIRED)
# Moreover, these lines are not needed while all boost deps are header only;
# set(Boost_USE_STATIC_LIBS ON); set(Boost_USE_STATIC_RUNTIME ON)
message(STATUS "Boost_INCLUDE_DIRS is " ${Boost_INCLUDE_DIRS})

find_package(Threads REQUIRED)

set(logging_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/logging/logging.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/logging/error.cpp
)

set(error_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/error/error.cpp
)

set(util_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/util/printiter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/util/stringutil.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/util/error.cpp
)

add_library(poprithms-util-objlib OBJECT 
  ${util_sources} 
  ${logging_sources} 
  ${error_sources}
)
# For each sub-project "foo" of poprithms a shared library called
# poprithms-foo is created and used to improve test link times.
add_library(poprithms-util SHARED  $<TARGET_OBJECTS:poprithms-util-objlib>)

set(schedule_src_dir ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/schedule)
set(anneal_source_dir ${schedule_src_dir}/anneal)
set(schedule_anneal_sources
 ${anneal_source_dir}/alloc.cpp
 ${anneal_source_dir}/allocweight.cpp
 ${anneal_source_dir}/error.cpp
 ${anneal_source_dir}/graph.cpp
 ${anneal_source_dir}/graphserialization.cpp
 ${anneal_source_dir}/logging.cpp
 ${anneal_source_dir}/op.cpp
 ${anneal_source_dir}/trackentry.cpp
 ${anneal_source_dir}/transitiveclosureoptimizations.cpp
)
set(dfs_source_dir ${schedule_src_dir}/dfs)
set(schedule_dfs_sources
  ${dfs_source_dir}/dfs.cpp
  ${dfs_source_dir}/error.cpp
)
set(supercon_source_dir ${schedule_src_dir}/supercon)
set(schedule_supercon_sources
  ${supercon_source_dir}/error.cpp
  ${supercon_source_dir}/graph.cpp
  ${supercon_source_dir}/logging.cpp
)
set(transitiveclosure_source_dir ${schedule_src_dir}/transitiveclosure)
set(schedule_transitive_closure_sources
  ${transitiveclosure_source_dir}/error.cpp
  ${transitiveclosure_source_dir}/logging.cpp
  ${transitiveclosure_source_dir}/transitiveclosure.cpp
)
add_library(poprithms-schedule-objlib OBJECT
  ${schedule_anneal_sources}
  ${schedule_dfs_sources}
  ${schedule_supercon_sources}
  ${schedule_transitive_closure_sources}
)
add_library(poprithms-schedule SHARED
  $<TARGET_OBJECTS:poprithms-schedule-objlib>
  $<TARGET_OBJECTS:poprithms-util-objlib>
)


set(outline_linear_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/outline/linear/error.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/outline/linear/graph.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/outline/linear/linearusings.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/outline/linear/logging.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/outline/linear/op.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/outline/linear/tensor.cpp
)
add_library(poprithms-outline-objlib OBJECT ${outline_linear_sources})
add_library(poprithms-outline SHARED
  $<TARGET_OBJECTS:poprithms-outline-objlib>
  $<TARGET_OBJECTS:poprithms-schedule-objlib>
  $<TARGET_OBJECTS:poprithms-util-objlib>
)


set(memory_nest_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/memory/nest/error.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/memory/nest/stripe.cpp
)
add_library(poprithms-memory-objlib OBJECT ${memory_nest_sources})
add_library(poprithms-memory SHARED
  $<TARGET_OBJECTS:poprithms-memory-objlib>
  $<TARGET_OBJECTS:poprithms-util-objlib>
)


add_library(poprithms SHARED 
  $<TARGET_OBJECTS:poprithms-schedule-objlib>
  $<TARGET_OBJECTS:poprithms-outline-objlib>
  $<TARGET_OBJECTS:poprithms-memory-objlib>
  $<TARGET_OBJECTS:poprithms-util-objlib>
)


add_library(poprithms-static STATIC
  $<TARGET_OBJECTS:poprithms-schedule-objlib>
  $<TARGET_OBJECTS:poprithms-outline-objlib>
  $<TARGET_OBJECTS:poprithms-memory-objlib>
  $<TARGET_OBJECTS:poprithms-util-objlib>
)


list(APPEND POPRITHMS_COMMON_INCLUDES ${Boost_INCLUDE_DIR})

set(POPRITHMS_COMMON_LIBRARIES ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

if (${POPRITHMS_USE_STACKTRACE})
  # Boost stacktrace requires linking with libdl
  list(APPEND POPRITHMS_COMMON_LIBRARIES ${CMAKE_DL_LIBS})
endif()

foreach(tgt IN ITEMS
  poprithms-schedule-objlib
  poprithms-schedule
  poprithms-outline-objlib
  poprithms-outline
  poprithms-memory-objlib
  poprithms-memory
  poprithms-util-objlib
  poprithms-util
  poprithms
  poprithms-static)

  target_include_directories (${tgt} SYSTEM PRIVATE ${POPRITHMS_COMMON_INCLUDES})
  target_link_libraries(${tgt} PRIVATE ${POPRITHMS_COMMON_LIBRARIES})
  target_include_directories(${tgt} PUBLIC
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

