include(GNUInstallDirs)

include(EnableCompilerWarnings)
option (POPRITHMS_WERROR "C++ compiler warnings become errors" TRUE)
if (${POPRITHMS_WERROR})
  message(STATUS "Appending -Werror compiler flag, because POPRITHMS_WERROR=TRUE")
  add_compile_options(-Werror)
endif ()


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
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/util/permutation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/util/printiter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/util/stringutil.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/util/error.cpp
)

set(ndarray_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/ndarray/shape.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/ndarray/dtype.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/ndarray/error.cpp
)


add_library(poprithms-util-objlib OBJECT 
  ${util_sources}
  ${ndarray_sources}
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
 ${anneal_source_dir}/transitiveclosureoptimizations.cpp
)
set(dfs_source_dir ${schedule_src_dir}/dfs)
set(schedule_dfs_sources
  ${dfs_source_dir}/dfs.cpp
  ${dfs_source_dir}/error.cpp
)
set(scc_source_dir ${schedule_src_dir}/scc)
set(schedule_scc_sources
  ${scc_source_dir}/scc.cpp
  ${scc_source_dir}/error.cpp
)
set(schedule_vanilla_source_dir ${schedule_src_dir}/vanilla)
set(schedule_vanilla_sources
  ${schedule_vanilla_source_dir}/vanilla.cpp
  ${schedule_vanilla_source_dir}/error.cpp
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
  ${schedule_scc_sources}
  ${schedule_vanilla_sources}
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


set(compute_src_dir ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/compute)
set(compute_host_sources
  ${compute_src_dir}/host/error.cpp
  ${compute_src_dir}/host/tensor.cpp
  ${compute_src_dir}/host/viewchange.cpp
  ${compute_src_dir}/host/numpyformatter.cpp
  ${compute_src_dir}/host/gridpointhelper.cpp
  ${compute_src_dir}/host/ieeehalf.cpp
  ${compute_src_dir}/host/origindata.cpp
  ${compute_src_dir}/host/basedata.cpp
  # instantiating templates on separate translation units
  # to parallelize their compilations.
  ${compute_src_dir}/host/types/float32float64.cpp
  ${compute_src_dir}/host/types/float16.cpp
  ${compute_src_dir}/host/types/int64uint64.cpp
  ${compute_src_dir}/host/types/int32uint32.cpp
  ${compute_src_dir}/host/types/int16uint16.cpp
  ${compute_src_dir}/host/types/int8uint8.cpp
  ${compute_src_dir}/host/types/bool.cpp
)
add_library(poprithms-compute-host-objlib OBJECT ${compute_host_sources})
add_library(poprithms-compute-host SHARED
  $<TARGET_OBJECTS:poprithms-compute-host-objlib>
  $<TARGET_OBJECTS:poprithms-util-objlib>
)



set(memory_src_dir ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/memory)
set(memory_nest_sources
  ${memory_src_dir}/nest/error.cpp
  ${memory_src_dir}/nest/logging.cpp
  ${memory_src_dir}/nest/region.cpp
  ${memory_src_dir}/nest/stripe.cpp
  ${memory_src_dir}/nest/sett.cpp
)

set(memory_alias_sources
	${memory_src_dir}/alias/error.cpp
	${memory_src_dir}/alias/graph.cpp
	${memory_src_dir}/alias/logging.cpp
	${memory_src_dir}/alias/node.cpp
	${memory_src_dir}/alias/nodes.cpp
  ${memory_src_dir}/alias/origins.cpp
  ${memory_src_dir}/alias/tensor.cpp
)

set(memory_inplace_sources
       ${memory_src_dir}/inplace/consumer.cpp
       ${memory_src_dir}/inplace/crossalias.cpp
       ${memory_src_dir}/inplace/tensor.cpp
       ${memory_src_dir}/inplace/error.cpp
       ${memory_src_dir}/inplace/graph.cpp
       ${memory_src_dir}/inplace/logging.cpp
       ${memory_src_dir}/inplace/op.cpp
       ${memory_src_dir}/inplace/ops.cpp
       ${memory_src_dir}/inplace/proposal.cpp
       ${memory_src_dir}/inplace/result.cpp
       ${memory_src_dir}/inplace/tensorid.cpp
       ${memory_src_dir}/inplace/tensormap.cpp
)



add_library(poprithms-memory-nest-objlib OBJECT ${memory_nest_sources})
add_library(poprithms-memory-nest SHARED
  $<TARGET_OBJECTS:poprithms-memory-nest-objlib>
  $<TARGET_OBJECTS:poprithms-util-objlib>
)

add_library(poprithms-memory-alias-objlib OBJECT ${memory_alias_sources})
add_library(poprithms-memory-alias SHARED
  $<TARGET_OBJECTS:poprithms-memory-alias-objlib>
  $<TARGET_OBJECTS:poprithms-memory-nest-objlib>
  $<TARGET_OBJECTS:poprithms-util-objlib>
)

add_library(poprithms-memory-inplace-objlib OBJECT ${memory_inplace_sources})
add_library(poprithms-memory-inplace SHARED
  $<TARGET_OBJECTS:poprithms-memory-inplace-objlib>
  $<TARGET_OBJECTS:poprithms-memory-alias-objlib>
  $<TARGET_OBJECTS:poprithms-memory-nest-objlib>
  $<TARGET_OBJECTS:poprithms-schedule-objlib>
  $<TARGET_OBJECTS:poprithms-util-objlib>
)


add_library(poprithms SHARED 
  $<TARGET_OBJECTS:poprithms-compute-host-objlib>
  $<TARGET_OBJECTS:poprithms-schedule-objlib>
  $<TARGET_OBJECTS:poprithms-outline-objlib>
  $<TARGET_OBJECTS:poprithms-memory-nest-objlib>
  $<TARGET_OBJECTS:poprithms-memory-alias-objlib>
  $<TARGET_OBJECTS:poprithms-memory-inplace-objlib>
  $<TARGET_OBJECTS:poprithms-util-objlib>
)

list(APPEND POPRITHMS_COMMON_INCLUDES ${Boost_INCLUDE_DIR})

set(POPRITHMS_COMMON_LIBRARIES ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

if (${POPRITHMS_USE_STACKTRACE})
  # Boost stacktrace requires linking with libdl
  list(APPEND POPRITHMS_COMMON_LIBRARIES ${CMAKE_DL_LIBS})
endif()

foreach(tgt IN ITEMS
  poprithms-compute-host-objlib
  poprithms-compute-host
  poprithms-schedule-objlib
  poprithms-schedule
  poprithms-outline-objlib
  poprithms-outline
  poprithms-memory-nest-objlib
  poprithms-memory-nest
  poprithms-memory-alias-objlib
  poprithms-memory-alias
  poprithms-memory-inplace-objlib
  poprithms-memory-inplace
  poprithms-util-objlib
  poprithms-util
  poprithms)

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
