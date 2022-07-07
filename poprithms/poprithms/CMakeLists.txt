include(GNUInstallDirs)

include(EnableCompilerWarnings)
option (POPRITHMS_WERROR "C++ compiler warnings become errors" TRUE)
if (${POPRITHMS_WERROR})
  message(STATUS "Appending -Werror compiler flag, because POPRITHMS_WERROR=TRUE")
  add_compile_options(-Werror)
endif ()



set(src_dir ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms)

set(logging_sources
  ${src_dir}/logging/timepartitionlogger.cpp
  ${src_dir}/logging/logging.cpp
  ${src_dir}/logging/error.cpp
)

set(error_sources
  ${src_dir}/error/error.cpp
)

set(util_sources
  ${src_dir}/util/permutation.cpp
  ${src_dir}/util/printiter.cpp
  ${src_dir}/util/stringutil.cpp
  ${src_dir}/util/interval.cpp
  ${src_dir}/util/error.cpp
)

set(ndarray_sources
  ${src_dir}/ndarray/accessors.cpp
  ${src_dir}/ndarray/broadcastsetter.cpp
  ${src_dir}/ndarray/dtype.cpp
  ${src_dir}/ndarray/error.cpp
  ${src_dir}/ndarray/shape.cpp
  ${src_dir}/ndarray/tensorinfo.cpp
)

add_library(poprithms-util-objlib OBJECT 
  ${util_sources}
  ${ndarray_sources}
  ${logging_sources}
  ${error_sources}
)

# For each sub-project "foo" of poprithms an object library and (most often)
# a shared library called poprithms-foo are created. Linking only the necessary
# components to tests improves test link times.
add_library(poprithms-util SHARED  $<TARGET_OBJECTS:poprithms-util-objlib>)

set(common_multiout_sources
  ${src_dir}/common/multiout/consumptionid.cpp
  ${src_dir}/common/multiout/optraversal.cpp
  ${src_dir}/common/multiout/error.cpp
  ${src_dir}/common/multiout/graph.cpp
  ${src_dir}/common/multiout/logging.cpp
  ${src_dir}/common/multiout/op.cpp
  ${src_dir}/common/multiout/opid.cpp
  ${src_dir}/common/multiout/tensorid.cpp
  ${src_dir}/common/multiout/removalevent.cpp
  ${src_dir}/common/multiout/optionaltensorid.cpp
  ${src_dir}/common/multiout/fwdedgemap.cpp
)
add_library(poprithms-common-multiout-objlib OBJECT ${common_multiout_sources})
add_library(poprithms-common-multiout SHARED
  $<TARGET_OBJECTS:poprithms-common-multiout-objlib>
  $<TARGET_OBJECTS:poprithms-util-objlib>
)


set(common_schedulable_sources
  ${src_dir}/common/schedulable/op.cpp
  ${src_dir}/common/schedulable/bidiredgemap.cpp
  ${src_dir}/common/schedulable/error.cpp
  ${src_dir}/common/schedulable/graph.cpp
  ${src_dir}/common/schedulable/subgraphid.cpp
)
add_library(poprithms-common-schedulable-objlib OBJECT ${common_schedulable_sources})
add_library(poprithms-common-schedulable SHARED
  $<TARGET_OBJECTS:poprithms-schedule-other-objlib>
  $<TARGET_OBJECTS:poprithms-common-multiout-objlib>
  $<TARGET_OBJECTS:poprithms-common-schedulable-objlib>
  $<TARGET_OBJECTS:poprithms-util-objlib>
)

set(common_compute_sources
  ${src_dir}/common/compute/device.cpp
  ${src_dir}/common/compute/devicetype.cpp
  ${src_dir}/common/compute/error.cpp
  ${src_dir}/common/compute/graph.cpp
  ${src_dir}/common/compute/iexecutable.cpp
  ${src_dir}/common/compute/initialvalues.cpp
  ${src_dir}/common/compute/memoryaliasmapper.cpp
  ${src_dir}/common/compute/op.cpp
  ${src_dir}/common/compute/opverifier.cpp
  ${src_dir}/common/compute/scheduler.cpp
  ${src_dir}/common/compute/simexecutable.cpp
  ${src_dir}/common/compute/simtensormap.cpp
  ${src_dir}/common/compute/subgraph.cpp
  ${src_dir}/common/compute/tensor.cpp

  ${src_dir}/common/compute/ops/binaryelementwise.cpp
  ${src_dir}/common/compute/ops/dynamic.cpp
  ${src_dir}/common/compute/ops/encode.cpp
  ${src_dir}/common/compute/ops/init.cpp
  ${src_dir}/common/compute/ops/interdevicecopy.cpp
  ${src_dir}/common/compute/ops/matmul.cpp
  ${src_dir}/common/compute/ops/ops.cpp
  ${src_dir}/common/compute/ops/reduce.cpp
  ${src_dir}/common/compute/ops/reffrom.cpp
  ${src_dir}/common/compute/ops/unaryelementwise.cpp
  ${src_dir}/common/compute/ops/viewchange.cpp
  ${src_dir}/common/compute/ops/withcallees.cpp
  ${src_dir}/common/compute/ops/withoutcallees.cpp

  ${src_dir}/common/compute/autodiff/coregraphmutator.cpp
  ${src_dir}/common/compute/autodiff/guidegraphinfo.cpp

  ${src_dir}/common/compute/prune/prune.cpp
)

add_library(poprithms-common-compute-objlib OBJECT ${common_compute_sources})
add_library(poprithms-common-compute SHARED
  $<TARGET_OBJECTS:poprithms-autodiff-automatic-objlib>
  $<TARGET_OBJECTS:poprithms-autodiff-base-objlib>
  $<TARGET_OBJECTS:poprithms-compute-host-objlib>
  $<TARGET_OBJECTS:poprithms-autodiff-base-objlib>
  $<TARGET_OBJECTS:poprithms-autodiff-automatic-objlib>
  $<TARGET_OBJECTS:poprithms-schedule-other-objlib>
  $<TARGET_OBJECTS:poprithms-memory-alias-objlib>
  $<TARGET_OBJECTS:poprithms-memory-chain-objlib>
  $<TARGET_OBJECTS:poprithms-memory-nest-objlib>
  $<TARGET_OBJECTS:poprithms-memory-unwind-objlib>
  $<TARGET_OBJECTS:poprithms-common-multiout-objlib>
  $<TARGET_OBJECTS:poprithms-common-schedulable-objlib>
  $<TARGET_OBJECTS:poprithms-common-compute-objlib>
  $<TARGET_OBJECTS:poprithms-program-objlib>
  $<TARGET_OBJECTS:poprithms-util-objlib>
)


set(schedule_src_dir ${src_dir}/schedule)

set(shift_source_dir ${src_dir}/schedule/shift)
set(schedule_shift_sources
 ${shift_source_dir}/alloc.cpp
 ${shift_source_dir}/allocweight.cpp
 ${shift_source_dir}/allocsimplifier.cpp
 ${shift_source_dir}/error.cpp
 ${shift_source_dir}/fromcache.cpp
 ${shift_source_dir}/graph.cpp
 ${shift_source_dir}/greedykahn.cpp
 ${shift_source_dir}/graphserialization.cpp
 ${shift_source_dir}/kahndecider.cpp
 ${shift_source_dir}/logging.cpp
 ${shift_source_dir}/op.cpp
 ${shift_source_dir}/schedulecache.cpp
 ${shift_source_dir}/scheduledgraph.cpp
 ${shift_source_dir}/settings.cpp
 ${shift_source_dir}/shiftandcost.cpp
 ${shift_source_dir}/summarywriter.cpp
 ${shift_source_dir}/transitiveclosureconstrainer.cpp
 ${shift_source_dir}/transitiveclosureoptimizer.cpp
 ${shift_source_dir}/transitiveclosureoptimizations.cpp
 ${shift_source_dir}/updatefromfirstfinal.cpp
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
set(connectedcomponents_source_dir ${schedule_src_dir}/connectedcomponents)
set(schedule_connectedcomponents_sources
  ${connectedcomponents_source_dir}/connectedcomponents.cpp
  ${connectedcomponents_source_dir}/error.cpp
)
set(schedule_vanilla_source_dir ${schedule_src_dir}/vanilla)
set(schedule_vanilla_sources
  ${schedule_vanilla_source_dir}/error.cpp
  ${schedule_vanilla_source_dir}/vanilla.cpp
  ${schedule_vanilla_source_dir}/pathcount.cpp
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
  ${transitiveclosure_source_dir}/partitionedtransitiveclosure.cpp
)

# we divide all of the scheduling projects into "shift" and "other". This is 
# because, currently, "shift" is much larger than the others. Should this 
# balance change in the future, we can further divide "other" into smaller 
# components. 
add_library(poprithms-schedule-shift-objlib OBJECT
  ${schedule_shift_sources}
)

add_library(poprithms-schedule-other-objlib OBJECT
  ${schedule_dfs_sources}
  ${schedule_scc_sources}
  ${schedule_connectedcomponents_sources}
  ${schedule_vanilla_sources}
  ${schedule_supercon_sources}
  ${schedule_transitive_closure_sources}
)

add_library(poprithms-schedule-shift SHARED
  $<TARGET_OBJECTS:poprithms-schedule-shift-objlib>
  $<TARGET_OBJECTS:poprithms-schedule-other-objlib>
  $<TARGET_OBJECTS:poprithms-util-objlib>
)

add_library(poprithms-schedule-other SHARED
  $<TARGET_OBJECTS:poprithms-schedule-other-objlib>
  $<TARGET_OBJECTS:poprithms-util-objlib>
)



set(outline_linear_sources
  ${src_dir}/outline/linear/error.cpp
  ${src_dir}/outline/linear/graph.cpp
  ${src_dir}/outline/linear/linearusings.cpp
  ${src_dir}/outline/linear/logging.cpp
  ${src_dir}/outline/linear/op.cpp
  ${src_dir}/outline/linear/tensor.cpp
)
add_library(poprithms-outline-objlib OBJECT ${outline_linear_sources})
add_library(poprithms-outline SHARED
  $<TARGET_OBJECTS:poprithms-outline-objlib>
  $<TARGET_OBJECTS:poprithms-schedule-other-objlib>
  $<TARGET_OBJECTS:poprithms-util-objlib>
)

set(autodiff_src_dir ${CMAKE_CURRENT_SOURCE_DIR}/src/poprithms/autodiff)
set(autodiff_base_sources
  ${autodiff_src_dir}/core/autodiff.cpp
  ${autodiff_src_dir}/core/gradinfo.cpp
  ${autodiff_src_dir}/core/summary.cpp
  ${autodiff_src_dir}/guide/graphinfo.cpp
  ${autodiff_src_dir}/guide/guide.cpp
  ${autodiff_src_dir}/guide/traversals.cpp
  ${autodiff_src_dir}/guide/objective.cpp
  ${autodiff_src_dir}/autodiff/error.cpp
)


add_library(poprithms-autodiff-base-objlib OBJECT ${autodiff_base_sources})
add_library(poprithms-autodiff-base SHARED
  $<TARGET_OBJECTS:poprithms-autodiff-base-objlib>
  $<TARGET_OBJECTS:poprithms-util-objlib>
  $<TARGET_OBJECTS:poprithms-common-multiout-objlib>)

# The top-level autodiff requires more components (scheduling, etc) of poprithms 
# and so is bundled separatedly to the rest of the autodiff project. 
set(autodiff_automatic_sources
  ${autodiff_src_dir}/automatic/call.cpp
  ${autodiff_src_dir}/automatic/gradinfos.cpp
  ${autodiff_src_dir}/automatic/iautomatic.cpp
)
add_library(poprithms-autodiff-automatic-objlib OBJECT ${autodiff_automatic_sources})
add_library(poprithms-autodiff-automatic SHARED
  $<TARGET_OBJECTS:poprithms-autodiff-automatic-objlib>
  $<TARGET_OBJECTS:poprithms-autodiff-base-objlib>
  $<TARGET_OBJECTS:poprithms-util-objlib>
  $<TARGET_OBJECTS:poprithms-common-multiout-objlib>
  $<TARGET_OBJECTS:poprithms-schedule-other-objlib>
  $<TARGET_OBJECTS:poprithms-common-schedulable-objlib>
  $<TARGET_OBJECTS:poprithms-program-objlib> 
)



set(compute_src_dir ${src_dir}/compute)
set(compute_host_sources
  ${compute_src_dir}/host/basedata.cpp
  ${compute_src_dir}/host/error.cpp
  ${compute_src_dir}/host/gridpointhelper.cpp
  ${compute_src_dir}/host/tensormapper.cpp
  ${compute_src_dir}/host/ieeehalf.cpp
  ${compute_src_dir}/host/numpyformatter.cpp
  ${compute_src_dir}/host/origindata.cpp
  ${compute_src_dir}/host/regionutil.cpp
  ${compute_src_dir}/host/serializer.cpp
  ${compute_src_dir}/host/tensor.cpp
  ${compute_src_dir}/host/viewchange.cpp
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
  $<TARGET_OBJECTS:poprithms-memory-nest-objlib>
  $<TARGET_OBJECTS:poprithms-util-objlib>
)

# we can decompose program into constituent level-2 projects if/when it gets large. 
set (program_src_dir ${src_dir}/program)
set (program_sources
   ${program_src_dir}/distributed/codelocation.cpp
   ${program_src_dir}/distributed/error.cpp
   ${program_src_dir}/distributed/helper.cpp
   ${program_src_dir}/distributed/program.cpp
   ${program_src_dir}/callstack/callstack.cpp
   ${program_src_dir}/callstack/querier.cpp
   ${program_src_dir}/callstack/error.cpp
   ${program_src_dir}/callstack/copyin.cpp
   ${program_src_dir}/callstack/copyout.cpp
   ${program_src_dir}/prune/error.cpp
   ${program_src_dir}/prune/prune.cpp
)

add_library(poprithms-program-objlib OBJECT ${program_sources})
add_library(poprithms-program SHARED
  $<TARGET_OBJECTS:poprithms-common-multiout-objlib>
  $<TARGET_OBJECTS:poprithms-common-schedulable-objlib>
  $<TARGET_OBJECTS:poprithms-program-objlib>
  $<TARGET_OBJECTS:poprithms-schedule-other-objlib>
  $<TARGET_OBJECTS:poprithms-util-objlib>
)


set(memory_src_dir ${src_dir}/memory)
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

set(memory_chain_sources
	${memory_src_dir}/chain/chain.cpp
	${memory_src_dir}/chain/error.cpp
	${memory_src_dir}/chain/logging.cpp
	${memory_src_dir}/chain/op.cpp
	${memory_src_dir}/chain/settutil.cpp
	${memory_src_dir}/chain/type.cpp
 )

set(memory_inplace_sources
   ${memory_src_dir}/inplace/crosslink.cpp
   ${memory_src_dir}/inplace/tensor.cpp
   ${memory_src_dir}/inplace/error.cpp
   ${memory_src_dir}/inplace/graph.cpp
   ${memory_src_dir}/inplace/logging.cpp
   ${memory_src_dir}/inplace/op.cpp
   ${memory_src_dir}/inplace/ops.cpp
   ${memory_src_dir}/inplace/proposal.cpp
   ${memory_src_dir}/inplace/result.cpp
   ${memory_src_dir}/inplace/tensormap.cpp
)

set(memory_unwind_sources
  ${memory_src_dir}/unwind/hosttensorhelper.cpp
  ${memory_src_dir}/unwind/logging.cpp
  ${memory_src_dir}/unwind/graph.cpp
  ${memory_src_dir}/unwind/error.cpp
  ${memory_src_dir}/unwind/path.cpp
  ${memory_src_dir}/unwind/lower.cpp
  ${memory_src_dir}/unwind/ops.cpp
  ${memory_src_dir}/unwind/op.cpp
  ${memory_src_dir}/unwind/solution.cpp
  ${memory_src_dir}/unwind/scheduledsolution.cpp
  ${memory_src_dir}/unwind/valuedtensorid.cpp
  ${memory_src_dir}/unwind/sumlike.cpp
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
  $<TARGET_OBJECTS:poprithms-common-multiout-objlib>
  $<TARGET_OBJECTS:poprithms-schedule-other-objlib>
  $<TARGET_OBJECTS:poprithms-util-objlib>
)

add_library(poprithms-memory-chain-objlib OBJECT ${memory_chain_sources})
add_library(poprithms-memory-chain SHARED
  $<TARGET_OBJECTS:poprithms-compute-host-objlib>
  $<TARGET_OBJECTS:poprithms-memory-chain-objlib>
  $<TARGET_OBJECTS:poprithms-memory-nest-objlib>
  $<TARGET_OBJECTS:poprithms-util-objlib>
)

add_library(poprithms-memory-inplace-objlib OBJECT ${memory_inplace_sources})
add_library(poprithms-memory-inplace SHARED
  $<TARGET_OBJECTS:poprithms-common-multiout-objlib>
  $<TARGET_OBJECTS:poprithms-memory-inplace-objlib>
  $<TARGET_OBJECTS:poprithms-memory-alias-objlib>
  $<TARGET_OBJECTS:poprithms-memory-nest-objlib>
  $<TARGET_OBJECTS:poprithms-schedule-other-objlib>
  $<TARGET_OBJECTS:poprithms-util-objlib>
)

add_library(poprithms-memory-unwind-objlib OBJECT ${memory_unwind_sources})
add_library(poprithms-memory-unwind SHARED
  $<TARGET_OBJECTS:poprithms-common-multiout-objlib>
  $<TARGET_OBJECTS:poprithms-common-schedulable-objlib>
  $<TARGET_OBJECTS:poprithms-schedule-other-objlib>
  $<TARGET_OBJECTS:poprithms-compute-host-objlib>
  $<TARGET_OBJECTS:poprithms-memory-unwind-objlib>
  $<TARGET_OBJECTS:poprithms-memory-chain-objlib>
  $<TARGET_OBJECTS:poprithms-memory-nest-objlib>
  $<TARGET_OBJECTS:poprithms-util-objlib>
)

add_library(poprithms SHARED 
  $<TARGET_OBJECTS:poprithms-autodiff-base-objlib>
  $<TARGET_OBJECTS:poprithms-autodiff-automatic-objlib>
  $<TARGET_OBJECTS:poprithms-common-multiout-objlib>
  $<TARGET_OBJECTS:poprithms-common-schedulable-objlib>
  $<TARGET_OBJECTS:poprithms-common-compute-objlib>
  $<TARGET_OBJECTS:poprithms-compute-host-objlib>
  $<TARGET_OBJECTS:poprithms-schedule-shift-objlib>
  $<TARGET_OBJECTS:poprithms-schedule-other-objlib>
  $<TARGET_OBJECTS:poprithms-outline-objlib>
  $<TARGET_OBJECTS:poprithms-memory-nest-objlib>
  $<TARGET_OBJECTS:poprithms-memory-alias-objlib>
  $<TARGET_OBJECTS:poprithms-memory-chain-objlib>
  $<TARGET_OBJECTS:poprithms-memory-inplace-objlib>
  $<TARGET_OBJECTS:poprithms-memory-unwind-objlib>
  $<TARGET_OBJECTS:poprithms-program-objlib>
  $<TARGET_OBJECTS:poprithms-util-objlib>
)


foreach(tgt IN ITEMS
  poprithms-common-multiout-objlib
  poprithms-common-multiout
  poprithms-common-schedulable-objlib
  poprithms-common-schedulable
  poprithms-common-compute-objlib
  poprithms-common-compute
  poprithms-compute-host-objlib
  poprithms-compute-host
  poprithms-schedule-shift-objlib
  poprithms-schedule-shift
  poprithms-schedule-other-objlib
  poprithms-schedule-other
  poprithms-outline-objlib
  poprithms-outline
  poprithms-autodiff-base-objlib
  poprithms-autodiff-base
  poprithms-autodiff-automatic-objlib
  poprithms-autodiff-automatic
  poprithms-memory-nest-objlib
  poprithms-memory-nest
  poprithms-memory-alias-objlib
  poprithms-memory-alias
  poprithms-memory-chain-objlib
  poprithms-memory-chain
  poprithms-memory-inplace-objlib
  poprithms-memory-inplace
  poprithms-memory-unwind-objlib
  poprithms-memory-unwind
  poprithms-program-objlib
  poprithms-program
  poprithms-util
  poprithms-util-objlib
  poprithms)

  target_include_directories (${tgt} SYSTEM PRIVATE ${POPRITHMS_COMMON_INCLUDES})
  target_link_libraries(${tgt} PRIVATE ${POPRITHMS_COMMON_LIBRARIES})
  target_include_directories(${tgt} 
    PRIVATE 
      ${poprithms_src_dir}/poprithms
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
