# Code which is common to tests, but
# does not belong in the main library.

set (schedule_test_util_source_dir
  ${CMAKE_CURRENT_SOURCE_DIR}/src/testutil/schedule)

set(schedule-test-util-sources
  ${schedule_test_util_source_dir}/base/randomdag.cpp
  ${schedule_test_util_source_dir}/shift/shiftcommandlineoptions.cpp
  ${schedule_test_util_source_dir}/shift/bifurcate_generator.cpp
  ${schedule_test_util_source_dir}/shift/branch_doubling_generator.cpp
  ${schedule_test_util_source_dir}/shift/diamond_generator.cpp
  ${schedule_test_util_source_dir}/shift/grid_generator.cpp
  ${schedule_test_util_source_dir}/shift/randomgraph.cpp
  ${schedule_test_util_source_dir}/shift/recompute_generator.cpp
  ${schedule_test_util_source_dir}/commandlineoptions.cpp
  ${schedule_test_util_source_dir}/transitiveclosure/randomedges.cpp
  ${schedule_test_util_source_dir}/transitiveclosure/transitiveclosurecommandlineoptions.cpp
)

add_library(schedule-test-util SHARED ${schedule-test-util-sources})
target_include_directories(schedule-test-util PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(schedule-test-util poprithms-schedule-shift poprithms-schedule-other)

set(memory-nest-test-util-sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/testutil/memory/nest/randomsett.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/testutil/memory/nest/randomregion.cpp
)
add_library(memory-nest-test-util SHARED ${memory-nest-test-util-sources})
target_include_directories(memory-nest-test-util PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(memory-nest-test-util poprithms-memory-nest)

set(memory-unwind-test-util-sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/testutil/memory/unwind/op.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/testutil/memory/unwind/graph.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/testutil/memory/unwind/fullstate.cpp
)
add_library(memory-unwind-test-util SHARED ${memory-unwind-test-util-sources})
target_include_directories(memory-unwind-test-util PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(memory-unwind-test-util poprithms-memory-unwind poprithms-common-schedulable)


set(common-schedulable-test-util-sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/testutil/common/schedulable/graph.cpp
)
add_library(common-schedulable-test-util SHARED ${common-schedulable-test-util-sources})
target_include_directories(common-schedulable-test-util PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(common-schedulable-test-util poprithms-common-schedulable)

set(autodiff-test-util-sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/testutil/autodiff/testop.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/testutil/autodiff/testgraphinfo.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/testutil/autodiff/testgraphmutator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/testutil/autodiff/finitedifference.cpp
)
add_library(autodiff-test-util SHARED ${autodiff-test-util-sources})
target_include_directories(autodiff-test-util PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(autodiff-test-util poprithms-autodiff-base 
                                         poprithms-autodiff-automatic 
                                         poprithms-common-multiout
                                         poprithms-compute-host)

set(program-test-util-sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/testutil/program/callstack/graph.cpp
)
add_library(program-test-util SHARED ${program-test-util-sources})
target_include_directories(program-test-util PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(program-test-util poprithms-program poprithms-common-multiout)


