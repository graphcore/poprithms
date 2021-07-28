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
target_link_libraries(schedule-test-util poprithms-schedule)

set(memory-nest-test-util-sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/testutil/memory/nest/randomsett.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/testutil/memory/nest/randomregion.cpp
)
add_library(memory-nest-test-util SHARED ${memory-nest-test-util-sources})
target_include_directories(memory-nest-test-util PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(memory-nest-test-util poprithms-memory-nest)



set(common-schedulable-test-util-sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/testutil/common/schedulable/schedulablegraph.cpp
)
add_library(common-schedulable-test-util SHARED ${common-schedulable-test-util-sources})
target_include_directories(common-schedulable-test-util PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(common-schedulable-test-util poprithms-common-schedulable)
