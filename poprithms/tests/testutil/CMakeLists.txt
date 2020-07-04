# Code which is common to tests, but

# does not belong in the main library.

set (schedule_test_util_source_dir
  ${CMAKE_CURRENT_SOURCE_DIR}/src/testutil/schedule)

set(schedule-test-util-sources
  ${schedule_test_util_source_dir}/anneal/annealcommandlineoptions.cpp
  ${schedule_test_util_source_dir}/anneal/bifurcate_generator.cpp
  ${schedule_test_util_source_dir}/anneal/branch_doubling_generator.cpp
  ${schedule_test_util_source_dir}/anneal/diamond_generator.cpp
  ${schedule_test_util_source_dir}/anneal/grid_generator.cpp
  ${schedule_test_util_source_dir}/anneal/randomgraph.cpp
  ${schedule_test_util_source_dir}/anneal/recompute_generator.cpp
  ${schedule_test_util_source_dir}/commandlineoptions.cpp
  ${schedule_test_util_source_dir}/transitiveclosure/randomedges.cpp
  ${schedule_test_util_source_dir}/transitiveclosure/transitiveclosurecommandlineoptions.cpp
)

add_library(schedule-test-util SHARED ${schedule-test-util-sources})
target_include_directories(schedule-test-util PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(schedule-test-util poprithms-schedule)

set(memory-test-util-sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/testutil/memory/nest/randomsett.cpp
)
add_library(memory-test-util SHARED ${memory-test-util-sources})
target_include_directories(memory-test-util PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(memory-test-util poprithms-memory)

