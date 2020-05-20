# Code which is common to tests, but
# does not belong in the main library.

set(test-util-sources
  ${CMAKE_CURRENT_SOURCE_DIR}/include/testutil/schedule/anneal/annealcommandlineoptions.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/testutil/schedule/anneal/bifurcate_generator.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/testutil/schedule/anneal/branch_doubling_generator.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/testutil/schedule/anneal/diamond_generator.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/testutil/schedule/anneal/grid_generator.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/testutil/schedule/anneal/randomgraph.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/testutil/schedule/anneal/recompute_generator.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/testutil/schedule/commandlineoptions.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/testutil/schedule/transitiveclosure/randomedges.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/testutil/schedule/transitiveclosure/transitiveclosurecommandlineoptions.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/testutil/schedule/anneal/annealcommandlineoptions.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/testutil/schedule/anneal/bifurcate_generator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/testutil/schedule/anneal/branch_doubling_generator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/testutil/schedule/anneal/diamond_generator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/testutil/schedule/anneal/grid_generator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/testutil/schedule/anneal/randomgraph.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/testutil/schedule/anneal/recompute_generator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/testutil/schedule/commandlineoptions.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/testutil/schedule/transitiveclosure/randomedges.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/testutil/schedule/transitiveclosure/transitiveclosurecommandlineoptions.cpp
)

add_library(poprithms-test-util SHARED ${test-util-sources})
target_include_directories(poprithms-test-util PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(poprithms-test-util poprithms)
