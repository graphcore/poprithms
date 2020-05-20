include(CMakeFindDependencyMacro)

get_filename_component(poprithms_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)

find_dependency(Threads REQUIRED)

if (NOT TARGET poprithms-static)
  include("${poprithms_CMAKE_DIR}/poprithms-static-targets.cmake")
endif()
