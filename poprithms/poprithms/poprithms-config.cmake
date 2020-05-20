include(CMakeFindDependencyMacro)

get_filename_component(poprithms_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)

if (NOT TARGET poprithms)
  include("${poprithms_CMAKE_DIR}/poprithms-targets.cmake")
endif()
