# All C++ code in project poprithms will be compiled as C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Compile with fpic on to support linking into a shared library.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Generate compile_commands.json file for IDE integration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Colorful ninja! Idea taken from
# @alasher: colored-c-compiler-output-with-ninja-clang-gcc-10bfe7f2b949
option (FORCE_COLORED "Produce ANSI-colored output (GNU/Clang only)." TRUE)
if (${FORCE_COLORED})
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
       add_compile_options (-fdiagnostics-color=always)
     elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang" OR
             "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" )
       add_compile_options (-fcolor-diagnostics)
    endif ()
endif ()


# Where to find Findxxxx.cmake files
# PROJECT_SOURCE_DIR is the directory with the most recent project() call, 
# which in this case is this directory 
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/poprithms/cmake/Modules)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/poprithms/cmake)

# Tests path
set(INSTALL_TESTS ${CMAKE_INSTALL_PREFIX}/tests)

set(poprithms_src_dir ${PROJECT_SOURCE_DIR}/poprithms/poprithms/src)

# Provide defaults for Boost_USE_STATIC* based on BUILD_SHARED_LIBS, to make
# views that are fully static (poplarview) or fully dynamic
# (colossus_tools_view) work out of the box. Views using a mix of static and
# dynamic linking (host_runtime_view) must explicitly pass Boost_USE_STATIC* to
# match how boost was built.
if(NOT BUILD_SHARED_LIBS)
  if(NOT DEFINED Boost_USE_STATIC_LIBS)
    set(Boost_USE_STATIC_LIBS ON)
  endif()
  if(NOT DEFINED Boost_USE_STATIC_RUNTIME)
    set(Boost_USE_STATIC_RUNTIME ON)
  endif()
endif()

# The header only components don't need to be listed after COMPONENTS. 
find_package(Boost REQUIRED COMPONENTS filesystem system serialization)
message(STATUS "Boost_INCLUDE_DIRS is " ${Boost_INCLUDE_DIRS})
message(STATUS "Boost_LIBRARIES is " ${Boost_LIBRARIES})

list(APPEND POPRITHMS_COMMON_INCLUDES ${Boost_INCLUDE_DIR})
set(POPRITHMS_COMMON_LIBRARIES ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

find_package(Threads REQUIRED)


if (${POPRITHMS_USE_STACKTRACE})
  # Boost stacktrace requires linking with libdl
  list(APPEND POPRITHMS_COMMON_LIBRARIES ${CMAKE_DL_LIBS})
endif()

add_subdirectory(poprithms)

# The cmake BUILD_TESTING option is automatically created when 
# CTest is included, and defaults to TRUE.
if (BUILD_TESTING)
  add_subdirectory(tests)
endif()

if(${POPRITHMS_BUILD_MOCKS})
  set(POPRITHMS_MOCKS_DIR ${PROJECT_SOURCE_DIR}/poprithms/tests/mocks)
  add_subdirectory(tests/mocks)
  if (BUILD_TESTING)
     add_subdirectory(tests/googletests)
  endif()
endif()

