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


# The header only components don't need to be listed after COMPONENTS. 
find_package(Boost REQUIRED COMPONENTS filesystem system)
# Moreover, these lines are not needed while all boost deps are header only;
# set(Boost_USE_STATIC_LIBS ON); set(Boost_USE_STATIC_RUNTIME ON)
message(STATUS "Boost_INCLUDE_DIRS is " ${Boost_INCLUDE_DIRS})
message(STATUS "Boost_LIBRARIES is " ${Boost_LIBRARIES})

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

# Find the libraries that correspond to the LLVM components that we wish to use
llvm_map_components_to_libnames(llvm_libs support core)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "LLVM_INCLUDE_DIRS is " ${LLVM_INCLUDE_DIRS})
message(STATUS "MLIR_INCLUDE_DIRS is " ${MLIR_INCLUDE_DIRS})
message(STATUS "LLVM_LIBRARY_DIR is " ${LLVM_LIBRARY_DIR})

find_package(Threads REQUIRED)

list(APPEND POPRITHMS_COMMON_INCLUDES ${Boost_INCLUDE_DIR} ${LLVM_INCLUDE_DIRS})

set(POPRITHMS_COMMON_LIBRARIES ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${llvm_libs})

if (${POPRITHMS_USE_STACKTRACE})
  # Boost stacktrace requires linking with libdl
  list(APPEND POPRITHMS_COMMON_LIBRARIES ${CMAKE_DL_LIBS})
endif()

add_subdirectory(poprithms)
add_subdirectory(tests)

