# All C++ code in project poprithms will be compiled as C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

add_subdirectory(poprithms)
add_subdirectory(tests)
