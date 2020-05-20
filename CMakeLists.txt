cmake_minimum_required(VERSION 3.10)
project(poprithms)

# see https://stackoverflow.com/questions/50468620
option(ENABLE_TESTS "Enable tests" ON)
if (${ENABLE_TESTS})
    enable_testing()
endif()

# Add an explicit exported symbols file (a linker flag)
if(APPLE)
  string(APPEND CMAKE_SHARED_LINKER_FLAGS
         " -Wl,-exported_symbols_list,"
         "${CMAKE_CURRENT_SOURCE_DIR}/exported_symbols_osx.lds")
else()
  string(APPEND CMAKE_SHARED_LINKER_FLAGS
         " -Wl,--version-script,"
         "${CMAKE_CURRENT_SOURCE_DIR}/exported_symbols_linux.lds")
endif()

option(POPRITHMS_USE_STACKTRACE "Enable boost stacktrace reports in error messages" ON)
if (${POPRITHMS_USE_STACKTRACE})
    # Building with Boost Stacktrace using the default header only implementation
    # Note this is only supported for any compiler on POSIX or MinGW.
    message(STATUS "Building poprithms with Boost Stacktrace")
    add_definitions(-DPOPRITHMS_USE_STACKTRACE -DBOOST_STACKTRACE_GNU_SOURCE_NOT_REQUIRED)
endif()

add_subdirectory(poprithms)
