cmake_minimum_required(VERSION 3.10)

# see https://stackoverflow.com/questions/50468620
option(ENABLE_TESTS "Enable tests" ON)
if (${ENABLE_TESTS})
    enable_testing()
endif()

add_subdirectory(poprithms)
