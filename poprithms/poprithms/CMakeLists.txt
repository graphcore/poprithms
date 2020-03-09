include(GNUInstallDirs)

include(EnableCompilerWarnings)

file(GLOB_RECURSE all_sources src/*cpp)

set(sources
  ${all_sources}
  )

find_package (Threads REQUIRED)

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_STATIC_RUNTIME ON)
find_package(Boost REQUIRED COMPONENTS graph)

option(SPD_LOGGING "Use SPD logging (instead of less tidy custom logging)" ON)
message(STATUS "SPD_LOGGING option is ${SPD_LOGGING}")
if (${SPD_LOGGING})
  # Make the error message for the case where Spdlog is not found robust
  find_package(Spdlog REQUIRED)
  message(STATUS "SPDLOG_INCLUD_DIR is " ${SPDLOG_INCLUDE_DIR})
  add_definitions(-DUSE_SPD_LOG)
endif()


add_library(poprithms-objlib OBJECT ${sources})
add_library(poprithms STATIC $<TARGET_OBJECTS:poprithms-objlib>)
add_library(poprithms-shared SHARED $<TARGET_OBJECTS:poprithms-objlib>)
message(STATUS "Boost_INLCUDE_DIRS is " ${Boost_INCLUDE_DIRS})

list(APPEND POPRITHMS_COMMON_INCLUDES ${Boost_INCLUDE_DIR} ${SPDLOG_INCLUDE_DIR} "include" )

target_include_directories (poprithms-objlib SYSTEM PUBLIC ${POPRITHMS_COMMON_INCLUDES})
target_include_directories (poprithms SYSTEM PUBLIC ${POPRITHMS_COMMON_INCLUDES})

set(POPRITHMS_COMMON_LIBRARIES
  ${Boost_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
)

target_link_libraries(poprithms 
  ${POPRITHMS_COMMON_LIBRARIES}
)

target_link_libraries(poprithms-shared
  ${POPRITHMS_COMMON_LIBRARIES}
)

install(TARGETS poprithms
        EXPORT poprithms
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT poprithms)

install(DIRECTORY "include/poprithms"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT poprithms)


