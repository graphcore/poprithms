include(GNUInstallDirs)

include(EnableCompilerWarnings)

# poprithms schedule files:
file(GLOB_RECURSE schedule_sources src/poprithms/schedule/*cpp)

set(sources
  ${schedule_sources}
  # other projects here
  )

add_library(poprithms-objlib OBJECT ${sources})
add_library(poprithms STATIC $<TARGET_OBJECTS:poprithms-objlib>)
add_library(poprithms-shared SHARED $<TARGET_OBJECTS:poprithms-objlib>)

target_include_directories (poprithms-objlib SYSTEM PUBLIC "include")
target_include_directories (poprithms SYSTEM PUBLIC "include")
target_include_directories (poprithms-shared SYSTEM PUBLIC "include")

target_link_libraries(poprithms 
  # These should preferably be inserted in algorithm / 
  # sub-project specific cmake files, so that 
  # users can build a sub-collection of algorithms
  # ${Boost_LIBRARIES}
  )

target_link_libraries(poprithms-shared
  # as per poprithms
  )


install(TARGETS poprithms
        EXPORT poprithms
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT poprithms)

install(DIRECTORY "include/poprithms"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT poprithms)
