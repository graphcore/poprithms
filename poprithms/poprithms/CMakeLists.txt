include(GNUInstallDirs)

include(EnableCompilerWarnings)

# poprithms schedule files:
file(GLOB_RECURSE schedule_sources src/poprithms/schedule/*cpp)

set(sources
  ${schedule_sources}
  # other projects here
  )

add_library(poprithms SHARED ${sources})

target_include_directories (poprithms SYSTEM PUBLIC 
  "include"
  )

target_link_libraries(poprithms
  # currently no requirements. Should these be algorithm specific? TODO(jn)
  # ${Boost_LIBRARIES}
  )


install(TARGETS poprithms
        EXPORT poprithms
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT poprithms)

install(DIRECTORY "include/poprithms"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT poprithms)
