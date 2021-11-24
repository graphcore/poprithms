# Add a test which links to the poprithms shared library. 
# Most testing in poprithms is done by linking to small
# sub-project libraries, to accelerate test link times. 
function(add_full_library_test_executable name source)
  add_executable(${name} ${source})
  target_include_directories(${name}
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      $<TARGET_PROPERTY:poprithms,INCLUDE_DIRECTORIES>)
  target_link_libraries(${name} poprithms)
endfunction()

function(add_full_library_test name source)
  add_full_library_test_executable(${name} ${source})
  set(executable_name ${name})
  add_test(NAME ${name}
           COMMAND ${name} ${ARGN}
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
endfunction()

function(add_full_library_test_from_params name exe)
  add_test(NAME ${name}
           COMMAND ${exe} ${ARGN}
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
endfunction()

include(EnableCompilerWarnings)

add_subdirectory(autodiff)

add_subdirectory(schedule)

add_subdirectory(compute)

add_subdirectory(common)

add_subdirectory(memory)

add_subdirectory(outline)

add_subdirectory(util)

add_subdirectory(verify_cxx_11_interface)

add_subdirectory(dependency_free_interface)

add_subdirectory(boost)

