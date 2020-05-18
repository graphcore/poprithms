### Overview

Algorithms used by frameworks. Algorithm specific information is in the notes directory

### Usage

The easiest way to use poprithms in CMake is via `find_package`, specifying on the `CMAKE_PREFIX_PATH` the location of the poprithms install.

```
find_package(poprithms CONFIG REQUIRED)
target_link_libraries(my_library PRIVATE poprithms)
```

Alternatively, for the static poprithms library

```
find_package(poprithms-static CONFIG REQUIRED)
target_link_libraries(my_library PRIVATE poprithms-static)
```

### FAQ

#### CMake can't find poprithms

An error like the following is caused when `poprithms-config.cmake` is not on the defined set of search paths.

```
CMake Error at CMakeLists.txt:X (find_package):
  By not providing "Findpoprithms.cmake" in CMAKE_MODULE_PATH this project
  has asked CMake to find a package configuration file provided by
  "poprithms", but CMake did not find one.

  Could not find a package configuration file provided by "poprithms" with
  any of the following names:

    poprithmsConfig.cmake
    poprithms-config.cmake

  Add the installation prefix of "poprithms" to CMAKE_PREFIX_PATH or set
  "poprithms_DIR" to a directory containing one of the above files.  If
  "poprithms" provides a separate development package or SDK, be sure it has
  been installed
```

To fix this specify `poprithms_DIR` (or `poprithms-static_DIR` for the static library) as a environment or CMake variable when configuring CMake to the location of `poprithms-config.cmake`.

`$ poprithms_DIR=XXX/install/poprithms/lib/cmake/ cmake ..`

where `XXX` is the path to the poprithms install.

See https://cmake.org/cmake/help/latest/command/find_package.html for more information.
