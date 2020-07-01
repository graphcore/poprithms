### Overview

Poprithms is a graph algorithms library used by ML frameworks. Algorithm specific information can be found in the notes sub-directory. 

### Prerequisites for building

* Boost, any version. 

### Configure with cmake 

Create a build directory:
```
mkdir build; cd build;
```

Configure cmake. If boost is installed in a standard location, 
```
cmake /path/to/poprithms/root/dir
```

If boost is installed somewhere unusual, 
```
cmake /path/to/poprithms/root/dir -DBOOST_ROOT=/path/to/boost/install 
```


By default, the C++ compiler flag `-Werror` is enabled. To disable `-Werror`, use 
```
cmake /path/to/poprithms/root/dir  -DPOPRITHMS_WERROR=OFF
```

The usual CMake flags can be used to set the install directory and generator:

```
cmake /path/to/poprithms/root/dir  -DPOPRITHMS_WERROR=OFF -BOOST_ROOT=/path/to/boost/install -DCMAKE_INSTALL_PREFIX=/my/install/dir -DCMAKE_GENERATOR="Ninja"
```

### Build the library

The library can be built from the build directory. If the generator is Ninja and you have an install directory set, then 
```
ninja install
```

will build and copy the poprithms shared library, header and configuration files into the install directory. 

### Build the documentation 

Currently poprithms does not build documentation. 

### Examples 

Currently the test directory serves as examples. 

### Prerequisites for building

* Boost, any version. 

### Configure with cmake 

Create a build directory:
```
mkdir build; cd build;
```

Configure cmake. If boost is installed in a standard location, 
```
cmake /path/to/poprithms/root/dir
```

If boost is installed somewhere unusual, 
```
cmake /path/to/poprithms/root/dir -DBOOST_ROOT=/path/to/boost/install 
```


By default, the C++ compiler flag `-Werror` is enabled. To disable `-Werror`, use 
```
cmake /path/to/poprithms/root/dir  -DPOPRITHMS_WERROR=OFF
```

The usual CMake flags can be used to set the install directory and generator:

```
cmake /path/to/poprithms/root/dir  -DPOPRITHMS_WERROR=OFF -BOOST_ROOT=/path/to/boost/install -DCMAKE_INSTALL_PREFIX=/my/install/dir -DCMAKE_GENERATOR="Ninja"
```

### Build the library

The library can be built from the build directory. If the generator is Ninja and you have an install directory set, then 
```
ninja install
```

will build and copy the poprithms shared library, header and configuration files into the install directory. 

### Build the documentation 

Currently poprithms does not build documentation. 

### Examples 

Currently the test directory serves as examples. 

### Using the poprithms library in CMake projects

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

### Troubleshooting

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

To fix this, specify `poprithms_DIR` (or `poprithms-static_DIR` for the static library) as a environment or CMake variable when configuring CMake to the location of `poprithms-config.cmake`.

`$ poprithms_DIR=XXX/install/poprithms/lib/cmake/ cmake ..`

where `XXX` is the path to the poprithms install.

See https://cmake.org/cmake/help/latest/command/find_package.html for more information.
