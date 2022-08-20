set(CMAKE_COMPILER_WARNINGS)
list(APPEND CMAKE_COMPILER_WARNINGS
     -Wall
     -pedantic
     -Wextra
     # T67853:
     # -Wweak-vtables
     -Wdisabled-optimization
     -Wshadow
     -Wformat=2
     -Wundef)

   # Previously, Weverything was used. But it is suggested at
   # clang.llvm.org/docs/UsersManual.html#cmdoption-weverything
   #that this is not required/recommended.
   #  if (CMAKE_${COMPILER}_COMPILER_ID MATCHES "Clang")
   #      list(APPEND CMAKE_COMPILER_WARNINGS
   #       -Weverything
   #        # These seem unavoidable:
   #       -Wno-exit-time-destructors
   #       -Wno-c++98-compat
   #       -Wno-c++98-compat-pedantic
   #       -Wno-padded
   #       -Wno-weak-vtables
   #       -Wno-global-constructors
   #       # Overwrite -Werror for this warning, i.e. do not
   #       # ever error for an unknown warning flag
   #       -Wno-error=unknown-warning-option)
   #  endif()

add_compile_options(${CMAKE_COMPILER_WARNINGS})
