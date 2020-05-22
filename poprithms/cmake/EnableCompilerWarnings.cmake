## Strict warning level
if (MSVC)
    # Use the highest warning level for visual studio.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /w")

else()
    foreach(COMPILER C CXX)
        set(CMAKE_COMPILER_WARNINGS)
        list(APPEND CMAKE_COMPILER_WARNINGS
            -Werror
            -Wall
            -Wextra
            -Wcomment
            -Wendif-labels
            -Wformat
            -Winit-self
            -Wreturn-type
            -Wsequence-point
            -Wshadow
            -Wswitch
            -Wtrigraphs
            -Wundef
            -Wuninitialized
            -Wunreachable-code
            -Wunused
            -Wno-sign-compare
        )
        if (CMAKE_${COMPILER}_COMPILER_ID MATCHES "Clang")
            list(APPEND CMAKE_COMPILER_WARNINGS
                -Weverything
                 # absolutely need these:
                -Wno-exit-time-destructors
                -Wno-c++98-compat
                -Wno-c++98-compat-pedantic
                -Wno-padded
                -Wno-weak-vtables
                # I like having a default in switch statement even
                # if all cases are covered. This is because if
                # someone adds a additional case (additional enum) which
                # is not covered in the switch statement, default
                # catches it.
                -Wno-covered-switch-default
                # This warning seems unavoidable
                # for non-trivial static construction.
                -Wno-global-constructors
                # Overwrite -Werror for this warning, i.e. do not
                # error for an unknown warning flag
                -Wno-error=unknown-warning-option
            )
        else()
            list(APPEND CMAKE_COMPILER_WARNINGS
                -Wno-missing-field-initializers
                -Wno-deprecated-declarations
            )
        endif()

        # An issue with older AppleClang compiler, not deleting for now
        # in case this issue crops up again.
        #
        # if (CMAKE_${COMPILER}_COMPILER_ID MATCHES "AppleClang")
        #     list(APPEND CMAKE_COMPILER_WARNINGS
        #         # Seems like best practise is to always return by value. See
        #         # c11-rvalues-and-move-semantics-confusion-return-statement
        #         # for stackoverflow discussion
        #         -Wno-return-std-move-in-c++11
        #       )
        # endif()

        add_compile_options(${CMAKE_COMPILER_WARNINGS})
    endforeach()
endif ()
