add_poprithms_unit_test(schedule_pathmatrix_correctness_0 
                        correctness_0.cpp)

add_poprithms_unit_test(schedule_pathmatrix_remove_redundant_0 
                        remove_redundant_0.cpp)

add_poprithms_unit_test(schedule_pathmatrix_get_unconstrained_0 
                        get_unconstrained_0.cpp)


add_poprithms_unit_test(relative_positions_0 
                        relative_positions_0.cpp)

add_poprithms_unit_test(schedule_pathmatrix_performance_0 
                        performance_0.cpp
                        N 600 
                        E 5 
                        D 20)

add_poprithms_unit_test(schedule_pathmatrix_bitset_performance_0 
                        bitset_performance_0.cpp
                        repeat 7
                        )
