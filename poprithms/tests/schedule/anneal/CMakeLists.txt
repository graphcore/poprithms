add_poprithms_unit_test(schedule_anneal_bifurcate bifurcate.cpp D 5)
add_poprithms_unit_test(schedule_anneal_grid grid.cpp N 5)
add_poprithms_unit_test(schedule_anneal_test_random test_random.cpp)
add_poprithms_unit_test(schedule_anneal_recompute recompute.cpp N 50 type sqrt)
add_poprithms_unit_test(schedule_anneal_random random.cpp N 100 E 5 D 10 graphSeed 1011)
#TODO(jn):
add_poprithms_unit_test(schedule_anneal_priorities priorities.cpp)
#TODO(jn):
add_poprithms_unit_test(schedule_anneal_tied tied.cpp)

