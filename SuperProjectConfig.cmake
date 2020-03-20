# See BOOSTSCRIPTS CMakeLists.txt for the motivation for specifying custom
# Boost library components. 

# Certain of these do not belong in poprithms' SuperProjectConfig, but 
# there is currently a design constraint in cbt (or boostscripts) which means 
# that the deps from different repos are not concatenated, but overwrite each 
# other. Forthis reason, I am putting all boost deps here

# The special one required by poprithms is graph


set(BOOST_COMPONENTS "graph system filesystem test program_options atomic container date_time log math regex thread timer")

list(APPEND BOOST_CMAKE_ARGS -DBoost_build_specific_components=${BOOST_COMPONENTS})
