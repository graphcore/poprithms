#define BOOST_TEST_MODULE boostsmoke_smoke_test_boost_unit_test_framework_cpp

// Note: the include directory must be like this for recent versions of
// boost
// #include <boost/test/included/unit_test.hpp>
// Previously (and in popart this somehow still works) this worked
// #include <boost/test/unit_test.hpp>
// Solution discovered here:
// https://stackoverflow.com/questions/33644088/linker-error-while-building-unit-tests-with-boost

#include <boost/test/included/unit_test.hpp>
#include <vector>
#include <poprithms/schedule/anneal/graph.hpp>

BOOST_AUTO_TEST_CASE(WeightAnchorTest0) {
  poprithms::schedule::anneal::Graph g;
  BOOST_CHECK(g.nOps() == 0);
}
