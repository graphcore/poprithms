#define BOOST_TEST_MODULE boostsmoke_smoke_test_boost_unit_test_framework_cpp

// Using this include path is 5x slower for compilation
// #include <boost/test/included/unit_test.hpp>
#include <vector>

#include <boost/test/unit_test.hpp>

#include <poprithms/schedule/anneal/graph.hpp>

BOOST_AUTO_TEST_CASE(WeightAnchorTest0) {
  poprithms::schedule::anneal::Graph g;
  BOOST_CHECK(g.nOps() == 0);
}
