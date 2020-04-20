#include <poprithms/schedule/anneal/error.hpp>
#include <poprithms/schedule/anneal/filteredschedule.hpp>
#include <poprithms/util/printiter.hpp>

namespace {
using namespace poprithms::schedule::anneal;
using namespace poprithms;
template <typename Filter>
void test(int c,
          const Graph &g,
          OpAddress a,
          const std::vector<OpAddress> &expected,
          Filter f) {

  auto sched0 = getFilteredSchedule(g, a, f);
  std::sort(sched0.begin(), sched0.end());
  if (sched0 != expected) {
    std::ostringstream oss;
    oss << "Failure in test case " << c << ". Expected : ";
    util::append(oss, expected);
    oss << "   Observed : ";
    util::append(oss, sched0);
    throw error(oss.str());
  }
}
} // namespace

int main() {

  //    0
  //   / \
  //  1   2
  //  |   |\
  //  3   4 6
  //   \ / \|
  //    5   7

  using namespace poprithms::schedule::anneal;

  Graph g;

  for (uint64_t i = 0; i < 8; ++i) {
    g.insertOp("op" + std::to_string(i));
  }
  g.insertConstraints({{0, 1},
                       {0, 2},
                       {1, 3},
                       {2, 4},
                       {2, 6},
                       {3, 5},
                       {4, 5},
                       {4, 7},
                       {6, 7}});

  test(0, g, 0, {0, 1, 2, 3, 4, 5, 6, 7}, [](OpAddress) { return true; });
  test(1, g, 1, {1, 3}, [](OpAddress) { return true; });
  test(2, g, 1, {1}, [](OpAddress i) { return i < 3; });
  test(3, g, 2, {2, 6}, [](OpAddress i) { return i != 4; });
  test(4, g, 2, {2, 4, 6, 7}, [](OpAddress i) { return true; });

  return 0;
}
