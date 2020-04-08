#include <algorithm>
#include <iostream>
#include <sstream>
#include <poprithms/schedule/pathmatrix/error.hpp>
#include <poprithms/schedule/pathmatrix/pathmatrix.hpp>

namespace {

using namespace poprithms::schedule::pathmatrix;

void test1() {

  //
  //       0
  //     / |
  //    1  |
  //    | /|
  //    2  |
  //   /| /|
  //  | |/ |
  //  |/|  |
  //  3  \ |
  //       4
  const Edges edges{{1, 2, 3, 4}, {2}, {3, 4}, {}, {}};
  const Edges critical{{1}, {2}, {3, 4}, {}, {}};

  PathMatrix pmat{edges};
  auto redundants = pmat.getFlattenedRedundants(edges);
  for (OpId from = 0; from < edges.size(); ++from) {
    for (auto to : edges[from]) {
      auto isRedundant =
          std::find(redundants.cbegin(),
                    redundants.cend(),
                    std::array<OpId, 2>{from, to}) != redundants.cend();
      auto expected =
          std::find(critical[from].cbegin(), critical[from].cend(), to) ==
          critical[from].cend();
      if (isRedundant != expected) {
        std::ostringstream oss;
        oss << "Incorrect redundancy for edge " << from << "->" << to
            << ", expected = " << expected;
        throw error(oss.str());
      }
    }
  }
}

} // namespace

int main() {
  test1();
  return 0;
}
