#include <algorithm>
#include <iostream>
#include <sstream>
#include <poprithms/schedule/pathmatrix/error.hpp>
#include <poprithms/schedule/pathmatrix/pathmatrix.hpp>

namespace {

using namespace poprithms::schedule::pathmatrix;

template <typename T> void assertUnion(const T &a0, const T &a1, T b) {
  auto c = a0;
  c.insert(c.end(), a1.cbegin(), a1.cend());
  std::sort(c.begin(), c.end());
  std::sort(b.begin(), b.end());
  if (c != b) {
    throw error("Failed to assert that union(a0, a1) == b");
  }
}

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
  PathMatrix pmat{{{1, 2, 3, 4}, {2}, {3, 4}, {}, {}}};
  if (pmat.getFwd() != Edges{{{1}, {2}, {3, 4}, {}, {}}}) {
    throw error("Incorrect post-redundant fwd edges");
  }
  const auto mb = Edges{{{}, {0}, {1}, {2}, {2, 2}}};
  if (pmat.getBwd() != Edges{{}, {0}, {1}, {2}, {2}}) {
    throw error("Incorrect post-redundant bwd edges");
  }
}

auto getAsVector(const Edges &m) {
  std::vector<std::array<OpId, 2>> v;
  for (uint64_t i = 0; i < m.size(); ++i) {
    for (auto j : m[i]) {
      v.push_back({i, j});
    }
  }
  return v;
}

auto getReversed(const std::vector<std::array<OpId, 2>> &v) {
  auto v2 = v;
  for (auto &x : v2) {
    std::swap(std::get<0>(x), std::get<1>(x));
  }
  return v2;
}

void test2() {
  Edges edges(130);
  uint64_t nEdges = 0;
  for (uint64_t i = 0; i < 100; ++i) {
    for (uint64_t j = 1; j < 25; ++j) {
      if (j % (1 + i % 10) == 0 || j % 7 == 0) {
        edges[i].push_back(i + j);
      }
    }
  }
  auto pmat = PathMatrix(edges);
  assertUnion(
      pmat.getFwdRedundant(), getAsVector(pmat.getFwd()), getAsVector(edges));
  assertUnion(pmat.getBwdRedundant(),
              getAsVector(pmat.getBwd()),
              getReversed(getAsVector(edges)));
}

} // namespace

int main() {
  test1();
  test2();
  return 0;
}
