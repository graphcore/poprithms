// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <vector>

#include <poprithms/ndarray/error.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/util/permutation.hpp>
#include <poprithms/util/printiter.hpp>

namespace {

using namespace poprithms::ndarray;

void assertFactorization(const Shape &from,
                         const Shape &to,
                         const std::vector<std::vector<uint64_t>> &expected_,
                         bool expectedOrthogonal) {

  std::vector<Dimensions> expected;
  expected.reserve(expected_.size());
  for (auto e : expected_) {
    expected.push_back(Dimensions(e));
  }

  const auto observed           = from.getReshapeFactorization(to);
  const auto observedOrthogonal = from.isOrthogonalReshape(to);

  const auto baseError = [&]() {
    std::ostringstream oss;
    oss << "Error in assertFactorization(from = " << from << ", to = " << to
        << ". expected = " << expected
        << ", expectedOrthogonal = " << expectedOrthogonal << ')'
        << ". observed = " << observed
        << ", and observedOrthogonal = " << observedOrthogonal << '.';
    return oss.str();
  };

  if (observed != expected) {
    throw error(baseError() + " Incorrect observed. ");
  }

  if (expectedOrthogonal != from.isOrthogonalReshape(to)) {
    throw error(baseError() + " Incorrect observedOrthogonal. ");
  }
}

} // namespace

int main() {
  /*
   *   2  3  from
   *   |  |
   *   2  3  to
   * */
  assertFactorization({2, 3}, {2, 3}, {{0}, {1}}, true);

  /*
   *     2  3  5  7   from
   *     |  |  |  |
   *     +--++-+--+
   *         |
   *        210       to
   * */
  assertFactorization({2, 3, 5, 7}, {210}, {{0, 1, 2, 3}}, true);

  /*
   *        210         from
   *         |
   *     +--++-+--+
   *     |  |  |  |
   *     2  3  5  7      to
   * */
  assertFactorization({210}, {2, 3, 5, 7}, {{0}, {0}, {0}, {0}}, true);

  /*
   *  2   2   2   2    from
   *   \ /     \ /
   *    4       4       to
   * */
  assertFactorization({2, 2, 2, 2}, {4, 4}, {{0, 1}, {2, 3}}, true);

  /*
   *   4      4        from
   *  / \    / \
   * 2   2 2    2       to
   * */
  assertFactorization({4, 4}, {2, 2, 2, 2}, {{0}, {0}, {1}, {1}}, true);

  /*
   *   2   3   5   7   this
   *    \  |  /|  /|
   *      10   7   3   to
   **/
  assertFactorization(
      {2, 3, 5, 7}, {10, 7, 3}, {{0, 1, 2}, {2, 3}, {3}}, false);

  /*
   *     6   2  4   this
   *    / \  \ /
   *   2   3  8     to
   *   */
  assertFactorization({6, 2, 4}, {2, 3, 8}, {{0}, {0}, {1, 2}}, true);

  /*
   *   2 3 5     6   5  4   this
   *   | | |    / \ / \ |
   *   2 3 5   4   5    6   to
   *   */
  assertFactorization({2, 3, 5, 6, 5, 4},
                      {2, 3, 5, 4, 5, 6},
                      {{0}, {1}, {2}, {3}, {3, 4}, {4, 5}},
                      false);

  /*
   *  2   3
   *  | / |
   *  3   2
   * */
  assertFactorization({2, 3}, {3, 2}, {{0, 1}, {1}}, false);

  /*
   *  3   3
   *  | \ |
   *  2   3
   * */
  assertFactorization({3, 2}, {2, 3}, {{0}, {0, 1}}, false);

  return 0;
}
