// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <random>
#include <sstream>

#include <poprithms/error/error.hpp>
#include <poprithms/schedule/vanilla/pathcount.hpp>
#include <poprithms/util/printiter.hpp>

namespace {

using namespace poprithms::schedule::vanilla;

std::ostream &operator<<(std::ostream &ost, const std::vector<uint64_t> &vs) {
  poprithms::util::append(ost, vs);
  return ost;
}

void verify(const Edges<uint64_t> &edges,
            CountType ct,
            const std::vector<uint64_t> &expected) {
  auto out =
      PathCounter::count(edges, ct, ErrorIfCycle::Yes, VerifyEdges::Yes);
  if (out != expected) {
    std::ostringstream oss;
    oss << "Error with " << ct << ". "
        << "Expected" << expected << " but observed " << out << ".";
    throw poprithms::test::error(oss.str());
  }
}

void test0() {

  // Base test, a chain of ops
  {
    Edges<uint64_t> edges{{1}, {2}, {3}, {}};
    verify(edges, CountType::Add, {1, 1, 1, 1});
    verify(edges, CountType::Min, {4, 3, 2, 1});
    verify(edges, CountType::Max, {4, 3, 2, 1});
  }

  // Test with multiple paths of different length:
  {
    Edges<uint64_t> edges{{1, 2}, {5}, {3}, {4}, {5}, {}};
    verify(edges, CountType::Add, {2, 1, 1, 1, 1, 1});
    verify(edges, CountType::Min, {3, 2, 4, 3, 2, 1});
    verify(edges, CountType::Max, {5, 2, 4, 3, 2, 1});
  }

  // Test where the schedule isn't just increasing ints:
  {
    Edges<uint64_t> edges{{}, {0}, {0, 1}, {2, 1, 0}, {3, 2, 1, 0}, {}};
    verify(edges, CountType::Add, {1, 1, 2, 4, 8, 1});
    verify(edges, CountType::Min, {1, 2, 2, 2, 2, 1});
    verify(edges, CountType::Max, {1, 2, 3, 4, 5, 1});
  }
}

} // namespace

int main() {
  test0();
  return 0;
}
