// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/memory/inplace/error.hpp>
#include <poprithms/memory/inplace/graph.hpp>

namespace {

using namespace poprithms::memory::inplace;
}

void test0() {
  Graph g;
  const auto x = g.variable({10, 10});

  //              11
  //              11
  //          /.  |   \.
  //        1.    .1   11
  //        ..    ..   11
  //        |     |    |
  //        nl    nl   nl
  //

  const auto x00 = g.settSample(
      x, AliasType::outplace(), {{10, 10}, {{{{1, 1, 0}}}, {{{1, 1, 0}}}}});

  const auto x11 = g.settSample(
      x, AliasType::outplace(), {{10, 10}, {{{{1, 1, 1}}}, {{{1, 1, 1}}}}});

  const auto xSS = g.settSample(
      x, AliasType::outplace(), {{10, 10}, {{{{2, 3, 0}}}, {{{2, 3, 0}}}}});

  const auto n00 = g.unary(x00, AliasType::outplace());
  const auto n11 = g.unary(x11, AliasType::outplace());
  const auto nSS = g.unary(xSS, AliasType::outplace());

  const auto &gStart = g;
  auto test          = [&gStart](const TensorIds &order,
                        std::vector<bool> expectedInplace) {
    auto g0      = gStart;
    auto results = g0.tryInplaces(Graph::createProposalsAllInplace(order),
                                  CheckParallelWriteable::Yes);
    for (uint64_t i = 0; i < expectedInplace.size(); ++i) {
      auto id        = order[i];
      bool isInplace = (g0.aliasType(id) != AliasType::outplace());
      if (isInplace != expectedInplace[i]) {
        std::ostringstream oss;
        oss << "Failure with input graph " << gStart
            << ", which was inplaced to " << g0 << ". Expected "
            << "inplace status of id " << id << " not observed. "
            << "This with order = " << order;
      }
    }
  };

  // Inplacing nSS prevents n00 and n11 from being inplaced:
  test({x00, x11, xSS, nSS, n00, n11},
       {true, true, true, true, false, false});

  // n00 and n11 can both be inplaced while nSS is outplace:
  test({x00, x11, xSS, n00, n11, nSS}, {true, true, true, true, true, false});

  // Same as above, but with non-linearities inplaced first:
  test({n00, n11, nSS, x00, x11, xSS}, {true, true, true, true, true, false});
}

int main() {

  test0();

  return 0;
}
