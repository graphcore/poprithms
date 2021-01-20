// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/memory/inplace/error.hpp>
#include <poprithms/memory/inplace/graph.hpp>
#include <poprithms/memory/inplace/tensor.hpp>

namespace {

using namespace poprithms::memory::inplace;
}

void test0() {
  Graph g;
  const auto x = Tensor::variable(g, {10, 10});

  //              11
  //              11
  //          /.  |   \.
  //        1.    .1   11    sett samples.
  //        ..    ..   11
  //        |     |    |
  //       mux   mux  mux   [x00, x11, xSS]
  //        |     |    |
  //       mux   mux  mux   [n00, n11, nSS]
  //        |     |    |
  //      unary unary unary
  //

  const auto x00 =
      x.settSample({{10, 10}, {{{{1, 1, 0}}}, {{{1, 1, 0}}}}}).closedMux();

  const auto x11 =
      x.settSample({{10, 10}, {{{{1, 1, 1}}}, {{{1, 1, 1}}}}}).closedMux();

  const auto xSS =
      x.settSample({{10, 10}, {{{{2, 3, 0}}}, {{{2, 3, 0}}}}}).closedMux();

  const auto n00 = x00.closedMux();
  const auto n11 = x11.closedMux();
  const auto nSS = xSS.closedMux();

  n00.modify();
  n11.modify();
  nSS.modify();

  const auto &gStart = g;
  auto test          = [&gStart](const Tensors &order,
                        std::vector<bool> expectedInplace) {
    auto g0 = gStart;
    auto results =
        g0.tryOpenings0(Tensor::opIds(order), CheckParallelWriteable::Yes);
    for (uint64_t i = 0; i < expectedInplace.size(); ++i) {
      auto id = order[i];
      if (id.muxIsOpen() != expectedInplace[i]) {
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
