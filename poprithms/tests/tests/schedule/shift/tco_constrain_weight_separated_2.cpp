// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>

#include <poprithms/schedule/shift/error.hpp>
#include <poprithms/schedule/shift/scheduledgraph.hpp>
#include <poprithms/schedule/shift/transitiveclosureoptimizations.hpp>

namespace {
void test(
    double w012,
    double w01,
    double w02,
    double w12, // seems unlikely, as not a clear "creator", but could happen
    double w13,
    double w23,
    double w0123) {

  std::cout << "In test with : " << std::endl;
  std::cout << "  w012 = " << w012 << std::endl;
  std::cout << "  w01  = " << w01 << std::endl;
  std::cout << "  w02  = " << w02 << std::endl;
  std::cout << "  w12  = " << w12 << std::endl;
  std::cout << "  w13  = " << w13 << std::endl;
  std::cout << "  w23  = " << w23 << std::endl;
  std::cout << "  w0123 = " << w0123 << std::endl;

  using namespace poprithms::schedule::shift;

  // The classic diamond:
  //
  //        0     .
  //       / \    .
  //      1   2   .
  //       \ /    .
  //        3     .
  //
  Graph g;
  auto ops = g.insertOps({"op0", "op1", "op2", "op3"});
  g.insertConstraint(ops[0], ops[1]);
  g.insertConstraint(ops[0], ops[2]);
  g.insertConstraint(ops[1], ops[3]);
  g.insertConstraint(ops[2], ops[3]);

  auto a0123 = g.insertAlloc(w0123);
  g.insertOpAlloc({ops[0], ops[1], ops[2], ops[3]}, a0123);

  auto a012 = g.insertAlloc(w012);
  g.insertOpAlloc({ops[0], ops[1], ops[2]}, a012);

  auto a01 = g.insertAlloc(w01);
  g.insertOpAlloc({ops[0], ops[1]}, a01);

  auto a02 = g.insertAlloc(w02);
  g.insertOpAlloc({ops[0], ops[2]}, a02);

  auto a12 = g.insertAlloc(w12);
  g.insertOpAlloc({ops[1], ops[2]}, a12);

  auto a13 = g.insertAlloc(w13);
  g.insertOpAlloc({ops[1], ops[3]}, a13);

  auto a23 = g.insertAlloc(w23);
  g.insertOpAlloc({ops[2], ops[3]}, a23);

  std::cout << "copying g" << std::endl;
  auto gBefore = g;

  auto tco = TransitiveClosureOptimizations::allOff();
  tco.withConstrainWeightSeparatedGroups().withMaxIterations(1);
  ScheduledGraph sg(std::move(g),
                    KahnTieBreaker::RANDOM,
                    tco,
                    RotationTermination::preStart());

  // expect the constraint 1 --> 2 to have been inserted
  auto edge12 = (w13 - w01 <= w23 - w02);
  std::vector<OpAddress> expectSameOuts{0, 3};
  if (edge12) {
    expectSameOuts.push_back(2);
  } else {
    expectSameOuts.push_back(1);
  }

  for (uint64_t i : expectSameOuts) {
    if (sg.getOp(i).getOuts() != gBefore.getOp(i).getOuts()) {
      std::ostringstream oss;
      oss << "Only expected Op 1 to change its outs, not " << i << ". ";
      throw error(oss.str());
    }
  }

  OpAddress from = edge12 ? 1 : 2;
  OpAddress to   = edge12 ? 2 : 1;
  auto outs      = sg.getOp(from).getOuts();
  std::sort(outs.begin(), outs.end());
  if (outs != std::vector<OpAddress>{to, 3}) {
    std::ostringstream oss;
    oss << "Expected {" << to << ",3} to be the outputs of " << from;
    throw error(oss.str());
  }
}
} // namespace

int main() {

  std::cout << "\n\n\n\n\n" << std::endl;
  test(/* 012 =  */ 19.,
       /* 01  =  */ 10.,
       /* 02  =  */ 0.,
       /* 12  =  */ 0.,
       /* 13  =  */ 0.,
       /* 23  =  */ 10.,
       /* 0123=  */ 0);

  std::cout << "\n\n\n\n\n" << std::endl;
  test(/* 012 =  */ 0.,
       /* 01  =  */ 0.,
       /* 02  =  */ 10.,
       /* 12  =  */ 0.,
       /* 13  =  */ 10.,
       /* 23  =  */ 0.,
       /* 0123=  */ 0);

  std::cout << "\n\n\n\n\n" << std::endl;
  test(/* 012 =  */ 0.,
       /* 01  =  */ 10.,
       /* 02  =  */ 0.,
       /* 12  =  */ 0.,
       /* 13  =  */ 0.,
       /* 23  =  */ 10.,
       /* 0123=  */ 0);

  std::cout << "\n\n\n\n\n" << std::endl;
  test(/* 012 =  */ 0.,
       /* 01  =  */ 10.,
       /* 02  =  */ 10.,
       /* 12  =  */ 0.,
       /* 13  =  */ 10.,
       /* 23  =  */ 10.,
       /* 0123=  */ 0);

  return 0;
}
