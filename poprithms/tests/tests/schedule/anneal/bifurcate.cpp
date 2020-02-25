#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>
#include <poprithms/schedule/anneal/error.hpp>
#include <poprithms/schedule/anneal/graph.hpp>
#include <poprithms/schedule/anneal/opalloc.hpp>
#include <testutil/schedule/anneal/annealcommandlineoptions.hpp>

int main(int argc, char **argv) {

  using namespace poprithms::schedule::anneal;
  auto opts = AnnealCommandLineOptions().getCommandLineOptionsMap(
      argc,
      argv,
      {"D"},
      {"The depth of the bifurcating-merging graph, The number of nodes "
       "grows as 2**D"});

  // A graph of Ops, where at each depth there are
  // d = 0   : 1 Op with 0 producers and 2 consumers
  // d = 1   : 2 Ops with 1 producer and 2 consumers
  // d = 2   : 4 Ops with 1 producer and 2 consumers
  // .
  // .
  // d = D   : 2^D Ops with 1 producer and 1 consumer
  // d = D+1 : 2^(D-1) Ops with 2 producers and 1 consumer
  // d = D+2 : 2^(D=2) Ops with 2 producers and 1 consumer
  // .
  // .
  // d = 2D-1 : 1 Op with 2 producers and 0 consumers
  //
  // For D = 4;
  //
  //                o
  //
  //        o0              o1
  //    o00     o01    o10      o11
  //                            / \
  //  o   o   o   o   o   o    o   o
  //                              / \
  // o o o o o o o o o o o o o o o   o
  //                              \ /
  //  o   o   o   o   o   o    o   o
  //                            \ /
  //    o       o       o        o
  //
  //        o               o
  //                o
  //
  // All Ops are non-inplace and produce 1 allocation of weight 1.
  //
  // It is easy to see that the maximum liveness of any schedule is an
  // integer in the range [D+2, 2^D+1].
  //
  // We test that these extrema are obtained with the annealing algorithm.
  //

  Graph g;

  // the root "o" in the figure above
  auto in0Mm = g.insertAlloc(1);
  auto inOp  = g.insertOp({}, {in0Mm}, "o");

  auto getFwdSplit = [&g](OpAlloc oa) {
    auto op  = oa.op;
    auto mm  = oa.alloc;
    auto mm0 = g.insertAlloc(1);
    auto op0 =
        g.insertOp({op}, {mm0, mm}, g.getOp(op).getDebugString() + "0");
    auto mm1 = g.insertAlloc(1);
    auto op1 =
        g.insertOp({op}, {mm1, mm}, g.getOp(op).getDebugString() + "1");
    return std::array<OpAlloc, 2>{OpAlloc{op0, mm0}, OpAlloc{op1, mm1}};
  };

  auto getBwdTie = [&g](std::array<OpAlloc, 2> oas) {
    auto oa0  = std::get<0>(oas);
    auto oa1  = std::get<1>(oas);
    auto dbs0 = g.getOp(oa0.op).getDebugString();
    auto dbs  = "y" + dbs0.substr(1, dbs0.size() - 2);
    auto mm   = g.insertAlloc(1);
    auto op   = g.insertOp({oa0.op, oa1.op}, {oa0.alloc, oa1.alloc, mm}, dbs);
    return OpAlloc{op, mm};
  };

  uint64_t D = std::stoi(opts.at("D"));

  std::vector<std::vector<OpAlloc>> xs;
  xs.push_back({{inOp, in0Mm}});
  while (xs.back().size() < (1 << D)) {
    std::vector<OpAlloc> newXs;
    for (auto x : xs.back()) {
      auto x2 = getFwdSplit(x);
      newXs.push_back(std::get<0>(x2));
      newXs.push_back(std::get<1>(x2));
    }
    xs.push_back(newXs);
  }

  while (xs.back().size() != 1UL) {
    std::vector<OpAlloc> newXs;
    for (uint64_t i = 0; i < xs.back().size() / 2; ++i) {
      auto iStart = 2 * i;
      newXs.push_back(getBwdTie(
          std::array<OpAlloc, 2>{xs.back()[iStart], xs.back()[iStart + 1]}));
    }
    xs.push_back(newXs);
  }

  g.insertOp({xs.back().back().op}, {xs.back().back().alloc}, "return");

  // set schedule and all related variables
  g.initialize();
  std::cout << g.getLivenessString() << std::endl;
  g.minSumLivenessAnneal(
      AnnealCommandLineOptions().getAlgoCommandLineOptionsMap(opts));

  auto finalMaxLiveness = g.getMaxLiveness();
  if (finalMaxLiveness != AllocWeight(D + 2, 0)) {
    throw poprithms::schedule::anneal::error(
        "expected final max liveness to be D + 2");
  }

  return 0;
}
