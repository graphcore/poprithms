// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/schedule/anneal/error.hpp>
#include <poprithms/schedule/anneal/graph.hpp>
#include <testutil/schedule/anneal/annealcommandlineoptions.hpp>
#include <testutil/schedule/anneal/branch_doubling_generator.hpp>

int main(int argc, char **argv) {

  using namespace poprithms::schedule::anneal;

  std::ostringstream oss;
  oss << "Offset from the power-2 growth of chain length. "
      << "In particular, each subbsequent is of length: "
      << "(sum of previous lengths) - 1 + offset.";

  auto opts = AnnealCommandLineOptions().getCommandLineOptionsMap(
      argc,
      argv,
      {"nBranches", "offset"},
      {"The number of branches from the root Op", oss.str()});
  auto nBranches = std::stoi(opts.at("nBranches"));
  int offset     = std::stoi(opts.at("offset"));

  auto g = getBranchDoublingGraph(nBranches, offset);

  g.initialize(
      KahnTieBreaker::RANDOM, 1011, TransitiveClosureOptimizations::allOn());
  g.minSumLivenessAnneal(
      AnnealCommandLineOptions().getAlgoCommandLineOptionsMap(opts));
  assertGlobalMinimumBranchDoubling(g, nBranches, offset);
  return 0;
}
