// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <poprithms/schedule/shift/error.hpp>
#include <poprithms/schedule/shift/graph.hpp>
#include <testutil/schedule/shift/branch_doubling_generator.hpp>
#include <testutil/schedule/shift/shiftcommandlineoptions.hpp>

int main(int argc, char **argv) {

  using namespace poprithms::schedule::shift;

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

  auto opts1 = AnnealCommandLineOptions().getAlgoCommandLineOptionsMap(opts);
  opts1.insert({"kahnTieBreaker", "Random"});
  opts1.insert({"seed", "1011"});
  opts1.insert({"allTCO", "1"});

  const auto sg = ScheduledGraph(std::move(g), opts1);

  assertGlobalMinimumBranchDoubling(sg, nBranches, offset);
  return 0;
}
