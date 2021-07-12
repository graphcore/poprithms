// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>

#include <poprithms/error/error.hpp>
#include <poprithms/schedule/transitiveclosure/transitiveclosure.hpp>
#include <testutil/schedule/transitiveclosure/randomedges.hpp>
#include <testutil/schedule/transitiveclosure/transitiveclosurecommandlineoptions.hpp>

using namespace poprithms::schedule::transitiveclosure;

int main(int argc, char **argv) {

  auto opts = TransitiveClosureCommandLineOptions().getCommandLineOptionsMap(
      argc,
      argv,
      {"N", "E", "D"},
      {"Number of Ops",
       "Number of out edges per Op",
       "Maximum inter-index edge length"});
  auto N   = std::stoi(opts.at("N"));
  auto E   = std::stoi(opts.at("E"));
  auto D   = std::stoi(opts.at("D"));
  auto fwd = getRandomEdges(N, E, D, 10111);

  auto start = std::chrono::high_resolution_clock::now();
  TransitiveClosure fem(fwd);
  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = stop - start;

  bool printConnectivity{false};
  if (printConnectivity) {
    std::cout << "\nConstraint Map. v[i][j] = 1 iff i->j is a constraint. \n"
              << std::endl;
    for (uint64_t from = 0; from < fem.nOps_u64(); ++from) {
      for (uint64_t to = 0; to < fem.nOps_u64(); ++to) {
        std::cout << fem.constrained(from, to);
      }
      std::cout << std::endl;
    }
  }

  std::cout << "Total time to construct TransitiveClosure = "
            << elapsed.count() << " [s]" << std::endl;
  return 0;
}
