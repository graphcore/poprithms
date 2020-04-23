#include <iostream>

#include <poprithms/schedule/anneal/error.hpp>
#include <poprithms/schedule/anneal/graph.hpp>
#include <poprithms/schedule/anneal/opalloc.hpp>
#include <testutil/schedule/anneal/annealcommandlineoptions.hpp>
#include <testutil/schedule/anneal/bifurcate_generator.hpp>

int main(int argc, char **argv) {

  using namespace poprithms::schedule::anneal;
  auto opts = AnnealCommandLineOptions().getCommandLineOptionsMap(
      argc,
      argv,
      {"D"},
      {"The depth of the bifurcating-merging graph, The number of nodes "
       "grows as 2**D"});

  uint64_t D = std::stoi(opts.at("D"));
  auto annos = AnnealCommandLineOptions().getAlgoCommandLineOptionsMap(opts);

  auto g = getBifurcatingGraph0(D);
  g.initialize();
  g.minSumLivenessAnneal(annos);

  assertGlobalMinimumBifurcatingGraph0(g, D);
  return 0;
}
