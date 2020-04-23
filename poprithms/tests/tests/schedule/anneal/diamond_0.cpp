#include <string>

#include <poprithms/schedule/anneal/error.hpp>
#include <poprithms/schedule/anneal/graph.hpp>
#include <testutil/schedule/anneal/annealcommandlineoptions.hpp>
#include <testutil/schedule/anneal/diamond_generator.hpp>

int main(int argc, char **argv) {

  using namespace poprithms::schedule::anneal;
  auto opts = AnnealCommandLineOptions().getCommandLineOptionsMap(
      argc, argv, {"N"}, {"The number of intermediate Ops in the diamond"});
  auto N      = std::stoull(opts.at("N"));
  Graph graph = getDiamondGraph0(N);
  graph.initialize(KahnTieBreaker::RANDOM);
  graph.minSumLivenessAnneal({});
  assertGlobalMinimumDiamondGraph0(graph, N);
}
