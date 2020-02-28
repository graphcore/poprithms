#include <iostream>
#include <numeric>
#include <string>
#include <poprithms/schedule/anneal/error.hpp>
#include <poprithms/schedule/anneal/graph.hpp>
#include <poprithms/schedule/anneal/opalloc.hpp>
#include <testutil/schedule/anneal/annealcommandlineoptions.hpp>
#include <testutil/schedule/anneal/recompute_generator.hpp>

int main(int argc, char **argv) {

  using namespace poprithms;
  using namespace poprithms::schedule::anneal;
  auto opts = AnnealCommandLineOptions().getCommandLineOptionsMap(
      argc,
      argv,
      {"N", "type"},
      {"The number of forward Ops",
       "The type of recomputation. Either sqrt: checkpoints at "
       "approximately every root(N) interval, or log: multi-depth "
       "recursion, where at each depth just the mid-point is checkpoint, "
       "and there approximately log(N) depths "});
  uint64_t nFwd             = std::stoi(opts.at("N"));
  std::string recomputeType = opts.at("type");
  std::vector<int> pattern;
  if (recomputeType == "sqrt") {
    pattern = getSqrtSeries(nFwd);
  } else if (recomputeType == "log") {
    pattern = getLogNSeries(nFwd);
  } else {
    throw poprithms::schedule::anneal::error(
        "Invalid type, log and sqrt are the current options");
  }
  auto g = getRecomputeGraph(pattern);
  g.initialize();
  std::cout << g.getLivenessString() << std::endl;

  g.minSumLivenessAnneal(
      AnnealCommandLineOptions().getAlgoCommandLineOptionsMap(opts));

  std::cout << g.getLivenessString() << std::endl;

  assertGlobalMinimumRecomputeGraph0(g);

  return 0;
}
