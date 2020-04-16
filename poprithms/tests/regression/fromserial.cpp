#include <fstream>
#include <sstream>
#include <streambuf>
#include <poprithms/logging/logging.hpp>
#include <poprithms/schedule/anneal/error.hpp>
#include <poprithms/schedule/anneal/graph.hpp>
#include <poprithms/schedule/anneal/logging.hpp>
#include <testutil/schedule/anneal/annealcommandlineoptions.hpp>

// Example use case:
//
// ./fromserial filename /path/to/graph17.json pmo yes

int main(int argc, char **argv) {

  using namespace poprithms;
  using namespace poprithms::schedule::anneal;
  auto opts = AnnealCommandLineOptions().getCommandLineOptionsMap(
      argc,
      argv,
      {"filename", "pmo"},
      {"The full path of the json serialized poprithms anneal Graph.",
       "If yes/1/true : apply all PathMatrixOptimizations during "
       "initialization. If no/0/false : do not apply any "
       "PathMatrixOptimizations during initialization."});

  logging::setGlobalLevel(logging::Level::Trace);
  logging::enableDeltaTime(true);
  logging::enableTotalTime(true);

  const auto optPMO = opts.at("pmo");
  bool applyPMOs;
  if (optPMO == "yes" || optPMO == "1" || optPMO == "true") {
    applyPMOs = true;
  } else if (optPMO == "no" || optPMO == "0" || optPMO == "false") {
    applyPMOs = false;
  } else {
    throw error("Invalid value for option \"pmo\", must be one of "
                "{no,0,false,yes,1,true} and not " +
                optPMO);
  }

  log().debug("Loading json file into buffer");
  std::ifstream jsfn(opts.at("filename"));
  if (!jsfn.is_open()) {
    throw error(std::string("Failed to open ") + opts.at("filename"));
  }

  std::stringstream buffer;
  buffer << jsfn.rdbuf();
  log().debug("Calling Graph::fromSerializationString");
  auto g = Graph::fromSerializationString(buffer.str());

  auto pmos = applyPMOs ? PathMatrixOptimizations::allOn()
                        : PathMatrixOptimizations::allOff();

  g.initialize(KahnTieBreaker::GREEDY, 1011, pmos);
  g.minSumLivenessAnneal(
      AnnealCommandLineOptions().getAlgoCommandLineOptionsMap(opts));
  return 0;
}
