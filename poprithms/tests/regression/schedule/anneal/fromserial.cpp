// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
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
// ./fromserial filename /path/to/graph17.json tco yes

int main(int argc, char **argv) {

  using namespace poprithms;
  using namespace poprithms::schedule::anneal;
  auto opts = AnnealCommandLineOptions().getCommandLineOptionsMap(
      argc,
      argv,
      {"filename", "tco"},
      {"The full path of the json serialized poprithms anneal Graph.",
       "If yes/1/true : apply all TransitiveClosureOptimizations during "
       "initialization. If no/0/false : do not apply any "
       "TransitiveClosureOptimizations during initialization."});

  logging::setGlobalLevel(logging::Level::Trace);
  logging::enableDeltaTime(true);
  logging::enableTotalTime(true);

  const auto optTCO = opts.at("tco");
  bool applyTCOs;
  if (optTCO == "yes" || optTCO == "1" || optTCO == "true") {
    applyTCOs = true;
  } else if (optTCO == "no" || optTCO == "0" || optTCO == "false") {
    applyTCOs = false;
  } else {
    throw error("Invalid value for option \"tco\", must be one of "
                "{no,0,false,yes,1,true} and not " +
                optTCO);
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

  auto tcos = applyTCOs ? TransitiveClosureOptimizations::allOn()
                        : TransitiveClosureOptimizations::allOff();

  g.initialize(KahnTieBreaker::GREEDY, 1011, tcos);
  g.minSumLivenessAnneal(
      AnnealCommandLineOptions().getAlgoCommandLineOptionsMap(opts));
  return 0;
}
