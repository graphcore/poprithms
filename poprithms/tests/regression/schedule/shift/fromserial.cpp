// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <fstream>
#include <sstream>
#include <streambuf>

#include <testutil/schedule/shift/shiftcommandlineoptions.hpp>

#include <poprithms/error/error.hpp>
#include <poprithms/logging/logging.hpp>
#include <poprithms/schedule/shift/logging.hpp>
#include <poprithms/schedule/shift/scheduledgraph.hpp>

// Example use case:
//
// ./fromserial filename /path/to/graph17.json tco yes

int main(int argc, char **argv) {

  using namespace poprithms::schedule::shift;
  auto opts = ShiftCommandLineOptions().getCommandLineOptionsMap(
      argc,
      argv,
      {"filename", "tco"},
      {"The full path of the json serialized poprithms shift Graph.",
       "If yes/1/true : apply all TransitiveClosureOptimizations during "
       "initialization. If no/0/false : do not apply any "
       "TransitiveClosureOptimizations during initialization."});

  {
    using namespace poprithms::logging;
    setGlobalLevel(Level::Trace);
    enableDeltaTime(true);
    enableTotalTime(true);
  }

  const auto optTCO = opts.at("tco");
  bool applyTCOs;
  if (optTCO == "yes" || optTCO == "1" || optTCO == "true") {
    applyTCOs = true;
  } else if (optTCO == "no" || optTCO == "0" || optTCO == "false") {
    applyTCOs = false;
  } else {
    throw poprithms::test::error(
        "Invalid value for option \"tco\", must be one of "
        "{no,0,false,yes,1,true} and not " +
        optTCO);
  }

  log().debug("Loading json file into buffer");
  std::ifstream jsfn(opts.at("filename"));
  if (!jsfn.is_open()) {
    throw poprithms::test::error(std::string("Failed to open ") +
                                 opts.at("filename"));
  }

  std::stringstream buffer;
  buffer << jsfn.rdbuf();
  log().debug("Calling Graph::fromSerializationString");
  auto g = Graph::fromSerializationString(buffer.str());

  auto tcos = applyTCOs ? TransitiveClosureOptimizations::allOn()
                        : TransitiveClosureOptimizations::allOff();

  auto opts1 = ShiftCommandLineOptions().getAlgoCommandLineOptionsMap(opts);

  opts1.insert({"kahnTieBreaker", "GREEDY"});
  opts1.insert({"kahnSeed", "1011"});
  if (applyTCOs) {
    opts1.insert({"allTCO", "1"});
  } else {
    opts1.insert({"allTCO", "0"});
  }

  auto sg = ScheduledGraph(std::move(g), opts1);

  return 0;
}
