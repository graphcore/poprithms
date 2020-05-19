// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <testutil/schedule/anneal/annealcommandlineoptions.hpp>

namespace poprithms {
namespace schedule {
namespace anneal {

const std::vector<std::string> &
AnnealCommandLineOptions::getAlgoCommandLineOptions() const {
  static std::vector<std::string> x{"debug",
                                    "seed",
                                    "pStayPut",
                                    "pHigherFallRate",
                                    "pClimb",
                                    "logging",
                                    "filterSusceptible"};
  return x;
}

} // namespace anneal
} // namespace schedule
} // namespace poprithms
