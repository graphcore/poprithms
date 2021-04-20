// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <testutil/schedule/shift/shiftcommandlineoptions.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

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

} // namespace shift
} // namespace schedule
} // namespace poprithms
