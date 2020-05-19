// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <testutil/schedule/transitiveclosure/transitiveclosurecommandlineoptions.hpp>

namespace poprithms {
namespace schedule {
namespace transitiveclosure {

const std::vector<std::string> &
TransitiveClosureCommandLineOptions::getAlgoCommandLineOptions() const {
  static std::vector<std::string> x{};
  return x;
}

} // namespace transitiveclosure
} // namespace schedule
} // namespace poprithms
