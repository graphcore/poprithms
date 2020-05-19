// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef TESTUTIL_SCHEDULE_EDGEMAP_EDGEMAPCOMMANDLINEOPTIONS_HPP
#define TESTUTIL_SCHEDULE_EDGEMAP_EDGEMAPCOMMANDLINEOPTIONS_HPP

#include <testutil/schedule/commandlineoptions.hpp>

namespace poprithms {
namespace schedule {
namespace transitiveclosure {

class TransitiveClosureCommandLineOptions : public CommandLineOptions {
public:
  virtual const std::vector<std::string> &
  getAlgoCommandLineOptions() const final;
};

} // namespace transitiveclosure
} // namespace schedule
} // namespace poprithms

#endif
