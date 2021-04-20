// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef TESTUTIL_SCHEDULE_ANNEAL_ANNEALCOMMANDLINEOPTIONS_HPP
#define TESTUTIL_SCHEDULE_ANNEAL_ANNEALCOMMANDLINEOPTIONS_HPP

#include <testutil/schedule/commandlineoptions.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

class AnnealCommandLineOptions : public CommandLineOptions {
public:
  virtual const std::vector<std::string> &
  getAlgoCommandLineOptions() const final;
};

} // namespace shift
} // namespace schedule
} // namespace poprithms

#endif
