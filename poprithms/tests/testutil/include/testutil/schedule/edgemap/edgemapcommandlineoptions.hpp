#ifndef TESTUTIL_SCHEDULE_EDGEMAP_EDGEMAPCOMMANDLINEOPTIONS_HPP
#define TESTUTIL_SCHEDULE_EDGEMAP_EDGEMAPCOMMANDLINEOPTIONS_HPP

#include <testutil/schedule/commandlineoptions.hpp>

namespace poprithms {
namespace schedule {
namespace edgemap {

class EdgeMapCommandLineOptions : public CommandLineOptions {
public:
  virtual const std::vector<std::string> &
  getAlgoCommandLineOptions() const final;
};

} // namespace edgemap
} // namespace schedule
} // namespace poprithms

#endif
