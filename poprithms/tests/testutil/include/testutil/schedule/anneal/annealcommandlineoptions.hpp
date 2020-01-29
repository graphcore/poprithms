#ifndef TESTUTIL_SCHEDULE_ANNEAL_ANNEALCOMMANDLINEOPTIONS_HPP
#define TESTUTIL_SCHEDULE_ANNEAL_ANNEALCOMMANDLINEOPTIONS_HPP

#include <testutil/schedule/commandlineoptions.hpp>

namespace poprithms {
namespace schedule {
namespace anneal {

class AnnealCommandLineOptions : public CommandLineOptions {
public:
  virtual const std::vector<std::string> &
  getAlgoCommandLineOptions() const final;
};

} // namespace anneal
} // namespace schedule
} // namespace poprithms

#endif
