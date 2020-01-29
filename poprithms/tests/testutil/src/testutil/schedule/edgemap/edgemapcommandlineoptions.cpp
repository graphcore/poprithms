#include <testutil/schedule/edgemap/edgemapcommandlineoptions.hpp>

namespace poprithms {
namespace schedule {
namespace edgemap {

const std::vector<std::string> &
EdgeMapCommandLineOptions::getAlgoCommandLineOptions() const {
  static std::vector<std::string> x{};
  return x;
}

} // namespace edgemap
} // namespace schedule
} // namespace poprithms
