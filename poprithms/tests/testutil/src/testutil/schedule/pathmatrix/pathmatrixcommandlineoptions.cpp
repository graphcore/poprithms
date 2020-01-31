#include <testutil/schedule/pathmatrix/pathmatrixcommandlineoptions.hpp>

namespace poprithms {
namespace schedule {
namespace pathmatrix {

const std::vector<std::string> &
PathMatrixCommandLineOptions::getAlgoCommandLineOptions() const {
  static std::vector<std::string> x{};
  return x;
}

} // namespace pathmatrix
} // namespace schedule
} // namespace poprithms
