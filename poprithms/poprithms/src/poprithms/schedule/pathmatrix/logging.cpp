#include <poprithms/logging/logging.hpp>
#include <poprithms/schedule/pathmatrix/logging.hpp>

namespace poprithms {
namespace schedule {
namespace pathmatrix {

poprithms::logging::Logger &log() {
  static poprithms::logging::Logger logger("pm");
  return logger;
}

} // namespace pathmatrix
} // namespace schedule
} // namespace poprithms
