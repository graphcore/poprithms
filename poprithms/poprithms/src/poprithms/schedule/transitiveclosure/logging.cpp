#include <poprithms/logging/logging.hpp>
#include <poprithms/schedule/transitiveclosure/logging.hpp>

namespace poprithms {
namespace schedule {
namespace transitiveclosure {

poprithms::logging::Logger &log() {
  static poprithms::logging::Logger logger("pm");
  return logger;
}

} // namespace transitiveclosure
} // namespace schedule
} // namespace poprithms
