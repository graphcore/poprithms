#include <poprithms/logging/logging.hpp>
#include <poprithms/schedule/anneal/logging.hpp>

namespace poprithms {
namespace schedule {
namespace anneal {

poprithms::logging::Logger &log() {
  static poprithms::logging::Logger logger("anneal");
  return logger;
}

} // namespace anneal
} // namespace schedule
} // namespace poprithms
