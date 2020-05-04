#include <poprithms/logging/logging.hpp>
#include <poprithms/outline/linear/logging.hpp>

namespace poprithms {
namespace outline {
namespace linear {

poprithms::logging::Logger &log() {
  static poprithms::logging::Logger logger("outline::linear");
  return logger;
}

} // namespace linear
} // namespace outline
} // namespace poprithms
