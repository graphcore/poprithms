#include <poprithms/logging/error.hpp>

namespace poprithms {
namespace logging {

poprithms::util::error error(const std::string &what) {
  static const std::string logging("logging");
  return poprithms::util::error(logging, what);
}

} // namespace logging
} // namespace poprithms