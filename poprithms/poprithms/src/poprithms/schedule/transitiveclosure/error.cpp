#include <poprithms/schedule/transitiveclosure/error.hpp>

namespace poprithms {
namespace schedule {
namespace transitiveclosure {

poprithms::util::error error(const std::string &what) {
  static const std::string transitiveclosure("schedule::transitiveclosure");
  return poprithms::util::error(transitiveclosure, what);
}

} // namespace transitiveclosure
} // namespace schedule
} // namespace poprithms
