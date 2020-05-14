#include <poprithms/schedule/pathmatrix/error.hpp>

namespace poprithms {
namespace schedule {
namespace pathmatrix {

poprithms::util::error error(const std::string &what) {
  static const std::string pathmatrix("schedule::pathmatrix");
  return poprithms::util::error(pathmatrix, what);
}

} // namespace pathmatrix
} // namespace schedule
} // namespace poprithms