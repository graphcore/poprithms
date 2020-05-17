#include <poprithms/schedule/dfs/error.hpp>

namespace poprithms {
namespace schedule {
namespace dfs {

poprithms::util::error error(const std::string &what) {
  static const std::string dfs("schedule::dfs");
  return poprithms::util::error(dfs, what);
}

} // namespace dfs
} // namespace schedule
} // namespace poprithms
