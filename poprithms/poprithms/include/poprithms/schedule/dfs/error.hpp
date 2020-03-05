#ifndef POPRITHMS_SCHEDULE_DFS_ERROR_HPP
#define POPRITHMS_SCHEDULE_DFS_ERROR_HPP

#include <stdexcept>
#include <string>

namespace poprithms {
namespace schedule {
namespace dfs {

class error : public std::runtime_error {
public:
  explicit error(const std::string &what)
      : std::runtime_error(std::string("poprithms::schedule::dfs error. ") +
                           what) {}

  explicit error(const char *what)
      : std::runtime_error(std::string("poprithms::schedule::dfs error. ") +
                           std::string(what)) {}
};

} // namespace dfs
} // namespace schedule
} // namespace poprithms

#endif
