#ifndef POPRITHMS_SCHEDULE_EDGEMAP_ERROR_HPP
#define POPRITHMS_SCHEDULE_EDGEMAP_ERROR_HPP

#include <stdexcept>
#include <string>

namespace poprithms {
namespace schedule {
namespace pathmatrix {

class error : public std::runtime_error {
public:
  explicit error(const std::string &what)
      : std::runtime_error(
            std::string("poprithms::schedule::pathmatrix error. ") + what) {}

  explicit error(const char *what)
      : std::runtime_error(
            std::string("poprithms::schedule::pathmatrix error. ") +
            std::string(what)) {}
};

} // namespace pathmatrix
} // namespace schedule
} // namespace poprithms

#endif
