#ifndef POPRITHMS_SCHEDULE_ANNEAL_ERROR_HPP
#define POPRITHMS_SCHEDULE_ANNEAL_ERROR_HPP

#include <stdexcept>
#include <string>

namespace poprithms {
namespace schedule {
namespace anneal {

class error : public std::runtime_error {
public:
  explicit error(const std::string &what)
      : std::runtime_error(
            std::string("poprithms::schedule::anneal error. ") + what) {}

  explicit error(const char *what)
      : std::runtime_error(
            std::string("poprithms::schedule::anneal error. ") +
            std::string(what)) {}
};

} // namespace anneal
} // namespace schedule
} // namespace poprithms

#endif
