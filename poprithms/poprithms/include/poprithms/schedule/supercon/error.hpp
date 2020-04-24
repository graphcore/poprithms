#ifndef POPRITHMS_SCHEDULE_SUPER_ERROR_HPP
#define POPRITHMS_SCHEDULE_SUPER_ERROR_HPP

#include <stdexcept>
#include <string>

namespace poprithms {
namespace schedule {
namespace supercon {

// see TODO(T19556) to factorize the various poprithms error classes 
class error : public std::runtime_error {
public:
  explicit error(const std::string &what)
      : std::runtime_error(
            std::string("poprithms::schedule::supercon error. ") + what) {}

  explicit error(const char *what)
      : std::runtime_error(
            std::string("poprithms::schedule::supercon error. ") +
            std::string(what)) {}
};

} // namespace supercon
} // namespace schedule
} // namespace poprithms

#endif
