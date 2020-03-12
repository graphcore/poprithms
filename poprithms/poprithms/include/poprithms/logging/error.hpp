#ifndef POPRITHMS_LOGGING_ERROR_HPP
#define POPRITHMS_LOGGING_ERROR_HPP

#include <stdexcept>
#include <string>

namespace poprithms {
namespace logging {

class error : public std::runtime_error {
public:
  explicit error(const std::string &what)
      : std::runtime_error(std::string("poprithms::logging error. ") + what) {
  }

  explicit error(const char *what)
      : std::runtime_error(std::string("poprithms::logging error. ") +
                           std::string(what)) {}
};

} // namespace logging
} // namespace poprithms

#endif
