#ifndef POPRITHMS_OUTLINE_LINEAR_ERROR_HPP
#define POPRITHMS_OUTLINE_LINEAR_ERROR_HPP

#include <stdexcept>
#include <string>

namespace poprithms {
namespace outline {
namespace linear {

// see TODO(T19556): create a base class for errors
class error : public std::runtime_error {
public:
  explicit error(const std::string &what)
      : std::runtime_error(std::string("poprithms::outline::linear error. ") +
                           what) {}

  explicit error(const char *what)
      : std::runtime_error(std::string("poprithms::outline::linear error. ") +
                           std::string(what)) {}
};

} // namespace linear
} // namespace outline
} // namespace poprithms

#endif
