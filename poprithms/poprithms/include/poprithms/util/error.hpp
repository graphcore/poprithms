#ifndef POPRITHMS_UTIL_ERROR_HPP
#define POPRITHMS_UTIL_ERROR_HPP

#include <sstream>
#include <stdexcept>
#include <string>

namespace poprithms {
namespace util {

class error : public std::runtime_error {
public:
  error(const std::string &base, const std::string &what)
      : std::runtime_error(formatMessage(base, what)) {}

private:
  static std::string formatMessage(const std::string &base,
                                   const std::string &what) {
    std::ostringstream oss;
    static constexpr auto root = "poprithms::";
    oss << root << base << " error. " << what;
    return oss.str();
  }
};

} // namespace util
} // namespace poprithms

#endif