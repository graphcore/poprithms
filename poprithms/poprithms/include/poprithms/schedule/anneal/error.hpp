#ifndef FWALGS_ERROR_ERROR_HPP_HPP
#define FWALGS_ERROR_ERROR_HPP_HPP

#include <stdexcept>

namespace poprithms {

class error : public std::runtime_error {
public:
  explicit error(const std::string &what) : std::runtime_error(what) {}
  explicit error(const char *what) : std::runtime_error(what) {}
};

} // namespace poprithms

#endif
