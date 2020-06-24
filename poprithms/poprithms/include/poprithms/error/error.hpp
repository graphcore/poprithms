// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_ERROR_ERROR_HPP
#define POPRITHMS_ERROR_ERROR_HPP

#include <sstream>
#include <stdexcept>
#include <string>

namespace poprithms {
namespace error {

class error : public std::runtime_error {
public:
  error(const std::string &base, const std::string &what)
      : std::runtime_error(formatMessage(base, what)) {}

private:
  static std::string formatMessage(const std::string &base,
                                   const std::string &what);
};

} // namespace error
} // namespace poprithms

#endif
