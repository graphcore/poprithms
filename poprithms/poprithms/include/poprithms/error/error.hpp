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
  error(const std::string &base, uint64_t id, const std::string &what)
      : std::runtime_error(formatMessage(base, id, what)), code_(id) {}

  error(const std::string &base, const std::string &what)
      : std::runtime_error(formatMessage(base, what)), code_(0ull) {}

  uint64_t code() const { return code_; }

private:
  static std::string formatMessage(const std::string &base,
                                   uint64_t id,
                                   const std::string &what);

  static std::string formatMessage(const std::string &base,
                                   const std::string &what);

  uint64_t code_;
};

} // namespace error

namespace test {
error::error error(const std::string &what);
}

} // namespace poprithms

#endif
