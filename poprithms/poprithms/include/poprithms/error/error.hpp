// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_ERROR_ERROR_HPP
#define POPRITHMS_ERROR_ERROR_HPP

#include <sstream>
#include <stdexcept>
#include <string>

namespace poprithms {
namespace error {

/**
 * the poprithms::error class has an optional Code field. Codes can be useful
 * when searching for information about errors 'in the wild', and for making
 * testing more robust.
 * */
class Code {
public:
  explicit Code(uint64_t v) : v_(v) {}
  uint64_t val() const { return v_; }

  bool operator<=(const Code &r) const { return val() <= r.val(); }
  bool operator<(const Code &r) const { return val() < r.val(); }
  bool operator==(const Code &r) const { return r.val() == val(); }
  bool operator!=(const Code &r) const { return r.val() != val(); }
  bool operator>=(const Code &r) const { return val() >= r.val(); }
  bool operator>(const Code &r) const { return val() > r.val(); }

private:
  uint64_t v_;
};

std::ostream &operator<<(std::ostream &, const Code &);

class error : public std::runtime_error {
public:
  /**
   * Construct an error with a Code.
   * */
  error(const std::string &base, Code code, const std::string &what)
      : std::runtime_error(formatMessage(base, code.val(), what)),
        code_(code) {}

  /**
   * Construct an error without a Code.
   * */
  error(const std::string &base, const std::string &what)
      : std::runtime_error(formatMessage(base, what)), code_(Code(0ull)) {}

  Code code() const { return code_; }

private:
  static std::string formatMessage(const std::string &base,
                                   uint64_t codeVal,
                                   const std::string &what);

  static std::string formatMessage(const std::string &base,
                                   const std::string &what);

  Code code_;

  virtual void noWeakVTables();

public:
  /**
   * Some classes have a dummy virtual method with an out-of-line definition.
   * For more information, see the compiler option Wweak-vtable.
   * */
  static std::string weakVTableMessage();
};

} // namespace error

namespace test {
error::error error(const std::string &what);
}

} // namespace poprithms

#endif
