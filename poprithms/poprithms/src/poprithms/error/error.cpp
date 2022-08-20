// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/error/error.hpp>

#ifdef POPRITHMS_USE_STACKTRACE
#include <boost/stacktrace.hpp>
#endif

namespace poprithms {
namespace error {

namespace {
std::string withStackTrace(const std::string &prefix) {

  std::ostringstream oss;
  oss << prefix;

#ifdef POPRITHMS_USE_STACKTRACE
  // Configure Boost Stacktrace
  static constexpr size_t numFramesToSkip = 3;
  static constexpr size_t maxDepth        = 16;
  boost::stacktrace::stacktrace st(numFramesToSkip, maxDepth);
  oss << "\n\n";

  for (size_t i = 0; i < st.size(); i++) {
    oss << "[" << i << "] " << st[i].name() << "\n";
  }

  oss << "\n\n";
#endif

  return oss.str();
} // namespace
} // namespace

std::ostream &operator<<(std::ostream &ost, const Code &c) {
  ost << c.val();
  return ost;
}

std::string error::formatMessage(const std::string &base,
                                 const uint64_t id,
                                 const std::string &what) {
  const std::string prefix = std::string("poprithms::") + base +
                             " error, code is POPRITHMS" +
                             std::to_string(id) + ". " + what;
  return withStackTrace(prefix);
}

std::string error::formatMessage(const std::string &base,
                                 const std::string &what) {
  const std::string prefix =
      std::string("poprithms::") + base + " error. " + what;
  return withStackTrace(prefix);
}

std::string error::weakVTableMessage() {
  return "This out-of-line virtual method implementation is to avoid "
         "vtable copies. It is a dummy method which should not be called. ";
}

void error::noWeakVTables() { throw error("error", weakVTableMessage()); }

} // namespace error

namespace test {
poprithms::error::error error(const std::string &what) {
  static const std::string t("test");
  return poprithms::error::error(t, what);
}
} // namespace test
} // namespace poprithms
