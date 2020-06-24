// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/error/error.hpp>

#ifdef POPRITHMS_USE_STACKTRACE
#include <boost/stacktrace.hpp>
#endif

namespace poprithms {
namespace error {
std::string error::formatMessage(const std::string &base,
                                 const std::string &what) {
  std::ostringstream oss;
  static constexpr auto root = "poprithms::";
  oss << root << base << " error. " << what;

#ifdef POPRITHMS_USE_STACKTRACE
  // Configure Boost Stacktrace
  static constexpr size_t numFramesToSkip = 3;
  static constexpr size_t maxDepth        = 8;
  boost::stacktrace::stacktrace st(numFramesToSkip, maxDepth);
  oss << "\n\n";

  for (size_t i = 0; i < st.size(); i++) {
    oss << "[" << i << "] " << st[i].name() << "\n";
  }

  oss << "\n\n";
#endif

  return oss.str();
}

} // namespace error
} // namespace poprithms
