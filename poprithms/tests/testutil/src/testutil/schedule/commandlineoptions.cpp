#include <algorithm>
#include <array>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <testutil/schedule/commandlineoptions.hpp>

namespace poprithms {
namespace schedule {

poprithms::schedule::CommandLineOptions::StringMap
CommandLineOptions::getAlgoCommandLineOptionsMap(const StringMap &m) const {
  const auto &algoOpts = getAlgoCommandLineOptions();
  StringMap m2;
  for (const auto &[k, v] : m) {
    if (std::find(algoOpts.cbegin(), algoOpts.cend(), k) != algoOpts.cend()) {
      m2.emplace(k, v);
    }
  }
  return m2;
}

std::string CommandLineOptions::getInfoString(
    const std::vector<std::string> &required,
    const std::vector<std::string> &requiredInfos) const {

  std::ostringstream oss;
  oss << "\n\nAlgorithm specific command-line options are:";
  for (auto x : getAlgoCommandLineOptions()) {
    oss << "\n      " << x;
  }
  oss << ".\n\nRequired command-line options are:";
  for (uint64_t i = 0; i < required.size(); ++i) {
    oss << "\n      " << required[i] << " : " << requiredInfos[i];
  }
  oss << ".\n\nExample use of command-line options:\n"
      << "      ./myProgram opt0 val0 opt1 val1 opt2 val3 (etc ect)\n";
  return oss.str();
}

CommandLineOptions::StringMap CommandLineOptions::getCommandLineOptionsMap(
    int argc,
    char **argv,
    const std::vector<std::string> &required,
    const std::vector<std::string> &requiredInfos) {

  if (required.size() != requiredInfos.size()) {
    throw std::runtime_error(
        "Error in getCommandLineOptionsMap : required and "
        "requiredInfos are not of the same size");
  }
  StringMap m;
  if (argc % 2 != 1) {
    throw std::runtime_error(
        "Invalid (modulo 2) number of command-line options. " +
        getInfoString(required, requiredInfos));
  }
  for (int i = 1; i < argc; i += 2) {
    if (m.find(argv[i]) != m.cend()) {
      throw std::runtime_error(
          "Repeated command-line arguments not allowed. " +
          getInfoString(required, requiredInfos));
    }
    m.emplace(argv[i], argv[i + 1]);
  }

  for (auto x : required) {
    if (m.find(x) == m.end()) {
      throw std::runtime_error("Required command-line option `" + x +
                               "' is missing.  " +
                               getInfoString(required, requiredInfos));
    }
  }

  const auto &allowed = getAlgoCommandLineOptions();

  for (const auto &[k, v] : m) {
    if (std::find(allowed.cbegin(), allowed.cend(), k) == allowed.cend() &&
        std::find(required.cbegin(), required.cend(), k) == required.cend()) {
      throw std::runtime_error("unrecognised command-line flag " + k);
    }
  }
  return m;
}

} // namespace schedule
} // namespace poprithms
