#include <algorithm>
#include <array>
#include <sstream>
#include <string>
#include <testutil/schedule/anneal/commandlineoptions.hpp>
#include <vector>
#include <poprithms/schedule/anneal/error.hpp>

namespace poprithms {
namespace schedule {
namespace anneal {

CommandLineOptions::StringMap
CommandLineOptions::getAnnealCommandLineOptionsMap(const StringMap &m) {
  const auto &annOpts = getAnnealCommandLineOptions();
  StringMap m2;
  for (const auto &[k, v] : m) {
    if (std::find(annOpts.cbegin(), annOpts.cend(), k) != annOpts.cend()) {
      m2.emplace(k, v);
    }
  }
  return m2;
}

const std::array<std::string, 6> &
CommandLineOptions::getAnnealCommandLineOptions() {
  static std::array<std::string, 6> x{
      "debug", "seed", "pStayPut", "pHigherFallRate", "pClimb", "logging"};
  return x;
}

namespace {
const std::string
getInfoString(const std::vector<std::string> &required,
              const std::vector<std::string> &requiredInfos) {

  std::ostringstream oss;
  oss << "Annealing command-line options are [";
  for (auto x : CommandLineOptions::getAnnealCommandLineOptions()) {
    oss << " " << x << " ";
  }
  oss << "].   Required command-line options are [ \n";
  for (uint64_t i = 0; i < required.size(); ++i) {
    oss << " " << required[i] << " : " << requiredInfos[i] << " \n";
  }
  oss << "]. Example use of command-line options: ./myProgram debug 0 pClimb "
         "2.0 (etc etc)";
  return oss.str();
}
} // namespace

CommandLineOptions::StringMap CommandLineOptions::getCommandLineOptionsMap(
    int argc,
    char **argv,
    const std::vector<std::string> &required,
    const std::vector<std::string> &requiredInfos) {

  if (required.size() != requiredInfos.size()) {
    throw error("Error in getCommandLineOptionsMap : required and "
                "requiredInfos are not of the same size");
  }
  StringMap m;
  if (argc % 2 != 1) {
    throw poprithms::error(
        "Invalid (modulo 2) number of command-line options. " +
        getInfoString(required, requiredInfos));
  }
  for (int i = 1; i < argc; i += 2) {
    if (m.find(argv[i]) != m.cend()) {
      throw poprithms::error("Repeated command-line arguments not allowed. " +
                             getInfoString(required, requiredInfos));
    }
    m.emplace(argv[i], argv[i + 1]);
  }

  for (auto x : required) {
    if (m.find(x) == m.end()) {
      throw poprithms::error("Required command-line option `" + x +
                             "' is missing.  " +
                             getInfoString(required, requiredInfos));
    }
  }

  const auto &allowed = getAnnealCommandLineOptions();

  for (const auto &[k, v] : m) {
    if (std::find(allowed.cbegin(), allowed.cend(), k) == allowed.cend() &&
        std::find(required.cbegin(), required.cend(), k) == required.cend()) {
      throw poprithms::error("unrecognised command-line flag " + k);
    }
  }
  return m;
}

} // namespace anneal
} // namespace schedule
} // namespace poprithms
