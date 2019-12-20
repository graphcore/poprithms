#include <array>
#include <map>
#include <string>

namespace poprithms {
namespace schedule {
namespace anneal {

class CommandLineOptions {

public:
  using StringMap = std::map<std::string, std::string>;

  // parse argv, and verify that all keys in required appear once
  static StringMap
  getCommandLineOptionsMap(int argc,
                           char **argv,
                           const std::vector<std::string> &required,
                           const std::vector<std::string> &requiredInfos);

  // the keys specific to schedule annealing
  static const std::array<std::string, 6> &getAnnealCommandLineOptions();

  // select all schedule annealing arguments from m
  static StringMap getAnnealCommandLineOptionsMap(const StringMap &m);
};

} // namespace anneal
} // namespace schedule
} // namespace poprithms
