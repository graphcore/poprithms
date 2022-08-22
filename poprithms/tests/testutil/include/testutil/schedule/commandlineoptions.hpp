// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef TESTUTIL_SCHEDULE_COMMANDLINEOPTIONS_HPP
#define TESTUTIL_SCHEDULE_COMMANDLINEOPTIONS_HPP

#include <map>
#include <string>
#include <vector>

namespace poprithms {
namespace schedule {

class CommandLineOptions {

public:
  using StringMap = std::map<std::string, std::string>;

  // parse argv, and verify that all keys in required appear once
  StringMap
  getCommandLineOptionsMap(int argc,
                           char **argv,
                           const std::vector<std::string> &required,
                           const std::vector<std::string> &requiredInfos);

  // the specific keys to the algorithm being tested
  virtual const std::vector<std::string> &
  getAlgoCommandLineOptions() const = 0;

  std::string
  getInfoString(const std::vector<std::string> &required,
                const std::vector<std::string> &requiredInfos) const;

  // select all algorithm specific arguments from m
  StringMap getAlgoCommandLineOptionsMap(const StringMap &m) const;

private:
  virtual void noWeakVTables();
};

} // namespace schedule
} // namespace poprithms

#endif
