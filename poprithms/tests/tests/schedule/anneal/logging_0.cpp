#include <iostream>
#include <sstream>

#include <poprithms/schedule/anneal/error.hpp>
#include <poprithms/schedule/anneal/logging.hpp>

int main() {

  using namespace poprithms::schedule::anneal;
  using namespace poprithms::logging;
  for (auto level : {Level::Off, Level::Info, Level::Debug, Level::Trace}) {
    std::cout << "\n\nSetting logger level to " << level << std::endl;
    log().setLevel(level);
    log().info("info info info");
    log().debug("debug debug debug debug");
    log().trace("trace trace trace trace trace");
  }

  for (auto level : {Level::Off, Level::Info, Level::Debug, Level::Trace}) {
    std::cout << "\n\nSetting global to to " << level << std::endl;
    setGlobalLevel(level);
    log().info("info info info");
    log().debug("debug debug debug debug");
    log().trace("trace trace trace trace trace");
  }

  log().setLevelDebug();
  if (log().shouldLogDebug()) {
    std::ostringstream oss;
    oss << "That's all for now, "
        << "folks!";
    log().debug(oss.str());
  } else {
    throw error("Expected to be able to log at debug-level.");
  }
  if (log().shouldLogTrace()) {
    throw error("Didn't expect to be able to log at trace-level.");
  }

  return 0;
}
