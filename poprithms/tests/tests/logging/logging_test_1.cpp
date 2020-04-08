#include <chrono>
#include <thread>
#include <poprithms/logging/error.hpp>
#include <poprithms/logging/logging.hpp>

int main() {
  using namespace poprithms::logging;

  auto pause = []() {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  };

  //  A     B     C     D
  //
  //  Off   --    --    --
  Logger A("a");
  A.setLevel(Level::Trace);

  pause();
  A.info("Line 1, no time");
  enableDeltaTime(true);

  pause();
  A.info("Line 2, just delta time");
  enableTotalTime(true);

  pause();
  A.debug("Line 3, delta and total time");

  pause();
  A.trace("Line 4, delta and total time");

  pause();
  A.info("Line 5, delta and total time");
  enableDeltaTime(false);
  enableTotalTime(false);

  pause();
  A.info("Line 5, no time logging");

  // Example output:
  //   [a] [info]  Line 1, no time
  //   [dt=0.002530] [a] [info]  Line 2, just delta time
  //   [T=0.007677] [dt=0.002587] [a] [debug] Line 3, delta and total time
  //   [T=0.010145] [dt=0.002468] [a] [trace] Line 4, delta and total time
  //   [T=0.012693] [dt=0.002548] [a] [info]  Line 5, delta and total time
  //   [a] [info]  Line 5, no time logging

  return 0;
}
