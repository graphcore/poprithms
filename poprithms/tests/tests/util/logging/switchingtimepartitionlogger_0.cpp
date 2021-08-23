// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>

#include <poprithms/error/error.hpp>
#include <poprithms/logging/timepartitionlogger.hpp>

namespace {

using namespace poprithms::logging;
using Event = TimePartitionLogger::Event;
using Type  = TimePartitionLogger::EventType;

auto &testLogger() {
  static SwitchingTimePartitionLogger logger("TimeInScopeLogger for testing");
  return logger;
}

void foo2() {
  auto a = testLogger().scopedStopwatch("foo2");
  std::this_thread::sleep_for(std::chrono::milliseconds(3));
}

void foo1() {
  auto b = testLogger().scopedStopwatch("foo1");
  foo2();
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
}

void foo0() {
  auto c = testLogger().scopedStopwatch("foo0");
  foo1();
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

void scopeStopwatch0() {
  foo0();

  std::cout << testLogger().eventsStr() << std::endl;

  testLogger().verifyEvents({{"foo0", Type::Start},
                             {"foo0", Type::Stop},
                             {"foo1", Type::Start},
                             {"foo1", Type::Stop},
                             {"foo2", Type::Start},
                             {"foo2", Type::Stop},
                             {"foo1", Type::Start},
                             {"foo1", Type::Stop},
                             {"foo0", Type::Start},
                             {"foo0", Type::Stop}});
}

void testPercentage() {
  auto c = SwitchingTimePartitionLogger();
  c.start("a");
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  c.start("b");
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  c.start("c");
  std::this_thread::sleep_for(std::chrono::milliseconds(4));

  auto countPerc = [](const std::string &x) {
    int n{0};
    for (auto c_ : x) {
      if (c_ == '%') {
        ++n;
      }
    }
    return n;
  };
  if (countPerc(c.str(100.)) != 3) {
    std::ostringstream oss;
    oss << "Counting the number of scopes which have at least 100 percent. "
        << "Expected 3 : total, accounted, and unaccounted, the 3 which "
           "always appear as the threshold is not applied to them. The log "
           "is\n "
        << c.str(100.);
    throw poprithms::test::error(oss.str());
  }

  if (countPerc(c.str(0.)) != 6) {
    std::ostringstream oss;
    oss << "Counting the number of scopes which have at least 0 percent. "
        << "Expected 6: total, accounted, unaccounted -- the 3 which always "
           "appear as the threshold is not applied to them -- and a, b, and "
           "c. The log is\n "
        << c.str(0.);
    throw poprithms::test::error(oss.str());
  }
}

void scopeStopwatch1() {
  SwitchingTimePartitionLogger logger("scopeStopwatch1");

  // start swBase
  auto swBase = logger.scopedStopwatch("swBase");
  {

    // stop swBase
    // start sw0
    auto sw0 = logger.scopedStopwatch("sw0");

    // stop sw0
    // start sw1
    auto sw1 = logger.scopedStopwatch("sw1");

    // C++ guarantees destruction in reverse order
    // (See chapter of C++ standard: 3.6.3 Termination)
    //
    // stop sw1
    // start sw0
    // stop  sw0
    // start swBase
  }

  {
    // stop swBase
    // start sw0
    auto sw0 = logger.scopedStopwatch("sw0");

    // stop sw0
    // start sw2
    auto sw2 = logger.scopedStopwatch("sw2");

    //  stop sw2
    // start sw0
    // stop sw0
    // start swBase
  }

  logger.verifyEvents({{"swBase", Type::Start},
                       {"swBase", Type::Stop},
                       {"sw0", Type::Start},
                       {"sw0", Type::Stop},
                       {"sw1", Type::Start},
                       {"sw1", Type::Stop},
                       {"sw0", Type::Start},
                       {"sw0", Type::Stop},
                       {"swBase", Type::Start},
                       {"swBase", Type::Stop},
                       {"sw0", Type::Start},
                       {"sw0", Type::Stop},
                       {"sw2", Type::Start},
                       {"sw2", Type::Stop},
                       {"sw0", Type::Start},
                       {"sw0", Type::Stop},
                       {"swBase", Type::Start}});

  std::cout << logger.eventsStr() << std::endl;
}

void moveScopeStopwatch0() {
  // Check there is no stopping conflict if you move a scope stopwatch.
  SwitchingTimePartitionLogger logger("moveScopeStopwatch0");

  {
    // start scoped stopwatch
    auto sw0 = logger.scopedStopwatch("sw");

    // move to another stopwatch.
    auto sw1 = std::move(sw0);

    // Both sw0 and sw1 will go out of scope here, but only sw1 should stop
    // "sw" on the logger. If sw0 also calls stop despite being moved then an
    // error will be thrown at this when the second stop is called.
  }
}

void testOrder0() {
  std::cout << "In test order 0" << std::endl;
  SwitchingTimePartitionLogger watcher("aSwitchingLogger");

  const uint64_t nScopes = 6;

  for (uint64_t i = 0; i < nScopes; ++i) {
    {
      const auto sw = watcher.scopedStopwatch("foo_" + std::to_string(i));
      std::this_thread::sleep_for(
          std::chrono::milliseconds(1 * (1 + (101 * i) % 5)));
    }
  }

  const double loggingPercentageThreshold{0.00};

  /**
   * Summary string looks something like:
   *
   *  Scope              Time [s]        Count  Percentage
   *  -----              --------        -----  ----------
   *  foo_4              0.006390            1        33 %
   *  foo_3              0.004140            1        21 %
   *  foo_2              0.003815            1        19 %
   *  foo_1              0.002541            1        13 %
   *  foo_5              0.001416            1         7 %
   *  foo_0              0.001183            1         6 %
   *  Total              0.019593          n/a       100 %
   *  Accounted for      0.019486          n/a        99 %
   *  Unaccounted for    0.000107          n/a         1 %
   *
   * We test that the times (second column) are sorted.
   */
  const auto summary = watcher.str(loggingPercentageThreshold);
  std::cout << "\n\n" << summary << std::endl;

  const auto x0 = summary.find('\n', 0);
  const auto x1 = summary.find('\n', x0 + 1);
  const auto x2 = summary.find('\n', x1 + 1);

  // Distance between lines
  const auto delta = x2 - x1;

  // range where the time (in seconds) is found.
  const auto y0 = summary.find("0.", 0);
  const auto y1 = summary.find(" ", y0);

  // extract all the times from the string
  std::vector<double> times;
  for (uint64_t i = 0; i < nScopes; ++i) {
    const auto x = std::string{summary.cbegin() + delta * i + y0 + 1,
                               summary.cbegin() + delta * i + y1};
    times.push_back(std::stod(x));
  }

  // to assert that they're sorted, we create a copy and sort that, then
  // compare for equivalence.
  auto sortedTimes = times;
  std::sort(sortedTimes.rbegin(), sortedTimes.rend());
  if (sortedTimes != times) {
    throw poprithms::test::error("Times not sorted: " + summary);
  }
}

// 23/08/2021: with nScopes = 100 and nSwitches = 1000000 this takes about
// 0.8 seconds.
void rapidFireTest0(uint64_t nScopes, uint64_t nSwitches) {
  SwitchingTimePartitionLogger s;

  auto x = s.scopedStopwatch("Main scope");

  std::vector<std::string> scopes;
  for (uint64_t i = 0; i < nScopes; ++i) {
    scopes.push_back("Timing scope number #" + std::to_string(i));
  }

  uint64_t j = 0;
  for (uint64_t i = 0; i < nSwitches; ++i) {
    const auto sw = s.scopedStopwatch(scopes[i % nScopes]);
    j += (i * i % 3 + i * (i + 1));
  }

  std::cout << "Getting summary" << std::endl;

  std::cout << s.str(0.0) << std::endl;
}

} // namespace

int main() {

  scopeStopwatch0();
  scopeStopwatch1();
  moveScopeStopwatch0();
  testPercentage();
  testOrder0();
  rapidFireTest0(10, 100);

  return 0;
}
