// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>

#include <poprithms/logging/error.hpp>
#include <poprithms/logging/timepartitionlogger.hpp>

namespace {

using namespace poprithms::logging;
using Event = TimePartitionLogger::Event;
using Type  = Event::Type;

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
  if (countPerc(c.str(100.)) != 2) {
    std::ostringstream oss;
    oss << "Counting the number of scopes which have at least 100 percent. "
        << "Expected 2 : total and unaccounted, the 2 which always appear as "
        << "the threshold is not applied to them. The log is\n "
        << c.str(100.);
    throw error(oss.str());
  }

  if (countPerc(c.str(0.)) != 5) {
    std::ostringstream oss;
    oss << "Counting the number of scopes which have at least 0 percent. "
        << "Expected 5: total and unaccounted, the 2 which always appear as "
        << "the threshold is not applied to them, and a, b, and c. The log "
           "is\n "
        << c.str(0.);
    throw error(oss.str());
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

void testConstructors0() {

  SwitchingTimePartitionLogger a("x", false);

  // Construct another Logger with base id of "x", but extended to make
  // unique.
  SwitchingTimePartitionLogger b("x", true);
  if (b.id() == "x") {
    throw error("b should not have id x, it should have been extended");
  }

  bool caught{false};
  try {
    SwitchingTimePartitionLogger c("x", false);
  } catch (const poprithms::error::error &) {
    caught = true;
  }
  if (!caught) {
    throw error(
        std::string(
            "c should not be constructible, the name x is already taken ") +
        "and the option to extend the ID is false. ");
  }

  SwitchingTimePartitionLogger d;
  ManualTimePartitionLogger e;
  SwitchingTimePartitionLogger f;
  if (f.id() == d.id() || e.id() == d.id()) {
    throw error("d, e, and f should all have unique ids");
  }
}

void testOrder0() {
  SwitchingTimePartitionLogger watcher("aSwitchingLogger", true);

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
   *     foo_9              : 0.133324 [s]    :    18 %
   *     foo_5              : 0.122544 [s]    :    16 %
   *     foo_1              : 0.115147 [s]    :    15 %
   *     foo_6              : 0.092928 [s]    :    12 %
   *
   * We test that the times (second column) are sorted.
   */
  const auto summary = watcher.str(loggingPercentageThreshold);

  const auto x0 = summary.find('\n', 0);
  const auto x1 = summary.find('\n', x0 + 1);

  // Distance between lines
  const auto delta = x1 - x0;

  // range where the time (in seconds) is found.
  const auto y0 = summary.find(':', 0);
  const auto y1 = summary.find("[s]", y0);

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
    throw error("Times not sorted: " + summary);
  }
}

} // namespace

int main() {

  scopeStopwatch0();
  scopeStopwatch1();
  moveScopeStopwatch0();
  testConstructors0();
  testPercentage();
  testOrder0();

  return 0;
}
