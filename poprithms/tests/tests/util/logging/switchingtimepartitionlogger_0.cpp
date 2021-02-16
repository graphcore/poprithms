// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <chrono>
#include <iostream>
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
    for (auto c : x) {
      if (c == '%') {
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

} // namespace

int main() {

  scopeStopwatch0();
  scopeStopwatch1();
  testConstructors0();
  testPercentage();

  return 0;
}
