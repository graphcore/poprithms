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
  std::cout << testLogger().str(0.0) << std::endl;
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
} // namespace

int main() {

  scopeStopwatch0();
  scopeStopwatch1();

  return 0;
}
