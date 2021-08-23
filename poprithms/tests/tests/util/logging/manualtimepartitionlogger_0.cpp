// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <chrono>
#include <iostream>
#include <thread>

#include <poprithms/error/error.hpp>
#include <poprithms/logging/timepartitionlogger.hpp>

namespace {

using namespace poprithms::logging;

using Event = TimePartitionLogger::Event;
using Type  = TimePartitionLogger::EventType;

void summarizerTest0() {
  ManualTimePartitionLogger summarizer("myManualTimePartitionLogger");

  const auto sw0 = "first-sw";
  const auto sw1 = "my-chrometer";
  const auto sw2 = "second-sw";

  summarizer.start(sw0);
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  summarizer.stop();

  summarizer.start(sw1);
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  summarizer.stop();

  summarizer.start(sw2);
  std::this_thread::sleep_for(std::chrono::milliseconds(3));
  summarizer.stop();

  const auto events = summarizer.events();

  summarizer.verifyEvents({
      {sw0, Type::Start},
      {sw0, Type::Stop},
      {sw1, Type::Start},
      {sw1, Type::Stop},
      {sw2, Type::Start},
      {sw2, Type::Stop},
  });

  // Print the events to std::cout.
  std::cout << summarizer.str(0.0) << std::endl;
}

// A global summarizer is constructed and accessed in the various parts of the
// codebase.
ManualTimePartitionLogger &summarizer() {
  static ManualTimePartitionLogger s("global-time-partitioner");
  return s;
}

void part0() {
  summarizer().start("part0");
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  summarizer().stop();
}

void part1() {
  summarizer().start("part1");
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  summarizer().stop();
}

void globalTest() {
  part0();
  part1();
  part0();
  part0();
  part1();

  const auto t0 = summarizer().get("part0");
  const auto t1 = summarizer().get("part1");

  // we test with 1 millisecond margin for error. Note that for this to fail,
  // the recorded time would need to be less than the pause, which is
  // presumably impossible.
  if (t0 < 2e-3) {
    std::ostringstream oss;
    oss << "part0 ran for a total for 3 milliseconds, "
        << "incorrect time of " << t0 << '.';
    throw poprithms::test::error(oss.str());
  }

  if (t1 < 3e-3) {
    std::ostringstream oss;
    oss << "part1 ran for a total for 4 milliseconds, "
        << "incorrect time of " << t1 << '.';
    throw poprithms::test::error(oss.str());
  }

  std::cout << summarizer().str(0.0) << std::endl;
}

void noDoubleStart() {

  ManualTimePartitionLogger s("noDoubleStartTest");
  s.start("scope0");
  bool caught{false};
  try {
    s.start("scope1");
  } catch (const poprithms::error::error &err) {
    caught = true;
  }
  if (!caught) {
    throw poprithms::test::error(
        "Failed in test that start cannot be called without a stop");
  }
}

void noDoubleStop() {

  ManualTimePartitionLogger s("noDoubleStopTest");
  s.start("scope0");
  s.stop();
  bool caught{false};
  try {
    s.stop();
  } catch (const poprithms::error::error &err) {
    caught = true;
  }
  if (!caught) {
    throw poprithms::test::error(
        "Failed in test that stop cannot be called without start");
  }
}

void timeRegisteredBeforeStop() {
  ManualTimePartitionLogger s("foo3");
  s.start("a");
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  if (s.get("a") < 1e-3) {
    throw poprithms::test::error(
        "Stop should not be required to get accurate time measurement. ");
  }
}

void twoManualTimePartitionLoggersWithSameId() {

  // You can construct PartitionLoggers with the same name, unlike Loggers.
  ManualTimePartitionLogger a("new101");
  ManualTimePartitionLogger b("new101");
}

void testScopedStopwatch0() {

  ManualTimePartitionLogger l("scopeStowatchTest0");
  { auto a = l.scopedStopwatch("a"); }

  { auto b = l.scopedStopwatch("b"); }

  auto c = l.scopedStopwatch("c");

  l.verifyEvents({{"a", Type::Start},
                  {"a", Type::Stop},
                  {"b", Type::Start},
                  {"b", Type::Stop},
                  {"c", Type::Start}});
}

void testReservedNames() {

  ManualTimePartitionLogger l("x");
  for (auto n : {"Total", "total", "Unaccounted for"}) {
    l.start(n);
    l.stop();
  }

  if (l.get("Total") > l.sinceConstruction()) {
    throw poprithms::test::error(
        "User chose to have Total as one of their scopes, not "
        "working as expected");
  }

  l.verifyEvents({{"Total", Type::Start},
                  {"Total", Type::Stop},
                  {"total", Type::Start},
                  {"total", Type::Stop},
                  {"Unaccounted for", Type::Start},
                  {"Unaccounted for", Type::Stop}});
}

} // namespace

int main() {

  poprithms::logging::setGlobalLevel(poprithms::logging::Level::Info);
  summarizerTest0();
  globalTest();
  noDoubleStart();
  noDoubleStop();
  timeRegisteredBeforeStop();
  twoManualTimePartitionLoggersWithSameId();
  testScopedStopwatch0();
  testReservedNames();
  poprithms::logging::setGlobalLevel(poprithms::logging::Level::Off);
}
